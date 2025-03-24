import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List
from einops import rearrange
import collections
from collections import defaultdict
from itertools import repeat
import unfoldNd

from stereoanyvideo.models.core.update import SequenceUpdateBlock3D
from stereoanyvideo.models.core.extractor import BasicEncoder, MultiBasicEncoder, DepthExtractor
from stereoanyvideo.models.core.corr import AAPC
from stereoanyvideo.models.core.utils.utils import InputPadder, interp

autocast = torch.cuda.amp.autocast

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class StereoAnyVideo(nn.Module):
    def __init__(self, mixed_precision=False):
        super(StereoAnyVideo, self).__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = 128
        self.context_dim = 128
        self.dropout = 0

        # feature network and update block
        self.cnet = BasicEncoder(output_dim=96, norm_fn='instance', dropout=self.dropout)
        self.fnet = BasicEncoder(output_dim=96, norm_fn='instance', dropout=self.dropout)
        self.depthnet = DepthExtractor()
        self.corr_mlp = Mlp(in_features=4 * 9 * 9, hidden_features=256, out_features=128)
        self.update_block = SequenceUpdateBlock3D(hidden_dim=self.hidden_dim, cor_planes=128, mask_size=4)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"time_embed"}

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def convex_upsample(self, flow, mask, rate=4):
        """ Upsample flow field [H/rate, W/rate, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def convex_upsample_3D(self, flow, mask, b, T, rate=4):
        """Upsample flow field from [T, H/rate, W/rate, 2] to [T, H, W, 2] using convex combination.

        unfoldNd repo: https://github.com/f-dangel/unfoldNd
        Run: pip install --user unfoldNd

        Args:
            flow: (N*T, C_flow, H, W)
            mask: (N*T, C_mask, H, W) or (N, 1, 27, 1, rate, rate, T, H, W)
            rate: int
        """
        flow = rearrange(flow, "(b t) c h w -> b c t h w", b=b, t=T)
        mask = rearrange(mask, "(b t) c h w -> b c t h w", b=b, t=T)

        N, _, T, H, W = flow.shape

        mask = mask.view(N, 1, 27, 1, rate, rate, T, H, W)  # (N, 1, 27, rate, rate, rate, T, H, W) if upsample T
        mask = torch.softmax(mask, dim=2)

        upsample = unfoldNd.UnfoldNd([3, 3, 3], padding=1)
        flow_upsampled = upsample(rate * flow)
        flow_upsampled = flow_upsampled.view(N, 2, 27, 1, 1, 1, T, H, W)
        flow_upsampled = torch.sum(mask * flow_upsampled, dim=2)
        flow_upsampled = flow_upsampled.permute(0, 1, 5, 2, 6, 3, 7, 4)
        flow_upsampled = flow_upsampled.reshape(N, 2, T, rate * H,
                                                rate * W)  # (N, 2, rate*T, rate*H, rate*W) if upsample T
        up_flow = rearrange(flow_upsampled, "b c t h w -> (b t) c h w")

        return up_flow

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        flow_u = torch.zeros([N, 1, H, W], dtype=torch.float)
        flow_v = torch.zeros([N, 1, H, W], dtype=torch.float)
        flow = torch.cat([flow_u, flow_v], dim=1).to(fmap.device)
        return flow

    def forward_batch_test(
        self, batch_dict, iters = 24, flow_init=None,
    ):
        kernel_size = 20
        stride = kernel_size // 2
        predictions = defaultdict(list)

        disp_preds = []
        video = batch_dict["stereo_video"]

        num_ims = len(video)
        print("video", video.shape)

        for i in range(0, num_ims, stride):
            left_ims = video[i : min(i + kernel_size, num_ims), 0]
            padder = InputPadder(left_ims.shape, divis_by=32)
            right_ims = video[i : min(i + kernel_size, num_ims), 1]
            left_ims, right_ims = padder.pad(left_ims, right_ims)
            if flow_init is not None:
                flow_init_ims = flow_init[i: min(i + kernel_size, num_ims)]
                flow_init_ims = padder.pad(flow_init_ims)[0]
                with autocast(enabled=self.mixed_precision):
                    disparities_forw = self.forward(
                        left_ims[None].cuda(),
                        right_ims[None].cuda(),
                        flow_init=flow_init_ims,
                        iters=iters,
                        test_mode=True,
                    )
            else:
                with autocast(enabled=self.mixed_precision):
                    disparities_forw = self.forward(
                        left_ims[None].cuda(),
                        right_ims[None].cuda(),
                        iters=iters,
                        test_mode=True,
                    )

            disparities_forw = padder.unpad(disparities_forw[:, 0])[:, None].cpu()

            if len(disp_preds) > 0 and len(disparities_forw) >= stride:

                if len(disparities_forw) < kernel_size:
                    disp_preds.append(disparities_forw[stride // 2 :])
                else:
                    disp_preds.append(disparities_forw[stride // 2 : -stride // 2])

            elif len(disp_preds) == 0:
                disp_preds.append(disparities_forw[: -stride // 2])

        predictions["disparity"] = (torch.cat(disp_preds).squeeze(1).abs())[:, :1]

        return predictions

    def forward(self, image1, image2, flow_init=None, iters=12, test_mode=False):
        b, T, c, h, w = image1.shape

        image1 = image1 / 255.0
        image2 = image2 / 255.0

        # Normalize using mean and std for ImageNet pre-trained models
        mean = torch.tensor([0.485, 0.456, 0.406], device=image1.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image1.device).view(1, 1, 3, 1, 1)

        image1 = (image1 - mean) / std
        image2 = (image2 - mean) / std
        image1 = image1.float()
        image2 = image2.float()

        # feature network
        with autocast(enabled=self.mixed_precision):
            fmap1_depth_feature = self.depthnet(image1)
            fmap2_depth_feature = self.depthnet(image2)
            fmap1_cnet_feature = self.cnet(image1.flatten(0, 1)).unflatten(0, (b, T))
            fmap1_fnet_feature = self.fnet(image1.flatten(0, 1)).unflatten(0, (b, T))
            fmap2_fnet_feature = self.fnet(image2.flatten(0, 1)).unflatten(0, (b, T))

        fmap1 = torch.cat((fmap1_depth_feature, fmap1_fnet_feature), dim=2).flatten(0, 1)
        fmap2 = torch.cat((fmap2_depth_feature, fmap2_fnet_feature), dim=2).flatten(0, 1)

        context = torch.cat((fmap1_depth_feature, fmap1_cnet_feature), dim=2).flatten(0, 1)

        with autocast(enabled=self.mixed_precision):
            net = torch.tanh(context)
            inp = torch.relu(context)

            s_net = F.avg_pool2d(net, 2, stride=2)
            s_inp = F.avg_pool2d(inp, 2, stride=2)

            # 1/4 -> 1/8
            # feature
            s_fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            s_fmap2 = F.avg_pool2d(fmap2, 2, stride=2)

            # 1/4 -> 1/16
            # feature
            ss_fmap1 = F.avg_pool2d(fmap1, 4, stride=4)
            ss_fmap2 = F.avg_pool2d(fmap2, 4, stride=4)

            ss_net = F.avg_pool2d(net, 4, stride=4)
            ss_inp = F.avg_pool2d(inp, 4, stride=4)

        # Correlation
        corr_fn = AAPC(fmap1, fmap2)
        s_corr_fn = AAPC(s_fmap1, s_fmap2)
        ss_corr_fn = AAPC(ss_fmap1, ss_fmap2)

        # cascaded refinement (1/16 + 1/8 + 1/4)
        flow_predictions = []
        flow = None
        flow_up = None

        if flow_init is not None:
            flow_init = flow_init.cuda()
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = scale * interp(flow_init, size=(fmap1.shape[2], fmap1.shape[3]))
        else:
            # init flow
            ss_flow = self.zero_init(ss_fmap1)

            # 1/16
            for itr in range(iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                ss_flow = ss_flow.detach()
                out_corrs = ss_corr_fn(ss_flow, None, small_patch=small_patch)  # 36 * H/16 * W/16
                out_corrs = self.corr_mlp(out_corrs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                with autocast(enabled=self.mixed_precision):
                    ss_net, up_mask, delta_flow = self.update_block(ss_net, ss_inp, out_corrs, ss_flow, t=T)

                ss_flow = ss_flow + delta_flow
                flow = self.convex_upsample_3D(ss_flow, up_mask, b, T, rate=4)  # 2 * H/4 * W/4
                flow_up = 4 * F.interpolate(flow, size=(4 * flow.shape[2], 4 * flow.shape[3]), mode='bilinear',
                                            align_corners=True)  # 2 * H/2 * W/2
                flow_predictions.append(flow_up[:, :1])

            scale = s_fmap1.shape[2] / flow.shape[2]
            s_flow = scale * interp(flow, size=(s_fmap1.shape[2], s_fmap1.shape[3]))

            # 1/8
            for itr in range(iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                s_flow = s_flow.detach()
                out_corrs = s_corr_fn(s_flow, None, small_patch=small_patch)
                out_corrs = self.corr_mlp(out_corrs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                with autocast(enabled=self.mixed_precision):
                    s_net, up_mask, delta_flow = self.update_block(s_net, s_inp, out_corrs, s_flow, t=T)

                s_flow = s_flow + delta_flow
                flow = self.convex_upsample_3D(s_flow, up_mask, b, T, rate=4)
                flow_up = 2 * F.interpolate(flow, size=(2 * flow.shape[2], 2 * flow.shape[3]), mode='bilinear',
                                            align_corners=True)
                flow_predictions.append(flow_up[:, :1])

            scale = fmap1.shape[2] / flow.shape[2]
            flow = scale * interp(flow, size=(fmap1.shape[2], fmap1.shape[3]))

        # 1/4
        for itr in range(iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = corr_fn(flow, None, small_patch=small_patch)
            out_corrs = self.corr_mlp(out_corrs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow, t=T)

            flow = flow + delta_flow
            flow_up = self.convex_upsample_3D(flow, up_mask, b, T, rate=4)
            flow_predictions.append(flow_up[:, :1])

        predictions = torch.stack(flow_predictions)
        predictions = rearrange(predictions, "d (b t) c h w -> d t b c h w", b=b, t=T)
        flow_up = predictions[-1]

        if test_mode:
            return flow_up

        return predictions


