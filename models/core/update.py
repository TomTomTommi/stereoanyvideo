from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from stereoanyvideo.models.core.attention import LoFTREncoderLayer


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead3D, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder3D(nn.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder3D, self).__init__()

        self.convc1 = nn.Conv3d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv3d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv3d(2, 128, 5, padding=2)
        self.convf2 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv = nn.Conv3d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU3D, self).__init__()
        self.convz1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convr1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convq1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )

        self.convz2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convr2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convq2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )

        self.convz3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convr3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convq3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # time
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SKSepConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SKSepConvGRU3D, self).__init__()
        self.convz1 = nn.Sequential(
            nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)),
        )
        self.convr1 = nn.Sequential(
            nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)),
        )
        self.convq1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )

        self.convz2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convr2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convq2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )

        self.convz3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convr3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convq3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # time
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8, attention_type=None):
        super(BasicUpdateBlock, self).__init__()
        self.attention_type = attention_type
        if attention_type is not None:
            if "update_time" in attention_type:
                self.time_attn = TimeAttnBlock(dim=256, num_heads=8)

            if "update_space" in attention_type:
                self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)

        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size ** 2 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True, t=1):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat((inp, motion_features), dim=1)
        if self.attention_type is not None:
            if "update_time" in self.attention_type:
                inp = self.time_attn(inp, T=t)
            if "update_space" in self.attention_type:
                inp = self.space_attn(inp, T=t)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        return x


class TimeAttnBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super(TimeAttnBlock, self).__init__()
        self.temporal_attn = Attention(dim, num_heads=8, qkv_bias=False, qk_scale=None)
        self.temporal_fc = nn.Linear(dim, dim)
        self.temporal_norm1 = nn.LayerNorm(dim)

        nn.init.constant_(self.temporal_fc.weight, 0)
        nn.init.constant_(self.temporal_fc.bias, 0)

    def forward(self, x, T=1):
        _, _, h, w = x.shape

        x = rearrange(x, "(b t) m h w -> (b h w) t m", h=h, w=w, t=T)
        res_temporal1 = self.temporal_attn(self.temporal_norm1(x))
        res_temporal1 = rearrange(
            res_temporal1, "(b h w) t m -> b (h w t) m", h=h, w=w, t=T
        )
        res_temporal1 = self.temporal_fc(res_temporal1)
        res_temporal1 = rearrange(
            res_temporal1, " b (h w t) m -> b t m h w", h=h, w=w, t=T
        )
        x = rearrange(x, "(b h w) t m -> b t m h w", h=h, w=w, t=T)
        x = x + res_temporal1
        x = rearrange(x, "b t m h w -> (b t) m h w", h=h, w=w, t=T)
        return x


class SpaceAttnBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super(SpaceAttnBlock, self).__init__()
        self.encoder_layer = LoFTREncoderLayer(dim, nhead=num_heads, attention="linear")

    def forward(self, x, T=1):
        _, _, h, w = x.shape
        x = rearrange(x, "(b t) m h w -> (b t) (h w) m", h=h, w=w, t=T)
        x = self.encoder_layer(x, x)
        x = rearrange(x, "(b t) (h w) m -> (b t) m h w", h=h, w=w, t=T)
        return x


class SequenceUpdateBlock3D(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8):
        super(SequenceUpdateBlock3D, self).__init__()

        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SKSepConvGRU3D(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead3D(hidden_dim, hidden_dim=256)
        self.mask3d = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim + 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim + 128, (mask_size ** 2) * (3 * 3 * 3), 1, padding=0),
        )
        self.time_attn = TimeAttnBlock(dim=256, num_heads=8)
        self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)

    def forward(self, net, inp, corrs, flows, t):
        motion_features = self.encoder(flows, corrs)
        inp_tensor = torch.cat([inp, motion_features], dim=1)

        inp_tensor = self.time_attn(inp_tensor, T=t)
        inp_tensor = self.space_attn(inp_tensor, T=t)

        net = rearrange(net, "(b t) c h w -> b c t h w", t=t)
        inp_tensor = rearrange(inp_tensor, "(b t) c h w -> b c t h w", t=t)

        net = self.gru(net, inp_tensor)

        delta_flow = self.flow_head(net)

        # scale mask to balance gradients
        mask = 0.25 * self.mask3d(net)
        net = rearrange(net, " b c t h w -> (b t) c h w")
        mask =rearrange(mask, " b c t h w -> (b t) c h w")
        delta_flow = rearrange(delta_flow, " b c t h w -> (b t) c h w")
        return net, mask, delta_flow