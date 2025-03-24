from dataclasses import dataclass
from typing import Dict, Optional, Union
from stereoanyvideo.evaluation.utils.ssim import SSIM
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2
from pytorch3d.utils import opencv_from_cameras_projection
from stereoanyvideo.models.raft_model import RAFTModel


@dataclass(eq=True, frozen=True)
class PerceptionMetric:
    metric: str
    depth_scaling_norm: Optional[str] = None
    suffix: str = ""
    index: str = ""

    def __str__(self):
        return (
            self.metric
            + self.index
            + (
                ("_norm_" + self.depth_scaling_norm)
                if self.depth_scaling_norm is not None
                else ""
            )
            + self.suffix
        )


def compute_flow(seq, is_seq=True):
    raft = RAFTModel().cuda()
    raft.eval()
    if is_seq:
        t, c, h, w = seq.size()
        flows_forward = []
        for i in range(t-1):
            flow_forward = raft.forward_fullres(seq[i][None], seq[i+1][None], iters=20)
            flows_forward.append(flow_forward)
        flows_forward = torch.cat(flows_forward, dim=0)
        return flows_forward

    else:
        img1, img2 = seq
        flow_forward = raft.forward_fullres(img1, img2, iters=20)
        return flow_forward

def flow_warp(x, flow):
    if flow.size(3) != 2:  # [B, H, W, 2]
        flow = flow.permute(0, 2, 3, 1)
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)
    return output

def eval_endpoint_error_sequence(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    crop: int = 0,
    mask_thr: float = 0.5,
    clamp_thr: float = 1e-5,
) -> Dict[str, torch.Tensor]:

    assert len(x.shape) == len(y.shape) == len(mask.shape) == 4, (
        x.shape,
        y.shape,
        mask.shape,
    )
    assert x.shape[0] == y.shape[0] == mask.shape[0], (x.shape, y.shape, mask.shape)

    # chuck out the border
    if crop > 0:
        if crop > min(y.shape[2:]) - crop:
            raise ValueError("Incorrect crop size.")
        y = y[:, :, crop:-crop, crop:-crop]
        x = x[:, :, crop:-crop, crop:-crop]
        mask = mask[:, :, crop:-crop, crop:-crop]

    y = y * (mask > mask_thr).float()
    x = x * (mask > mask_thr).float()
    y[torch.isnan(y)] = 0

    results = {}
    for epe_name in ("epe", "temp_epe"):
        if epe_name == "epe":
            endpoint_error = (mask * (x - y) ** 2).sum(dim=1).sqrt()
        elif epe_name == "temp_epe":
            delta_mask = mask[:-1] * mask[1:]
            # endpoint_error = (
            #     (delta_mask * ((x[:-1] - x[1:]) - (y[:-1] - y[1:])) ** 2)
            #     .sum(dim=1)
            #     .sqrt()
            # )
            endpoint_error = (
                (delta_mask * ((x[:-1] - x[1:]).abs() - (y[:-1] - y[1:]).abs()) ** 2)
                .sum(dim=1)
                .sqrt()
            )

        # epe_nonzero = endpoint_error != 0
        nonzero = torch.count_nonzero(endpoint_error)
        epe_mean = endpoint_error.sum() / torch.clamp(
            nonzero, clamp_thr
        )  # average error for all the sequence pixels
        epe_inv_accuracy_05px = (endpoint_error > 0.5).sum() / torch.clamp(
            nonzero, clamp_thr
        )
        epe_inv_accuracy_1px = (endpoint_error > 1).sum() / torch.clamp(
            nonzero, clamp_thr
        )
        epe_inv_accuracy_2px = (endpoint_error > 2).sum() / torch.clamp(
            nonzero, clamp_thr
        )
        epe_inv_accuracy_3px = (endpoint_error > 3).sum() / torch.clamp(
            nonzero, clamp_thr
        )

        results[f"{epe_name}_mean"] = epe_mean[None]
        results[f"{epe_name}_bad_0.5px"] = epe_inv_accuracy_05px[None] * 100
        results[f"{epe_name}_bad_1px"] = epe_inv_accuracy_1px[None] * 100
        results[f"{epe_name}_bad_2px"] = epe_inv_accuracy_2px[None] * 100
        results[f"{epe_name}_bad_3px"] = epe_inv_accuracy_3px[None] * 100
    return results


def eval_TCC_sequence(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    crop: int = 0,
    mask_thr: float = 0.5,
) -> Dict[str, torch.Tensor]:

    assert len(x.shape) == len(y.shape) == len(mask.shape) == 4, (
        x.shape,
        y.shape,
        mask.shape,
    )
    assert x.shape[0] == y.shape[0] == mask.shape[0], (x.shape, y.shape, mask.shape)
    t, c, h, w = x.shape
    # chuck out the border
    if crop > 0:
        if crop > min(y.shape[2:]) - crop:
            raise ValueError("Incorrect crop size.")
        y = y[:, :, crop:-crop, crop:-crop]
        x = x[:, :, crop:-crop, crop:-crop]
        mask = mask[:, :, crop:-crop, crop:-crop]

    y = y * (mask > mask_thr).float()
    x = x * (mask > mask_thr).float()
    x[torch.isnan(x)] = 0
    y[torch.isnan(y)] = 0

    ssim_loss = SSIM(1.0, nonnegative_ssim=True)
    delta_mask = mask[:-1] * mask[1:]

    tcc = 0
    for i in range(t-1):
        tcc += ssim_loss((torch.abs(x[i][None] - x[i+1][None]) * delta_mask[i]).expand(-1, 3, -1, -1),
                          (torch.abs(y[i][None] - y[i+1][None]) * delta_mask[i]).expand(-1, 3, -1, -1))
    tcc = tcc / (t-1)

    return tcc

def eval_TCM_sequence(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    crop: int = 0,
    mask_thr: float = 0.5,
) -> Dict[str, torch.Tensor]:

    assert len(x.shape) == len(y.shape) == len(mask.shape) == 4, (
        x.shape,
        y.shape,
        mask.shape,
    )
    assert x.shape[0] == y.shape[0] == mask.shape[0], (x.shape, y.shape, mask.shape)

    t, c, h, w = x.shape
    # chuck out the border
    if crop > 0:
        if crop > min(y.shape[2:]) - crop:
            raise ValueError("Incorrect crop size.")
        y = y[:, :, crop:-crop, crop:-crop]
        x = x[:, :, crop:-crop, crop:-crop]
        mask = mask[:, :, crop:-crop, crop:-crop]

    y = y * (mask > mask_thr).float()
    x = x * (mask > mask_thr).float()
    y[torch.isnan(y)] = 0

    ssim_loss = SSIM(1.0, nonnegative_ssim=True, size_average=False)
    delta_mask = mask[:-1] * mask[1:]

    tcm = 0
    for i in range(t-1):
        dmax = torch.max(y[i][None].view(1, -1), -1)[0].view(1, 1, 1, 1).expand(-1, 3, -1, -1)
        dmin = torch.min(y[i][None].view(1, -1), -1)[0].view(1, 1, 1, 1).expand(-1, 3, -1, -1)

        x_norm = (x[i][None].expand(-1, 3, -1, -1) - dmin) / (dmax - dmin) * 255.
        x2_norm = (x[i+1][None].expand(-1, 3, -1, -1) - dmin) / (dmax - dmin) * 255.
        x_flow = compute_flow([x_norm.cuda(), x2_norm.cuda()], is_seq=False).cpu()

        y_norm = (y[i][None].expand(-1, 3, -1, -1) - dmin) / (dmax - dmin) * 255.
        y2_norm = (y[i+1][None].expand(-1, 3, -1, -1) - dmin) / (dmax - dmin) * 255.
        y_flow = compute_flow([y_norm.cuda(), y2_norm.cuda()], is_seq=False).cpu()

        flow_mask = torch.sum(y_flow > 250, 1, keepdim=True) == 0

        mask = delta_mask[i][None] * flow_mask
        mask = mask.expand(-1, 3, -1, -1)
        if torch.sum(mask) > 0:
            tcm += torch.mean(ssim_loss(
                torch.cat((x_flow, torch.ones_like(x_flow[:, 0, None, ...])), 1) * mask,
                torch.cat((y_flow, torch.ones_like(x_flow[:, 0, None, ...])), 1) * mask)[:, :2])
    tcm = tcm / (t-1)

    return tcm


def eval_OPW_sequence(
    img: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    crop: int = 0,
    mask_thr: float = 0.5,
    clamp_thr: float = 1e-5,
) -> Dict[str, torch.Tensor]:

    assert len(x.shape) == len(y.shape) == len(mask.shape) == 4, (
        x.shape,
        y.shape,
        mask.shape,
    ) # T, 1, H, W
    assert x.shape[0] == y.shape[0] == mask.shape[0], (x.shape, y.shape, mask.shape)

    t, c, h, w = img[:, 0].shape
    # chuck out the border
    if crop > 0:
        if crop > min(y.shape[2:]) - crop:
            raise ValueError("Incorrect crop size.")
        y = y[:, :, crop:-crop, crop:-crop]
        x = x[:, :, crop:-crop, crop:-crop]
        mask = mask[:, :, crop:-crop, crop:-crop]

    y = y * (mask > mask_thr).float()
    x = x * (mask > mask_thr).float()
    y[torch.isnan(y)] = 0
    delta_mask = mask[:-1] * mask[1:]
    depth_mask_30 = torch.sum(y > 30, 1, keepdim=True) == 0
    depth_mask_30 = depth_mask_30[:-1] * depth_mask_30[1:]
    depth_mask_50 = torch.sum(y > 50, 1, keepdim=True) == 0
    depth_mask_50 = depth_mask_50[:-1] * depth_mask_50[1:]
    depth_mask_100 = torch.sum(y > 100, 1, keepdim=True) == 0
    depth_mask_100 = depth_mask_100[:-1] * depth_mask_100[1:]

    flow = compute_flow(img[:, 0].cuda()).cpu()
    warped_disp = flow_warp(x[1:], flow)
    warped_img = flow_warp(img[:, 0][1:].float(), flow)

    flow_mask = torch.sum(flow > 250, 1, keepdim=True) == 0

    delta_mask = delta_mask * torch.exp(-50. * torch.sqrt(
        ((warped_img / 255. - img[:, 0][:-1].float() / 255.) ** 2).sum(dim=1, keepdim=True))) * flow_mask * (
                          warped_disp > 0) > 1e-2
    opw_err = torch.abs(warped_disp - x[:-1]) * delta_mask
    opw_err_30 = torch.abs(warped_disp - x[:-1]) * delta_mask * depth_mask_30
    opw_err_50 = torch.abs(warped_disp - x[:-1]) * delta_mask * depth_mask_50
    opw_err_100 = torch.abs(warped_disp - x[:-1]) * delta_mask * depth_mask_100

    opw = 0
    opw_30 = 0
    opw_50 = 0
    opw_100 = 0
    for i in range(t-1):
        if torch.sum(delta_mask[i]) > 0:
            opw += torch.sum(opw_err[i]) / torch.sum(delta_mask[i])
        if torch.sum(delta_mask[i] * depth_mask_30[i]) > 0:
            opw_30 += torch.sum(opw_err_30[i]) / torch.sum(delta_mask[i] * depth_mask_30[i])
        if torch.sum(delta_mask[i] * depth_mask_50[i]) > 0:
            opw_50 += torch.sum(opw_err_50[i]) / torch.sum(delta_mask[i] * depth_mask_50[i])
        if torch.sum(delta_mask[i] * depth_mask_100[i]) > 0:
            opw_100 += torch.sum(opw_err_100[i]) / torch.sum(delta_mask[i] * depth_mask_100[i])
    opw = opw / (t - 1)
    opw_30 = opw_30 / (t - 1)
    opw_50 = opw_50 / (t - 1)
    opw_100 = opw_100 / (t - 1)
    return opw, opw_30, opw_50, opw_100


def eval_RTC_sequence(
    img: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    crop: int = 0,
    mask_thr: float = 0.5,
    clamp_thr: float = 1e-5,
) -> Dict[str, torch.Tensor]:

    assert len(x.shape) == len(y.shape) == len(mask.shape) == 4, (
        x.shape,
        y.shape,
        mask.shape,
    ) # T, 1, H, W
    assert x.shape[0] == y.shape[0] == mask.shape[0], (x.shape, y.shape, mask.shape)

    t, c, h, w = img[:, 0].shape
    # chuck out the border
    if crop > 0:
        if crop > min(y.shape[2:]) - crop:
            raise ValueError("Incorrect crop size.")
        y = y[:, :, crop:-crop, crop:-crop]
        x = x[:, :, crop:-crop, crop:-crop]
        mask = mask[:, :, crop:-crop, crop:-crop]

    y = y * (mask > mask_thr).float()
    x = x * (mask > mask_thr).float()
    y[torch.isnan(y)] = 0

    flow = compute_flow(img[:, 0].cuda()).cpu()
    delta_mask = mask[:-1] * mask[1:]

    warped_disp = flow_warp(x[1:], flow)
    warped_img = flow_warp(img[:, 0][1:], flow)

    flow_mask = torch.sum(flow > 250, 1, keepdim=True) == 0
    depth_mask = torch.sum(y > 30, 1, keepdim=True) == 0
    depth_mask = depth_mask[:-1] * depth_mask[1:]

    delta_mask = delta_mask * torch.exp(-50. * torch.sqrt(
        ((warped_img / 255. - img[:, 0][:-1] / 255.) ** 2).sum(dim=1, keepdim=True))) * flow_mask * (
                         warped_disp > 0) > 1e-2
    tau = 1.01

    x1 = x[:-1]  / warped_disp
    x2 = warped_disp / x[:-1]

    x1[torch.isinf(x1)] = -1e10
    x2[torch.isinf(x2)] = -1e10
    x = torch.max(torch.cat((x1, x2), 1), 1)[0] < tau

    rtc_err = x[:, None] * delta_mask
    rtc_err_30 = x[:, None] * delta_mask * depth_mask
    rtc = 0
    rtc_30 = 0
    for i in range(t-1):
        if torch.sum(delta_mask[i]) > 0:
            rtc += torch.sum(rtc_err[i]) / torch.sum(delta_mask[i])
        if torch.sum(delta_mask[i] * depth_mask[i]) > 0:
            rtc_30 += torch.sum(rtc_err_30[i]) / torch.sum(delta_mask[i] * depth_mask[i])
    rtc = rtc / (t-1)
    rtc_30 = rtc_30 / (t - 1)
    return rtc, rtc_30


def depth2disparity_scale(left_camera, right_camera, image_size_tensor):
    # # opencv camera matrices
    (_, T1, K1), (_, T2, _) = [
        opencv_from_cameras_projection(
            f,
            image_size_tensor,
        )
        for f in (left_camera, right_camera)
    ]
    fix_baseline = T1[0][0] - T2[0][0]
    focal_length_px = K1[0][0][0]
    # following this https://github.com/princeton-vl/RAFT-Stereo#converting-disparity-to-depth
    return focal_length_px * fix_baseline


def depth_to_pcd(
    depth_map,
    img,
    focal_length,
    cx,
    cy,
    step: int = None,
    inv_extrinsic=None,
    mask=None,
    filter=False,
):
    __, w, __ = img.shape
    if step is None:
        step = int(w / 100)
    Z = depth_map[::step, ::step]
    colors = img[::step, ::step, :]

    Pixels_Y = torch.arange(Z.shape[0]).to(Z.device) * step
    Pixels_X = torch.arange(Z.shape[1]).to(Z.device) * step

    X = (Pixels_X[None, :] - cx) * Z / focal_length
    Y = (Pixels_Y[:, None] - cy) * Z / focal_length

    inds = Z > 0

    if mask is not None:
        inds = inds * (mask[::step, ::step] > 0)

    X = X[inds].reshape(-1)
    Y = Y[inds].reshape(-1)
    Z = Z[inds].reshape(-1)
    colors = colors[inds]
    pcd = torch.stack([X, Y, Z]).T

    if inv_extrinsic is not None:
        pcd_ext = torch.vstack([pcd.T, torch.ones((1, pcd.shape[0])).to(Z.device)])
        pcd = (inv_extrinsic @ pcd_ext)[:3, :].T

    if filter:
        pcd, filt_inds = filter_outliers(pcd)
        colors = colors[filt_inds]
    return pcd, colors


def filter_outliers(pcd, sigma=3):
    mean = pcd.mean(0)
    std = pcd.std(0)
    inds = ((pcd - mean).abs() < sigma * std)[:, 2]
    pcd = pcd[inds]
    return pcd, inds


def eval_batch(batch_dict, predictions, scale) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Produce performance metrics for a single batch of perception
    predictions.
    Args:
        frame_data: A PixarFrameData object containing the input to the new view
            synthesis method.
        preds: A PerceptionPrediction object with the predicted data.
    Returns:
        results: A dictionary holding evaluation metrics.
    """
    results = {}

    assert "disparity" in predictions
    mask_now = torch.ones_like(batch_dict["fg_mask"])

    mask_now = mask_now * batch_dict["disparity_mask"]

    eval_flow_traj_output = eval_endpoint_error_sequence(
        predictions["disparity"], batch_dict["disparity"], mask_now
    )
    for epe_name in ("epe", "temp_epe"):
        results[PerceptionMetric(f"disp_{epe_name}_mean")] = eval_flow_traj_output[
            f"{epe_name}_mean"
        ]

        results[PerceptionMetric(f"disp_{epe_name}_bad_3px")] = eval_flow_traj_output[
            f"{epe_name}_bad_3px"
        ]

        results[PerceptionMetric(f"disp_{epe_name}_bad_2px")] = eval_flow_traj_output[
            f"{epe_name}_bad_2px"
        ]

        results[PerceptionMetric(f"disp_{epe_name}_bad_1px")] = eval_flow_traj_output[
            f"{epe_name}_bad_1px"
        ]

        results[PerceptionMetric(f"disp_{epe_name}_bad_0.5px")] = eval_flow_traj_output[
            f"{epe_name}_bad_0.5px"
        ]
    if "endpoint_error_per_pixel" in eval_flow_traj_output:
        results["disp_endpoint_error_per_pixel"] = eval_flow_traj_output[
            "endpoint_error_per_pixel"
        ]

    # disparity to depth
    depth = scale / predictions["disparity"].clamp(min=1e-10)

    eval_TCC_output = eval_TCC_sequence(
        depth, scale / batch_dict["disparity"].clamp(min=1e-10), mask_now
    )
    results[PerceptionMetric("disp_TCC")] = eval_TCC_output[None]

    eval_TCM_output = eval_TCM_sequence(
        depth, scale / batch_dict["disparity"].clamp(min=1e-10), mask_now
    )
    results[PerceptionMetric("disp_TCM")] = eval_TCM_output[None]

    eval_OPW_output, eval_OPW_30_output, eval_OPW_50_output, eval_OPW_100_output = eval_OPW_sequence(
        batch_dict["stereo_video"], depth, scale / batch_dict["disparity"].clamp(min=1e-10), mask_now
    )
    results[PerceptionMetric("disp_OPW")] = eval_OPW_output[None]
    results[PerceptionMetric("disp_OPW_100")] = eval_OPW_100_output[None]
    results[PerceptionMetric("disp_OPW_50")] = eval_OPW_50_output[None]
    if eval_OPW_30_output > 0:
        results[PerceptionMetric("disp_OPW_30")] = eval_OPW_30_output[None]
    else:
        results[PerceptionMetric("disp_OPW_30")] = torch.tensor([0.0])

    eval_RTC_output, eval_RTC_30_output = eval_RTC_sequence(
        batch_dict["stereo_video"], depth, scale / batch_dict["disparity"].clamp(min=1e-10), mask_now
    )
    results[PerceptionMetric("disp_RTC")] = eval_RTC_output[None]
    if eval_RTC_30_output > 0:
        results[PerceptionMetric("disp_RTC_30")] = eval_RTC_30_output[None]
    else:
        results[PerceptionMetric("disp_RTC_30")] = torch.tensor([0.0])

    return (results, len(predictions["disparity"]))
