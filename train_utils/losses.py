import torch
from einops import rearrange
import torch.nn.functional as F


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """Loss function defined over sequence of flow predictions"""
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt().unsqueeze(1)

    if len(valid.shape) != len(flow_gt.shape):
        valid = valid.unsqueeze(1)

    valid = (valid >= 0.5) & (mag < max_flow)

    if valid.shape != flow_gt.shape:
        valid = torch.cat([valid, valid], dim=1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert (
            not torch.isnan(flow_preds[i]).any()
            and not torch.isinf(flow_preds[i]).any()
        )
        if n_predictions == 1:
            i_weight = 1
        else:
            # We adjust the loss_gamma so it is consistent for any number of iterations
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

        flow_pred = flow_preds[i].clone()
        if valid.shape[1] == 1 and flow_preds[i].shape[1] == 2:
            flow_pred = flow_pred[:, :1]

        i_loss = (flow_pred - flow_gt).abs()

        assert i_loss.shape == valid.shape, [
            i_loss.shape,
            valid.shape,
            flow_gt.shape,
            flow_pred.shape,
        ]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    valid = valid[:, 0]
    epe = epe.view(-1)
    epe = epe[valid.reshape(epe.shape)]

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics


def temporal_loss(flow_preds, flow_preds2, flow_gt, flow_gt2, valid, loss_gamma=0.9, max_flow=700):
    assert len(flow_preds) == len(flow_preds2)
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt().unsqueeze(1)

    if len(valid.shape) != len(flow_gt.shape):
        valid = valid.unsqueeze(1)

    valid = (valid >= 0.5) & (mag < max_flow)
    if valid.shape != flow_gt.shape:
        valid = torch.cat([valid, valid], dim=1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert (
            not torch.isnan(flow_preds[i]).any()
            and not torch.isinf(flow_preds[i]).any()
        )
        assert (
                not torch.isnan(flow_preds2[i]).any()
                and not torch.isinf(flow_preds2[i]).any()
        )
        if n_predictions == 1:
            i_weight = 1
        else:
            # We adjust the loss_gamma so it is consistent for any number of iterations
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

        flow_pred = flow_preds[i].clone()
        flow_pred2 = flow_preds2[i].clone()
        if valid.shape[1] == 1 and flow_preds[i].shape[1] == 2:
            flow_pred = flow_pred[:, :1]
            flow_pred2 = flow_pred2[:, :1]
        i_loss = ((flow_pred2 - flow_pred).abs() - (flow_gt2 - flow_gt).abs()).abs()

        assert i_loss.shape == valid.shape, [
            i_loss.shape,
            valid.shape,
            flow_gt.shape,
            flow_pred.shape,
        ]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    tepe = torch.sum(((flow_preds2[-1] - flow_preds[-1]) -  - (flow_gt2 - flow_gt)) ** 2, dim=1).sqrt()

    mask = (flow_gt2 - flow_gt) < 5
    valid = mask * valid
    valid = valid[:, 0]
    tepe = tepe.view(-1)
    tepe = tepe[valid.reshape(tepe.shape)]

    metrics = {
        "tepe": tepe.mean().item(),
    }
    return flow_loss, metrics

def compute_flow(Flow_Model, seq):
    n, t, c, h, w = seq.size()
    flows_forward = []
    flows_backward = []
    for i in range(t-1):
        # i-th flow_backward denotes seq[i+1] towards seq[i]
        flow_backward = Flow_Model.forward_fullres(seq[:,i], seq[:,i+1])
        # i-th flow_forward denotes seq[i] towards seq[i+1]
        flow_forward = Flow_Model.forward_fullres(seq[:,i+1], seq[:,i])
        flows_backward.append(flow_backward)
        flows_forward.append(flow_forward)
    flows_forward = torch.stack(flows_forward, dim=1)
    flows_backward = torch.stack(flows_backward, dim=1)

    return flows_forward, flows_backward


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


def bidirectional_alignment(seq, flows_backward, flows_forward):
    b, T, *_ = seq.shape

    # seq_backward = seq[:, 1:, ...]
    # seq_forward = seq[:, :T - 1, ...]
    # seq_backward = rearrange(seq_backward, "b t c h w -> (b t) c h w")
    # seq_forward = rearrange(seq_forward, "b t c h w -> (b t) c h w")
    # flows_forward = rearrange(flows_forward, "b t c h w -> (b t) c h w")
    # flows_backward = rearrange(flows_backward, "b t c h w -> (b t) c h w")
    # seq_backward = flow_warp(seq_backward, flows_backward)
    # seq_forward = flow_warp(seq_forward, flows_forward)
    # seq_backward = rearrange(seq_backward, "(b t) c h w -> b t c h w", b=b, t=T - 1)
    # seq_forward = rearrange(seq_forward, "(b t) c h w -> b t c h w", b=b, t=T - 1)
    # output_backward = torch.cat((seq_backward, seq[:, -1:]), dim=1)
    # output_forward = torch.cat((seq[:, :1], seq_forward), dim=1)

    output_backward = []
    for i in range(1, T):
        feat_prop = flow_warp(seq[:, i], flows_backward[:, i-1])
        output_backward.append(feat_prop)
    output_backward.append(seq[:, T - 1])
    output_backward = torch.stack(output_backward, dim=1)

    # forward-time process
    output_forward = [seq[:, 0]]
    for i in range(T - 1):
        feat_prop = flow_warp(seq[:, i], flows_forward[:, i])
        output_forward.append(feat_prop)
    output_forward = torch.stack(output_forward, dim=1)

    return output_backward, output_forward


def consistency_loss(seq, disparities, Flow_Model, alpha=50):
    b, T, *_ = seq.shape
    # compute optical flow
    flows_forward, flows_backward = compute_flow(Flow_Model, seq)

    seq_backward, seq_forward = bidirectional_alignment(seq, flows_backward, flows_forward)
    disparities_backward, disparities_forward = bidirectional_alignment(disparities, flows_backward, flows_forward)

    diff_disparities_back = torch.abs(disparities_backward - disparities)
    diff_disparities_for = torch.abs(disparities_forward - disparities)
    diff_seq_back = (seq_backward - seq) ** 2
    diff_seq_for = (seq_forward - seq) ** 2

    mask_seq_back = torch.exp(-(alpha * diff_seq_back))
    mask_seq_for = torch.exp(-(alpha * diff_seq_for))
    mask_seq_back = torch.sum(mask_seq_back, dim=2, keepdim=True)
    mask_seq_for = torch.sum(mask_seq_for, dim=2, keepdim=True)
    temporal_loss_back = torch.mul(mask_seq_back, diff_disparities_back)
    temporal_loss_for = torch.mul(mask_seq_for, diff_disparities_for)
    temporal_loss = torch.mean(temporal_loss_back) + torch.mean(temporal_loss_for)

    return temporal_loss
