import torch
import torch.nn.functional as F
from einops import rearrange


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    if H > 1:
        ygrid = 2 * ygrid/(H - 1) - 1
    img = img.contiguous()
    grid = torch.cat([xgrid, ygrid], dim=-1).contiguous()
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class AAPC:
    """
    Implementation of All-in-All-Pair Correlation.
    """
    def __init__(self, fmap1, fmap2, att=None):
        self.fmap1 = fmap1
        self.fmap2 = fmap2

        self.att = att
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)

    def __call__(self, flow, extra_offset, small_patch=False):

        corr = self.correlation(self.fmap1, self.fmap2, flow, small_patch)

        return corr

    def correlation(self, left_feature, right_feature, flow, small_patch):
        flow[:, 1:] = 0
        coords = self.coords - flow
        coords = coords.permute(0, 2, 3, 1)
        right_feature = bilinear_sampler(right_feature, coords)

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        N, C, H, W = left_feature.size()
        lefts = torch.split(left_feature, [C // 4] * 4, dim=1)
        rights = torch.split(right_feature, [C // 4] * 4, dim=1)
        corrs = []
        for i in range(len(psize_list)):
            corr = self.get_correlation(lefts[i], rights[i], psize_list[i], dilate_list[i])
            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)
        return final_corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):

        N, C, H, W = left_feature.size()

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        left_pad = F.pad(left_feature, [padx, padx, pady, pady], mode='replicate')
        right_pad = F.pad(right_feature, [padx, padx, pady, pady], mode='replicate')

        corr_list = []
        for dy1 in range(0, pady * 2 + 1, di_y):
            for dx1 in range(0, padx * 2 + 1, di_x):
                left_crop = left_pad[:, :, dy1:dy1 + H, dx1:dx1 + W]

                for dy2 in range(0, pady * 2 + 1, di_y):
                    for dx2 in range(0, padx * 2 + 1, di_x):
                        right_crop = right_pad[:, :, dy2:dy2 + H, dx2:dx2 + W]
                        assert right_crop.size() == left_crop.size()
                        corr = (left_crop * right_crop).sum(dim=1, keepdim=True)  # Sum over channels
                        corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)

        return corr_final