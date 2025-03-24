import os
import numpy as np
import cv2
from collections import defaultdict
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig
from pytorch3d.implicitron.tools.config import Configurable
from stereoanyvideo.evaluation.utils.eval_utils import depth2disparity_scale, eval_batch
from stereoanyvideo.evaluation.utils.utils import (
    PerceptionPrediction,
    pretty_print_perception_metrics,
    visualize_batch,
)


def depth_to_colormap(depth, colormap='jet', eps=1e-3, scale_vmin=1.0):
    valid = (depth > eps) & (depth < 1e4)
    vmin = depth[valid].min() * scale_vmin
    vmax = depth[valid].max()
    if colormap=='jet':
        cmap = plt.cm.jet
    else:
        cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth))
    depth[~valid] = 1
    return np.ascontiguousarray(depth[...,:3] * 255, dtype=np.uint8)


class Evaluator(Configurable):
    """
    A class defining the DynamicStereo evaluator.

    Args:
        eps: Threshold for converting disparity to depth.
    """

    eps = 1e-5

    def setup_visualization(self, cfg: DictConfig) -> None:
        # Visualization
        self.visualize_interval = cfg.visualize_interval
        self.render_bin_size = cfg.render_bin_size
        self.exp_dir = cfg.exp_dir
        if self.visualize_interval > 0:
            self.visualize_dir = os.path.join(cfg.exp_dir, "visualisations")

    @torch.no_grad()
    def evaluate_sequence(
        self,
        model,
        model_stabilizer,
        test_dataloader: torch.utils.data.DataLoader,
        is_real_data: bool = False,
        step=None,
        writer=None,
        train_mode=False,
        interp_shape=None,
        exp_dir=None,
    ):
        model.eval()
        per_batch_eval_results = []

        if self.visualize_interval > 0:
            os.makedirs(self.visualize_dir, exist_ok=True)

        for batch_idx, sequence in enumerate(tqdm(test_dataloader)):
            batch_dict = defaultdict(list)
            batch_dict["stereo_video"] = sequence["img"]
            if not is_real_data:
                batch_dict["disparity"] = sequence["disp"][:, 0].abs()
                batch_dict["disparity_mask"] = sequence["valid_disp"][:, :1]

                if "mask" in sequence:
                    batch_dict["fg_mask"] = sequence["mask"][:, :1]
                else:
                    batch_dict["fg_mask"] = torch.ones_like(
                        batch_dict["disparity_mask"]
                    )

            elif interp_shape is not None:
                left_video = batch_dict["stereo_video"][:, 0]
                left_video = F.interpolate(
                    left_video, tuple(interp_shape), mode="bilinear"
                )
                right_video = batch_dict["stereo_video"][:, 1]
                right_video = F.interpolate(
                    right_video, tuple(interp_shape), mode="bilinear"
                )
                batch_dict["stereo_video"] = torch.stack([left_video, right_video], 1)

            if model_stabilizer is not None:
                predictions = model.forward_stabilizer(batch_dict, model_stabilizer)
            elif train_mode:
                predictions = model.forward_batch_test(batch_dict)
            else:
                predictions = model(batch_dict)

            assert "disparity" in predictions
            predictions["disparity"] = predictions["disparity"][:, :1].clone().cpu()
            if not is_real_data:
                predictions["disparity"] = predictions["disparity"] * (
                    batch_dict["disparity_mask"].round()
                )

                batch_eval_result, seq_length = eval_batch(batch_dict, predictions, sequence["depth2disp_scale"][0])
                per_batch_eval_results.append((batch_eval_result, seq_length))
                pretty_print_perception_metrics(batch_eval_result)

            if (self.visualize_interval > 0) and (
                batch_idx % self.visualize_interval == 0
            ):
                perception_prediction = PerceptionPrediction()

                pred_disp = predictions["disparity"]
                pred_disp[pred_disp < self.eps] = self.eps

                scale = sequence["depth2disp_scale"][0]
                perception_prediction.depth_map = (scale / pred_disp).cuda()

                perspective_cameras = []
                if "viewpoint" in sequence:
                    for cam in sequence["viewpoint"]:
                        perspective_cameras.append(cam[0])
                        perception_prediction.perspective_cameras = perspective_cameras

                if "stereo_original_video" in batch_dict:
                    batch_dict["stereo_video"] = batch_dict[
                        "stereo_original_video"
                    ].clone()

                for k, v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_dict[k] = v.cuda()

                visualize_batch(
                    batch_dict,
                    perception_prediction,
                    output_dir=self.visualize_dir,
                    sequence_name=sequence["metadata"][0][0][0],
                    step=step,
                    writer=writer,
                    render_bin_size=self.render_bin_size,
                )
                filename = os.path.join(self.visualize_dir, sequence["metadata"][0][0][0])
                if not os.path.isdir(filename):
                    os.mkdir(filename)
                disparity_list = pred_disp.data.cpu().numpy()
                depth_list = perception_prediction.depth_map.data.cpu().numpy()
                np.save(f"{filename}_depth_list.npy", depth_list)

                video_disparity = cv2.VideoWriter(
                    f"{filename}_disparity.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps=30,
                    frameSize=(
                        batch_dict["stereo_video"][:, 0][0].shape[2], batch_dict["stereo_video"][:, 0][0].shape[1]),
                    isColor=True,
                )

                disparity_vis = depth_to_colormap(disparity_list[:, 0], eps=self.eps, colormap='inferno')
                for i in range(disparity_list.shape[0]):
                    filename_temp = filename + '/disparity_' + str(i).zfill(3) + '.png'
                    disparity_vis[i] = cv2.cvtColor(disparity_vis[i], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename_temp, disparity_vis[i])
                    video_disparity.write(disparity_vis[i])
                video_disparity.release()

        return per_batch_eval_results
