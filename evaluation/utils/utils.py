from collections import defaultdict
import configparser
import os
import math
from typing import Optional, List
import torch
import cv2
import numpy as np
from dataclasses import dataclass
from tabulate import tabulate
import logging

from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.utils import (
    opencv_from_cameras_projection,
)
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from stereoanyvideo.evaluation.utils.eval_utils import depth_to_pcd


@dataclass
class PerceptionPrediction:
    """
    Holds the tensors that describe a result of any perception module.
    """

    depth_map: Optional[torch.Tensor] = None
    disparity: Optional[torch.Tensor] = None
    image_rgb: Optional[torch.Tensor] = None
    fg_probability: Optional[torch.Tensor] = None


def aggregate_eval_results(per_batch_eval_results, reduction="mean"):

    total_length = 0
    aggregate_results = defaultdict(list)
    for result in per_batch_eval_results:
        if isinstance(result, tuple):
            reduction = "sum"
            length = result[1]
            total_length += length
            result = result[0]
        for metric, val in result.items():
            if reduction == "sum":
                aggregate_results[metric].append(val * length)

    if reduction == "mean":
        return {k: torch.cat(v).mean().item() for k, v in aggregate_results.items()}
    elif reduction == "sum":
        return {
            k: torch.cat(v).sum().item() / float(total_length)
            for k, v in aggregate_results.items()
        }


def aggregate_and_print_results(
    per_batch_eval_results: List[dict],
):
    print("")
    result = aggregate_eval_results(
        per_batch_eval_results,
    )
    pretty_print_perception_metrics(result)
    result = {str(k): v for k, v in result.items()}

    print("")
    return result


def pretty_print_perception_metrics(results):

    metrics = sorted(list(results.keys()), key=lambda x: x.metric)

    print("===== Perception results =====")
    print(
        tabulate(
            [[metric, results[metric]] for metric in metrics],
        )
    )
    logging.info("===== Perception results =====")
    logging.info(tabulate(
            [[metric, results[metric]] for metric in metrics],
        ))


def read_calibration(calibration_file, resolution_string):
    # ported from https://github.com/stereolabs/zed-open-capture/
    # blob/dfa0aee51ccd2297782230a05ca59e697df496b2/examples/include/calibration.hpp#L4172

    zed_resolutions = {
        "2K": (1242, 2208),
        "FHD": (1080, 1920),
        "HD": (720, 1280),
        # "qHD": (540, 960),
        "VGA": (376, 672),
    }
    assert resolution_string in zed_resolutions.keys()
    image_height, image_width = zed_resolutions[resolution_string]

    # Open camera configuration file
    assert os.path.isfile(calibration_file)
    calib = configparser.ConfigParser()
    calib.read(calibration_file)

    # Get translations
    T = np.zeros((3, 1))
    T[0] = float(calib["STEREO"]["baseline"])
    T[1] = float(calib["STEREO"]["ty"])
    T[2] = float(calib["STEREO"]["tz"])

    baseline = T[0]

    # Get left parameters
    left_cam_cx = float(calib[f"LEFT_CAM_{resolution_string}"]["cx"])
    left_cam_cy = float(calib[f"LEFT_CAM_{resolution_string}"]["cy"])
    left_cam_fx = float(calib[f"LEFT_CAM_{resolution_string}"]["fx"])
    left_cam_fy = float(calib[f"LEFT_CAM_{resolution_string}"]["fy"])
    left_cam_k1 = float(calib[f"LEFT_CAM_{resolution_string}"]["k1"])
    left_cam_k2 = float(calib[f"LEFT_CAM_{resolution_string}"]["k2"])
    left_cam_p1 = float(calib[f"LEFT_CAM_{resolution_string}"]["p1"])
    left_cam_p2 = float(calib[f"LEFT_CAM_{resolution_string}"]["p2"])
    left_cam_k3 = float(calib[f"LEFT_CAM_{resolution_string}"]["k3"])

    # Get right parameters
    right_cam_cx = float(calib[f"RIGHT_CAM_{resolution_string}"]["cx"])
    right_cam_cy = float(calib[f"RIGHT_CAM_{resolution_string}"]["cy"])
    right_cam_fx = float(calib[f"RIGHT_CAM_{resolution_string}"]["fx"])
    right_cam_fy = float(calib[f"RIGHT_CAM_{resolution_string}"]["fy"])
    right_cam_k1 = float(calib[f"RIGHT_CAM_{resolution_string}"]["k1"])
    right_cam_k2 = float(calib[f"RIGHT_CAM_{resolution_string}"]["k2"])
    right_cam_p1 = float(calib[f"RIGHT_CAM_{resolution_string}"]["p1"])
    right_cam_p2 = float(calib[f"RIGHT_CAM_{resolution_string}"]["p2"])
    right_cam_k3 = float(calib[f"RIGHT_CAM_{resolution_string}"]["k3"])

    # Get rotations
    R_zed = np.zeros(3)
    R_zed[0] = float(calib["STEREO"][f"rx_{resolution_string.lower()}"])
    R_zed[1] = float(calib["STEREO"][f"cv_{resolution_string.lower()}"])
    R_zed[2] = float(calib["STEREO"][f"rz_{resolution_string.lower()}"])

    R = cv2.Rodrigues(R_zed)[0]

    # Left
    cameraMatrix_left = np.array(
        [[left_cam_fx, 0, left_cam_cx], [0, left_cam_fy, left_cam_cy], [0, 0, 1]]
    )
    distCoeffs_left = np.array(
        [left_cam_k1, left_cam_k2, left_cam_p1, left_cam_p2, left_cam_k3]
    )

    # Right
    cameraMatrix_right = np.array(
        [
            [right_cam_fx, 0, right_cam_cx],
            [0, right_cam_fy, right_cam_cy],
            [0, 0, 1],
        ]
    )
    distCoeffs_right = np.array(
        [right_cam_k1, right_cam_k2, right_cam_p1, right_cam_p2, right_cam_k3]
    )

    # Stereo
    R1, R2, P1, P2, Q = cv2.stereoRectify(
        cameraMatrix1=cameraMatrix_left,
        distCoeffs1=distCoeffs_left,
        cameraMatrix2=cameraMatrix_right,
        distCoeffs2=distCoeffs_right,
        imageSize=(image_width, image_height),
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=(image_width, image_height),
        alpha=0,
    )[:5]

    # Precompute maps for cv::remap()
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        cameraMatrix_left,
        distCoeffs_left,
        R1,
        P1,
        (image_width, image_height),
        cv2.CV_32FC1,
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        cameraMatrix_right,
        distCoeffs_right,
        R2,
        P2,
        (image_width, image_height),
        cv2.CV_32FC1,
    )

    zed_calib = {
        "map_left_x": map_left_x,
        "map_left_y": map_left_y,
        "map_right_x": map_right_x,
        "map_right_y": map_right_y,
        "pose_left": P1,
        "pose_right": P2,
        "baseline": baseline,
        "image_width": image_width,
        "image_height": image_height,
    }

    return zed_calib


def filter_depth_discontinuities(pcd, depth_map, threshold=5):
    """
    Removes points that belong to high-depth discontinuity regions.

    Args:
        pcd (torch.Tensor): Nx3 point cloud tensor.
        depth_map (torch.Tensor): HxW depth map.
        threshold (float): Depth change threshold.

    Returns:
        torch.Tensor: Filtered point cloud.
    """
    # Compute depth differences in x and y directions
    depth_diff_x = torch.abs(depth_map[:, 1:] - depth_map[:, :-1])  # Shape (H, W-1)
    depth_diff_y = torch.abs(depth_map[1:, :] - depth_map[:-1, :])  # Shape (H-1, W)

    # Initialize mask with all True (valid points)
    mask = torch.ones_like(depth_map, dtype=torch.bool)  # Shape (H, W)

    # Apply filtering: set False where depth difference is too large
    mask[:, :-1] &= depth_diff_x <= threshold  # X-direction filtering
    mask[:-1, :] &= depth_diff_y <= threshold  # Y-direction filtering

    # Flatten mask to match point cloud size
    mask_flat = mask.flatten()[: pcd.shape[0]]

    return pcd[mask_flat]  # Return only valid points



def visualize_batch(
    batch_dict: dict,
    preds: PerceptionPrediction,
    output_dir: str,
    ref_frame: int = 0,
    only_foreground=False,
    step=0,
    sequence_name=None,
    writer=None,
    render_bin_size=None
):
    os.makedirs(output_dir, exist_ok=True)

    outputs = {}

    if preds.depth_map is not None:
        device = preds.depth_map.device

        pcd_global_seq = []
        H, W = batch_dict["stereo_video"].shape[3:]

        for i in range(len(batch_dict["stereo_video"])):
            if hasattr(preds, 'perspective_cameras'):
                R, T, K = opencv_from_cameras_projection(
                    preds.perspective_cameras[i],
                    torch.tensor([H, W])[None].to(device),
                )  # 1x3x3, 1x3, 1x3x3
            else:
                raise KeyError(f"R T K not found!")
            extrinsic_3x4_0 = torch.cat([R[0], T[0, :, None]], dim=1)
            extr_matrix = torch.cat(
                [
                    extrinsic_3x4_0,
                    torch.Tensor([[0, 0, 0, 1]]).to(extrinsic_3x4_0.device),
                ],
                dim=0,
            )
            inv_extr_matrix = extr_matrix.inverse().to(device)
            pcd, colors = depth_to_pcd(
                preds.depth_map[i, 0],
                batch_dict["stereo_video"][i][0].permute(1, 2, 0),
                K[0][0][0],
                K[0][0][2],
                K[0][1][2],
                step=1,
                inv_extrinsic=inv_extr_matrix,
                mask=batch_dict["fg_mask"][i, 0] if only_foreground else None,
                filter=False,
            )
            R, T = inv_extr_matrix[None, :3, :3], inv_extr_matrix[None, :3, 3]
            pcd_global_seq.append((pcd, colors, (R, T, preds.perspective_cameras[i])))

        raster_settings = PointsRasterizationSettings(
            image_size=[H, W],
            radius=0.003,
            points_per_pixel=10,
        )
        R, T, cam_ = pcd_global_seq[ref_frame][2]
        median_depth = preds.depth_map.median()
        cam_.cuda()

        for mode in ["angle_15", "angle_-15", "angle_0", "changing_angle"]:
            res = []
            for t, (pcd, color, __) in enumerate(pcd_global_seq):

                if mode == "changing_angle":
                    angle = math.cos((math.pi) * (t / 60)) * 15
                elif mode == "angle_15":
                    angle = 15
                elif mode == "angle_-15":
                    angle = -15
                elif mode == "angle_0":
                    angle = 0

                delta_x = median_depth * math.sin(math.radians(angle))
                delta_z = median_depth * (1 - math.cos(math.radians(angle)))

                cam = cam_.clone()
                cam.R = torch.bmm(
                    cam.R,
                    RotateAxisAngle(angle=angle, axis="Y", device=device).get_matrix()[
                        :, :3, :3
                    ],
                )
                cam.T[0, 0] = cam.T[0, 0] - delta_x
                cam.T[0, 2] = cam.T[0, 2] - delta_z + median_depth / 2.0

                rasterizer = PointsRasterizer(
                    cameras=cam, raster_settings=raster_settings
                )
                renderer = PointsRenderer(
                    rasterizer=rasterizer,
                    compositor=AlphaCompositor(background_color=(1, 1, 1)),
                )
                pcd_copy = pcd.clone()
                point_cloud = Pointclouds(points=[pcd_copy], features=[color / 255.0])

                images = renderer(point_cloud)
                res.append(images[0, ..., :3].cpu())
            res = torch.stack(res)
            video = (res * 255).numpy().astype(np.uint8)
            save_name = f"{sequence_name}_reconstruction_{step}_mode_{mode}_"

            if writer is None:
                outputs[mode] = video
            if only_foreground:
                save_name += "fg_only"
            else:
                save_name += "full_scene"
            video_out = cv2.VideoWriter(
                os.path.join(
                    output_dir,
                    f"{save_name}.mp4",
                ),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps=30,
                frameSize=(res.shape[2], res.shape[1]),
                isColor=True,
            )
            filename = os.path.join(output_dir, sequence_name + '_img_')
            if not os.path.isdir(filename + str(mode)):
                os.mkdir(filename + str(mode))
            for i in range(len(video)):
                filename_temp = filename + str(mode) + '/' + str(i).zfill(3) + '.png'
                cv2.imwrite(filename_temp, cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB))
                video_out.write(cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB))
            video_out.release()

            if writer is not None:
                writer.add_video(
                    f"{sequence_name}_reconstruction_mode_{mode}",
                    (res * 255).permute(0, 3, 1, 2).to(torch.uint8)[None],
                    global_step=step,
                    fps=30,
                )

    return outputs
