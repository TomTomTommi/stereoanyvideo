import os
import copy
import gzip
import logging
import torch
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import os.path as osp
from glob import glob
import cv2
import re
from scipy.spatial.transform import Rotation as R

from collections import defaultdict
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    load_dataclass,
)

from stereoanyvideo.datasets import frame_utils
from stereoanyvideo.evaluation.utils.eval_utils import depth2disparity_scale
from stereoanyvideo.datasets.augmentor import SequenceDispFlowAugmentor, SequenceDispSparseFlowAugmentor


@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    """A dataclass used to load annotations from json."""

    camera_name: Optional[str] = None


class StereoSequenceDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = (
            aug_params.pop("img_pad", None) if aug_params is not None else None
        )
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SequenceDispSparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = SequenceDispFlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader
        self.depth_reader = self._load_depth
        self.is_test = False
        self.sample_list = []
        self.extra_info = []
        self.depth_eps = 1e-5

    def _load_depth(self, depth_path):
        if depth_path[-3:] == "npy":
            return self._load_npy_depth(depth_path)
        elif depth_path[-3:] == "png":
            if "kitti_depth" in depth_path:
                return self._load_kitti_depth(depth_path)
            elif "vkitti2" in depth_path:
                return self._load_vkitti2(depth_path)
            else:
                return self._load_16big_png_depth(depth_path)
        else:
            raise ValueError("Other format depth is not implemented")

    def _load_npy_depth(self, depth_npy):
        depth = np.load(depth_npy)
        return depth

    def _load_vkitti2(self, depth_png):
        depth_image = cv2.imread(depth_png, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_in_meters = depth_image.astype(np.float32) / 100.0
        depth_in_meters[depth_image == 0] = -1.

        return depth_in_meters

    def _load_kitti_depth(self, depth_png):
        # depth_image = cv2.imread(depth_png, cv2.IMREAD_UNCHANGED)
        # depth_in_meters = depth_image.astype(np.float32) / 256.0
        depth_image = np.array(Image.open(depth_png), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_image) > 255)

        depth_in_meters = depth_image.astype(np.float32) / 256.
        depth_in_meters[depth_image == 0] = -1.

        return depth_in_meters

    def _load_16big_png_depth(self, depth_png):
        with Image.open(depth_png) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        return depth

    def load_tartanair_pose(self, filepath, index=0):
        poses = np.loadtxt(filepath)
        tx, ty, tz, qx, qy, qz, qw = poses[index]

        # Quaternion to rotation matrix
        r = R.from_quat([qx, qy, qz, qw])
        R_mat = r.as_matrix()

        # Assemble 4x4 pose matrix
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = [tx, ty, tz]

        return T

    def parse_txt_file(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read()

        # Regex patterns
        intrinsic_pattern = re.compile(r"Intrinsic:\s*\[\[([^\]]+)\]\s*\[\s*([^\]]+)\]\s*\[\s*([^\]]+)\]\]")
        frame_pattern = re.compile(r"Frame (\d+): Pose: ([\w\d]+)\n([\s\S]+?)(?=Frame|\Z)")

        # Extract intrinsic matrix (K)
        intrinsic_match = intrinsic_pattern.search(data)
        if intrinsic_match:
            K = np.array([list(map(float, row.split())) for row in intrinsic_match.groups()])
        else:
            raise ValueError("Intrinsic matrix not found in the file")

        # Extract frames and compute R and T
        frames = []
        for frame_match in frame_pattern.finditer(data):
            frame_number = int(frame_match.group(1))
            pose_id = frame_match.group(2)
            pose_matrix = np.array([list(map(float, row.split())) for row in frame_match.group(3).strip().split('\n')])

            # Decompose pose matrix into R and T
            R = pose_matrix[:3, :3]  # The upper-left 3x3 part is the rotation matrix
            T = pose_matrix[:3, 3]  # The first three elements of the fourth column is the translation vector

            frames.append({
                'frame_number': frame_number,
                'pose_id': pose_id,
                'pose_matrix': pose_matrix,
                'R': R,
                'T': T
            })

        return K, frames

    def _get_pytorch3d_camera(
        self, entry_viewpoint, image_size, scale: float
    ) -> PerspectiveCameras:
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(
            entry_viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)
        half_image_size_wh_orig = (
            torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
        )

        # first, we convert from the dataset's NDC convention to pixels
        format = entry_viewpoint.intrinsics_format
        if format.lower() == "ndc_norm_image_bounds":
            # this is e.g. currently used in CO3D for storing intrinsics
            rescale = half_image_size_wh_orig
        elif format.lower() == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {format}")

        # principal point and focal length in pixels
        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale

        # now, convert from pixels to PyTorch3D v0.5+ NDC convention
        # if self.image_height is None or self.image_width is None:
        out_size = list(reversed(image_size))

        half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
        half_min_image_size_output = half_image_size_output.min()

        # rescaled principal point and focal length in ndc
        principal_point = (
            half_image_size_output - principal_point_px * scale
        ) / half_min_image_size_output
        focal_length = focal_length_px * scale / half_min_image_size_output
        return PerspectiveCameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        )

    def _get_pytorch3d_camera_from_blender(self, R, T, K, image_size, scale: float) -> PerspectiveCameras:
        assert R is not None and T is not None and K is not None
        assert R.shape == (3, 3), f"Expected R to be 3x3, but got {R.shape}"
        assert T.shape == (3,), f"Expected T to be a 3-element vector, but got {T.shape}"
        assert K.shape == (3, 3), f"Expected K to be 3x3, but got {K.shape}"

        # Extract principal point and focal length from K
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        principal_point = torch.tensor([cx, cy], dtype=torch.float)
        focal_length = torch.tensor([fx, fy], dtype=torch.float)

        half_image_size_wh_orig = (
                torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
        )

        # Adjust principal point and focal length in pixels
        principal_point_px = principal_point * scale
        focal_length_px = focal_length * scale

        # Convert from pixels to PyTorch3D NDC convention
        principal_point = (principal_point_px - half_image_size_wh_orig) / half_image_size_wh_orig
        half_min_image_size_output = half_image_size_wh_orig.min()
        focal_length = focal_length_px / half_min_image_size_output

        R = R.T @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        T = T @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

        # Convert R and T to PyTorch tensors
        R_tensor = torch.tensor(R, dtype=torch.float).unsqueeze(0)  # Add batch dimension
        T_tensor = torch.tensor(T, dtype=torch.float).view(1, 3)  # Ensure T is a 1x3 tensor

        # Return PerspectiveCameras object
        return PerspectiveCameras(
            focal_length=focal_length.unsqueeze(0),  # Add batch dimension
            principal_point=principal_point.unsqueeze(0),  # Add batch dimension
            R=R_tensor,
            T=T_tensor,
        )

    def _get_output_tensor(self, sample):
        output_tensor = defaultdict(list)
        sample_size = len(sample["image"]["left"])
        output_tensor_keys = ["img", "disp", "valid_disp", "mask"]
        add_keys = ["viewpoint", "metadata"]
        for add_key in add_keys:
            if add_key in sample:
                output_tensor_keys.append(add_key)

        for key in output_tensor_keys:
            output_tensor[key] = [[] for _ in range(sample_size)]

        if "viewpoint" in sample:
            viewpoint_left = self._get_pytorch3d_camera(
                sample["viewpoint"]["left"][0],
                sample["metadata"]["left"][0][1],
                scale=1.0,
            )
            viewpoint_right = self._get_pytorch3d_camera(
                sample["viewpoint"]["right"][0],
                sample["metadata"]["right"][0][1],
                scale=1.0,
            )
            depth2disp_scale = depth2disparity_scale(
                viewpoint_left,
                viewpoint_right,
                torch.Tensor(sample["metadata"]["left"][0][1])[None],
            )
            output_tensor["depth2disp_scale"] = [depth2disp_scale]

        if "camera" in sample:
            output_tensor["viewpoint"] = [[] for _ in range(sample_size)]
            # InfinigenSV
            if sample["camera"]["left"][0][-3:] == "npz":
                # Note that the K, R, T is based on Blender world Matrix
                camera_left = np.load(sample["camera"]["left"][0])
                camera_right = np.load(sample["camera"]["right"][0])
                camera_left_RT = camera_left['T']
                camera_right_RT = camera_right['T']
                camera_left_K = camera_left['K']
                camera_right_K = camera_right['K']
                camera_left_T = camera_left['T'][:3, 3]
                camera_left_R = camera_left['T'][:3, :3]
                fix_baseline = np.linalg.norm(camera_left_RT[:3, 3] - camera_right_RT[:3, 3])
                focal_length_px = camera_left_K[0][0]
                depth2disp_scale = focal_length_px * fix_baseline

            # Sintel
            elif sample["camera"]["left"][0][-3:] == "cam":
                TAG_FLOAT = 202021.25
                f = open(sample["camera"]["left"][0], 'rb')
                check = np.fromfile(f, dtype=np.float32, count=1)[0]
                assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
                    TAG_FLOAT, check)
                camera_left_K = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
                camera_left_RT = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))
                fix_baseline = 0.1 # From the MPI Sintel dataset website, the baseline of the cameras = 10cm = 0.1m
                focal_length_px = camera_left_K[0][0]
                depth2disp_scale = focal_length_px * fix_baseline
                camera_left_T = camera_left_RT[:3, 3]
                camera_left_R = camera_left_RT[:3, :3]

            # Spring
            elif any(filename in path for path in sample["camera"]["left"] for filename in ["focaldistance.txt", "extrinsics.txt", "intrinsics.txt"]):
                for path in sample["camera"]["left"]:
                    if "intrinsics.txt" in path:
                        intrinsics_path = path
                    elif "extrinsics.txt" in path:
                        extrinsics_path = path

                fx, fy, cx, cy = np.loadtxt(intrinsics_path)[0]
                # Build the 3x3 intrinsic matrix
                camera_left_K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                focal_length_px = camera_left_K[0][0]
                fix_baseline = 0.065  # From the dataset website, the baseline of the cameras = 6.5cm = 0.065m
                depth2disp_scale = focal_length_px * fix_baseline
                camera_left_RT = np.loadtxt(extrinsics_path).reshape(-1, 4, 4)[0]
                camera_left_T = camera_left_RT[:3, 3]
                camera_left_R = camera_left_RT[:3, :3]

            # TartanAir
            elif sample["camera"]["left"][0][-13:] == "pose_left.txt":
                fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
                # Build the 3x3 intrinsic matrix
                camera_left_K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                focal_length_px = camera_left_K[0][0]
                fix_baseline = 0.25
                depth2disp_scale = focal_length_px * fix_baseline
                camera_left_RT = self.load_tartanair_pose(sample["camera"]["left"][0], index=0)
                camera_left_T = camera_left_RT[:3, 3]
                camera_left_R = camera_left_RT[:3, :3]

            # KITTI Depth
            elif sample["camera"]["left"][0][-20:] == "calib_cam_to_cam.txt":
                calib_data = {}
                with open(sample["camera"]["left"][0], 'r') as f:
                    for line in f:
                        key, value = line.split(':', 1)
                        calib_data[key.strip()] = value.strip()

                P_key = 'P_rect_02'
                if P_key in calib_data:
                    P_values = np.array([float(x) for x in calib_data[P_key].split()])
                    P_matrix = P_values.reshape(3, 4)
                else:
                    raise KeyError(f"Projection matrix for camera not found in calibration data")
                focal_length_px = P_matrix[0, 0]

                T_key1 = 'T_02'
                T_key2 = 'T_03'
                if T_key1 in calib_data and T_key2 in calib_data:
                    T1 = np.array([float(x) for x in calib_data[T_key1].split()])
                    T2 = np.array([float(x) for x in calib_data[T_key2].split()])
                    baseline = np.linalg.norm(T1 - T2)
                else:
                    raise KeyError(f"Translation vectors for cameras not found in calibration data")

                R_key1 = 'R_rect_02'
                R_key2 = 'R_rect_03'
                if R_key1 in calib_data and R_key2 in calib_data:
                    R1 = np.array([float(x) for x in calib_data[R_key1].split()]).reshape(3, 3)
                    R2 = np.array([float(x) for x in calib_data[R_key2].split()]).reshape(3, 3)
                else:
                    raise KeyError(f"Rotation vectors for cameras not found in calibration data")

                depth2disp_scale = focal_length_px * baseline
                camera_left_K = P_matrix[:, :3]
                camera_left_T = T1
                camera_left_R = R1

            # VKITTI2
            elif sample["camera"]["left"][0][-13:] == "intrinsic.txt":
                baseline = 0.532725
                with open(sample["camera"]["left"][0], 'r') as f:
                    line = f.readlines()[1]
                    values = line.strip().split()
                    frame = int(values[0])
                    camera_id = int(values[1])
                    fx = float(values[2])
                    fy = float(values[3])
                    cx = float(values[4])
                    cy = float(values[5])
                    # Construct the intrinsic matrix
                    camera_left_K = torch.tensor([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]], dtype=torch.float32)
                depth2disp_scale = camera_left_K[0, 0] * baseline

                with open(sample["camera"]["left"][0].replace("intrinsic.txt", "extrinsic.txt"), 'r') as f:
                    line = f.readlines()[1]
                    values = line.strip().split()
                    frame = int(values[0])
                    camera_id = int(values[1])
                    # Extract rotation (3x3) and translation (3x1)
                    camera_left_R = np.array([
                        [float(values[2]), float(values[3]), float(values[4])],
                        [float(values[6]), float(values[7]), float(values[8])],
                        [float(values[10]), float(values[11]), float(values[12])]
                    ], dtype=np.float32)

                    camera_left_T = np.array([
                        float(values[5]),
                        float(values[9]),
                        float(values[13])
                    ], dtype=np.float32)

            # SouthKensington
            elif sample["camera"]["left"][0][-3:] == "txt":
                camera_left_K, frames = self.parse_txt_file(sample["camera"]["left"][0])
                fix_baseline = 0.12
                camera_left_R = frames[0]['R']
                camera_left_T = frames[0]['T']
                focal_length_px = camera_left_K[0][0]
                depth2disp_scale = focal_length_px * fix_baseline
            else:
                raise ValueError("Other format camera is not implemented")

            output_tensor["depth2disp_scale"] = [depth2disp_scale]
            output_tensor["RTK"] = [camera_left_R, camera_left_T, camera_left_K]

        for i in range(sample_size):
            for cam in ["left", "right"]:
                if "mask" in sample and cam in sample["mask"]:
                    mask = frame_utils.read_gen(sample["mask"][cam][i])
                    mask = np.array(mask) / 255.0
                    output_tensor["mask"][i].append(mask)

                if "viewpoint" in sample and cam in sample["viewpoint"]:
                    viewpoint = self._get_pytorch3d_camera(
                        sample["viewpoint"][cam][i],
                        sample["metadata"][cam][i][1],
                        scale=1.0,
                    )
                    output_tensor["viewpoint"][i].append(viewpoint)
                if "camera" in sample:
                    # InfinigenSV
                    if sample["camera"]["left"][0][-3:] == "npz" and cam in sample["camera"]:
                        # Note that the K, R, T is based on Blender world Matrix
                        camera = np.load(sample["camera"][cam][i])
                        camera_K = camera['K']
                        camera_T = camera['T'][:3, 3]
                        camera_R = camera['T'][:3, :3]
                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                    # Sintel
                    elif sample["camera"]["left"][0][-3:] == "cam" and cam in sample["camera"]:
                        f = open(sample["camera"]["left"][0], 'rb')
                        check = np.fromfile(f, dtype=np.float32, count=1)[0]
                        assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
                            TAG_FLOAT, check)
                        camera_K = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
                        camera_RT = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))
                        camera_T = camera_RT[:3, 3]
                        camera_R = camera_RT[:3, :3]
                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                    # TartanAir
                    elif sample["camera"]["left"][0][-13:] == "pose_left.txt":
                        fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
                        # Build the 3x3 intrinsic matrix
                        camera_left_K = np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ])
                        focal_length_px = camera_left_K[0][0]
                        fix_baseline = 0.25
                        depth2disp_scale = focal_length_px * fix_baseline
                        camera_left_RT = self.load_tartanair_pose(sample["camera"]["left"][0], index=i)
                        camera_left_T = camera_left_RT[:3, 3]
                        camera_left_R = camera_left_RT[:3, :3]

                    # Spring
                    elif any(filename in path for path in sample["camera"]["left"] for filename
                             in ["focaldistance.txt", "extrinsics.txt", "intrinsics.txt"]) and cam in sample["camera"]:

                        for path in sample["camera"]["left"]:
                            if "intrinsics.txt" in path:
                                intrinsics_path = path
                            elif "extrinsics.txt" in path:
                                extrinsics_path = path

                        fx, fy, cx, cy = np.loadtxt(intrinsics_path)[0]
                        # Build the 3x3 intrinsic matrix
                        camera_K = np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ])
                        focal_length_px = camera_left_K[0][0]
                        fix_baseline = 0.065  # From the dataset website, the baseline of the cameras = 6.5cm = 0.065m
                        depth2disp_scale = focal_length_px * fix_baseline
                        camera_RT = np.loadtxt(extrinsics_path).reshape(-1, 4, 4)[i]
                        camera_T = camera_RT[:3, 3]
                        camera_R = camera_RT[:3, :3]
                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                    # KITTI Depth
                    elif sample["camera"]["left"][0][-20:] == "calib_cam_to_cam.txt":
                        calib_data = {}
                        with open(sample["camera"]["left"][0], 'r') as f:
                            for line in f:
                                key, value = line.split(':', 1)
                                calib_data[key.strip()] = value.strip()

                        P_key = 'P_rect_02'
                        if P_key in calib_data:
                            P_values = np.array([float(x) for x in calib_data[P_key].split()])
                            P_matrix = P_values.reshape(3, 4)
                        else:
                            raise KeyError(f"Projection matrix for camera not found in calibration data")
                        focal_length_px = P_matrix[0, 0]

                        T_key1 = 'T_02'
                        T_key2 = 'T_03'
                        if T_key1 in calib_data and T_key2 in calib_data:
                            T1 = np.array([float(x) for x in calib_data[T_key1].split()])
                            T2 = np.array([float(x) for x in calib_data[T_key2].split()])
                            baseline = np.linalg.norm(T1 - T2)
                        else:
                            raise KeyError(f"Translation vectors for cameras not found in calibration data")

                        R_key1 = 'R_rect_02'
                        R_key2 = 'R_rect_03'
                        if R_key1 in calib_data and R_key2 in calib_data:
                            R1 = np.array([float(x) for x in calib_data[R_key1].split()]).reshape(3, 3)
                            R2 = np.array([float(x) for x in calib_data[R_key2].split()]).reshape(3, 3)
                        else:
                            raise KeyError(f"Rotation vectors for cameras not found in calibration data")

                        depth2disp_scale = focal_length_px * baseline
                        camera_K = P_matrix[:, :3]
                        camera_T = T1
                        camera_R = R1
                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                    # VKITTI2
                    elif sample["camera"]["left"][0][-13:] == "intrinsic.txt":
                        with open(sample["camera"]["left"][0], 'r') as f:
                            line = f.readlines()[1+i]
                            values = line.strip().split()
                            frame = int(values[0])
                            camera_id = int(values[1])
                            fx = float(values[2])
                            fy = float(values[3])
                            cx = float(values[4])
                            cy = float(values[5])
                            # Construct the intrinsic matrix
                            camera_K = torch.tensor([[fx, 0, cx],
                                                          [0, fy, cy],
                                                          [0, 0, 1]], dtype=torch.float32)
                        with open(sample["camera"]["left"][0].replace("intrinsic.txt", "extrinsic.txt"), 'r') as f:
                            line = f.readlines()[1+i]
                            values = line.strip().split()
                            frame = int(values[0])
                            camera_id = int(values[1])
                            # Extract rotation (3x3) and translation (3x1)
                            camera_R = np.array([
                                [float(values[2]), float(values[3]), float(values[4])],
                                [float(values[6]), float(values[7]), float(values[8])],
                                [float(values[10]), float(values[11]), float(values[12])]
                            ], dtype=np.float32)

                            camera_T = np.array([
                                float(values[5]),
                                float(values[9]),
                                float(values[13])
                            ], dtype=np.float32)
                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                    # SouthKensington
                    elif sample["camera"]["left"][0][-3:] == "txt" and cam in sample["camera"]:
                        camera_left_K, frames = self.parse_txt_file(sample["camera"]["left"][0])

                        camera_K = camera_left_K
                        camera_R = frames[i]['R']
                        camera_T = frames[i]['T']
                        viewpoint = self._get_pytorch3d_camera_from_blender(
                            camera_R, camera_T, camera_K,
                            sample["metadata"][cam][i][1],
                            scale=1.0,
                        )
                        output_tensor["viewpoint"][i].append(viewpoint)

                if "metadata" in sample and cam in sample["metadata"]:
                    metadata = sample["metadata"][cam][i]
                    output_tensor["metadata"][i].append(metadata)

                if cam in sample["image"]:
                    img = frame_utils.read_gen(sample["image"][cam][i])
                    img = np.array(img).astype(np.uint8)

                    # grayscale images
                    if len(img.shape) == 2:
                        img = np.tile(img[..., None], (1, 1, 3))
                    else:
                        img = img[..., :3]
                    output_tensor["img"][i].append(img)

                if cam in sample["disparity"]:
                    disp = self.disparity_reader(sample["disparity"][cam][i])
                    if isinstance(disp, tuple):
                        disp, valid_disp = disp
                    else:
                        valid_disp = disp < 512
                    disp = np.array(disp).astype(np.float32)
                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)
                    # disp = np.stack([disp, np.zeros_like(disp)], axis=-1)

                    output_tensor["disp"][i].append(disp)
                    output_tensor["valid_disp"][i].append(valid_disp)

                elif "depth" in sample and cam in sample["depth"]:
                    depth = self.depth_reader(sample["depth"][cam][i])
                    depth_mask = depth < self.depth_eps
                    depth[depth_mask] = self.depth_eps
                    disp = depth2disp_scale / depth
                    disp[depth_mask] = 0
                    valid_disp = (disp < 512) * (1 - depth_mask)
                    disp = np.array(disp).astype(np.float32)
                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)
                    output_tensor["disp"][i].append(disp)
                    output_tensor["valid_disp"][i].append(valid_disp)

        return output_tensor

    def __getitem__(self, index):
        im_tensor = {"img"}
        sample = self.sample_list[index]
        if self.is_test:
            sample_size = len(sample["image"]["left"])
            im_tensor["img"] = [[] for _ in range(sample_size)]
            for i in range(sample_size):
                for cam in ["left", "right"]:
                    img = frame_utils.read_gen(sample["image"][cam][i])
                    img = np.array(img).astype(np.uint8)[..., :3]
                    img = torch.from_numpy(img).permute(2, 0, 1).float()
                    im_tensor["img"][i].append(img)
            im_tensor["img"] = torch.stack(im_tensor["img"])
            return im_tensor, self.extra_info[index]

        index = index % len(self.sample_list)
        try:
            output_tensor = self._get_output_tensor(sample)
        except:
            logging.warning(f"Exception in loading sample {index}!")
            index = np.random.randint(len(self.sample_list))
            logging.info(f"New index is {index}")
            sample = self.sample_list[index]
            output_tensor = self._get_output_tensor(sample)

        sample_size = len(sample["image"]["left"])
        if self.augmentor is not None:
            if self.sparse:
                output_tensor["img"], output_tensor["disp"], output_tensor["valid_disp"] = self.augmentor(
                    output_tensor["img"], output_tensor["disp"], output_tensor["valid_disp"]
                )
            else:
                output_tensor["img"], output_tensor["disp"] = self.augmentor(
                    output_tensor["img"], output_tensor["disp"]
                )
        for i in range(sample_size):
            for cam in (0, 1):
                if cam < len(output_tensor["img"][i]):
                    img = (
                        torch.from_numpy(output_tensor["img"][i][cam])
                        .permute(2, 0, 1)
                        .float()
                    )
                    if self.img_pad is not None:
                        padH, padW = self.img_pad
                        img = F.pad(img, [padW] * 2 + [padH] * 2)
                    output_tensor["img"][i][cam] = img

                if cam < len(output_tensor["disp"][i]):
                    disp = (
                        torch.from_numpy(output_tensor["disp"][i][cam])
                        .permute(2, 0, 1)
                        .float()
                    )

                    if self.sparse:
                        valid_disp = torch.from_numpy(
                            output_tensor["valid_disp"][i][cam]
                        )
                    else:
                        valid_disp = (
                            (disp[0].abs() < 512)
                            & (disp[1].abs() < 512)
                            & (disp[0].abs() != 0)
                        )
                    disp = disp[:1]

                    output_tensor["disp"][i][cam] = disp
                    output_tensor["valid_disp"][i][cam] = valid_disp.float()

                if "mask" in output_tensor and cam < len(output_tensor["mask"][i]):
                    mask = torch.from_numpy(output_tensor["mask"][i][cam]).float()
                    output_tensor["mask"][i][cam] = mask

                if "viewpoint" in output_tensor and cam < len(
                    output_tensor["viewpoint"][i]
                ):

                    viewpoint = output_tensor["viewpoint"][i][cam]
                    output_tensor["viewpoint"][i][cam] = viewpoint

        res = {}
        if "viewpoint" in output_tensor and self.split != "train":
            res["viewpoint"] = output_tensor["viewpoint"]
        if "metadata" in output_tensor and self.split != "train":
            res["metadata"] = output_tensor["metadata"]
        if "depth2disp_scale" in output_tensor and self.split != "train":
            res["depth2disp_scale"] = output_tensor["depth2disp_scale"]
        if "RTK" in output_tensor and self.split != "train":
            res["RTK"] = output_tensor["RTK"]

        for k, v in output_tensor.items():
            if k != "viewpoint" and k != "metadata" and k != "depth2disp_scale" and k != "RTK":
                for i in range(len(v)):
                    if len(v[i]) > 0:
                        v[i] = torch.stack(v[i])
                if len(v) > 0 and (len(v[0]) > 0):
                    res[k] = torch.stack(v)
        return res

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.sample_list = v * copy_of_self.sample_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.sample_list)


class DynamicReplicaDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root="./data/datasets/dynamic_replica_data",
        split="train",
        sample_len=-1,
        only_first_n_samples=-1,
    ):
        super(DynamicReplicaDataset, self).__init__(aug_params)
        self.root = root
        self.sample_len = sample_len
        self.split = split

        frame_annotations_file = f"frame_annotations_{split}.jgz"

        with gzip.open(
            osp.join(root, split, frame_annotations_file), "rt", encoding="utf8"
        ) as zipfile:
            frame_annots_list = load_dataclass(
                zipfile, List[DynamicReplicaFrameAnnotation]
            )
        seq_annot = defaultdict(lambda: defaultdict(list))
        for frame_annot in frame_annots_list:
            seq_annot[frame_annot.sequence_name][frame_annot.camera_name].append(
                frame_annot
            )
        for seq_name in seq_annot.keys():
            try:
                filenames = defaultdict(lambda: defaultdict(list))
                for cam in ["left", "right"]:
                    for framedata in seq_annot[seq_name][cam]:
                        im_path = osp.join(root, split, framedata.image.path)
                        depth_path = osp.join(root, split, framedata.depth.path)
                        mask_path = osp.join(root, split, framedata.mask.path)

                        assert os.path.isfile(im_path), im_path
                        if self.split == 'train':
                            assert os.path.isfile(depth_path), depth_path
                        assert os.path.isfile(mask_path), mask_path

                        filenames["image"][cam].append(im_path)
                        if os.path.isfile(depth_path):
                            filenames["depth"][cam].append(depth_path)
                        filenames["mask"][cam].append(mask_path)

                        filenames["viewpoint"][cam].append(framedata.viewpoint)
                        filenames["metadata"][cam].append(
                            [framedata.sequence_name, framedata.image.size]
                        )

                        for k in filenames.keys():
                            assert (
                                len(filenames[k][cam])
                                == len(filenames["image"][cam])
                                > 0
                            ), framedata.sequence_name

                seq_len = len(filenames["image"][cam])

                print("seq_len", seq_name, seq_len)
                if split == "train":
                    for ref_idx in range(0, seq_len, 3):
                        step = 1 if self.sample_len == 1 else np.random.randint(1, 6)
                        if ref_idx + step * self.sample_len < seq_len:
                            sample = defaultdict(lambda: defaultdict(list))
                            for cam in ["left", "right"]:
                                for idx in range(
                                    ref_idx, ref_idx + step * self.sample_len, step
                                ):
                                    for k in filenames.keys():
                                        if "mask" not in k:
                                            sample[k][cam].append(
                                                filenames[k][cam][idx]
                                            )

                            self.sample_list.append(sample)
                else:
                    step = self.sample_len if self.sample_len > 0 else seq_len
                    counter = 0
                    for ref_idx in range(0, seq_len, step):
                        sample = defaultdict(lambda: defaultdict(list))
                        for cam in ["left", "right"]:
                            for idx in range(ref_idx, ref_idx + step):
                                for k in filenames.keys():
                                    sample[k][cam].append(filenames[k][cam][idx])

                        self.sample_list.append(sample)
                        counter += 1
                        if only_first_n_samples > 0 and counter >= only_first_n_samples:
                            break
            except Exception as e:
                print(e)
                print("Skipping sequence", seq_name)

        assert len(self.sample_list) > 0, "No samples found"
        print(f"Added {len(self.sample_list)} from Dynamic Replica {split}")
        logging.info(f"Added {len(self.sample_list)} from Dynamic Replica {split}")


class InfinigenStereoVideoDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root="./data/datasets/InfinigenStereo",
        split="train",
        sample_len=-1,
        only_first_n_samples=-1,
    ):
        super(InfinigenStereoVideoDataset, self).__init__(aug_params)
        self.root = root
        self.sample_len = sample_len
        self.split = split

        sequence = sorted(
            glob(osp.join(root, self.split, "*"))
        )
        for i in range(len(sequence)):
            sequence_name = os.path.basename(sequence[i])
            try:
                filenames = defaultdict(lambda: defaultdict(list))
                for cam in ["left", "right"]:
                    suffix = "camera_0/" if cam == "left" else "camera_1/"
                    im_path_list = sorted(glob(osp.join(sequence[i], "frames/Image/", suffix, "*.png")))
                    depth_path_list = sorted(glob(osp.join(sequence[i], "frames/Depth/", suffix, "*.npy")))
                    camera_list = sorted(glob(osp.join(sequence[i], "frames/camview/", suffix, "*.npz")))
                    for j in range(len(im_path_list)):
                        im_path = im_path_list[j]
                        depth_path = depth_path_list[j]
                        camera_path = camera_list[j]
                        assert os.path.isfile(im_path), im_path
                        assert os.path.isfile(depth_path), depth_path
                        filenames["image"][cam].append(im_path)
                        filenames["depth"][cam].append(depth_path)
                        filenames["camera"][cam].append(camera_path)
                        filenames["metadata"][cam].append([sequence_name , (720,1280)])

                        for k in filenames.keys():
                            assert (
                                    len(filenames[k][cam])
                                    == len(filenames["image"][cam])
                                    > 0
                            ), sequence_name
                seq_len = len(filenames["image"][cam])

                print("seq_len", sequence_name, seq_len)
                if self.split == "train":
                    for ref_idx in range(0, seq_len, 3):
                        step = 1 if self.sample_len == 1 else np.random.randint(1, 6)
                        if ref_idx + step * self.sample_len < seq_len:
                            sample = defaultdict(lambda: defaultdict(list))
                            for cam in ["left", "right"]:
                                for idx in range(
                                    ref_idx, ref_idx + step * self.sample_len, step
                                ):
                                    for k in filenames.keys():
                                        if "mask" not in k:
                                            sample[k][cam].append(
                                                filenames[k][cam][idx]
                                            )

                            self.sample_list.append(sample)
                else:
                    step = self.sample_len if (self.sample_len > 0) and (self.sample_len < seq_len) else seq_len
                    print("sample_step", step)
                    counter = 0
                    for ref_idx in range(0, seq_len, step):
                        sample = defaultdict(lambda: defaultdict(list))
                        for cam in ["left", "right"]:
                            for idx in range(ref_idx, ref_idx + step):
                                for k in filenames.keys():
                                    sample[k][cam].append(filenames[k][cam][idx])

                        self.sample_list.append(sample)
                        counter += 1
                        if only_first_n_samples > 0 and counter >= only_first_n_samples:
                            break
            except Exception as e:
                print(e)
                print("Skipping sequence", sequence_name)
        assert len(self.sample_list) > 0, "No samples found"
        print(f"Added {len(self.sample_list)} from Infinigen Stereo Video {split}")
        logging.info(f"Added {len(self.sample_list)} from Infinigen Stereo Video {split}")


class SouthKensingtonStereoVideoDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root="./data/datasets/indoor5/dynamic/",
        split="test",
        subroot="",
        sample_len=-1,
        only_first_n_samples=-1,
    ):
        super(SouthKensingtonStereoVideoDataset, self).__init__(aug_params)
        self.root = root
        self.split = split
        self.sample_len = sample_len

        sequence = sorted(
            glob(osp.join(root, "*"))
        )
        for i in range(len(sequence)):
            sequence_name = os.path.basename(sequence[i])
            try:
                filenames = defaultdict(lambda: defaultdict(list))
                for cam in ["left", "right"]:
                    im_path_list = sorted(glob(osp.join(sequence[i], "images", cam, "*.png")))
                    camera_path = glob(osp.join(sequence[i], "*.txt"))[0]

                    for j in range(len(im_path_list)):
                        im_path = im_path_list[j]
                        assert os.path.isfile(im_path), im_path
                        filenames["image"][cam].append(im_path)
                        filenames["camera"][cam].append(camera_path)
                        filenames["metadata"][cam].append([sequence_name , (720,1280)])

                        for k in filenames.keys():
                            assert (
                                    len(filenames[k][cam])
                                    == len(filenames["image"][cam])
                                    > 0
                            ), sequence_name
                seq_len = len(filenames["image"][cam])
                print("seq_len", sequence_name, seq_len)

                step = self.sample_len if (self.sample_len > 0) and (self.sample_len < seq_len) else seq_len
                print("sample_step", step)
                counter = 0
                for ref_idx in range(0, seq_len, step):
                    sample = defaultdict(lambda: defaultdict(list))
                    for cam in ["left", "right"]:
                        for idx in range(ref_idx, ref_idx + step):
                            for k in filenames.keys():
                                sample[k][cam].append(filenames[k][cam][idx])

                    self.sample_list.append(sample)
                    counter += 1
                    if only_first_n_samples > 0 and counter >= only_first_n_samples:
                        break
            except Exception as e:
                print(e)
                print("Skipping sequence", sequence_name)
        assert len(self.sample_list) > 0, "No samples found"
        print(f"Added {len(self.sample_list)} from SouthKensington Stereo Video")
        logging.info(f"Added {len(self.sample_list)} from SouthKensington Stereo Video")


class KITTIDepthDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root="./data/datasets/",
        split="train",
        sample_len=-1,
        only_first_n_samples=-1,
    ):
        super().__init__(aug_params, sparse=True)
        # super(KITTIDepthDataset, self).__init__(aug_params)
        image_root = osp.join(root, "kitti_depth", "input")
        gt_root = osp.join(root, "kitti_depth", "gt_depth")
        self.sample_len = sample_len
        self.split = split
        # Following CODD: https://github.com/facebookresearch/CODD
        val_split = ['2011_10_03_drive_0042_sync']  # 1 scene
        test_split = ['2011_09_26_drive_0002_sync', '2011_09_26_drive_0005_sync',
                      '2011_09_26_drive_0013_sync', '2011_09_26_drive_0020_sync',
                      '2011_09_26_drive_0023_sync', '2011_09_26_drive_0036_sync',
                      '2011_09_26_drive_0079_sync', '2011_09_26_drive_0095_sync',
                      '2011_09_26_drive_0113_sync', '2011_09_28_drive_0037_sync',
                      '2011_09_29_drive_0026_sync', '2011_09_30_drive_0016_sync',
                      '2011_10_03_drive_0047_sync']  # 13 scenes

        sequence_root = sorted(glob(osp.join(gt_root, "*")))
        train_list = []
        val_list = []
        test_list = []
        for i in range(len(sequence_root)):
            sequence_name = os.path.basename(os.path.normpath(sequence_root[i]))
            if sequence_name in test_split:
                test_list.append(sequence_root[i])
            elif sequence_name in val_split:
                val_list.append(sequence_root[i])
            else:
                train_list.append(sequence_root[i])

        if self.split == "train":
            sequence_split = train_list
        elif self.split == "val":
            sequence_split = val_list
        elif self.split == "test":
            sequence_split = test_list
        else:
            raise ValueError("Wrong Split: ", self.split)

        for i in range(len(sequence_split)):
            sequence_name = os.path.basename(os.path.normpath(sequence_split[i]))
            sequence_camera = osp.join(image_root, sequence_name[:10], "calib_cam_to_cam.txt")
            try:
                filenames = defaultdict(lambda: defaultdict(list))
                for cam in ["left", "right"]:
                    suffix = "image_02/" if cam == "left" else "image_03/"
                    depth_path_list = sorted(
                        glob(osp.join(gt_root, sequence_name, "proj_depth", "groundtruth", suffix, "*.png")))
                    for j in range(len(depth_path_list)):
                        depth_path = depth_path_list[j]
                        assert os.path.isfile(depth_path), depth_path
                        filenames["depth"][cam].append(depth_path)

                        # find the corresponding images
                        im_name = os.path.basename(os.path.normpath(depth_path))
                        im_path = osp.join(image_root, sequence_name[:10], sequence_name, suffix, "data", im_name)
                        assert os.path.isfile(im_path), im_path
                        filenames["image"][cam].append(im_path)
                        filenames["camera"][cam].append(sequence_camera)
                        filenames["metadata"][cam].append([sequence_name, (370,1224)])
                        for k in filenames.keys():
                            assert (
                                    len(filenames[k][cam])
                                    == len(filenames["depth"][cam])
                                    > 0
                            ), sequence_name
                seq_len = len(filenames["image"][cam])
                print("seq_len", sequence_name, seq_len)
                if self.split == "train":
                    for ref_idx in range(0, seq_len, 3):
                        step = 1 if self.sample_len == 1 else np.random.randint(1, 6)
                        if ref_idx + step * self.sample_len < seq_len:
                            sample = defaultdict(lambda: defaultdict(list))
                            for cam in ["left", "right"]:
                                for idx in range(
                                        ref_idx, ref_idx + step * self.sample_len, step
                                ):
                                    for k in filenames.keys():
                                        if "mask" not in k:
                                            sample[k][cam].append(
                                                filenames[k][cam][idx]
                                            )
                            self.sample_list.append(sample)
                else:
                    step = self.sample_len if (self.sample_len > 0) and (self.sample_len < seq_len) else seq_len
                    print("sample_step", step)
                    counter = 0
                    for ref_idx in range(0, seq_len, step):
                        sample = defaultdict(lambda: defaultdict(list))
                        for cam in ["left", "right"]:
                            for idx in range(ref_idx, ref_idx + step):
                                for k in filenames.keys():
                                    sample[k][cam].append(filenames[k][cam][idx])

                        self.sample_list.append(sample)
                        counter += 1
                        if only_first_n_samples > 0 and counter >= only_first_n_samples:
                            break
            except Exception as e:
                print(e)
                print("Skipping sequence", sequence_name)
        assert len(self.sample_list) > 0, "No samples found"
        print(f"Added {len(self.sample_list)} from  KITTI Depth {split}")
        logging.info(f"Added {len(self.sample_list)} from KITTI Depth {split}")


def split_train_valid(path_list, valid_keywords):
    path_list_init = path_list
    for kw in valid_keywords:
        path_list = list(filter(lambda s: kw not in s, path_list))
    train_path_list = sorted(path_list)
    valid_path_list = sorted(list(set(path_list_init) - set(train_path_list)))
    return train_path_list, valid_path_list


class TartanAirDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root="./data/datasets/TartanAir/",
        split="train",
        sample_len=-1,
        only_first_n_samples=-1,
    ):
        super().__init__(aug_params, sparse=False)
        self.sample_len = sample_len
        self.split = split

        # Each entry is (scene, motion, part)
        test_entries = [
            ("abandonedfactory", "Easy", "P002"),
            ("abandonedfactory", "Hard", "P002"),
            ("amusement", "Easy", "P007"),
            ("amusement", "Hard", "P007"),
            ("carwelding", "Hard", "P003"),
            ("endofworld", "Easy", "P006"),
            ("endofworld", "Hard", "P006"),
            ("gascola", "Easy", "P001"),
            ("gascola", "Hard", "P001"),
            ("hospital", "Hard", "P042"),
            ("office", "Easy", "P006"),
            ("office", "Hard", "P006"),
            ("office2", "Easy", "P004"),
            ("office2", "Hard", "P004"),
            ("oldtown", "Hard", "P006"),
            ("soulcity", "Easy", "P008"),
            ("soulcity", "Hard", "P008"),
        ]

        scene_root = sorted(glob(osp.join(root, "*")))

        sequence_root_list = []
        test_set = []
        train_set = []
        for i in range(len(scene_root)):
            sequence_root_list += sorted(glob(osp.join(scene_root[i], "Easy", "*"))) + sorted(glob(osp.join(scene_root[i], "Hard", "*")))

        for path in sequence_root_list:
            parts = path.split("/")
            if len(parts) < 5:
                continue  # skip malformed paths

            scene, motion, part = parts[-3], parts[-2], parts[-1]
            if (scene, motion, part) in test_entries:
                test_set.append(path)
            else:
                train_set.append(path)

        if self.split == "train":
            sequence_root_list = train_set
        elif self.split == "test":
            sequence_root_list = test_set
        else:
            raise KeyError(f"Wrong Split!")

        for i in range(len(sequence_root_list)):
            filenames = defaultdict(lambda: defaultdict(list))
            sequence_root = sequence_root_list[i]
            parts = os.path.normpath(sequence_root).split(os.sep)
            sequence_name = "_".join(parts[-3:])
            try:
                for cam in ['left', 'right']:
                    depth_path_list = sorted(glob(osp.join(sequence_root, "depth_left/", "*.npy")))
                    im_path_list = sorted(glob(osp.join(sequence_root, f"image_{cam}/", "*.png")))
                    pose_path = os.path.join(sequence_root, f"pose_{cam}.txt")
                    assert len(depth_path_list) == len(im_path_list), [len(depth_path_list), len(im_path_list)]
                    for j in range(len(depth_path_list)):
                        depth_path = depth_path_list[j]
                        assert os.path.isfile(depth_path), depth_path
                        filenames["depth"][cam].append(depth_path)
                        im_path = im_path_list[j]
                        assert os.path.isfile(im_path), im_path
                        filenames["image"][cam].append(im_path)
                        filenames["camera"][cam].append(pose_path)
                        filenames["metadata"][cam].append([sequence_name, (480,640)])
                        for k in filenames.keys():
                            assert (
                                    len(filenames[k][cam])
                                    == len(filenames["depth"][cam])
                                    > 0
                            ), sequence_name
                seq_len = len(filenames["image"][cam])
                print("seq_len", sequence_name, seq_len)

                if self.split == "train":
                    for ref_idx in range(0, seq_len, 3):
                        step = 1 if self.sample_len == 1 else np.random.randint(1, 6)
                        if ref_idx + step * self.sample_len < seq_len:
                            sample = defaultdict(lambda: defaultdict(list))
                            for cam in ["left", "right"]:
                                for idx in range(
                                        ref_idx, ref_idx + step * self.sample_len, step
                                ):
                                    for k in filenames.keys():
                                        if "mask" not in k:
                                            sample[k][cam].append(
                                                filenames[k][cam][idx]
                                            )
                            self.sample_list.append(sample)
                else:
                    step = self.sample_len if (self.sample_len > 0) and (self.sample_len < seq_len) else seq_len
                    print("sample_step", step)
                    counter = 0
                    for ref_idx in range(0, seq_len, step):
                        sample = defaultdict(lambda: defaultdict(list))
                        for cam in ["left", "right"]:
                            for idx in range(ref_idx, ref_idx + step):
                                for k in filenames.keys():
                                    sample[k][cam].append(filenames[k][cam][idx])

                        self.sample_list.append(sample)
                        counter += 1
                        if only_first_n_samples > 0 and counter >= only_first_n_samples:
                            break
            except Exception as e:
                print(e)
                print("Skipping sequence", sequence_name)
        assert len(self.sample_list) > 0, "No samples found"
        print(f"Added {len(self.sample_list)} from  TarTanAir  {split}")
        logging.info(f"Added {len(self.sample_list)} from TarTanAir {split}")


class VKITTI2Dataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root="./data/datasets/vkitti2/",
        split="train",
        sample_len=-1,
        only_first_n_samples=-1,
    ):
        super().__init__(aug_params, sparse=False)
        self.sample_len = sample_len
        self.split = split
        if self.split == "train":
            sequence_name_list = []
            scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
            variations = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right',
                          'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
            for scene in scenes:
                for variation in variations:
                    sequence_name = f"{scene}_{variation}"
                    sequence_name_list.append(sequence_name)
        elif self.split == "test":
            sequence_name_list = ["Scene01_15-deg-left", "Scene02_30-deg-right", "Scene06_fog", "Scene18_morning", "Scene20_rain"]
        else:
            raise KeyError(f"Wrong Split!")

        for i in range(len(sequence_name_list)):
            filenames = defaultdict(lambda: defaultdict(list))
            sequence_name = sequence_name_list[i]
            scene, variation = sequence_name.split("_")
            try:
                for cam in [('left', 0), ('right', 1)]:
                    depth_path_list = sorted(glob(osp.join(root, f"{scene}/{variation}/frames/depth/Camera_{cam[1]}/", "*.png")))
                    im_path_list = sorted(glob(osp.join(root, f"{scene}/{variation}/frames/rgb/Camera_{cam[1]}/", "*.jpg")))
                    intrinsic_path = os.path.join(root, f"{scene}/{variation}/intrinsic.txt")
                    assert len(depth_path_list) == len(im_path_list), [len(depth_path_list), len(im_path_list)]
                    for j in range(len(depth_path_list)):
                        depth_path = depth_path_list[j]
                        assert os.path.isfile(depth_path), depth_path
                        filenames["depth"][cam[0]].append(depth_path)
                        im_path = im_path_list[j]
                        assert os.path.isfile(im_path), im_path
                        filenames["image"][cam[0]].append(im_path)
                        filenames["camera"][cam[0]].append(intrinsic_path)
                        filenames["metadata"][cam[0]].append([sequence_name, (375,1242)])
                        for k in filenames.keys():
                            assert (
                                    len(filenames[k][cam[0]])
                                    == len(filenames["depth"][cam[0]])
                                    > 0
                            ), sequence_name
                seq_len = len(filenames["image"][cam[0]])
                print("seq_len", sequence_name, seq_len)

                if self.split == "train":
                    for ref_idx in range(0, seq_len, 3):
                        step = 1 if self.sample_len == 1 else np.random.randint(1, 6)
                        if ref_idx + step * self.sample_len < seq_len:
                            sample = defaultdict(lambda: defaultdict(list))
                            for cam in ["left", "right"]:
                                for idx in range(
                                        ref_idx, ref_idx + step * self.sample_len, step
                                ):
                                    for k in filenames.keys():
                                        if "mask" not in k:
                                            sample[k][cam].append(
                                                filenames[k][cam][idx]
                                            )
                            self.sample_list.append(sample)
                else:
                    step = self.sample_len if (self.sample_len > 0) and (self.sample_len < seq_len) else seq_len
                    print("sample_step", step)
                    counter = 0
                    for ref_idx in range(0, seq_len, step):
                        sample = defaultdict(lambda: defaultdict(list))
                        for cam in ["left", "right"]:
                            for idx in range(ref_idx, ref_idx + step):
                                for k in filenames.keys():
                                    sample[k][cam].append(filenames[k][cam][idx])

                        self.sample_list.append(sample)
                        counter += 1
                        if only_first_n_samples > 0 and counter >= only_first_n_samples:
                            break
            except Exception as e:
                print(e)
                print("Skipping sequence", sequence_name)
        assert len(self.sample_list) > 0, "No samples found"
        print(f"Added {len(self.sample_list)} from  VKITTI2  {split}")
        logging.info(f"Added {len(self.sample_list)} from VKITTI2 {split}")


class SequenceSpringDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        sample_len=-1,
        root="./data/datasets/Spring",
    ):
        super(SequenceSpringDataset, self).__init__(aug_params)
        self.split = "test"
        self.sample_len = sample_len
        original_length = len(self.sample_list)
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        camera_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(root, f"train_frame_{cam}/*"))
            )

        cam = "left"
        disparity_paths[cam] = sorted(
                glob(osp.join(root, f"train_disp1_{cam}/*"))
            )

        camera_paths[cam] = sorted(
                glob(osp.join(root, "train_cam_data/*"))
            )

        num_seq = len(image_paths["left"])
        # for each sequence
        for seq_idx in range(num_seq):
            sequence_name = os.path.basename(image_paths[cam][seq_idx])
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                sample["image"][cam] = sorted(
                    glob(osp.join(image_paths[cam][seq_idx], f"frame_{cam}", "*.png"))
                )[:sample_len]
                # for _ in range(len(sample["image"][cam])):
                for _ in range(sample_len):
                    sample["metadata"][cam].append([sequence_name, (1080, 1920)])

            cam = "left"
            sample["disparity"][cam] = sorted(
                glob(osp.join(disparity_paths[cam][seq_idx], f"disp1_{cam}", "*.dsp5"))
            )[:sample_len]
            sample["camera"][cam] = sorted(
                glob(osp.join(camera_paths[cam][seq_idx], "cam_data", "*.txt"))
            )
            self.sample_list.append(sample)
            seq_len = len(sample["image"][cam])
            print("seq_len", sequence_name, seq_len)
        logging.info(
            f"Added {len(self.sample_list) - original_length} from Spring Dataset"
        )


class SequenceSceneFlowDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root="./data/datasets",
        dstype="frames_cleanpass",
        sample_len=1,
        things_test=False,
        add_things=True,
        add_monkaa=True,
        add_driving=True,
    ):
        super(SequenceSceneFlowDataset, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype
        self.sample_len = sample_len
        if things_test:
            self._add_things("TEST")
        else:
            if add_things:
                self._add_things("TRAIN")
            if add_monkaa:
                self._add_monkaa()
            if add_driving:
                self._add_driving()

    def _add_things(self, split="TRAIN"):
        """Add FlyingThings3D data"""

        original_length = len(self.sample_list)
        root = osp.join(self.root, "FlyingThings3D")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(root, self.dstype, split, f"*/*/{cam}/"))
            )
            disparity_paths[cam] = [
                path.replace(self.dstype, "disparity") for path in image_paths[cam]
            ]
        # Choose a random subset of 400 images for validation
        # state = np.random.get_state()
        # np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(image_paths["left"]))[:40])
        # np.random.set_state(state)
        # np.random.seed(0)
        num_seq = len(image_paths["left"])
        num = 0
        for seq_idx in range(num_seq):
            # if (split == "TEST" and seq_idx in val_idxs) or (
            #     split == "TRAIN" and not seq_idx in val_idxs
            # ):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(
                    glob(osp.join(image_paths[cam][seq_idx], "*.png"))
                )
                disparities[cam] = sorted(
                    glob(osp.join(disparity_paths[cam][seq_idx], "*.pfm"))
                )
            num = num + len(images["left"])
            self._append_sample(images, disparities)
        print(num)
        assert len(self.sample_list) > 0, "No samples found"
        print(
            f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dstype}"
        )
        logging.info(
            f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dstype}"
        )

    def _add_monkaa(self):
        """Add FlyingThings3D data"""

        original_length = len(self.sample_list)
        root = osp.join(self.root, "Monkaa")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(glob(osp.join(root, self.dstype, f"*/{cam}/")))
            disparity_paths[cam] = [
                path.replace(self.dstype, "disparity") for path in image_paths[cam]
            ]

        num_seq = len(image_paths["left"])

        for seq_idx in range(num_seq):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], "*.png")))
                disparities[cam] = sorted(
                    glob(osp.join(disparity_paths[cam][seq_idx], "*.pfm"))
                )

            self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        print(
            f"Added {len(self.sample_list) - original_length} from Monkaa {self.dstype}"
        )
        logging.info(
            f"Added {len(self.sample_list) - original_length} from Monkaa {self.dstype}"
        )

    def _add_driving(self):
        """Add FlyingThings3D data"""

        original_length = len(self.sample_list)
        root = osp.join(self.root, "Driving")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(root, self.dstype, f"*/*/*/{cam}/"))
            )
            disparity_paths[cam] = [
                path.replace(self.dstype, "disparity") for path in image_paths[cam]
            ]

        num_seq = len(image_paths["left"])
        for seq_idx in range(num_seq):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], "*.png")))
                disparities[cam] = sorted(
                    glob(osp.join(disparity_paths[cam][seq_idx], "*.pfm"))
                )

            self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        print(
            f"Added {len(self.sample_list) - original_length} from Driving {self.dstype}"
        )
        logging.info(
            f"Added {len(self.sample_list) - original_length} from Driving {self.dstype}"
        )

    def _append_sample(self, images, disparities):
        seq_len = len(images["left"])
        for ref_idx in range(0, seq_len - self.sample_len):
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                for idx in range(ref_idx, ref_idx + self.sample_len):
                    sample["image"][cam].append(images[cam][idx])
                    sample["disparity"][cam].append(disparities[cam][idx])
            self.sample_list.append(sample)

            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                for idx in range(ref_idx, ref_idx + self.sample_len):
                    sample["image"][cam].append(images[cam][seq_len - idx - 1])
                    sample["disparity"][cam].append(disparities[cam][seq_len - idx - 1])
            self.sample_list.append(sample)


class SequenceSintelStereo(StereoSequenceDataset):
    def __init__(
        self,
        dstype="clean",
        aug_params=None,
        root="./data/datasets",
    ):
        super().__init__(
            aug_params, sparse=True, reader=frame_utils.readDispSintelStereo
        )
        self.dstype = dstype
        self.split = "test"
        original_length = len(self.sample_list)
        image_root = osp.join(root, "sintel_stereo", "training")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        camera_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(image_root, f"{self.dstype}_{cam}/*"))
            )

        cam = "left"
        disparity_paths[cam] = [
            path.replace(f"{self.dstype}_{cam}", "disparities")
            for path in image_paths[cam]
        ]
        camera_paths[cam] = [
            path.replace(f"{self.dstype}_{cam}", "camdata_left")
            for path in image_paths[cam]
        ]

        num_seq = len(image_paths["left"])
        # for each sequence
        for seq_idx in range(num_seq):
            sequence_name = os.path.basename(image_paths[cam][seq_idx])
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                sample["image"][cam] = sorted(
                    glob(osp.join(image_paths[cam][seq_idx], "*.png"))
                )
                for _ in range(len(sample["image"][cam])):
                    sample["metadata"][cam].append([sequence_name, (436, 1024)])

            cam = "left"
            sample["disparity"][cam] = sorted(
                glob(osp.join(disparity_paths[cam][seq_idx], "*.png"))
            )
            sample["camera"][cam] = sorted(
                glob(osp.join(camera_paths[cam][seq_idx], "*.cam"))
            )

            for im1, disp, camera in zip(sample["image"][cam], sample["disparity"][cam], sample["camera"][cam]):
                assert (
                    im1.split("/")[-1].split(".")[0]
                    == disp.split("/")[-1].split(".")[0]
                    == camera.split("/")[-1].split(".")[0]
                ), (im1.split("/")[-1].split(".")[0], disp.split("/")[-1].split(".")[0], camera.split("/")[-1].split(".")[0])
            self.sample_list.append(sample)

        logging.info(
            f"Added {len(self.sample_list) - original_length} from SintelStereo {self.dstype}"
        )


def fetch_dataloader(args):
    """Create the data loader for the corresponding training set"""

    aug_params = {
        "crop_size": args.image_size,
        "min_scale": args.spatial_scale[0],
        "max_scale": args.spatial_scale[1],
        "do_flip": False,
        "yjitter": not args.noyjitter,
    }
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None

    add_monkaa = "monkaa" in args.train_datasets
    add_driving = "driving" in args.train_datasets
    add_things = "things" in args.train_datasets
    add_dynamic_replica = "dynamic_replica" in args.train_datasets
    add_infinigensv = "infinigen_stereovideo" in args.train_datasets
    add_kittidepth = "kitti_depth" in args.train_datasets
    add_vkitti2 = "vkitti2" in args.train_datasets
    add_tartanair = "tartanair" in args.train_datasets
    new_dataset = None

    if add_monkaa or add_driving or add_things:
        # clean_dataset = SequenceSceneFlowDataset(
        #     aug_params,
        #     dstype="frames_cleanpass",
        #     sample_len=args.sample_len,
        #     add_monkaa=add_monkaa,
        #     add_driving=add_driving,
        #     add_things=add_things,
        # )

        final_dataset = SequenceSceneFlowDataset(
            aug_params,
            dstype="frames_finalpass",
            sample_len=args.sample_len,
            add_monkaa=add_monkaa,
            add_driving=add_driving,
            add_things=add_things,
        )
        # new_dataset = clean_dataset + final_dataset

        new_dataset = final_dataset

    if add_dynamic_replica:
        dr_dataset = DynamicReplicaDataset(
            aug_params, split="train", sample_len=args.sample_len
        )
        if new_dataset is None:
            new_dataset = dr_dataset
        else:
            new_dataset = new_dataset + dr_dataset

    if add_infinigensv:
        infinigensv_dataset = InfinigenStereoVideoDataset(
            aug_params, split="train", sample_len=args.sample_len
        )
        if new_dataset is None:
            new_dataset = infinigensv_dataset
        else:
            new_dataset = new_dataset + infinigensv_dataset + infinigensv_dataset + infinigensv_dataset

    if add_kittidepth:
        kittidepth_dataset = KITTIDepthDataset(
            aug_params, split="train", sample_len=args.sample_len
        )
        if new_dataset is None:
            new_dataset = kittidepth_dataset
        else:
            new_dataset = new_dataset + kittidepth_dataset

    if add_vkitti2:
        vkitti2_dataset = VKITTI2Dataset(
            aug_params, split="train", sample_len=args.sample_len
        )
        if new_dataset is None:
            new_dataset = vkitti2_dataset
        else:
            new_dataset = new_dataset + vkitti2_dataset

    if add_tartanair:
        tartanair_dataset = TartanAirDataset(
            aug_params, split="train", sample_len=args.sample_len
        )
        if new_dataset is None:
            new_dataset = tartanair_dataset
        else:
            new_dataset = new_dataset + tartanair_dataset

    logging.info(f"Adding {len(new_dataset)} samples in total")
    train_dataset = (
        new_dataset if train_dataset is None else train_dataset + new_dataset
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    logging.info("Training with %d image pairs" % len(train_dataset))
    return train_loader
