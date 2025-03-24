from types import SimpleNamespace
from typing import ClassVar
import torch.nn.functional as F

from pytorch3d.implicitron.tools.config import Configurable
import torch
import importlib
import sys
import os

autocast = torch.cuda.amp.autocast

class RAFTModel(Configurable, torch.nn.Module):
    MODEL_CONFIG_NAME: ClassVar[str] = "RAFTModel"

    def __post_init__(self):
        super().__init__()
        thirdparty_raft_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/RAFT"))
        sys.path.append(thirdparty_raft_path)
        raft = importlib.import_module(
            "stereoanyvideo.third_party.RAFT.core.raft"
        )
        self.raft_utils = importlib.import_module(
            "stereoanyvideo.third_party.RAFT.core.utils.utils"
        )

        self.model_weights: str = "./third_party/RAFT/models/raft-things.pth"

        model_args = SimpleNamespace(
            mixed_precision=False,
            small=False,
            dropout=0.0,
        )
        self.args = model_args
        self.model = raft.RAFT(model_args).cuda()

        state_dict = torch.load(self.model_weights, map_location="cpu")
        weight_dict = {}
        for k,v in state_dict.items():
            temp_k = k.replace('module.', '') if 'module' in k else k
            weight_dict[temp_k] = v
        self.model.load_state_dict(weight_dict, strict=True)


    def forward(self, image1, image2, iters=10):
        left_image_rgb = image1.cuda()
        right_image_rgb = image2.cuda()
        padder = self.raft_utils.InputPadder(left_image_rgb.shape)
        left_image_rgb, right_image_rgb = padder.pad(
            left_image_rgb, right_image_rgb
        )
        with autocast(enabled=self.args.mixed_precision):
            flow, flow_up = self.model(left_image_rgb, right_image_rgb, iters=iters, test_mode=True)

        flow_up = padder.unpad(flow_up)
        return 0.25 * F.interpolate(flow_up, size=(flow_up.shape[2] // 4, flow_up.shape[3] // 4), mode="bilinear",
        align_corners=True)

    def forward_fullres(self, image1, image2, iters=20):
        left_image_rgb = image1.cuda()
        right_image_rgb = image2.cuda()
        padder = self.raft_utils.InputPadder(left_image_rgb.shape)
        left_image_rgb, right_image_rgb = padder.pad(
            left_image_rgb, right_image_rgb
        )
        with autocast(enabled=self.args.mixed_precision):
            flow, flow_up = self.model(left_image_rgb.contiguous(), right_image_rgb.contiguous(), iters=iters, test_mode=True)

        flow_up = padder.unpad(flow_up)
        return flow_up