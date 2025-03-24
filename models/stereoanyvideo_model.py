from typing import ClassVar

import torch
import torch.nn.functional as F
from pytorch3d.implicitron.tools.config import Configurable
from stereoanyvideo.models.core.stereoanyvideo import StereoAnyVideo


class StereoAnyVideoModel(Configurable, torch.nn.Module):

    MODEL_CONFIG_NAME: ClassVar[str] = "StereoAnyVideoModel"
    model_weights: str = "./checkpoints/StereoAnyVideo_MIX.pth"

    def __post_init__(self):
        super().__init__()

        self.mixed_precision = False
        model = StereoAnyVideo(mixed_precision=self.mixed_precision)

        state_dict = torch.load(self.model_weights, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

        self.model = model
        self.model.to("cuda")
        self.model.eval()

    def forward(self, batch_dict, iters=20):

        return self.model.forward_batch_test(batch_dict, iters=iters)