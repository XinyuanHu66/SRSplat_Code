from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        # prediction: DecoderOutput,
        prediction: torch.Tensor,
        batch: BatchedExample,
        gaussians,
        global_step: int,
        image: Float[Tensor, "b v c h w"],
    ) -> Float[Tensor, ""]:
        # delta = prediction.color - batch["target"]["image_HR"]
        delta = prediction - image
        
        return self.cfg.weight * (delta**2).mean()
