"""DINOv2 ViTs via torch.hub (downloads weights on first use)."""

from __future__ import annotations

import torch
from torchvision import transforms as T

from ..registry import BACKBONES
from .base import Backbone

_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}


@BACKBONES.register("dinov2")
class DINOv2Backbone(Backbone):
    def __init__(self, model: str = "dinov2_vitb14", image_size: int = 224):
        super().__init__()
        self.model_name, self.image_size = model, image_size
        self.model = torch.hub.load("facebookresearch/dinov2", model).eval()
        self.dim = _DIMS[model]
        self.patch_grid = image_size // 14

    def cache_key(self):
        return {"name": "dinov2", "model": self.model_name, "image_size": self.image_size}

    def transform(self):
        return T.Compose([T.Resize((self.image_size, self.image_size)), T.ToTensor(), T.Normalize(*_IMAGENET)])

    def to(self, device):
        self.model.to(device)
        return super().to(device)

    @torch.no_grad()
    def _forward(self, images):
        return self.model.forward_features(images.to(self.device))

    def encode_images(self, images):
        return self._forward(images)["x_norm_clstoken"].float()

    def encode_patches(self, images):
        return self._forward(images)["x_norm_patchtokens"].float()
