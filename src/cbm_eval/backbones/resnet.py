"""torchvision ResNets with ImageNet weights; patch features are the final conv map."""

from __future__ import annotations

import torch
import torchvision
from torchvision import transforms as T

from ..registry import BACKBONES
from .base import Backbone

_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@BACKBONES.register("resnet")
class ResNetBackbone(Backbone):
    def __init__(self, model: str = "resnet50", weights: str = "DEFAULT", image_size: int = 224):
        super().__init__()
        self.model_name, self.weights, self.image_size = model, weights, image_size
        net = getattr(torchvision.models, model)(weights=weights)
        self.dim = net.fc.in_features
        net.fc = torch.nn.Identity()
        self.trunk = torch.nn.Sequential(*list(net.children())[:-2]).eval()
        self.patch_grid = image_size // 32

    def cache_key(self):
        return {"name": "resnet", "model": self.model_name, "weights": self.weights, "image_size": self.image_size}

    def transform(self):
        return T.Compose([T.Resize((self.image_size, self.image_size)), T.ToTensor(), T.Normalize(*_IMAGENET)])

    def to(self, device):
        self.trunk.to(device)
        return super().to(device)

    @torch.no_grad()
    def encode_patches(self, images):
        fmap = self.trunk(images.to(self.device))  # (B, D, h, w)
        return fmap.flatten(2).transpose(1, 2).float()

    def encode_images(self, images):
        return self.encode_patches(images).mean(1)
