"""torchvision Inception-v3 with ImageNet weights: the backbone of Koh et al., "Concept Bottleneck
Models" (ICML 2020).

Frozen for feature caching like every backbone; ``trainable``, so a training stage with
``finetune:`` can optimize a copy end to end as Koh et al. do. Patch features are the final
``Mixed_7c`` map (8x8 at 299 px). Deviations from Koh et al.: standard ImageNet input
normalization (their code scales inputs by mean 0.5 / std 2 before ``transform_input``), the
auxiliary classifier is not used, and eval images are resized rather than center-cropped.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torchvision import transforms as T

from ..registry import BACKBONES
from .base import Backbone

_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_TRUNK = ("Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
          "maxpool2", "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d",
          "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c")


class InceptionEncoder(nn.Module):
    """Inception-v3 up to the pooled 2048-d features, with the network's own dropout (active
    only in train mode, i.e. while fine-tuning)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.transform_input = net.transform_input
        self.trunk = nn.Sequential(*(getattr(net, name) for name in _TRUNK))
        self.dropout = net.dropout

    def patches(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:  # ImageNet-normalized -> the [-1, 1] range of the TF port
            x = torch.cat([x[:, 0:1] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5,
                           x[:, 1:2] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5,
                           x[:, 2:3] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5], 1)
        return self.trunk(x).flatten(2).transpose(1, 2)  # (B, P, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.patches(x).mean(1))  # (B, 2048)


@BACKBONES.register("inception")
class InceptionBackbone(Backbone):
    """``weights`` is a torchvision ``Inception_V3_Weights`` name; they load on first use."""

    trainable = True

    def __init__(self, weights: str | None = "IMAGENET1K_V1", image_size: int = 299):
        super().__init__()
        self.kw = {"weights": weights, "image_size": image_size}
        self.dim, self.patch_grid = 2048, (image_size - 299) // 32 + 8
        self._encoder: InceptionEncoder | None = None

    def cache_key(self):
        return {"name": "inception", **self.kw}

    def transform(self):
        size = self.kw["image_size"]
        return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(*_IMAGENET)])

    def train_transform(self):
        """Koh et al.'s augmentation (after Cui et al., 2018): color jitter, random resized crop, flip."""
        return T.Compose([T.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                          T.RandomResizedCrop(self.kw["image_size"]), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(*_IMAGENET)])

    @property
    def frozen(self) -> InceptionEncoder:
        if self._encoder is None:
            import torchvision

            net = torchvision.models.inception_v3(weights=self.kw["weights"], aux_logits=True,
                                                  init_weights=self.kw["weights"] is None)
            self._encoder = InceptionEncoder(net).eval().requires_grad_(False).to(self.device)
        return self._encoder

    def encoder(self) -> nn.Module:
        return copy.deepcopy(self.frozen).requires_grad_(True)

    def to(self, device):
        if self._encoder is not None:
            self._encoder.to(device)
        return super().to(device)

    @torch.no_grad()
    def encode_patches(self, images):
        return self.frozen.patches(images.to(self.device)).float()

    def encode_images(self, images):
        return self.encode_patches(images).mean(1)
