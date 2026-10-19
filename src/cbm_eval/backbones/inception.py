"""torchvision Inception-v3 with ImageNet weights: the backbone of Koh et al., "Concept Bottleneck
Models" (ICML 2020).

Frozen for feature caching like every backbone; ``trainable``, so a training stage with
``finetune:`` can optimize a copy end to end as Koh et al. do. Patch features are the final
``Mixed_7c`` map (8x8 at 299 px). The weights are the authors' (``inception_v3_google-1a9a5a14``
equals torchvision's ``IMAGENET1K_V1``). By default inputs get ImageNet normalization and eval images
are resized to a square; ``mean: [0.5, 0.5, 0.5], std: [2, 2, 2]`` and ``eval_resize: false``
reproduce Koh et al.'s code (which scales inputs that way before ``transform_input`` and
center-crops eval images without resizing). The auxiliary tap (``aux_dim`` 768) is the pooled output
of the ImageNet-trained ``AuxLogits`` convolutions on ``Mixed_6e``, without their classifier; a
training stage with ``aux:`` puts Koh et al.'s auxiliary concept head on it.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T

from ..registry import BACKBONES
from .base import Backbone

_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_TRUNK = ("Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3",
          "maxpool2", "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d",
          "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c")
_AUX_AT = _TRUNK.index("Mixed_6e") + 1  # the auxiliary tap reads the trunk up to here


class InceptionAuxTap(nn.Module):
    """torchvision's ``InceptionAux`` without its classifier: ``Mixed_6e`` maps (B, 768, 17, 17) -> (B, 768)."""

    def __init__(self, aux: nn.Module):
        super().__init__()
        self.conv0, self.conv1 = aux.conv0, aux.conv1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.conv0(F.avg_pool2d(x, kernel_size=5, stride=3)))
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)


class InceptionEncoder(nn.Module):
    """Inception-v3 up to the pooled 2048-d features, with the network's own dropout (active
    only in train mode, i.e. while fine-tuning). With ``aux`` it also keeps the auxiliary tap."""

    def __init__(self, net: nn.Module, aux: bool = False):
        super().__init__()
        self.transform_input = net.transform_input
        self.trunk = nn.Sequential(*(getattr(net, name) for name in _TRUNK))
        self.dropout = net.dropout
        self.aux = InceptionAuxTap(net.AuxLogits) if aux else None

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:  # ImageNet-normalized -> the [-1, 1] range of the TF port
            x = torch.cat([x[:, 0:1] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5,
                           x[:, 1:2] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5,
                           x[:, 2:3] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5], 1)
        return x

    def patches(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(self._input(x)).flatten(2).transpose(1, 2)  # (B, P, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.patches(x).mean(1))  # (B, 2048)

    def aux_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.aux(self.trunk[:_AUX_AT](self._input(x)))  # (B, 768)

    def forward_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """((B, 2048), (B, 768)): ``forward`` and the auxiliary tap, from one pass through the trunk."""
        mid = self.trunk[:_AUX_AT](self._input(x))
        final = self.trunk[_AUX_AT:](mid).flatten(2).mean(2)
        return self.dropout(final), self.aux(mid)


@BACKBONES.register("inception")
class InceptionBackbone(Backbone):
    """``weights`` is a torchvision ``Inception_V3_Weights`` name; they load on first use.
    ``mean`` / ``std`` normalize inputs (None: ImageNet's). ``eval_resize``: eval images are resized
    to ``image_size`` squared; if False they are center-cropped to it at their own resolution."""

    trainable = True
    aux_dim = 768

    def __init__(self, weights: str | None = "IMAGENET1K_V1", image_size: int = 299,
                 mean: list[float] | None = None, std: list[float] | None = None, eval_resize: bool = True):
        super().__init__()
        self.kw = {"weights": weights, "image_size": image_size}
        # Only non-default preprocessing enters the key, so features cached before these options stay valid.
        self.kw |= {k: v for k, v, default in (("mean", mean, None), ("std", std, None),
                                               ("eval_resize", eval_resize, True)) if v != default}
        self.norm = T.Normalize(mean or _IMAGENET[0], std or _IMAGENET[1])
        self.eval_resize = eval_resize
        self.dim, self.patch_grid = 2048, (image_size - 299) // 32 + 8
        self._encoder: InceptionEncoder | None = None

    def cache_key(self):
        return {"name": "inception", **self.kw}

    def transform(self):
        size = self.kw["image_size"]
        geometry = T.Resize((size, size)) if self.eval_resize else T.CenterCrop(size)
        return T.Compose([geometry, T.ToTensor(), self.norm])

    def train_transform(self):
        """Koh et al.'s augmentation (after Cui et al., 2018): color jitter, random resized crop, flip."""
        return T.Compose([T.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                          T.RandomResizedCrop(self.kw["image_size"]), T.RandomHorizontalFlip(),
                          T.ToTensor(), self.norm])

    @property
    def frozen(self) -> InceptionEncoder:
        if self._encoder is None:
            import torchvision

            net = torchvision.models.inception_v3(weights=self.kw["weights"], aux_logits=True,
                                                  init_weights=self.kw["weights"] is None)
            self._encoder = InceptionEncoder(net, aux=True).eval().requires_grad_(False).to(self.device)
        return self._encoder

    def encoder(self, aux: bool = False) -> nn.Module:
        enc = copy.deepcopy(self.frozen)
        if not aux:  # keeps the state dict (``encoder.pt``) to the encoder proper
            enc.aux = None
        return enc.requires_grad_(True)

    def to(self, device):
        if self._encoder is not None:
            self._encoder.to(device)
        return super().to(device)

    @torch.no_grad()
    def encode_patches(self, images):
        return self.frozen.patches(images.to(self.device)).float()

    def encode_images(self, images):
        return self.encode_patches(images).mean(1)

    @torch.no_grad()
    def encode_aux(self, images):
        return self.frozen.aux_features(images.to(self.device)).float()
