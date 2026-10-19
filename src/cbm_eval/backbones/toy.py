"""Fixed random projection of latent patches; pairs with the synthetic dataset for tests."""

from __future__ import annotations

import torch
from torch import nn

from ..data.synthetic import toy_latent
from ..registry import BACKBONES
from .base import Backbone


class _ToyEncoder(nn.Module):
    def __init__(self, proj: torch.Tensor, aux_proj: torch.Tensor | None = None):
        super().__init__()
        self.proj = nn.Parameter(proj.clone())
        self.aux_proj = None if aux_proj is None else nn.Parameter(aux_proj.clone())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images.float() @ self.proj).mean(1)

    def forward_aux(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self(images)
        return x, x @ self.aux_proj


@BACKBONES.register("toy")
class ToyBackbone(Backbone):
    """``aux_dim`` adds an auxiliary tap: a fixed projection of the image features, so that an
    auxiliary loss reaches the fine-tuned projection as Koh et al.'s reaches the Inception trunk."""

    has_text = True
    trainable = True  # the projection can be fine-tuned, so tests cover ``finetune:`` offline

    def __init__(self, latent_dim: int = 64, dim: int = 128, grid: int = 4, seed: int = 0,
                 aux_dim: int | None = None):
        super().__init__()
        self.latent_dim, self.dim, self.patch_grid, self.seed, self.aux_dim = latent_dim, dim, grid, seed, aux_dim
        g = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(latent_dim, dim, generator=g) / latent_dim**0.5
        self.aux_proj = None if aux_dim is None else torch.randn(dim, aux_dim, generator=g) / dim**0.5

    def cache_key(self):
        key = {"name": "toy", "latent_dim": self.latent_dim, "dim": self.dim, "seed": self.seed}
        return key if self.aux_dim is None else key | {"aux_dim": self.aux_dim}  # older caches stay valid

    def encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        return images.float() @ self.proj

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_patches(images).mean(1)

    def encode_aux(self, images: torch.Tensor) -> torch.Tensor:
        if self.aux_proj is None:
            return super().encode_aux(images)
        return self.encode_images(images) @ self.aux_proj

    def encoder(self, aux: bool = False) -> nn.Module:
        if aux and self.aux_proj is None:
            raise ValueError("ToyBackbone(aux_dim=None) has no auxiliary tap; set `backbone.aux_dim`")
        return _ToyEncoder(self.proj, self.aux_proj if aux else None).to(self.device)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([toy_latent(t, self.latent_dim) for t in texts]) @ self.proj
