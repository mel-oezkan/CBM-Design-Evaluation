"""Fixed random projection of latent patches; pairs with the synthetic dataset for tests."""

from __future__ import annotations

import torch

from ..data.synthetic import toy_latent
from ..registry import BACKBONES
from .base import Backbone


@BACKBONES.register("toy")
class ToyBackbone(Backbone):
    has_text = True

    def __init__(self, latent_dim: int = 64, dim: int = 128, grid: int = 4, seed: int = 0):
        super().__init__()
        self.latent_dim, self.dim, self.patch_grid, self.seed = latent_dim, dim, grid, seed
        g = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(latent_dim, dim, generator=g) / latent_dim**0.5

    def cache_key(self):
        return {"name": "toy", "latent_dim": self.latent_dim, "dim": self.dim, "seed": self.seed}

    def encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        return images.float() @ self.proj

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_patches(images).mean(1)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([toy_latent(t, self.latent_dim) for t in texts]) @ self.proj
