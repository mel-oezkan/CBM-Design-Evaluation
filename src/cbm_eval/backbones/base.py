"""Backbone interface: frozen image encoders, optionally with a text tower in the same space.

Backbones with ``trainable = True`` can also hand out a fine-tunable copy of their image encoder
(``encoder()``), which a training stage with ``finetune:`` optimizes end to end; the backbone
itself, and its cached features, stay frozen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class Backbone(ABC):
    dim: int
    patch_grid: int | None = None  # side length of the patch grid, if patch features are available
    has_text: bool = False
    trainable: bool = False  # True if ``encoder()`` returns a fine-tunable copy (training ``finetune:``)

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    @abstractmethod
    def cache_key(self) -> dict[str, Any]: ...

    def transform(self) -> Callable | None:
        return None

    def train_transform(self) -> Callable | None:
        """Transform for images seen while fine-tuning (augmentation); defaults to ``transform()``."""
        return self.transform()

    def encoder(self) -> torch.nn.Module:
        """A fresh, trainable copy of the image encoder: ``transform()``-ed images (B, ...) -> (B, D),
        equal to ``encode_images`` in eval mode."""
        raise NotImplementedError(f"{type(self).__name__} cannot be fine-tuned (trainable = False)")

    def to(self, device: torch.device) -> "Backbone":
        self.device = device
        return self

    @abstractmethod
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """(B, ...) -> (B, D) global image features."""

    def encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        """(B, ...) -> (B, P, D) patch features in the same space as ``encode_images``."""
        raise NotImplementedError(f"{type(self).__name__} does not provide patch features")

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """list[str] -> (T, D) text embeddings in the image feature space."""
        raise NotImplementedError(f"{type(self).__name__} has no text encoder")
