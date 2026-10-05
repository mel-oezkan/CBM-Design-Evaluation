"""Backbone interface: frozen image encoders, optionally with a text tower in the same space."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class Backbone(ABC):
    dim: int
    patch_grid: int | None = None  # side length of the patch grid, if patch features are available
    has_text: bool = False

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    @abstractmethod
    def cache_key(self) -> dict[str, Any]: ...

    def transform(self) -> Callable | None:
        return None

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
