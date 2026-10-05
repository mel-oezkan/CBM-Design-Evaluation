"""Data containers passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch


@dataclass
class FeatureSplit:
    """Cached backbone outputs and annotations for one dataset split.

    ``patch_features`` and ``part_masks`` are only present when localization is evaluated.
    ``concept_labels`` holds human annotations aligned with ``Dataset.concept_names``.
    """

    features: torch.Tensor  # (N, D)
    labels: torch.Tensor  # (N,)
    attrs: torch.Tensor | None = None  # (N,) spurious attribute, e.g. background
    groups: torch.Tensor | None = None  # (N,) group id = label * n_attrs + attr
    concept_labels: torch.Tensor | None = None  # (N, K_h) in {0, 1}
    patch_features: torch.Tensor | None = None  # (N, P, D)
    part_masks: torch.Tensor | None = None  # (N, K_h, P) bool: where each annotated concept is
    paths: list[str] | None = None  # image paths, for VLM discovery

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class PartSplit:
    """Part annotations for one split, in the frame of the concept maps (see ``data/parts.py``).

    ``concept_keypoints`` / ``concept_segs`` say which dataset concepts (``Dataset.concept_names``)
    belong to each keypoint / segmentation group.
    """

    points: torch.Tensor  # (N, Q, max_points, 2) normalized (x, y); NaN if not visible
    keypoint_groups: list[str]
    concept_keypoints: torch.Tensor  # (K_h, Q) bool
    segs: torch.Tensor | None = None  # (N, G, S, S) bool; all False if the image has no mask for a part
    seg_groups: list[str] = field(default_factory=list)
    concept_segs: torch.Tensor | None = None  # (K_h, G) bool

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class Bag:
    """Marks the real instances in a zero-padded ``(N, M, ...)`` batch of per-image instance sets.

    ``area`` is each instance's share of the image, used by area-normalized MIL heads.
    """

    mask: torch.Tensor  # (N, M) bool
    area: torch.Tensor | None = None  # (N, M)

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, idx) -> "Bag":
        return Bag(self.mask[idx], None if self.area is None else self.area[idx])

    def to(self, device) -> "Bag":
        return Bag(self.mask.to(device), None if self.area is None else self.area.to(device))

    def rows(self, x: torch.Tensor) -> torch.Tensor:
        """(N, M, D) -> the (n_real, D) rows of real instances."""
        return x[self.mask]

    def mean(self, values: torch.Tensor) -> torch.Tensor:
        """Mean of (N, M) per-instance values over real instances."""
        m = self.mask.to(values.dtype)
        return (values * m).sum() / m.sum().clamp_min(1)

    def max(self, values: torch.Tensor) -> torch.Tensor:
        """(N, M, K) -> (N, K) max over real instances (0 for an image without any)."""
        out = values.masked_fill(~self.mask[..., None], float("-inf")).amax(1)
        return torch.where(self.mask.any(1, keepdim=True), out, torch.zeros_like(out))


@dataclass
class InstanceSplit:
    """Per-image instance sets (patches or segments) for one split, zero-padded to M per image.

    ``teacher_features`` is only stored when the teacher differs from the backbone. ``regions``
    marks which cells of the backbone patch grid each instance covers (for localization);
    with ``grid_aligned`` instance i *is* patch i and ``regions`` is left out.
    """

    features: torch.Tensor  # (N, M, D) in backbone space
    mask: torch.Tensor  # (N, M) bool
    area: torch.Tensor  # (N, M) share of the image
    teacher_features: torch.Tensor | None = None  # (N, M, D_t)
    regions: torch.Tensor | None = None  # (N, M, G*G) bool
    concepts: torch.Tensor | None = None  # (N, M) index of the concept that proposed a segment, -1 if none
    grid_aligned: bool = False

    def __len__(self) -> int:
        return len(self.features)

    @property
    def bag(self) -> Bag:
        return Bag(self.mask, self.area)


@dataclass
class ConceptSet:
    """Candidate concepts. ``vectors`` are directions in backbone feature space (e.g. SAE latents)."""

    names: list[str]
    vectors: torch.Tensor | None = None  # (K, D)
    meta: list[dict[str, Any]] = field(default_factory=list)  # per-concept provenance

    def __post_init__(self) -> None:
        if not self.meta:
            self.meta = [{} for _ in self.names]
        if len(self.meta) != len(self.names):
            raise ValueError("meta must have one entry per concept")
        if self.vectors is not None and len(self.vectors) != len(self.names):
            raise ValueError("vectors must have one row per concept")

    def __len__(self) -> int:
        return len(self.names)

    def subset(self, idx: list[int]) -> "ConceptSet":
        return replace(
            self,
            names=[self.names[i] for i in idx],
            vectors=None if self.vectors is None else self.vectors[idx],
            meta=[self.meta[i] for i in idx],
        )

    def to_json(self) -> dict[str, Any]:
        return {"names": self.names, "meta": self.meta, "has_vectors": self.vectors is not None}
