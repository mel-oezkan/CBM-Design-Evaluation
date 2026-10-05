"""Dataset interface. Datasets yield raw samples; ``features.py`` turns them into cached tensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

SPLITS = ("train", "val", "test")


@dataclass
class PartAnnotation:
    """One image's part annotations in pixel coordinates of the original image.

    ``points`` maps a keypoint group (e.g. "wing") to its visible keypoints (left and right wing);
    ``masks`` maps a segmentation group to the mask files whose union is that part.
    """

    size: tuple[int, int]  # (w, h)
    points: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    masks: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class PartSpec:
    """Which parts a dataset annotates, and which concepts (``concept_names``) belong to each."""

    keypoint_groups: list[str]
    seg_groups: list[str]
    concept_keypoints: torch.Tensor  # (K_h, Q) bool
    concept_segs: torch.Tensor  # (K_h, G) bool
    max_points: int = 2  # keypoints per group (left / right)
    source: str = ""  # where the masks come from; part of the cache key

    def cache_key(self) -> dict[str, Any]:
        return {"kp": self.keypoint_groups, "seg": self.seg_groups, "source": self.source,
                "ck": self.concept_keypoints.int().tolist(), "cs": self.concept_segs.int().tolist()}


@dataclass
class Sample:
    image: Any  # path (str) for real datasets, tensor for synthetic ones
    label: int
    attr: int = 0
    concepts: list[int] | None = None  # human concept annotations, aligned with concept_names
    keypoints: dict[int, tuple[float, float, float, float]] | None = None  # concept idx -> (x, y, w, h) of image
    parts: PartAnnotation | None = None  # keypoints and segmentation masks per body part


class ImageDataset(ABC):
    """A labelled dataset with an optional spurious attribute and optional concept annotations.

    ``splits`` are the splits ``samples`` serves. A test-only dataset (a shifted version of a
    training dataset, e.g. CUB as paintings) sets ``("test",)`` and can only be used under
    ``eval_datasets:``, where its classes and concepts are matched to the training dataset's by name.
    """

    name: str = "dataset"
    n_attrs: int = 1
    splits: tuple[str, ...] = SPLITS

    @property
    @abstractmethod
    def class_names(self) -> list[str]: ...

    @property
    def concept_names(self) -> list[str] | None:
        """Names of human-annotated concepts, or None if the dataset has no annotations."""
        return None

    def class_concepts(self) -> torch.Tensor | None:
        """(C, K_h) class-level concept labels (rows ``class_names``, columns ``concept_names``), or
        None. Eval datasets without their own concept annotations inherit labels through this."""
        return None

    @abstractmethod
    def samples(self, split: str) -> list[Sample]: ...

    def cache_key(self) -> dict[str, Any]:
        """Everything that changes the data; used to key the feature cache."""
        return {"name": self.name}

    def part_spec(self) -> PartSpec | None:
        """Part groups and the concept -> part mapping, or None if the dataset has no part annotations."""
        return None

    def torch_split(self, split: str, transform: Callable | None, grid: int | None = None) -> TorchDataset:
        return _SampleDataset(self.samples(split), transform, len(self.concept_names or []), grid)


def keypoint_mask(keypoints, n_concepts: int, grid: int, radius: int = 1) -> torch.Tensor:
    """(K, grid*grid) bool mask marking patches within ``radius`` cells of each concept keypoint.

    Assumes images are resized (not cropped) to a square, so normalized coordinates map
    directly onto the patch grid.
    """
    mask = torch.zeros(n_concepts, grid, grid, dtype=torch.bool)
    for k, (x, y, w, h) in (keypoints or {}).items():
        cx = min(int(x / w * grid), grid - 1)
        cy = min(int(y / h * grid), grid - 1)
        mask[k, max(cy - radius, 0) : cy + radius + 1, max(cx - radius, 0) : cx + radius + 1] = True
    return mask.flatten(1)


class _SampleDataset(TorchDataset):
    def __init__(self, samples: list[Sample], transform, n_concepts: int, grid: int | None):
        self.samples, self.transform, self.n_concepts, self.grid = samples, transform, n_concepts, grid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.samples[i]
        image = Image.open(s.image).convert("RGB") if isinstance(s.image, str) else s.image
        if self.transform is not None:
            image = self.transform(image)
        item = {"image": image, "label": s.label, "attr": s.attr, "index": i}
        if s.concepts is not None:
            item["concepts"] = torch.tensor(s.concepts, dtype=torch.float32)
        if self.grid is not None and self.n_concepts:
            item["parts"] = keypoint_mask(s.keypoints, self.n_concepts, self.grid)
        return item
