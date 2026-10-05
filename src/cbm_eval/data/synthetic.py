"""Synthetic spurious-correlation dataset with known concepts and concept locations.

Each "image" is a grid of latent patch vectors. A present concept places its prototype in one
random patch (giving ground-truth localization); the spurious attribute adds a background
vector to every patch and is correlated with the label on train/val but not on test. Pair it
with the ``toy`` backbone, whose text encoder maps a concept name to that same prototype, so
every stage (including text-based alignment) runs end-to-end without downloads.
"""

from __future__ import annotations

import hashlib

import torch

from ..registry import DATASETS
from .base import ImageDataset, Sample


def toy_latent(name: str, dim: int) -> torch.Tensor:
    """Deterministic unit vector for a name; shared by the synthetic data and the toy backbone."""
    seed = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    v = torch.randn(dim, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


@DATASETS.register("synthetic")
class SyntheticDataset(ImageDataset):
    name = "synthetic"
    n_attrs = 2

    def __init__(
        self,
        n_classes: int = 4,
        n_concepts: int = 24,
        n_distractors: int = 8,
        latent_dim: int = 64,
        grid: int = 4,
        sizes: dict[str, int] | None = None,
        spurious_corr: float = 0.9,
        p_on: float = 0.8,
        p_off: float = 0.1,
        noise: float = 0.3,
        concept_strength: float = 2.0,
        background_strength: float = 0.25,
        residual_strength: float = 0.5,
        data_seed: int = 0,
    ):
        self.kw = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.n_classes, self.n_concepts, self.latent_dim, self.grid = n_classes, n_concepts, latent_dim, grid
        self.sizes = sizes or {"train": 1200, "val": 400, "test": 800}
        self.spurious_corr, self.noise, self.concept_strength = spurious_corr, noise, concept_strength
        self.background_strength, self.residual_strength = background_strength, residual_strength
        self._concepts = [f"concept_{j:02d}" for j in range(n_concepts)]
        # Distractors are plausible-looking concept names that never appear in the images.
        self.distractors = [f"distractor_{j:02d}" for j in range(n_distractors)]
        g = torch.Generator().manual_seed(data_seed)
        on = torch.rand(n_classes, n_concepts, generator=g) < 0.35
        self.class_concept_prob = torch.where(on, torch.tensor(p_on), torch.tensor(p_off))
        self.data_seed = data_seed
        self._cache: dict[str, list[Sample]] = {}

    @property
    def class_names(self) -> list[str]:
        return [f"class_{c}" for c in range(self.n_classes)]

    @property
    def concept_names(self) -> list[str]:
        return self._concepts

    def cache_key(self):
        return {"name": self.name, **self.kw}

    def samples(self, split: str) -> list[Sample]:
        if split not in self._cache:
            self._cache[split] = self._generate(split)
        return self._cache[split]

    def _generate(self, split: str) -> list[Sample]:
        n = self.sizes[split]
        g = torch.Generator().manual_seed(self.data_seed * 1000 + ["train", "val", "test"].index(split) + 1)
        P, L = self.grid * self.grid, self.latent_dim
        protos = torch.stack([toy_latent(c, L) for c in self._concepts])
        backgrounds = [toy_latent(f"background_{a}", L) for a in range(self.n_attrs)]
        residuals = [toy_latent(f"residual_{c}", L) for c in range(self.n_classes)]
        rho = self.spurious_corr if split != "test" else 1.0 / self.n_attrs

        out = []
        for _ in range(n):
            y = int(torch.randint(self.n_classes, (1,), generator=g))
            aligned = torch.rand(1, generator=g).item() < rho
            a = y % self.n_attrs if aligned else int(torch.randint(self.n_attrs, (1,), generator=g))
            present = (torch.rand(self.n_concepts, generator=g) < self.class_concept_prob[y]).int()
            x = self.noise * torch.randn(P, L, generator=g) + self.background_strength * backgrounds[a]
            keypoints = {}
            for k in present.nonzero().flatten().tolist():
                p = int(torch.randint(P, (1,), generator=g))
                x[p] += self.concept_strength * protos[k]
                row, col = divmod(p, self.grid)
                keypoints[k] = (col + 0.5, row + 0.5, self.grid, self.grid)
            x[int(torch.randint(P, (1,), generator=g))] += self.residual_strength * residuals[y]
            out.append(Sample(image=x, label=y, attr=a, concepts=present.tolist(), keypoints=keypoints))
        return out
