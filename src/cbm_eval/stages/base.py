"""One interface per stage. Each registered variant implements exactly one of these."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import torch
from torch import nn

from ..structures import Bag, ConceptSet

if TYPE_CHECKING:
    from ..context import Context

TargetType = Literal["binary", "continuous"]


class Discovery(ABC):
    """Proposes a candidate concept set."""

    @abstractmethod
    def discover(self, ctx: "Context") -> ConceptSet: ...


class Filter(ABC):
    """Prunes a concept set. Filters are chained in config order."""

    @abstractmethod
    def filter(self, concepts: ConceptSet, ctx: "Context") -> ConceptSet: ...


@dataclass
class AlignedConcepts:
    """How concept neurons are tied to concept semantics.

    Either via supervision targets (``targets_fn``) or by construction (``init_weight`` + ``frozen``).
    """

    concepts: ConceptSet
    target_type: TargetType | None = None
    targets_fn: Callable[[str], torch.Tensor] | None = None
    init_weight: torch.Tensor | None = None  # (K, D) in backbone feature space
    frozen: bool = False
    _cache: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    @property
    def has_targets(self) -> bool:
        return self.targets_fn is not None

    def targets(self, split: str) -> torch.Tensor | None:
        if self.targets_fn is None:
            return None
        if split not in self._cache:
            self._cache[split] = self.targets_fn(split).float()
        return self._cache[split]


class Alignment(ABC):
    @abstractmethod
    def fit(self, concepts: ConceptSet, ctx: "Context") -> AlignedConcepts: ...


def instance_rows(x: torch.Tensor, bag: Bag | None = None) -> torch.Tensor:
    """(N, D) features as they are; (N, M, D) instance bags -> the (n_real, D) real instances."""
    if x.dim() == 2:
        return x
    return bag.rows(x) if bag is not None else x.flatten(0, -2)


class ConceptLayer(nn.Module):
    """Backbone features -> concept predictions ``c_hat`` (in target space) -> predictor input.

    Subclasses (one per Generation variant) only decide how ``c_hat`` becomes the representation.
    Inputs are (N, D) image features or (N, M, D) instance bags; everything broadcasts over M.
    """

    def __init__(self, in_dim: int, aligned: AlignedConcepts):
        super().__init__()
        k = len(aligned.concepts)
        self.n_concepts, self.target_type = k, aligned.target_type
        self.linear = nn.Linear(in_dim, k)
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("c_mean", torch.zeros(k))
        self.register_buffer("c_std", torch.ones(k))
        self.frozen = aligned.frozen
        if aligned.init_weight is not None:
            w = aligned.init_weight.float()
            with torch.no_grad():
                self.linear.weight.copy_(w / w.norm(dim=1, keepdim=True).clamp_min(1e-8))
                self.linear.bias.zero_()
        if self.frozen:
            self.linear.requires_grad_(False)

    rep_dim: int

    def fit_input_stats(self, x: torch.Tensor, bag: Bag | None = None) -> None:
        x = instance_rows(x, bag)
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-6))

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor, bag: Bag | None = None) -> None:
        """Standardize untrained (frozen / target-free) concept scores using training statistics."""
        if self.target_type == "continuous" or self.target_type is None:
            raw = self.linear(self.normalize(instance_rows(x, bag)))
            self.c_mean.copy_(raw.mean(0))
            self.c_std.copy_(raw.std(0).clamp_min(1e-6))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def concept_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw per-concept scores; works on (N, D) features and (N, P, D) patch features alike."""
        return (self.linear(self.normalize(x)) - self.c_mean) / self.c_std

    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits) if self.target_type == "binary" else logits

    def concept_params(self) -> list[nn.Parameter]:
        return [p for p in self.linear.parameters() if p.requires_grad]

    def rep_params(self) -> list[nn.Parameter]:
        """Parameters that shape the representation but are not concept predictors (e.g. CEM embeddings)."""
        return []

    @abstractmethod
    def represent(self, c_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor, intervene: tuple[torch.Tensor, torch.Tensor] | None = None):
        """``intervene = (mask, values)`` replaces concept predictions. On instance bags, an
        image-level (N, K) mask or value applies to every instance of the image."""
        c_hat = self.activate(self.concept_logits(x))
        if intervene is not None:
            mask, values = intervene
            if c_hat.dim() == 3:
                mask = mask[:, None] if mask.dim() == 2 else mask
                values = values[:, None] if values.dim() == 2 else values
            c_hat = torch.where(mask, values.to(c_hat), c_hat)
        return c_hat, self.represent(c_hat, x)

    def binarize(self, c_hat: torch.Tensor) -> torch.Tensor:
        return (c_hat > (0.5 if self.target_type == "binary" else 0.0)).float()


class Generation(ABC):
    """Chooses the concept representation passed to the predictor (scalar, embedding, binary)."""

    @abstractmethod
    def build(self, in_dim: int, aligned: AlignedConcepts) -> ConceptLayer: ...


class PredictorHead(nn.Module):
    """Concept representation (+ optionally raw features) -> class logits.

    ``optimizer``/``lr`` override the training stage's optimizer for this head's parameters
    (sparse heads need plain SGD so the proximal step is a true proximal-gradient update).
    ``bag`` is set when inputs are (N, M, ·) instance bags; only bag-aware heads pool over M.
    """

    optimizer: str = "adam"
    lr: float | None = None

    def forward(self, rep: torch.Tensor, x_norm: torch.Tensor, bag: Bag | None = None) -> torch.Tensor:
        raise NotImplementedError

    def phases(self) -> list[list[nn.Parameter]]:
        """Parameter groups fitted one after another (most heads have a single phase)."""
        return [list(self.parameters())]

    def penalty(self) -> torch.Tensor:
        return torch.zeros(())

    def proximal_step(self) -> None:
        """Called after each optimizer step (sparse heads soft-threshold here)."""

    def stats(self) -> dict[str, float]:
        return {}


class Predictor(ABC):
    supports_bags: bool = False  # True when the head pools an instance axis (needed with `instances:`)

    @abstractmethod
    def build(self, rep_dim: int, num_classes: int, feat_dim: int) -> PredictorHead: ...


class Training(ABC):
    """Fits the concept layer and predictor head; returns a log of training metrics.

    ``trains_backbone``: when True, ``PipelineBuilder`` passes ``encoder`` (a trainable copy of the
    backbone's image encoder, see ``Backbone.encoder``) and the stage fits it end to end on images
    from ``ctx.image_split``; afterwards the run's inputs are that encoder's features.
    ``num_workers`` is the dataloader width for those images.
    """

    trains_backbone: bool = False
    num_workers: int = 0

    @abstractmethod
    def fit(self, layer: ConceptLayer, head: PredictorHead, aligned: AlignedConcepts,
            ctx: "Context", encoder: nn.Module | None = None) -> dict[str, float]: ...
