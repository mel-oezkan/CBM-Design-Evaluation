"""Training: independent, sequential, or joint fitting of the concept layer and predictor head.

All variants train on cached (frozen) backbone features, on CPU by default (``device: auto`` uses
a GPU when there is one). Inputs are (N, D) image features or (N, M, D) instance bags.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import TRAINING
from ..structures import Bag
from ..utils import resolve_device
from .base import AlignedConcepts, ConceptLayer, PredictorHead, Training

CONCEPT_LOSSES = ("auto", "mse", "bce", "cosine")


def concept_loss(layer: ConceptLayer, x: torch.Tensor, targets: torch.Tensor, bag: Bag | None = None,
                 kind: str = "auto") -> torch.Tensor:
    """Mean concept loss over images, or over the real instances of a bag.

    ``kind``: ``auto`` (BCE for binary targets, else MSE), ``mse``, ``bce``, or ``cosine``
    (1 - cosine between each concept vector and its target vector, as in SEG-MIL-CBM).
    Image-level (N, K) targets on a bag supervise the max over instances (standard MIL).
    """
    logits = layer.concept_logits(x)
    if logits.dim() == 3 and targets.dim() == 2:
        logits = bag.max(logits) if bag is not None else logits.amax(1)
        bag = None
    if kind == "auto":
        kind = "bce" if layer.target_type == "binary" else "mse"
    if kind == "bce":
        per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(-1)
    elif kind == "mse":
        per = (logits - targets).pow(2).mean(-1)
    elif kind == "cosine":
        per = 1 - F.cosine_similarity(logits, targets, dim=-1, eps=1e-8)
    else:
        raise ValueError(f"Unknown concept loss '{kind}' (use one of {CONCEPT_LOSSES})")
    return bag.mean(per) if bag is not None else per.mean()


@dataclass
class Batch:
    """Inputs, labels, concept targets and (for instance bags) the bag, indexed together."""

    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor | None = None
    bag: Bag | None = None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> "Batch":
        return Batch(self.x[idx], self.y[idx], None if self.t is None else self.t[idx],
                     None if self.bag is None else self.bag[idx])

    def to(self, device) -> "Batch":
        return Batch(self.x.to(device), self.y.to(device), None if self.t is None else self.t.to(device),
                     None if self.bag is None else self.bag.to(device))


ParamGroup = tuple[list[nn.Parameter], str, float]  # (params, "adam" | "sgd", lr)


def _optimizers(groups: list[ParamGroup]) -> list[torch.optim.Optimizer]:
    opts = []
    for params, kind, lr in groups:
        params = [p for p in params if p.requires_grad]
        if params:
            opts.append(torch.optim.SGD(params, lr=lr) if kind == "sgd" else torch.optim.Adam(params, lr=lr))
    return opts


def optimize(groups: list[ParamGroup], modules: list[nn.Module], n: int,
             train_loss: Callable[[torch.Tensor], torch.Tensor], val_loss: Callable[[], float],
             epochs: int, batch_size: int, patience: int, seed: int,
             after_step: Callable[[], None] | None = None) -> dict[str, float]:
    """Minibatch training with early stopping on ``val_loss``; restores the best weights."""
    opts = _optimizers(groups)
    if not opts:
        return {"epochs": 0, "val_loss": val_loss()}
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, epoch = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        for m in modules:
            m.train()
        for idx in torch.randperm(n, generator=g).split(batch_size):
            loss = train_loss(idx)
            for opt in opts:
                opt.zero_grad()
            loss.backward()
            for opt in opts:
                opt.step()
            if after_step is not None:
                after_step()
        for m in modules:
            m.eval()
        with torch.no_grad():
            v = val_loss()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = [copy.deepcopy(m.state_dict()) for m in modules]
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        for m, s in zip(modules, best_state):
            m.load_state_dict(s)
    return {"epochs": epoch, "val_loss": best}


class _Base(Training):
    def __init__(self, epochs: int = 200, concept_epochs: int | None = None, batch_size: int = 256,
                 lr: float = 1e-3, patience: int = 10, concept_loss: str = "auto", device: str = "cpu"):
        if concept_loss not in CONCEPT_LOSSES:
            raise ValueError(f"Unknown concept loss '{concept_loss}' (use one of {CONCEPT_LOSSES})")
        self.epochs, self.concept_epochs = epochs, concept_epochs or epochs
        self.batch_size, self.lr, self.patience = batch_size, lr, patience
        self.concept_loss, self.device = concept_loss, device

    def fit(self, layer, head, aligned, ctx):
        device = resolve_device(self.device)
        tr, va = (_batch(ctx, aligned, s).to(device) for s in ("train", "val"))
        layer.to(device)
        head.to(device)
        try:
            return self._fit(layer, head, aligned, tr, va, ctx.seed)
        finally:
            layer.cpu()
            head.cpu()

    def _fit(self, layer: ConceptLayer, head: PredictorHead, aligned: AlignedConcepts, tr: Batch, va: Batch,
             seed: int) -> dict[str, float]:
        raise NotImplementedError

    def _prepare(self, layer: ConceptLayer, aligned: AlignedConcepts, tr: Batch) -> None:
        layer.fit_input_stats(tr.x, tr.bag)
        if not aligned.has_targets:
            layer.calibrate(tr.x, tr.bag)

    def _concept_loss(self, layer: ConceptLayer, b: Batch) -> torch.Tensor:
        return concept_loss(layer, b.x, b.t, b.bag, self.concept_loss)

    def _task_loss(self, logits: torch.Tensor, b: Batch) -> torch.Tensor:
        """Class logits (N, C) -> scalar task loss. The one place every variant computes it; override
        it in a subclass (registered under a new name) to try another task loss."""
        return F.cross_entropy(logits, b.y)

    def _fit_concepts(self, layer, aligned, tr: Batch, va: Batch, seed) -> dict[str, float]:
        if not aligned.has_targets or layer.frozen:
            return {}
        log = optimize([(layer.concept_params(), "adam", self.lr)], [layer], len(tr),
                       lambda idx: self._concept_loss(layer, tr[idx]),
                       lambda: float(self._concept_loss(layer, va)),
                       self.concept_epochs, self.batch_size, self.patience, seed)
        return {"concept_epochs": log["epochs"], "concept_val_loss": log["val_loss"]}

    def _head_group(self, head: PredictorHead, params: list[nn.Parameter]):
        return (params, head.optimizer, head.lr or self.lr)

    def _fit_head(self, layer, head, tr: Batch, va: Batch, rep_fn: Callable[[Batch], torch.Tensor],
                  seed) -> dict[str, float]:
        out: dict[str, float] = {}

        def task_loss(b: Batch) -> torch.Tensor:
            return self._task_loss(head(rep_fn(b), layer.normalize(b.x), b.bag), b)

        for i, phase in enumerate(head.phases()):
            groups = [self._head_group(head, phase)] + ([(layer.rep_params(), "adam", self.lr)] if i == 0 else [])
            log = optimize(groups, [layer, head], len(tr), lambda idx: task_loss(tr[idx]) + head.penalty(),
                           lambda: float(task_loss(va)), self.epochs, self.batch_size,
                           self.patience, seed + i, after_step=head.proximal_step)
            out[f"head_phase{i}_epochs"] = log["epochs"]
            out["head_val_loss"] = log["val_loss"]
        return out


def _batch(ctx, aligned: AlignedConcepts, split: str) -> Batch:
    x, bag = ctx.inputs(split)
    return Batch(x.float(), ctx.split(split).labels, aligned.targets(split), bag)


@TRAINING.register("sequential")
class Sequential(_Base):
    """Fit concepts to targets, freeze, then fit the predictor on predicted concepts."""

    def _fit(self, layer, head, aligned, tr, va, seed):
        self._prepare(layer, aligned, tr)
        log = self._fit_concepts(layer, aligned, tr, va, seed)

        def rep(b):
            with torch.no_grad():
                c_hat = layer.activate(layer.concept_logits(b.x))
            return layer.represent(c_hat, b.x)

        return log | self._fit_head(layer, head, tr, va, rep, seed + 1)


@TRAINING.register("independent")
class Independent(_Base):
    """Fit concepts to targets; fit the predictor on the *true* concept targets."""

    def fit(self, layer, head, aligned, ctx):
        if not aligned.has_targets:
            raise RuntimeError("independent training needs concept targets (not available with this alignment)")
        return super().fit(layer, head, aligned, ctx)

    def _fit(self, layer, head, aligned, tr, va, seed):
        self._prepare(layer, aligned, tr)
        log = self._fit_concepts(layer, aligned, tr, va, seed)
        return log | self._fit_head(layer, head, tr, va, lambda b: layer.represent(b.t, b.x), seed + 1)


@TRAINING.register("joint")
class Joint(_Base):
    """End-to-end: task loss + ``concept_weight`` x concept loss (when targets exist)."""

    def __init__(self, concept_weight: float = 1.0, **kw):
        super().__init__(**kw)
        self.concept_weight = concept_weight

    def _fit(self, layer, head, aligned, tr, va, seed):
        self._prepare(layer, aligned, tr)
        use_c = aligned.has_targets and self.concept_weight > 0

        def total(b: Batch) -> torch.Tensor:
            _, rep = layer(b.x)
            loss = self._task_loss(head(rep, layer.normalize(b.x), b.bag), b)
            if use_c:
                loss = loss + self.concept_weight * self._concept_loss(layer, b)
            return loss

        groups = [(layer.concept_params() + layer.rep_params(), "adam", self.lr),
                  self._head_group(head, list(head.parameters()))]
        log = optimize(groups, [layer, head], len(tr), lambda idx: total(tr[idx]) + head.penalty(),
                       lambda: float(total(va)), self.epochs, self.batch_size, self.patience, seed,
                       after_step=head.proximal_step)
        out = {"joint_epochs": log["epochs"], "joint_val_loss": log["val_loss"]}
        if use_c:
            with torch.no_grad():
                out["concept_val_loss"] = float(self._concept_loss(layer, va))
        return out
