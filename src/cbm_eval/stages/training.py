"""Training: independent, sequential, or joint fitting of the concept layer and predictor head.

By default all variants train on cached (frozen) backbone features, on CPU (``device: auto`` uses
a GPU when there is one). Inputs are (N, D) image features or (N, M, D) instance bags.

With ``finetune: {...}`` a variant also trains the backbone end to end (Koh et al., 2020): images
go through a trainable copy of the backbone's encoder, which is optimized together with the concept
layer (the x -> c phase of independent / sequential) or with the concept layer and head (joint).
The predictor of independent / sequential is then fitted on the fine-tuned encoder's features.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

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


@dataclass(frozen=True)
class Finetune:
    """End-to-end fine-tuning of the backbone encoder (a training stage's ``finetune:``).

    The encoder gets its own optimizer (``sgd`` with ``momentum``, or ``adam``; both with
    ``weight_decay``) at ``lr``, decayed by ``lr_gamma`` every ``lr_step`` epochs if set; the concept
    layer and head keep the stage's optimizers. Batches of ``batch_size`` training images, augmented
    with the backbone's ``train_transform`` if ``augment``. Early stopping on the validation loss
    with ``patience``, restoring the best encoder, layer and head.
    """

    epochs: int = 100
    lr: float = 1e-3
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 4e-5
    lr_step: int | None = None
    lr_gamma: float = 0.1
    batch_size: int = 64
    patience: int = 10
    augment: bool = True
    num_workers: int = 4

    def __post_init__(self):
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError(f"Unknown finetune optimizer '{self.optimizer}' (use 'sgd' or 'adam')")

    def make_optimizer(self, params) -> torch.optim.Optimizer:
        if self.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)


class _ImageFeed:
    """Train / val images encoded on the fly by the trainable encoder, as ``Batch``es of features."""

    def __init__(self, ctx, aligned: AlignedConcepts, encoder: nn.Module, spec: Finetune, device, seed: int):
        self.encoder, self.spec, self.device = encoder, spec, device
        self.labels = {s: ctx.split(s).labels for s in ("train", "val")}
        self.targets = {s: aligned.targets(s) for s in ("train", "val")}
        g = torch.Generator().manual_seed(seed)
        pin = device.type == "cuda"
        ordered = lambda split: DataLoader(ctx.image_split(split), batch_size=2 * spec.batch_size, shuffle=False,
                                           num_workers=spec.num_workers, pin_memory=pin)
        self.loaders = {  # "train": augmented, shuffled training batches; the rest in dataset order
            "train": DataLoader(ctx.image_split("train", augment=spec.augment), batch_size=spec.batch_size,
                                shuffle=True, drop_last=True, generator=g, num_workers=spec.num_workers, pin_memory=pin),
            "val": ordered("val"),
            "train_ordered": ordered("train"),
        }

    def batches(self, split: str, loader: str | None = None) -> Iterator[Batch]:
        """Image batches -> feature batches; gradients reach the encoder in train mode."""
        t = self.targets[split]
        for item in self.loaders[loader or split]:
            idx = item["index"]
            yield Batch(self.encoder(item["image"].to(self.device, non_blocking=True)).float(),
                        self.labels[split][idx].to(self.device), None if t is None else t[idx].to(self.device))

    @torch.no_grad()
    def encode(self, split: str) -> Batch:
        """The whole split, in order, under the eval-mode encoder (for fitting a head on fixed features)."""
        self.encoder.eval()
        x = torch.cat([b.x for b in self.batches(split, "train_ordered" if split == "train" else split)])
        t = self.targets[split]
        return Batch(x, self.labels[split].to(self.device), None if t is None else t.to(self.device))


def optimize_images(groups: list[ParamGroup], modules: list[nn.Module], feed: _ImageFeed,
                    train_loss: Callable[[Batch], torch.Tensor], val_loss: Callable[[Batch], torch.Tensor],
                    after_step: Callable[[], None] | None = None) -> dict[str, float]:
    """``optimize`` over image batches, with the encoder fine-tuned alongside ``modules``."""
    spec, encoder = feed.spec, feed.encoder
    opts = _optimizers(groups)
    enc_opt = spec.make_optimizer([p for p in encoder.parameters() if p.requires_grad])
    modules = [encoder] + modules
    best, best_state, bad, epoch = float("inf"), None, 0, 0
    for epoch in range(1, spec.epochs + 1):
        if spec.lr_step:
            for g in enc_opt.param_groups:
                g["lr"] = spec.lr * spec.lr_gamma ** ((epoch - 1) // spec.lr_step)
        for m in modules:
            m.train()
        for b in feed.batches("train"):
            loss = train_loss(b)
            for opt in opts + [enc_opt]:
                opt.zero_grad()
            loss.backward()
            for opt in opts + [enc_opt]:
                opt.step()
            if after_step is not None:
                after_step()
        for m in modules:
            m.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for b in feed.batches("val"):
                total, n = total + float(val_loss(b)) * len(b), n + len(b)
            v = total / max(n, 1)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = [{k: t.detach().cpu().clone() for k, t in m.state_dict().items()} for m in modules]
        else:
            bad += 1
            if bad >= spec.patience:
                break
    if best_state is not None:
        for m, s in zip(modules, best_state):
            m.load_state_dict(s)
    for m in modules:
        m.eval()
    return {"epochs": epoch, "val_loss": best}


class _Base(Training):
    def __init__(self, epochs: int = 200, concept_epochs: int | None = None, batch_size: int = 256,
                 lr: float = 1e-3, patience: int = 10, concept_loss: str = "auto", device: str = "cpu",
                 finetune: dict | None = None):
        if concept_loss not in CONCEPT_LOSSES:
            raise ValueError(f"Unknown concept loss '{concept_loss}' (use one of {CONCEPT_LOSSES})")
        self.epochs, self.concept_epochs = epochs, concept_epochs or epochs
        self.batch_size, self.lr, self.patience = batch_size, lr, patience
        self.concept_loss, self.device = concept_loss, device
        self.finetune = Finetune(**finetune) if finetune else None  # unknown keys raise TypeError

    @property
    def trains_backbone(self) -> bool:
        return self.finetune is not None

    @property
    def num_workers(self) -> int:
        return self.finetune.num_workers if self.finetune else 0

    def fit(self, layer, head, aligned, ctx, encoder=None):
        device = resolve_device(self.device)
        tr, va = (_batch(ctx, aligned, s).to(device) for s in ("train", "val"))
        feed = None
        if self.finetune is not None:
            if encoder is None:
                raise RuntimeError("`stages.training.finetune` is set but no encoder was passed; run it through "
                                   "PipelineBuilder with a trainable backbone")
            if tr.bag is not None:
                raise ValueError("`stages.training.finetune` does not support instance bags; remove `instances:` "
                                 "or `finetune:`")
            feed = _ImageFeed(ctx, aligned, encoder.to(device), self.finetune, device, ctx.seed)
        layer.to(device)
        head.to(device)
        try:
            return self._fit(layer, head, aligned, tr, va, ctx.seed, feed)
        finally:
            layer.cpu()
            head.cpu()

    def _fit(self, layer: ConceptLayer, head: PredictorHead, aligned: AlignedConcepts, tr: Batch, va: Batch,
             seed: int, feed: _ImageFeed | None = None) -> dict[str, float]:
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

    def _fit_concepts(self, layer, aligned, tr: Batch, va: Batch, seed,
                      feed: _ImageFeed | None = None) -> dict[str, float]:
        if feed is not None:  # x -> c end to end: encoder + concept layer on the concept loss
            if not aligned.has_targets:
                raise ValueError("`finetune:` with independent / sequential training needs concept targets "
                                 "(an alignment with targets, e.g. `human`); use `joint` to fine-tune without")
            log = optimize_images([(layer.concept_params(), "adam", self.lr)], [layer], feed,
                                  lambda b: self._concept_loss(layer, b), lambda b: self._concept_loss(layer, b))
            return {"finetune_epochs": log["epochs"], "concept_val_loss": log["val_loss"]}
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

    def _fit(self, layer, head, aligned, tr, va, seed, feed=None):
        self._prepare(layer, aligned, tr)
        log = self._fit_concepts(layer, aligned, tr, va, seed, feed)
        if feed is not None:  # the head sees the fine-tuned encoder's features
            tr, va = feed.encode("train"), feed.encode("val")

        def rep(b):
            with torch.no_grad():
                c_hat = layer.activate(layer.concept_logits(b.x))
            return layer.represent(c_hat, b.x)

        return log | self._fit_head(layer, head, tr, va, rep, seed + 1)


@TRAINING.register("independent")
class Independent(_Base):
    """Fit concepts to targets; fit the predictor on the *true* concept targets."""

    def fit(self, layer, head, aligned, ctx, encoder=None):
        if not aligned.has_targets:
            raise RuntimeError("independent training needs concept targets (not available with this alignment)")
        return super().fit(layer, head, aligned, ctx, encoder)

    def _fit(self, layer, head, aligned, tr, va, seed, feed=None):
        self._prepare(layer, aligned, tr)
        log = self._fit_concepts(layer, aligned, tr, va, seed, feed)
        if feed is not None:
            tr, va = feed.encode("train"), feed.encode("val")
        return log | self._fit_head(layer, head, tr, va, lambda b: layer.represent(b.t, b.x), seed + 1)


@TRAINING.register("joint")
class Joint(_Base):
    """End-to-end: task loss + ``concept_weight`` x concept loss (when targets exist)."""

    def __init__(self, concept_weight: float = 1.0, **kw):
        super().__init__(**kw)
        self.concept_weight = concept_weight

    def _fit(self, layer, head, aligned, tr, va, seed, feed=None):
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
        if feed is not None:  # encoder, concept layer and head together, on images
            log = optimize_images(groups, [layer, head], feed, lambda b: total(b) + head.penalty(), total,
                                  after_step=head.proximal_step)
            va = feed.encode("val")
            out = {"finetune_epochs": log["epochs"], "joint_val_loss": log["val_loss"]}
        else:
            log = optimize(groups, [layer, head], len(tr), lambda idx: total(tr[idx]) + head.penalty(),
                           lambda: float(total(va)), self.epochs, self.batch_size, self.patience, seed,
                           after_step=head.proximal_step)
            out = {"joint_epochs": log["epochs"], "joint_val_loss": log["val_loss"]}
        if use_c:
            with torch.no_grad():
                out["concept_val_loss"] = float(self._concept_loss(layer, va))
        return out


class _ImbalanceWeightedConcepts:
    """Koh et al. (2020): concept k's BCE is scaled by its negative/positive ratio on the training
    targets, ``(1 - p_k) / p_k`` (their ``BCEWithLogitsLoss(weight=...)``, not ``pos_weight``), then
    averaged over concepts. Needs binary (N, K) image-level targets."""

    def _prepare(self, layer, aligned, tr):
        super()._prepare(layer, aligned, tr)
        if aligned.has_targets:
            if aligned.target_type != "binary" or tr.t.dim() != 2:
                raise ValueError(f"'{type(self).__name__}' needs binary image-level concept targets (e.g. "
                                 "`alignment: human` without `instances:`); set `stages.training.name` to the "
                                 "unweighted variant (independent / sequential / joint) for this alignment")
            pos = tr.t.float().mean(0)
            self.concept_weights = (1 - pos) / pos.clamp_min(1e-6)

    def _concept_loss(self, layer, b):
        per = F.binary_cross_entropy_with_logits(layer.concept_logits(b.x), b.t, reduction="none")
        return (per * self.concept_weights.to(per)).mean(-1).mean()


@TRAINING.register("sequential_weighted")
class SequentialWeighted(_ImbalanceWeightedConcepts, Sequential):
    """``sequential`` with Koh et al.'s imbalance-weighted concept BCE."""


@TRAINING.register("independent_weighted")
class IndependentWeighted(_ImbalanceWeightedConcepts, Independent):
    """``independent`` with Koh et al.'s imbalance-weighted concept BCE."""


@TRAINING.register("joint_weighted")
class JointWeighted(_ImbalanceWeightedConcepts, Joint):
    """``joint`` with Koh et al.'s imbalance-weighted concept BCE."""
