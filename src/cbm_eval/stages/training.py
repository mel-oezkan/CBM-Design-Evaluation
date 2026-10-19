"""Training: independent, sequential, or joint fitting of the concept layer and predictor head.

By default all variants train on cached (frozen) backbone features, on CPU (``device: auto`` uses
a GPU when there is one). Inputs are (N, D) image features or (N, M, D) instance bags.

With ``finetune: {...}`` a variant also trains the backbone end to end (Koh et al., 2020): images
go through a trainable copy of the backbone's encoder, which is optimized together with the concept
layer (the x -> c phase of independent / sequential) or with the concept layer and head (joint).
The predictor of independent / sequential is then fitted on the fine-tuned encoder's features.

With ``aux: {...}`` a variant also trains Koh et al.'s auxiliary concept head on the backbone's
auxiliary tap (``Backbone.aux_dim``): every training step adds ``aux.weight`` x the step's loss on that
head (see ``Aux``). On a frozen backbone this only changes a joint model, through the shared head.
"""

from __future__ import annotations

import copy
import dataclasses
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
    return concept_loss_from_logits(layer.concept_logits(x), targets, bag, kind, layer.target_type)


def concept_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, bag: Bag | None = None,
                             kind: str = "auto", target_type: str | None = None) -> torch.Tensor:
    """``concept_loss`` of concept logits (N, K) or (N, M, K) that are already computed."""
    if logits.dim() == 3 and targets.dim() == 2:
        logits = bag.max(logits) if bag is not None else logits.amax(1)
        bag = None
    if kind == "auto":
        kind = "bce" if target_type == "binary" else "mse"
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
    """Inputs, labels, concept targets and (for instance bags) the bag, indexed together.

    On frozen features, work that cannot change during a phase is done once over the whole split:
    ``xn`` holds ``x`` normalized by the concept layer, ``c`` a frozen layer's concept predictions.
    ``x`` (and ``xn``) are None once neither the concept layer nor the head reads them.
    ``xa`` (N, D_aux) are auxiliary-tap features (training ``aux:``); only training batches carry them.
    """

    x: torch.Tensor | None
    y: torch.Tensor
    t: torch.Tensor | None = None
    bag: Bag | None = None
    xn: torch.Tensor | None = None
    c: torch.Tensor | None = None
    xa: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.y)

    def _map(self, fn) -> "Batch":
        return Batch(**{f.name: None if (v := getattr(self, f.name)) is None else fn(v)
                        for f in dataclasses.fields(self)})

    def __getitem__(self, idx) -> "Batch":
        return self._map(lambda v: v[idx])

    def to(self, device) -> "Batch":
        return self._map(lambda v: v.to(device))

    def normalized(self, layer: ConceptLayer) -> torch.Tensor:
        """The features through ``layer.normalize``: ``xn`` if precomputed, else computed now."""
        return self.xn if self.xn is not None else layer.normalize(self.x)

    def concept_logits(self, layer: ConceptLayer, xn: torch.Tensor | None = None) -> torch.Tensor:
        """``layer``'s concept logits (N, K) or (N, M, K), from ``xn`` if the caller already normalized the
        features. On a zero-padded bag only the real instances are computed (the padding stays 0)."""
        xn = self.xn if xn is None else xn
        if self.bag is not None and self.bag.padded:
            return self.bag.on_rows(layer.concept_logits, self.x) if xn is None else \
                self.bag.on_rows(layer.logits_from_normalized, xn)
        return layer.logits_from_normalized(self.normalized(layer) if xn is None else xn)

    def without_x(self) -> "Batch":
        return dataclasses.replace(self, x=None, xn=None, xa=None)


OPTIMIZERS = ("adam", "sgd")


@dataclass(frozen=True)
class OptimizerSpec:
    """The optimizer of one parameter group: ``adam``, or ``sgd`` with ``momentum``; both with ``weight_decay``."""

    kind: str = "adam"
    lr: float = 1e-3
    momentum: float = 0.0
    weight_decay: float = 0.0

    def build(self, params: list[nn.Parameter]) -> torch.optim.Optimizer:
        if self.kind == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)


ParamGroup = tuple[list[nn.Parameter], OptimizerSpec]


def _optimizers(groups: list[ParamGroup]) -> list[torch.optim.Optimizer]:
    opts = []
    for params, spec in groups:
        params = [p for p in params if p.requires_grad]
        if params:
            opts.append(spec.build(params))
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
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown finetune optimizer '{self.optimizer}' (use 'sgd' or 'adam')")

    def make_optimizer(self, params) -> torch.optim.Optimizer:
        return OptimizerSpec(self.optimizer, self.lr, self.momentum, self.weight_decay).build(params)


@dataclass(frozen=True)
class Aux:
    """Koh et al.'s auxiliary concept head (a training stage's ``aux:``).

    A second x -> c linear map on the backbone's auxiliary tap (``Backbone.aux_dim``: Inception-v3's
    ``AuxLogits`` convolutions on ``Mixed_6e``), its weights drawn from a normal truncated at +-2
    ``init_std`` (None: PyTorch's default init). Every training step adds ``weight`` x the step's loss
    on its concept logits: the concept loss, and in joint training also the task loss of those concepts
    through the shared head (Koh et al.'s ``End2EndModel``). Validation and model selection read the
    main branch only, and the head is discarded after training. Fine-tuning also trains the tap.
    """

    weight: float = 0.4
    init_std: float | None = 0.001

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("`stages.training.aux.weight` must be positive; remove `aux:` to train without it")


class _AuxHead(nn.Module):
    """Auxiliary concept logits (N, K) from auxiliary-tap features (N, D_aux): a linear map on features
    standardized with training statistics, as ``ConceptLayer`` standardizes its inputs."""

    def __init__(self, in_dim: int, k: int, init_std: float | None):
        super().__init__()
        self.linear = nn.Linear(in_dim, k)
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        if init_std is not None:
            nn.init.trunc_normal_(self.linear.weight, std=init_std, a=-2 * init_std, b=2 * init_std)

    def fit_input_stats(self, xa: torch.Tensor) -> None:
        self.x_mean.copy_(xa.mean(0))
        self.x_std.copy_(xa.std(0).clamp_min(1e-6))

    def forward(self, xa: torch.Tensor) -> torch.Tensor:
        return self.linear((xa - self.x_mean) / self.x_std)


def _params(module: nn.Module | None) -> list[nn.Parameter]:
    return [] if module is None else list(module.parameters())


class _ImageFeed:
    """Train / val images encoded on the fly by the trainable encoder, as ``Batch``es of features."""

    def __init__(self, ctx, aligned: AlignedConcepts, encoder: nn.Module, spec: Finetune, device, seed: int,
                 aux: bool = False):
        self.encoder, self.spec, self.device, self.aux = encoder, spec, device, aux
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
        """Image batches -> feature batches; gradients reach the encoder in train mode. With ``aux``,
        training batches also carry the encoder's auxiliary-tap features ``xa``."""
        t = self.targets[split]
        for item in self.loaders[loader or split]:
            idx = item["index"]
            images = item["image"].to(self.device, non_blocking=True)
            if self.aux and (loader or split) == "train":
                x, xa = self.encoder.forward_aux(images)
                xa = xa.float()
            else:
                x, xa = self.encoder(images), None
            yield Batch(x.float(), self.labels[split][idx].to(self.device),
                        None if t is None else t[idx].to(self.device), xa=xa)

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
    """Shared by every variant. ``optimizer`` (``adam``, or ``sgd`` with ``momentum``; both with
    ``weight_decay``) at ``lr`` fits the concept layer and every head that does not bring its own
    (``PredictorHead.optimizer``, e.g. the sparse head's plain SGD). ``aux: {weight, init_std}`` adds
    Koh et al.'s auxiliary concept head (see ``Aux``); it needs ``finetune:`` unless the variant passes
    the auxiliary concepts through the head (``aux_reaches_head``, joint)."""

    aux_reaches_head: bool = False  # True if the auxiliary concepts also train the head (joint)

    def __init__(self, epochs: int = 200, concept_epochs: int | None = None, batch_size: int = 256,
                 lr: float = 1e-3, patience: int = 10, concept_loss: str = "auto", device: str = "cpu",
                 finetune: dict | None = None, optimizer: str = "adam", momentum: float = 0.0,
                 weight_decay: float = 0.0, aux: dict | None = None):
        if concept_loss not in CONCEPT_LOSSES:
            raise ValueError(f"Unknown concept loss '{concept_loss}' (use one of {CONCEPT_LOSSES})")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{optimizer}' in `stages.training.optimizer` (use one of {OPTIMIZERS})")
        if momentum and optimizer != "sgd":  # would change the run_id without changing the run
            raise ValueError("`stages.training.momentum` needs `stages.training.optimizer: sgd`")
        self.epochs, self.concept_epochs = epochs, concept_epochs or epochs
        self.batch_size, self.lr, self.patience = batch_size, lr, patience
        self.optimizer, self.momentum, self.weight_decay = optimizer, momentum, weight_decay
        self.concept_loss, self.device = concept_loss, device
        self.finetune = Finetune(**finetune) if finetune else None  # unknown keys raise TypeError
        self.aux = Aux(**aux) if aux is not None else None  # `aux: {}` means the defaults, not "off"
        if self.aux is not None and self.finetune is None and not self.aux_reaches_head:
            raise ValueError(f"`stages.training.aux` would only train a discarded auxiliary concept layer: on a frozen "
                             f"backbone '{type(self).__name__}' training is unchanged by it. Add `finetune:` or "
                             "remove `aux:`")

    @property
    def trains_backbone(self) -> bool:
        return self.finetune is not None

    @property
    def needs_aux(self) -> bool:
        return self.aux is not None

    @property
    def num_workers(self) -> int:
        return self.finetune.num_workers if self.finetune else 0

    def fit(self, layer, head, aligned, ctx, encoder=None):
        device = resolve_device(self.device)
        if self.aux is not None and not aligned.has_targets:
            raise ValueError("`stages.training.aux` trains auxiliary concept predictions on concept targets; use an "
                             "alignment with targets (e.g. `human`) or remove `aux:`")
        if self.aux is not None and self.aux_reaches_head and (layer.represent_uses_x or head.uses_features):
            raise ValueError("`stages.training.aux` passes the auxiliary concepts through the head alone, so it "
                             "needs a generation and predictor that read only concepts (e.g. `generation: logits`, "
                             "`predictor: dense`); remove `aux:` for this combination")
        tr = _batch(ctx, aligned, "train", aux=self.aux is not None).to(device)
        va = _batch(ctx, aligned, "val").to(device)
        aux_head = None
        if self.aux is not None:
            if tr.bag is not None:
                raise ValueError("`stages.training.aux` does not support instance bags; remove `instances:` or `aux:`")
            # built here, after the layer and head, so that runs without `aux:` draw the same random numbers
            aux_head = _AuxHead(tr.xa.shape[1], layer.n_concepts, self.aux.init_std).to(device)
            aux_head.fit_input_stats(tr.xa)
        feed = None
        if self.finetune is not None:
            if encoder is None:
                raise RuntimeError("`stages.training.finetune` is set but no encoder was passed; run it through "
                                   "PipelineBuilder with a trainable backbone")
            if tr.bag is not None:
                raise ValueError("`stages.training.finetune` does not support instance bags; remove `instances:` "
                                 "or `finetune:`")
            feed = _ImageFeed(ctx, aligned, encoder.to(device), self.finetune, device, ctx.seed,
                              aux=aux_head is not None)
        layer.to(device)
        head.to(device)
        try:
            return self._fit(layer, head, aligned, tr, va, ctx.seed, feed, aux_head)
        finally:
            layer.cpu()
            head.cpu()

    def _fit(self, layer: ConceptLayer, head: PredictorHead, aligned: AlignedConcepts, tr: Batch, va: Batch,
             seed: int, feed: _ImageFeed | None = None, aux_head: _AuxHead | None = None) -> dict[str, float]:
        raise NotImplementedError

    def _prepare(self, layer: ConceptLayer, aligned: AlignedConcepts, tr: Batch) -> None:
        layer.fit_input_stats(tr.x, tr.bag)
        if not aligned.has_targets:
            layer.calibrate(tr.x, tr.bag)

    @staticmethod
    @torch.no_grad()
    def _normalized(layer: ConceptLayer, *batches: Batch) -> list[Batch]:
        """The batches with ``xn`` precomputed (after ``_prepare`` fixed the input statistics). Instance
        bags are normalized per minibatch instead, since a second copy of a patch bag can run to gigabytes."""
        return [b if b.x.dim() == 3 else dataclasses.replace(b, xn=layer.normalize(b.x)) for b in batches]

    def _concept_loss(self, layer: ConceptLayer, b: Batch, logits: torch.Tensor | None = None) -> torch.Tensor:
        """Concept loss of a batch; ``logits`` default to ``b.concept_logits(layer)`` (the auxiliary head
        passes its own)."""
        logits = b.concept_logits(layer) if logits is None else logits
        return concept_loss_from_logits(logits, b.t, b.bag, self.concept_loss, layer.target_type)

    def _aux_loss(self, aux_head: _AuxHead, layer: ConceptLayer, head: PredictorHead | None, b: Batch) -> torch.Tensor:
        """The auxiliary head's loss on a training batch, before ``aux.weight``: its concept loss. Joint
        training adds the task loss of its concepts through the shared head."""
        return self._concept_loss(layer, b, aux_head(b.xa))

    def _aux_term(self, aux_head: _AuxHead | None, layer: ConceptLayer, head: PredictorHead | None,
                  b: Batch) -> torch.Tensor | float:
        """``aux.weight`` x ``_aux_loss`` on batches that carry auxiliary features (training batches); else 0."""
        if aux_head is None or b.xa is None:
            return 0.0
        return self.aux.weight * self._aux_loss(aux_head, layer, head, b)

    def _task_loss(self, logits: torch.Tensor, b: Batch) -> torch.Tensor:
        """Class logits (N, C) -> scalar task loss. The one place every variant computes it; override
        it in a subclass (registered under a new name) to try another task loss."""
        return F.cross_entropy(logits, b.y)

    def _fit_concepts(self, layer, aligned, tr: Batch, va: Batch, seed, feed: _ImageFeed | None = None,
                      aux_head: _AuxHead | None = None) -> dict[str, float]:
        group = (layer.concept_params() + _params(aux_head), self._optimizer())

        def loss(b: Batch) -> torch.Tensor:
            return self._concept_loss(layer, b) + self._aux_term(aux_head, layer, None, b)

        if feed is not None:  # x -> c end to end: encoder + concept layer on the concept loss
            if not aligned.has_targets:
                raise ValueError("`finetune:` with independent / sequential training needs concept targets "
                                 "(an alignment with targets, e.g. `human`); use `joint` to fine-tune without")
            log = optimize_images([group], [layer], feed, loss, loss)
            return {"finetune_epochs": log["epochs"], "concept_val_loss": log["val_loss"]}
        if not aligned.has_targets or layer.frozen:
            return {}
        log = optimize([group], [layer], len(tr), lambda idx: loss(tr[idx]),
                       lambda: float(self._concept_loss(layer, va)),
                       self.concept_epochs, self.batch_size, self.patience, seed)
        return {"concept_epochs": log["epochs"], "concept_val_loss": log["val_loss"]}

    def _optimizer(self, lr: float | None = None) -> OptimizerSpec:
        return OptimizerSpec(self.optimizer, lr or self.lr, self.momentum, self.weight_decay)

    def _head_group(self, head: PredictorHead, params: list[nn.Parameter]) -> ParamGroup:
        if head.optimizer is not None:  # the head's own optimizer, as it asks for it (plain, no decay)
            return params, OptimizerSpec(head.optimizer, head.lr or self.lr)
        return params, self._optimizer(head.lr)

    def _fit_head(self, layer, head, tr: Batch, va: Batch, rep_fn: Callable[[Batch], torch.Tensor],
                  seed) -> dict[str, float]:
        """Fit the head on ``rep_fn(batch)``, which reads features only through ``layer.represent``."""
        out: dict[str, float] = {}
        if not (layer.represent_uses_x or head.uses_features):
            tr, va = tr.without_x(), va.without_x()  # minibatches then index (N, K) tensors only

        def task_loss(b: Batch) -> torch.Tensor:
            return self._task_loss(head(rep_fn(b), b.normalized(layer) if head.uses_features else None, b.bag), b)

        for i, phase in enumerate(head.phases()):
            groups = [self._head_group(head, phase)] + ([(layer.rep_params(), self._optimizer())] if i == 0 else [])
            log = optimize(groups, [layer, head], len(tr), lambda idx: task_loss(tr[idx]) + head.penalty(),
                           lambda: float(task_loss(va)), self.epochs, self.batch_size,
                           self.patience, seed + i, after_step=head.proximal_step)
            out[f"head_phase{i}_epochs"] = log["epochs"]
            out["head_val_loss"] = log["val_loss"]
        return out


def _batch(ctx, aligned: AlignedConcepts, split: str, aux: bool = False) -> Batch:
    x, bag = ctx.inputs(split)
    return Batch(x.float(), ctx.split(split).labels, aligned.targets(split), bag,
                 xa=ctx.aux_features(split).float() if aux else None)


@TRAINING.register("sequential")
class Sequential(_Base):
    """Fit concepts to targets, freeze, then fit the predictor on predicted concepts."""

    def _fit(self, layer, head, aligned, tr, va, seed, feed=None, aux_head=None):
        self._prepare(layer, aligned, tr)
        tr, va = self._normalized(layer, tr, va)
        log = self._fit_concepts(layer, aligned, tr, va, seed, feed, aux_head)
        if feed is not None:  # the head sees the fine-tuned encoder's features
            tr, va = self._normalized(layer, feed.encode("train"), feed.encode("val"))
        tr, va = self._frozen_concepts(layer, tr), self._frozen_concepts(layer, va)
        return log | self._fit_head(layer, head, tr, va, lambda b: layer.represent(b.c, b.x), seed + 1)

    @torch.no_grad()
    def _frozen_concepts(self, layer: ConceptLayer, b: Batch) -> Batch:
        """``b`` with the frozen concept layer's predictions ``c``, computed once rather than every step."""
        if b.xn is not None:
            c = layer.activate(layer.logits_from_normalized(b.xn))
        else:  # instance bags, in chunks so that the normalized bag never exists in full
            c = torch.cat([layer.activate(b[i:i + self.batch_size].concept_logits(layer))
                           for i in range(0, len(b), self.batch_size)])
        return dataclasses.replace(b, c=c)


@TRAINING.register("independent")
class Independent(_Base):
    """Fit concepts to targets; fit the predictor on the *true* concept targets."""

    def fit(self, layer, head, aligned, ctx, encoder=None):
        if not aligned.has_targets:
            raise RuntimeError("independent training needs concept targets (not available with this alignment)")
        return super().fit(layer, head, aligned, ctx, encoder)

    def _fit(self, layer, head, aligned, tr, va, seed, feed=None, aux_head=None):
        self._prepare(layer, aligned, tr)
        tr, va = self._normalized(layer, tr, va)
        log = self._fit_concepts(layer, aligned, tr, va, seed, feed, aux_head)
        if feed is not None:
            tr, va = self._normalized(layer, feed.encode("train"), feed.encode("val"))
        return log | self._fit_head(layer, head, tr, va, lambda b: layer.represent(b.t, b.x), seed + 1)


@TRAINING.register("joint")
class Joint(_Base):
    """End-to-end: task loss + ``concept_weight`` x concept loss (when targets exist). With ``aux:``
    the auxiliary concepts also go through the head, so ``aux`` changes the head on a frozen backbone too."""

    aux_reaches_head = True

    def __init__(self, concept_weight: float = 1.0, **kw):
        super().__init__(**kw)
        self.concept_weight = concept_weight

    def _aux_loss(self, aux_head, layer, head, b):
        """The auxiliary concepts go through the head as ``activate``-d logits. They skip the layer's
        ``logits_from_normalized``, so a generation that changes its scores there needs a matching
        change here (none does; with targets, ``c_mean`` / ``c_std`` stay 0 / 1)."""
        logits = aux_head(b.xa)
        loss = self._task_loss(head(layer.represent(layer.activate(logits), None), None, b.bag), b)
        if self.concept_weight > 0:
            loss = loss + self.concept_weight * self._concept_loss(layer, b, logits)
        return loss

    def _fit(self, layer, head, aligned, tr, va, seed, feed=None, aux_head=None):
        self._prepare(layer, aligned, tr)
        tr, va = self._normalized(layer, tr, va)
        use_c = aligned.has_targets and self.concept_weight > 0

        def total(b: Batch) -> torch.Tensor:
            xn = b.normalized(layer) if head.uses_features else None
            rep = layer.represent(layer.activate(b.concept_logits(layer, xn)), b.x)  # = layer(b.x)[1]
            loss = self._task_loss(head(rep, xn, b.bag), b)
            if use_c:
                loss = loss + self.concept_weight * self._concept_loss(layer, b)
            return loss + self._aux_term(aux_head, layer, head, b)

        groups = [(layer.concept_params() + layer.rep_params() + _params(aux_head), self._optimizer()),
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

    def _concept_loss(self, layer, b, logits=None):
        logits = b.concept_logits(layer) if logits is None else logits
        per = F.binary_cross_entropy_with_logits(logits, b.t, reduction="none")
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
