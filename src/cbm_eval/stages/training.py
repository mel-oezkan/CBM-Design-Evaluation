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

With ``augment: {...}`` the backbone stays frozen but the phases that read features (x -> c, joint)
see augmented images: each step reads one of ``views`` cached augmented copies of every image.

Every phase keeps the weights of its best epoch by ``select`` (``concept_select`` for the x -> c
phase): the lowest validation loss (default), the highest validation accuracy, or the highest
running training accuracy, which ``train_on_val`` (``concept_train_on_val``) needs because it also
trains on the validation split (Koh et al. retrain on train + val this way).
"""

from __future__ import annotations

import copy
import dataclasses
import time
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
SELECTIONS = ("val_loss", "val_acc", "train_acc")


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


def concept_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fraction of (image, concept) pairs where ``logit >= 0`` (``sigmoid >= 0.5``) matches a binary
    (N, K) target: Koh et al.'s concept accuracy."""
    return ((logits >= 0) == (targets > 0.5)).float().mean()


def class_accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(-1) == y).float().mean()


@dataclass
class Batch:
    """Inputs, labels, concept targets and (for instance bags) the bag, indexed together.

    On frozen features, work that cannot change during a phase is done once over the whole split:
    ``xn`` holds ``x`` normalized by the concept layer, ``z`` a frozen layer's concept logits.
    ``x`` (and ``xn``) are None once neither the concept layer nor the head reads them.
    ``xa`` (N, D_aux) are auxiliary-tap features (training ``aux:``); only batches that are trained on carry them.
    ``views`` (N, V, D) are augmented copies of ``x`` (training ``augment:``); ``view`` picks one per image.
    """

    x: torch.Tensor | None
    y: torch.Tensor
    t: torch.Tensor | None = None
    bag: Bag | None = None
    xn: torch.Tensor | None = None
    z: torch.Tensor | None = None
    xa: torch.Tensor | None = None
    views: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.y)

    def _map(self, fn) -> "Batch":
        return Batch(**{f.name: None if (v := getattr(self, f.name)) is None else fn(v)
                        for f in dataclasses.fields(self)})

    def __getitem__(self, idx) -> "Batch":
        return self._map(lambda v: v[idx])

    def to(self, device) -> "Batch":
        return self._map(lambda v: v.to(device))

    @staticmethod
    def cat(a: "Batch", b: "Batch") -> "Batch":
        """Two splits as one, e.g. train + val when training on both."""
        if a.bag is not None or b.bag is not None:
            raise ValueError("`train_on_val` does not support instance bags; remove `instances:` or set "
                             "`stages.training.train_on_val` / `concept_train_on_val` to false")
        return Batch(**{f.name: None if (u := getattr(a, f.name)) is None else torch.cat([u, getattr(b, f.name)])
                        for f in dataclasses.fields(a)})

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

    def view(self, layer: ConceptLayer, g: torch.Generator) -> "Batch":
        """One augmented view per image, drawn with ``g``, as ``x`` (and ``xn``); ``self`` without views."""
        if self.views is None:
            return self
        pick = torch.randint(self.views.shape[1], (len(self),), generator=g).to(self.views.device)
        x = self.views[torch.arange(len(self), device=self.views.device), pick].float()
        return dataclasses.replace(self, x=x, xn=layer.normalize(x), views=None)

    def without_x(self) -> "Batch":
        return dataclasses.replace(self, x=None, xn=None, xa=None, views=None)


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
# A minibatch's (loss, accuracy); the accuracy is None unless model selection reads it.
Step = tuple[torch.Tensor, torch.Tensor | None]


def _optimizers(groups: list[ParamGroup]) -> list[torch.optim.Optimizer]:
    opts = []
    for params, spec in groups:
        params = [p for p in params if p.requires_grad]
        if params:
            opts.append(spec.build(params))
    return opts


class _Selection:
    """Model selection over epochs: keeps the weights of the best epoch by ``select`` and asks to stop
    after ``patience`` epochs without improvement. ``val_loss`` improves by more than 1e-5; the
    accuracies improve when strictly higher (as Koh et al.'s ``best_val_acc < acc``). ``log`` (e.g.
    ``History.phase``) receives every epoch's numbers and whether it is the best so far."""

    def __init__(self, select: str, patience: int, modules: list[nn.Module], to_cpu: bool = False,
                 log: Callable[..., None] | None = None):
        self.select, self.patience, self.modules, self.to_cpu, self.log = select, patience, modules, to_cpu, log
        self.best = float("inf") if select == "val_loss" else float("-inf")
        self.state, self.bad, self.best_epoch, self.val_loss = None, 0, 0, float("nan")

    def update(self, epoch: int, train_acc: float | None, val: tuple[float, float | None] | None,
               **extra: float) -> bool:
        """Record an epoch's running training accuracy and (validation loss, accuracy); True to stop.
        ``extra`` (training loss, seconds, learning rates) only goes to ``log``."""
        val_loss, val_acc = val if val is not None else (float("nan"), None)
        if self.select == "val_loss":
            score, better = val_loss, val_loss < self.best - 1e-5
        else:
            score = train_acc if self.select == "train_acc" else val_acc
            better = score > self.best
        if self.log is not None:
            self.log(epoch=epoch, train_acc=train_acc, val_loss=None if val is None else val_loss, val_acc=val_acc,
                     best=bool(better), **extra)
        if better:
            self.best, self.bad, self.best_epoch, self.val_loss = score, 0, epoch, val_loss
            self.state = [{k: t.detach().cpu().clone() for k, t in m.state_dict().items()} if self.to_cpu
                          else copy.deepcopy(m.state_dict()) for m in self.modules]
            return False
        self.bad += 1
        return self.bad >= self.patience

    def finish(self, epoch: int) -> dict[str, float]:
        """Restore the best weights; the log of the phase."""
        if self.state is not None:
            for m, s in zip(self.modules, self.state):
                m.load_state_dict(s)
        out = {"epochs": epoch, "val_loss": self.best if self.select == "val_loss" else self.val_loss,
               "best_epoch": self.best_epoch}
        return out if self.select == "val_loss" else out | {self.select: self.best}


def _acc_flags(select: str) -> tuple[bool, bool]:
    """Whether training / validation steps must report an accuracy under ``select``."""
    return select == "train_acc", select == "val_acc"


def _lr(opt: torch.optim.Optimizer) -> float:
    return opt.param_groups[0]["lr"]


def optimize(groups: list[ParamGroup], modules: list[nn.Module], n: int,
             train_step: Callable[[torch.Tensor], Step], val_step: Callable[[], Step] | None,
             epochs: int, batch_size: int, patience: int, seed: int,
             after_step: Callable[[], None] | None = None, select: str = "val_loss",
             drop_last: bool = False, log: Callable[..., None] | None = None) -> dict[str, float]:
    """Minibatch training over ``n`` examples (``train_step(idx)``), with model selection by ``select``
    (``val_step()`` on the validation split; None when it is trained on too); restores the best
    weights. ``drop_last`` skips each epoch's last, incomplete minibatch. ``log`` receives each
    epoch's numbers (see ``_Selection``)."""
    opts = _optimizers(groups)
    if not opts:
        return {"epochs": 0, "val_loss": float(val_step()[0]) if val_step is not None else float("nan")}
    g = torch.Generator().manual_seed(seed)
    sel = _Selection(select, patience, modules, log=log)
    epoch = 0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        for m in modules:
            m.train()
        batches = torch.randperm(n, generator=g).split(batch_size)
        if drop_last and len(batches) > 1 and len(batches[-1]) < batch_size:
            batches = batches[:-1]
        correct, seen, loss_sum, n_seen = 0.0, 0, torch.zeros(()), 0
        for idx in batches:
            loss, acc = train_step(idx)
            loss_sum, n_seen = loss_sum.to(loss.device) + loss.detach() * len(idx), n_seen + len(idx)
            for opt in opts:
                opt.zero_grad()
            loss.backward()
            for opt in opts:
                opt.step()
            if after_step is not None:
                after_step()
            if acc is not None:
                correct, seen = correct + float(acc) * len(idx), seen + len(idx)
        for m in modules:
            m.eval()
        with torch.no_grad():
            val = None if val_step is None else tuple(None if v is None else float(v) for v in val_step())
        if sel.update(epoch, correct / seen if seen else None, val, train_loss=float(loss_sum) / max(n_seen, 1),
                      lr=_lr(opts[0]), seconds=time.perf_counter() - start):
            break
    return sel.finish(epoch)


@dataclass(frozen=True)
class Finetune:
    """End-to-end fine-tuning of the backbone encoder (a training stage's ``finetune:``).

    The encoder gets its own optimizer (``sgd`` with ``momentum``, or ``adam``; both with
    ``weight_decay``) at ``lr``, decayed by ``lr_gamma`` every ``lr_step`` epochs if set; the concept
    layer and head keep the stage's optimizers. Batches of ``batch_size`` training images, augmented
    with the backbone's ``train_transform`` if ``augment``. Early stopping with ``patience`` by the
    stage's ``select`` / ``concept_select``, restoring the best encoder, layer and head.
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


@dataclass(frozen=True)
class Augment:
    """Augmentation on a frozen backbone (a training stage's ``augment:``).

    ``views`` augmented copies of every training image (the backbone's ``train_transform``) are
    encoded once by the frozen backbone and cached (``Context.augmented``); every step of the x -> c
    and joint phases reads one of them per image, drawn at random. This is a finite stand-in for
    augmenting afresh every epoch, as fine-tuning does. ``seed`` fixes the views, so every run with
    the same backbone shares them; ``batch_size`` and ``num_workers`` only batch the encoding.
    """

    views: int = 32
    seed: int = 0
    batch_size: int = 128
    num_workers: int = 4

    def __post_init__(self):
        if self.views < 1:
            raise ValueError("`stages.training.augment.views` must be at least 1")


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
                    train_step: Callable[[Batch], Step], val_step: Callable[[Batch], Step],
                    after_step: Callable[[], None] | None = None, select: str = "val_loss",
                    log: Callable[..., None] | None = None) -> dict[str, float]:
    """``optimize`` over image batches, with the encoder fine-tuned alongside ``modules``."""
    spec, encoder = feed.spec, feed.encoder
    opts = _optimizers(groups)
    enc_opt = spec.make_optimizer([p for p in encoder.parameters() if p.requires_grad])
    modules = [encoder] + modules
    sel = _Selection(select, spec.patience, modules, to_cpu=True, log=log)
    epoch = 0
    for epoch in range(1, spec.epochs + 1):
        start = time.perf_counter()
        if spec.lr_step:
            for g in enc_opt.param_groups:
                g["lr"] = spec.lr * spec.lr_gamma ** ((epoch - 1) // spec.lr_step)
        for m in modules:
            m.train()
        correct, seen, loss_sum, n_seen = 0.0, 0, torch.zeros(()), 0
        for b in feed.batches("train"):
            loss, acc = train_step(b)
            loss_sum, n_seen = loss_sum.to(loss.device) + loss.detach() * len(b), n_seen + len(b)
            for opt in opts + [enc_opt]:
                opt.zero_grad()
            loss.backward()
            for opt in opts + [enc_opt]:
                opt.step()
            if after_step is not None:
                after_step()
            if acc is not None:
                correct, seen = correct + float(acc) * len(b), seen + len(b)
        for m in modules:
            m.eval()
        with torch.no_grad():
            total, hits, n, has_acc = 0.0, 0.0, 0, False
            for b in feed.batches("val"):
                loss, acc = val_step(b)
                total, n = total + float(loss) * len(b), n + len(b)
                if acc is not None:
                    hits, has_acc = hits + float(acc) * len(b), True
            val = (total / max(n, 1), hits / max(n, 1) if has_acc else None)
        if sel.update(epoch, correct / seen if seen else None, val, train_loss=float(loss_sum) / max(n_seen, 1),
                      lr=_lr(opts[0]) if opts else None, encoder_lr=_lr(enc_opt), seconds=time.perf_counter() - start):
            break
    log = sel.finish(epoch)
    for m in modules:
        m.eval()
    return log


class _Base(Training):
    """Shared by every variant. ``optimizer`` (``adam``, or ``sgd`` with ``momentum``; both with
    ``weight_decay``) at ``lr`` fits the concept layer and every head that does not bring its own
    (``PredictorHead.optimizer``, e.g. the sparse head's plain SGD).

    The x -> c phase of independent / sequential training may differ from the head's: ``concept_lr``,
    ``concept_weight_decay``, ``concept_epochs``, ``concept_select`` and ``concept_train_on_val``
    (each None: the stage-wide setting). ``select`` is ``val_loss``, ``val_acc`` (task accuracy for
    heads and joint, concept accuracy for x -> c) or ``train_acc`` (the running accuracy over an
    epoch's minibatches); ``train_on_val`` trains on train + val and needs ``train_acc``.
    ``drop_last`` skips each epoch's last incomplete minibatch. ``augment: {views, ...}`` trains the
    x -> c and joint phases on augmented views of the images (see ``Augment``). ``aux: {weight,
    init_std}`` adds Koh et al.'s auxiliary concept head (see ``Aux``); it needs ``finetune:`` unless
    the variant passes the auxiliary concepts through the head (``aux_reaches_head``, joint).
    """

    aux_reaches_head: bool = False  # True if the auxiliary concepts also train the head (joint)

    def __init__(self, epochs: int = 200, concept_epochs: int | None = None, batch_size: int = 256,
                 lr: float = 1e-3, patience: int = 10, concept_loss: str = "auto", device: str = "cpu",
                 finetune: dict | None = None, optimizer: str = "adam", momentum: float = 0.0,
                 weight_decay: float = 0.0, concept_lr: float | None = None,
                 concept_weight_decay: float | None = None, select: str = "val_loss",
                 concept_select: str | None = None, train_on_val: bool = False,
                 concept_train_on_val: bool | None = None, drop_last: bool = False, augment: dict | None = None,
                 aux: dict | None = None):
        if concept_loss not in CONCEPT_LOSSES:
            raise ValueError(f"Unknown concept loss '{concept_loss}' (use one of {CONCEPT_LOSSES})")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{optimizer}' in `stages.training.optimizer` (use one of {OPTIMIZERS})")
        if momentum and optimizer != "sgd":  # would change the run_id without changing the run
            raise ValueError("`stages.training.momentum` needs `stages.training.optimizer: sgd`")
        for key, value in (("select", select), ("concept_select", concept_select or select)):
            if value not in SELECTIONS:
                raise ValueError(f"Unknown `stages.training.{key}` '{value}' (use one of {SELECTIONS})")
        self.epochs, self.concept_epochs = epochs, concept_epochs or epochs
        self.batch_size, self.lr, self.patience = batch_size, lr, patience
        self.optimizer, self.momentum, self.weight_decay = optimizer, momentum, weight_decay
        self.concept_lr, self.concept_weight_decay = concept_lr, concept_weight_decay
        self.select, self.concept_select = select, concept_select or select
        self.train_on_val = train_on_val
        self.concept_train_on_val = train_on_val if concept_train_on_val is None else concept_train_on_val
        self.drop_last = drop_last
        self.concept_loss, self.device = concept_loss, device
        self.finetune = Finetune(**finetune) if finetune else None  # unknown keys raise TypeError
        self.augment = Augment(**augment) if augment else None
        self.aux = Aux(**aux) if aux is not None else None  # `aux: {}` means the defaults, not "off"
        if self.aux is not None and self.finetune is None and not self.aux_reaches_head:
            raise ValueError(f"`stages.training.aux` would only train a discarded auxiliary concept layer: on a frozen "
                             f"backbone '{type(self).__name__}' training is unchanged by it. Add `finetune:` or "
                             "remove `aux:`")
        if self.aux is not None and self.augment is not None:
            raise ValueError("`stages.training.aux` reads the auxiliary tap of the clean images, but `augment:` trains "
                             "on augmented views of them; remove `aux:` or `augment:`")
        for key, on_val, sel in (("train_on_val", self.train_on_val, self.select),
                                 ("concept_train_on_val", self.concept_train_on_val, self.concept_select)):
            if on_val and sel != "train_acc":
                raise ValueError(f"`stages.training.{key}` leaves no validation split to select on; set "
                                 f"`stages.training.{key.replace('train_on_val', 'select')}: train_acc`")
        if self.finetune is not None and (self.augment is not None or self.train_on_val or self.concept_train_on_val):
            raise ValueError("`stages.training.finetune` augments images itself and selects on the validation "
                             "split; remove `augment:` / `train_on_val` / `concept_train_on_val`, or `finetune:`")

    @property
    def trains_backbone(self) -> bool:
        return self.finetune is not None

    @property
    def needs_aux(self) -> bool:
        return self.aux is not None

    @property
    def num_workers(self) -> int:
        return self.finetune.num_workers if self.finetune else 0

    @property
    def _val_views(self) -> bool:
        """Whether a phase reads augmented views of the validation images (it trains on them)."""
        return self.train_on_val or self.concept_train_on_val

    def _views(self, ctx, split: str) -> torch.Tensor | None:
        a = self.augment
        if a is None or (split == "val" and not self._val_views):
            return None
        return ctx.augmented(split, a.views, a.seed, a.batch_size, a.num_workers)

    def fit(self, layer, head, aligned, ctx, encoder=None):
        device = resolve_device(self.device)
        self._history = ctx.history  # each phase reports its epochs here (see ``History``)
        if self.aux is not None and not aligned.has_targets:
            raise ValueError("`stages.training.aux` trains auxiliary concept predictions on concept targets; use an "
                             "alignment with targets (e.g. `human`) or remove `aux:`")
        if self.aux is not None and self.aux_reaches_head and (layer.represent_uses_x or head.uses_features):
            raise ValueError("`stages.training.aux` passes the auxiliary concepts through the head alone, so it "
                             "needs a generation and predictor that read only concepts (e.g. `generation: logits`, "
                             "`predictor: dense`); remove `aux:` for this combination")
        aux = self.aux is not None
        tr = _batch(ctx, aligned, "train", aux=aux)
        va = _batch(ctx, aligned, "val", aux=aux and (self.train_on_val or self.concept_train_on_val))
        if self.augment is not None and tr.bag is not None:
            raise ValueError("`stages.training.augment` augments whole images and does not support instance bags; "
                             "remove `instances:` or `augment:`")
        tr, va = (dataclasses.replace(b, views=self._views(ctx, s)).to(device) for b, s in ((tr, "train"), (va, "val")))
        aux_head = None
        if aux:
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
        """``aux.weight`` x ``_aux_loss`` of a training batch; 0 for a batch without auxiliary features."""
        if aux_head is None or b.xa is None:
            return 0.0
        return self.aux.weight * self._aux_loss(aux_head, layer, head, b)

    def _task_loss(self, logits: torch.Tensor, b: Batch) -> torch.Tensor:
        """Class logits (N, C) -> scalar task loss. The one place every variant computes it; override
        it in a subclass (registered under a new name) to try another task loss."""
        return F.cross_entropy(logits, b.y)

    def _fit_concepts(self, layer, aligned, tr: Batch, va: Batch, seed, feed: _ImageFeed | None = None,
                      aux_head: _AuxHead | None = None) -> dict[str, float]:
        train_acc, val_acc = _acc_flags(self.concept_select)
        if (train_acc or val_acc) and aligned.has_targets and (aligned.target_type != "binary" or tr.bag is not None):
            raise ValueError(f"`stages.training.concept_select: {self.concept_select}` measures the accuracy of binary "
                             "image-level concept predictions; use `val_loss` for this alignment or instance bags")
        # concept accuracy is defined for binary image-level targets; the history records it even if unselected
        acc = aligned.has_targets and aligned.target_type == "binary" and tr.bag is None

        def step(b: Batch, want_acc: bool, train: bool = False) -> Step:
            logits = b.concept_logits(layer)
            loss = self._concept_loss(layer, b, logits)
            if train:
                loss = loss + self._aux_term(aux_head, layer, None, b)
            return loss, concept_accuracy(logits, b.t) if want_acc else None

        group = (layer.concept_params() + _params(aux_head),
                 self._optimizer(self.concept_lr, self.concept_weight_decay))
        if feed is not None:  # x -> c end to end: encoder + concept layer on the concept loss
            if not aligned.has_targets:
                raise ValueError("`finetune:` with independent / sequential training needs concept targets "
                                 "(an alignment with targets, e.g. `human`); use `joint` to fine-tune without")
            log = optimize_images([group], [layer], feed, lambda b: step(b, acc, True), lambda b: step(b, acc),
                                  select=self.concept_select, log=self._history.phase("concept"))
            return {"finetune_epochs": log["epochs"], "concept_val_loss": log["val_loss"]} | _extra(log, "concept")
        if not aligned.has_targets or layer.frozen:
            return {}
        if self.concept_train_on_val:
            tr, va = Batch.cat(tr, va), None
        g = torch.Generator().manual_seed(seed + 1_000)  # draws the augmented view of each image
        log = optimize([group], [layer], len(tr), lambda idx: step(tr[idx].view(layer, g), acc, True),
                       None if va is None else lambda: step(va, acc),
                       self.concept_epochs, self.batch_size, self.patience, seed,
                       select=self.concept_select, drop_last=self.drop_last, log=self._history.phase("concept"))
        return {"concept_epochs": log["epochs"], "concept_val_loss": log["val_loss"]} | _extra(log, "concept")

    def _optimizer(self, lr: float | None = None, weight_decay: float | None = None) -> OptimizerSpec:
        return OptimizerSpec(self.optimizer, lr or self.lr, self.momentum,
                             self.weight_decay if weight_decay is None else weight_decay)

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
        else:
            tr, va = (dataclasses.replace(b, xa=None, views=None) for b in (tr, va))
        if self.train_on_val:
            tr, va = Batch.cat(tr, va), None

        def task_step(b: Batch) -> Step:
            logits = head(rep_fn(b), b.normalized(layer) if head.uses_features else None, b.bag)
            return self._task_loss(logits, b), class_accuracy(logits, b.y)

        def train_step(idx) -> Step:
            loss, acc = task_step(tr[idx])
            return loss + head.penalty(), acc

        for i, phase in enumerate(head.phases()):
            groups = [self._head_group(head, phase)] + ([(layer.rep_params(), self._optimizer())] if i == 0 else [])
            log = optimize(groups, [layer, head], len(tr), train_step,
                           None if va is None else lambda: task_step(va), self.epochs, self.batch_size,
                           self.patience, seed + i, after_step=head.proximal_step, select=self.select,
                           drop_last=self.drop_last, log=self._history.phase(f"head_phase{i}"))
            out[f"head_phase{i}_epochs"] = log["epochs"]
            out["head_val_loss"] = log["val_loss"]
            out |= _extra(log, "head")
        return out


def _extra(log: dict[str, float], prefix: str) -> dict[str, float]:
    """A phase log's selection entries beyond epochs and validation loss, under ``prefix``."""
    return {f"{prefix}_{k}": v for k, v in log.items() if k not in ("epochs", "val_loss")}


def _batch(ctx, aligned: AlignedConcepts, split: str, aux: bool = False) -> Batch:
    x, bag = ctx.inputs(split)
    return Batch(x.float(), ctx.split(split).labels, aligned.targets(split), bag,
                 xa=ctx.aux_features(split).float() if aux else None)


@TRAINING.register("sequential")
class Sequential(_Base):
    """Fit concepts to targets, freeze, then fit the predictor on predicted concepts.

    ``head_on_augmented`` (needs ``augment:``): the predictor trains and is selected on the concept
    predictions for one augmented view of each train / val image instead of the clean image, as in
    Koh et al.'s code (its ``ExtractConcepts`` step applies the training transform to every split)."""

    def __init__(self, head_on_augmented: bool = False, **kw):
        super().__init__(**kw)
        if head_on_augmented and self.augment is None:
            raise ValueError("`stages.training.head_on_augmented` needs `stages.training.augment`")
        self.head_on_augmented = head_on_augmented

    @property
    def _val_views(self) -> bool:
        return super()._val_views or self.head_on_augmented

    def _fit(self, layer, head, aligned, tr, va, seed, feed=None, aux_head=None):
        self._prepare(layer, aligned, tr)
        tr, va = self._normalized(layer, tr, va)
        log = self._fit_concepts(layer, aligned, tr, va, seed, feed, aux_head)
        if feed is not None:  # the head sees the fine-tuned encoder's features
            tr, va = self._normalized(layer, feed.encode("train"), feed.encode("val"))
        if self.head_on_augmented:
            g = torch.Generator().manual_seed(seed + 2_000)
            tr, va = tr.view(layer, g), va.view(layer, g)
        tr, va = self._frozen_concepts(layer, tr), self._frozen_concepts(layer, va)
        return log | self._fit_head(layer, head, tr, va, lambda b: layer.from_logits(b.z, b.x)[1], seed + 1)

    @torch.no_grad()
    def _frozen_concepts(self, layer: ConceptLayer, b: Batch) -> Batch:
        """``b`` with the frozen concept layer's logits ``z``, computed once rather than every step."""
        layer.eval()  # no dropout, also when the x -> c phase was skipped
        if b.xn is not None:
            z = layer.logits_from_normalized(b.xn)
        else:  # instance bags, in chunks so that the normalized bag never exists in full
            z = torch.cat([b[i:i + self.batch_size].concept_logits(layer) for i in range(0, len(b), self.batch_size)])
        return dataclasses.replace(b, z=z)


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
    the auxiliary concepts also go through the head, so ``aux`` changes the head on a frozen backbone too.

    ``normalize_loss`` divides the sum by ``1 + concept_weight``, as Koh et al.'s ``-normalize_loss``
    divides by ``1 + lambda * K`` (their concept loss sums over the K concepts, so their lambda * K is
    our ``concept_weight``), auxiliary terms included. Under SGD this equals scaling ``lr`` and
    ``weight_decay``, not ``lr`` alone."""

    aux_reaches_head = True

    def __init__(self, concept_weight: float = 1.0, normalize_loss: bool = False, **kw):
        super().__init__(**kw)
        self.concept_weight, self.normalize_loss = concept_weight, normalize_loss

    def _aux_loss(self, aux_head, layer, head, b):
        """The auxiliary concepts go through the head via ``layer.from_logits``. They skip the layer's
        ``logits_from_normalized``, so a generation that changes its scores there needs a matching
        change here (none does; with targets, ``c_mean`` / ``c_std`` stay 0 / 1)."""
        logits = aux_head(b.xa)
        loss = self._task_loss(head(layer.from_logits(logits, None)[1], None, b.bag), b)
        if self.concept_weight > 0:
            loss = loss + self.concept_weight * self._concept_loss(layer, b, logits)
        return loss

    def _fit(self, layer, head, aligned, tr, va, seed, feed=None, aux_head=None):
        self._prepare(layer, aligned, tr)
        tr, va = self._normalized(layer, tr, va)
        use_c = aligned.has_targets and self.concept_weight > 0

        def total(b: Batch, train: bool = False) -> Step:
            xn = b.normalized(layer) if head.uses_features else None
            c_logits = b.concept_logits(layer, xn)
            rep = layer.from_logits(c_logits, b.x)[1]  # = layer(b.x)[1]
            logits = head(rep, xn, b.bag)
            loss = self._task_loss(logits, b)
            if use_c:  # under dropout, the concept loss must see the mask the head saw
                loss = loss + self.concept_weight * self._concept_loss(
                    layer, b, c_logits if layer.dropout is not None else None)
            if train:
                loss = loss + self._aux_term(aux_head, layer, head, b)
            if self.normalize_loss:  # Koh et al. divide the whole loss, auxiliary terms included
                loss = loss / (1 + (self.concept_weight if use_c else 0.0))
            return loss, class_accuracy(logits, b.y)

        def train_step(b: Batch) -> Step:
            loss, acc = total(b, train=True)
            return loss + head.penalty(), acc

        groups = [(layer.concept_params() + layer.rep_params() + _params(aux_head), self._optimizer()),
                  self._head_group(head, list(head.parameters()))]
        if feed is not None:  # encoder, concept layer and head together, on images
            log = optimize_images(groups, [layer, head], feed, train_step, total, after_step=head.proximal_step,
                                  select=self.select, log=self._history.phase("joint"))
            va = feed.encode("val")
            out = {"finetune_epochs": log["epochs"], "joint_val_loss": log["val_loss"]}
        else:
            tr_fit, va_fit = (Batch.cat(tr, va), None) if self.train_on_val else (tr, va)
            g = torch.Generator().manual_seed(seed + 1_000)  # draws the augmented view of each image
            log = optimize(groups, [layer, head], len(tr_fit), lambda idx: train_step(tr_fit[idx].view(layer, g)),
                           None if va_fit is None else lambda: total(va_fit), self.epochs,
                           self.batch_size, self.patience, seed, after_step=head.proximal_step,
                           select=self.select, drop_last=self.drop_last, log=self._history.phase("joint"))
            out = {"joint_epochs": log["epochs"], "joint_val_loss": log["val_loss"]}
        out |= _extra(log, "joint")
        if use_c:
            with torch.no_grad():
                out["concept_val_loss"] = float(self._concept_loss(layer, va))
        return out


class _ImbalanceWeightedConcepts:
    """Koh et al. (2020): concept k's BCE is scaled by its negative/positive ratio on the training
    targets, ``(1 - p_k) / p_k`` (their ``BCEWithLogitsLoss(weight=...)``, not ``pos_weight``), then
    averaged over concepts. Needs binary (N, K) image-level targets. The ratios come from the
    training split only, also when training on train + val (as theirs come from ``train.pkl``)."""

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
