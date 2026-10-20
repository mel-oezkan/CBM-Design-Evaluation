"""Generation: what the predictor sees for each concept -- a score, an embedding, or a binary bit."""

from __future__ import annotations

import torch
from torch import nn

from ..registry import GENERATION
from .base import AlignedConcepts, ConceptLayer, Generation


class ScoreLayer(ConceptLayer):
    represent_uses_x = False

    def __init__(self, in_dim, aligned, **kw):
        super().__init__(in_dim, aligned, **kw)
        self.rep_dim = self.n_concepts

    def represent(self, c_hat, x):
        return c_hat


class LogitScoreLayer(ConceptLayer):
    """The predictor sees concept logits rather than probabilities: Koh et al. (2020) connect f to
    the logits for sequential and joint CBMs. ``c_hat`` stays a probability (so interventions keep
    working). By default the head reads it mapped back with ``logit`` clamped at ``eps``: logits
    past +-logit(1 - eps) (+-13.8 at 1e-6) reach the head clamped and get no task-loss gradient.
    ``raw`` passes the unclamped logits, as Koh et al. do; only intervened values and true targets
    (``independent``) still go through the clamped ``logit``. Either way an intervened 0/1 becomes
    +-logit(1 - eps) instead of Koh et al.'s 5th/95th-percentile logits."""

    represent_uses_x = False

    def __init__(self, in_dim, aligned, eps: float, raw: bool = False, **kw):
        super().__init__(in_dim, aligned, **kw)
        self.rep_dim, self.eps, self.raw = self.n_concepts, eps, raw

    def represent(self, c_hat, x):
        return torch.logit(c_hat, eps=self.eps) if self.target_type == "binary" else c_hat

    def from_logits(self, logits, x, intervene=None):
        if not self.raw or self.target_type != "binary":
            return super().from_logits(logits, x, intervene)
        c_hat = self.activate(logits)
        if intervene is None:
            return c_hat, logits
        c_hat, mask = self.intervened(c_hat, intervene)
        return c_hat, torch.where(mask, self.represent(c_hat, x), logits)


class BagOfConceptsLayer(ConceptLayer):
    """Hard 0/1 concept presence; straight-through gradients during joint training."""

    represent_uses_x = False

    def __init__(self, in_dim, aligned, **kw):
        super().__init__(in_dim, aligned, **kw)
        self.rep_dim = self.n_concepts

    def represent(self, c_hat, x):
        hard = self.binarize(c_hat)
        return hard + c_hat - c_hat.detach()


class EmbeddingLayer(ConceptLayer):
    """Concept Embedding Model: per concept, mix a positive and negative embedding by its probability."""

    def __init__(self, in_dim, aligned, emb_dim: int = 16, **kw):
        super().__init__(in_dim, aligned, **kw)
        self.emb_dim = emb_dim
        self.embed = nn.Linear(in_dim, self.n_concepts * emb_dim * 2)
        self.rep_dim = self.n_concepts * emb_dim

    def rep_params(self):
        return list(self.embed.parameters())

    def represent(self, c_hat, x):
        p = c_hat if self.target_type == "binary" else torch.sigmoid(c_hat)
        e = self.embed(self.normalize(x)).view(*x.shape[:-1], self.n_concepts, 2, self.emb_dim)
        mixed = p.unsqueeze(-1) * e[..., 0, :] + (1 - p).unsqueeze(-1) * e[..., 1, :]
        return mixed.flatten(-2)


class _LinearOptions(Generation):
    """Options of the x -> c linear layer, shared by every variant: ``dropout`` on its input while
    training, ``standardize`` (z-score the features with training statistics, or use them raw) and
    ``init_std`` (truncated-normal weights; None: PyTorch's default). Koh et al. (2020) use 0.5,
    raw features and 0.1."""

    def __init__(self, dropout: float = 0.0, standardize: bool = True, init_std: float | None = None):
        self.dropout, self.standardize, self.init_std = dropout, standardize, init_std

    @property
    def layer_kw(self) -> dict:
        return {"dropout": self.dropout, "standardize": self.standardize, "init_std": self.init_std}


@GENERATION.register("scores")
class Scores(_LinearOptions):
    """Concept probabilities (binary targets) or scores (continuous) into the predictor. ``dropout``,
    ``standardize`` and ``init_std`` configure the x -> c linear layer, as for every generation."""

    def build(self, in_dim: int, aligned: AlignedConcepts) -> ConceptLayer:
        return ScoreLayer(in_dim, aligned, **self.layer_kw)


@GENERATION.register("logits")
class LogitScores(_LinearOptions):
    """Concept logits into the predictor (Koh et al.'s sequential and joint CBMs); see ``LogitScoreLayer``.
    ``raw``: the head reads the unclamped concept logits (default False keeps the clamp at ``eps``).
    ``dropout``, ``standardize`` and ``init_std`` configure the x -> c linear layer."""

    def __init__(self, eps: float = 1e-6, raw: bool = False, **kw):
        super().__init__(**kw)
        self.eps, self.raw = eps, raw

    def build(self, in_dim, aligned):
        return LogitScoreLayer(in_dim, aligned, self.eps, self.raw, **self.layer_kw)


@GENERATION.register("boc")
class BagOfConcepts(_LinearOptions):
    """Hard 0/1 concepts into the predictor, with straight-through gradients in joint training."""

    def build(self, in_dim, aligned):
        return BagOfConceptsLayer(in_dim, aligned, **self.layer_kw)


@GENERATION.register("embeddings")
class Embeddings(_LinearOptions):
    """CEM-style concept embeddings of ``emb_dim`` per concept, mixed by the concept's probability."""

    def __init__(self, emb_dim: int = 16, **kw):
        super().__init__(**kw)
        self.emb_dim = emb_dim

    def build(self, in_dim, aligned):
        return EmbeddingLayer(in_dim, aligned, self.emb_dim, **self.layer_kw)
