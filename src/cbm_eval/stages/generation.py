"""Generation: what the predictor sees for each concept -- a score, an embedding, or a binary bit."""

from __future__ import annotations

import torch
from torch import nn

from ..registry import GENERATION
from .base import AlignedConcepts, ConceptLayer, Generation


class ScoreLayer(ConceptLayer):
    def __init__(self, in_dim, aligned):
        super().__init__(in_dim, aligned)
        self.rep_dim = self.n_concepts

    def represent(self, c_hat, x):
        return c_hat


class BagOfConceptsLayer(ConceptLayer):
    """Hard 0/1 concept presence; straight-through gradients during joint training."""

    def __init__(self, in_dim, aligned):
        super().__init__(in_dim, aligned)
        self.rep_dim = self.n_concepts

    def represent(self, c_hat, x):
        hard = self.binarize(c_hat)
        return hard + c_hat - c_hat.detach()


class EmbeddingLayer(ConceptLayer):
    """Concept Embedding Model: per concept, mix a positive and negative embedding by its probability."""

    def __init__(self, in_dim, aligned, emb_dim: int = 16):
        super().__init__(in_dim, aligned)
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


@GENERATION.register("scores")
class Scores(Generation):
    def build(self, in_dim: int, aligned: AlignedConcepts) -> ConceptLayer:
        return ScoreLayer(in_dim, aligned)


@GENERATION.register("boc")
class BagOfConcepts(Generation):
    def build(self, in_dim, aligned):
        return BagOfConceptsLayer(in_dim, aligned)


@GENERATION.register("embeddings")
class Embeddings(Generation):
    def __init__(self, emb_dim: int = 16):
        self.emb_dim = emb_dim

    def build(self, in_dim, aligned):
        return EmbeddingLayer(in_dim, aligned, self.emb_dim)
