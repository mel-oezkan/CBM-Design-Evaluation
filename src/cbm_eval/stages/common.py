"""Helpers shared by several stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..structures import ConceptSet

if TYPE_CHECKING:
    from ..context import Context


def concept_embeddings(concepts: ConceptSet, ctx: "Context") -> torch.Tensor:
    """Unit-norm concept embeddings: SAE-style vectors if present, else teacher text embeddings."""
    if concepts.vectors is not None:
        return F.normalize(concepts.vectors.float(), dim=1)
    return F.normalize(ctx.encode_text(concepts.names), dim=1)


def concept_activations(concepts: ConceptSet, ctx: "Context", split: str) -> torch.Tensor:
    """(N, K) image-concept similarity: cosine to teacher text, or projection onto concept vectors."""
    if concepts.vectors is not None:
        train = ctx.split("train").features
        x = (ctx.split(split).features - train.mean(0)) / train.std(0).clamp_min(1e-6)
        return x @ F.normalize(concepts.vectors.float(), dim=1).T
    img = F.normalize(ctx.teacher_split(split).features.float(), dim=1)
    return img @ concept_embeddings(concepts, ctx).T


def match_names(names: list[str], reference: list[str] | None) -> list[int | None]:
    """Index of each name in ``reference`` (case-insensitive), or None."""
    if not reference:
        return [None] * len(names)
    lookup = {r.lower(): i for i, r in enumerate(reference)}
    return [lookup.get(n.lower()) for n in names]
