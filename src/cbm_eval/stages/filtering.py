"""Filtering: prune a candidate concept set. Configured as a chain, applied in order."""

from __future__ import annotations

import random
import re

import torch

from ..registry import FILTERING
from ..structures import ConceptSet
from .base import Filter
from .common import concept_activations, concept_embeddings
from .scorers import SCORERS


def _normalize(name: str) -> str:
    name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in name.split()]
    return " ".join(words)


@FILTERING.register("rules")
class RuleFilter(Filter):
    """Text rules: length limits, near-duplicate names, and concepts that name a class."""

    def __init__(self, max_words: int = 4, min_chars: int = 2, remove_class_names: bool = True,
                 dedupe: bool = True, blocklist: list[str] | None = None):
        self.max_words, self.min_chars = max_words, min_chars
        self.remove_class_names, self.dedupe = remove_class_names, dedupe
        self.blocklist = {_normalize(b) for b in blocklist or []}

    def filter(self, concepts, ctx):
        classes = [_normalize(c) for c in ctx.class_names]
        keep, seen = [], set()
        for i, name in enumerate(concepts.names):
            norm = _normalize(name)
            if len(norm) < self.min_chars or len(norm.split()) > self.max_words or norm in self.blocklist:
                continue
            if self.remove_class_names and concepts.vectors is None and any(
                    re.search(rf"\b{re.escape(c)}\b", norm) or re.search(rf"\b{re.escape(norm)}\b", c)
                    for c in classes):
                continue
            if self.dedupe and norm in seen:
                continue
            seen.add(norm)
            keep.append(i)
        return concepts.subset(keep)


@FILTERING.register("clip")
class CLIPFilter(Filter):
    """Embedding-space filters from Label-free CBM: too close to a class name, near-duplicates,
    and concepts that never activate strongly on training images."""

    def __init__(self, class_sim: float | None = 0.85, dedupe_sim: float | None = 0.9,
                 min_top_activation: float | None = None, top_k_images: int = 5):
        self.class_sim, self.dedupe_sim = class_sim, dedupe_sim
        self.min_top_activation, self.top_k_images = min_top_activation, top_k_images

    def filter(self, concepts, ctx):
        emb = concept_embeddings(concepts, ctx)
        keep = list(range(len(concepts)))
        if self.class_sim is not None and concepts.vectors is None:
            cls = torch.nn.functional.normalize(ctx.encode_text(ctx.class_names), dim=1)
            max_sim = (emb @ cls.T).max(1).values
            keep = [i for i in keep if max_sim[i] < self.class_sim]
        if self.min_top_activation is not None:
            act = concept_activations(concepts, ctx, "train")
            top = act.topk(min(self.top_k_images, len(act)), dim=0).values.mean(0)
            keep = [i for i in keep if top[i] >= self.min_top_activation]
        if self.dedupe_sim is not None:
            kept: list[int] = []
            for i in keep:
                if not kept or (emb[kept] @ emb[i]).max() < self.dedupe_sim:
                    kept.append(i)
            keep = kept
        return concepts.subset(keep)


@FILTERING.register("dino")
class DINOFilter(Filter):
    """Visual grounding check: keep concepts a detector finds in a plausible fraction of images."""

    def __init__(self, scorer: dict | str = "grounding_dino", threshold: float = 0.3, min_rate: float = 0.01,
                 max_rate: float = 0.95, split: str = "val"):
        self.scorer = SCORERS.build(scorer)
        self.threshold, self.min_rate, self.max_rate, self.split = threshold, min_rate, max_rate, split

    def filter(self, concepts, ctx):
        rate = (self.scorer.score(concepts, ctx, self.split) > self.threshold).float().mean(0)
        return concepts.subset([i for i in range(len(concepts)) if self.min_rate <= rate[i] <= self.max_rate])


@FILTERING.register("select")
class SelectFilter(Filter):
    """Pick ``k`` concepts: random, highest-variance, most class-discriminative, or submodular
    (facility-location coverage of the candidate set + discriminability, as in LaBo)."""

    def __init__(self, k: int = 50, method: str = "submodular", alpha: float = 1.0):
        if method not in ("random", "variance", "discriminative", "submodular"):
            raise ValueError(f"Unknown selection method '{method}'")
        self.k, self.method, self.alpha = k, method, alpha

    def filter(self, concepts, ctx):
        if len(concepts) <= self.k:
            return concepts
        if self.method == "random":
            return concepts.subset(sorted(random.Random(ctx.seed).sample(range(len(concepts)), self.k)))
        act = concept_activations(concepts, ctx, "train")
        if self.method == "variance":
            return concepts.subset(sorted(act.var(0).topk(self.k).indices.tolist()))
        disc = self._discriminability(act, ctx.split("train").labels, ctx.num_classes)
        if self.method == "discriminative":
            return concepts.subset(sorted(disc.topk(self.k).indices.tolist()))
        return concepts.subset(sorted(self._greedy(concept_embeddings(concepts, ctx), disc)))

    @staticmethod
    def _discriminability(act: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Max over classes of the standardized gap between in-class and overall mean activation."""
        z = (act - act.mean(0)) / act.std(0).clamp_min(1e-6)
        class_means = torch.stack([z[labels == c].mean(0) for c in range(n_classes) if (labels == c).any()])
        d = class_means.max(0).values
        return (d - d.min()) / (d.max() - d.min()).clamp_min(1e-6)

    def _greedy(self, emb: torch.Tensor, disc: torch.Tensor) -> list[int]:
        sim = (emb @ emb.T).clamp_min(0)
        chosen: list[int] = []
        coverage = torch.zeros(len(emb))
        for _ in range(self.k):
            gain = torch.clamp(sim - coverage[None, :], min=0).sum(1) / len(emb) + self.alpha * disc
            if chosen:
                gain[chosen] = -float("inf")
            j = int(gain.argmax())
            chosen.append(j)
            coverage = torch.maximum(coverage, sim[j])
        return chosen
