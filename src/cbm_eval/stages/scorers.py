"""Per-image concept presence scorers, used by DINO-based filtering and alignment.

``grounding_dino`` runs an open-vocabulary detector; ``oracle`` reads human annotations
(useful as an upper bound and for tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..registry import Registry
from ..structures import ConceptSet
from ..utils import stable_hash
from .common import match_names

if TYPE_CHECKING:
    from ..context import Context

SCORERS: Registry = Registry("concept scorer")


@SCORERS.register("oracle")
class OracleScorer:
    def score(self, concepts: ConceptSet, ctx: "Context", split: str) -> torch.Tensor:
        fs = ctx.split(split)
        if fs.concept_labels is None:
            raise RuntimeError("oracle scorer needs a dataset with concept annotations")
        idx = match_names(concepts.names, ctx.dataset.concept_names)
        out = torch.zeros(len(fs), len(concepts))
        for k, j in enumerate(idx):
            if j is not None:
                out[:, k] = fs.concept_labels[:, j]
        return out


@SCORERS.register("grounding_dino")
class GroundingDINOScorer:
    """Max box confidence per concept (``uv sync --extra hf``). Results are cached on disk."""

    def __init__(self, model: str = "IDEA-Research/grounding-dino-tiny", box_threshold: float = 0.25,
                 concepts_per_prompt: int = 16, batch_size: int = 8):
        self.model_id, self.box_threshold = model, box_threshold
        self.concepts_per_prompt, self.batch_size = concepts_per_prompt, batch_size
        self.detector = None

    def score(self, concepts: ConceptSet, ctx: "Context", split: str) -> torch.Tensor:
        key = {"model": self.model_id, "data": ctx.dataset.cache_key(), "split": split, "concepts": concepts.names}
        path = ctx.cache_dir / "scores" / f"gdino-{split}-{stable_hash(key)}.pt"
        if path.exists():
            return torch.load(path)
        from PIL import Image

        from ..grounding import GroundingDINODetector

        if self.detector is None:
            self.detector = GroundingDINODetector(self.model_id, self.box_threshold, device=ctx.device)
        paths = ctx.split(split).paths
        if not paths or paths[0] is None:
            raise RuntimeError("grounding_dino needs image paths")
        out = torch.zeros(len(paths), len(concepts))
        chunks = [list(range(i, min(i + self.concepts_per_prompt, len(concepts))))
                  for i in range(0, len(concepts), self.concepts_per_prompt)]
        for start in range(0, len(paths), self.batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start : start + self.batch_size]]
            for chunk in chunks:
                names = [concepts.names[k] for k in chunk]
                for b, dets in enumerate(self.detector.detect(images, [names] * len(images))):
                    for d in dets:
                        if d.label >= 0:
                            k = chunk[d.label]
                            out[start + b, k] = max(out[start + b, k].item(), d.score)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
        return out
