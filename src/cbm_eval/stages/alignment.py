"""Alignment: tie each concept neuron to its concept's meaning.

``clip``/``dino``/``human`` supply supervision targets (pseudo-labels or annotations);
``weights`` fixes the neuron's weight vector to the concept's text embedding (no targets);
``segment_clip`` gives every instance of a bag its own CLIP targets (SEG-MIL-CBM).
With instance bags, image-level targets supervise the max over each image's instances.
"""

from __future__ import annotations

import warnings

import torch.nn.functional as F

from ..registry import ALIGNMENT
from .base import AlignedConcepts, Alignment
from .common import concept_activations, match_names
from .scorers import SCORERS


@ALIGNMENT.register("clip")
class CLIPAlignment(Alignment):
    """Continuous pseudo-labels: teacher image-text similarity, standardized on train (LF-CBM)."""

    def __init__(self, power: int = 1):
        self.power = power

    def fit(self, concepts, ctx):
        train = concept_activations(concepts, ctx, "train") ** self.power
        mean, std = train.mean(0), train.std(0).clamp_min(1e-6)

        def targets(split):
            return ((concept_activations(concepts, ctx, split) ** self.power) - mean) / std

        return AlignedConcepts(concepts, target_type="continuous", targets_fn=targets)


@ALIGNMENT.register("weights")
class WeightAlignment(Alignment):
    """Concept neurons are the concepts' embeddings in backbone space (zero-shot, LaBo-style init)."""

    def __init__(self, frozen: bool = True):
        self.frozen = frozen

    def fit(self, concepts, ctx):
        if concepts.vectors is not None:
            weight = concepts.vectors
        elif ctx.backbone.has_text:
            weight = ctx.encode_text(concepts.names, which="backbone")
        else:
            raise RuntimeError("weights alignment needs concept vectors or a backbone with a text encoder")
        return AlignedConcepts(concepts, target_type=None, init_weight=weight, frozen=self.frozen)


@ALIGNMENT.register("dino")
class DINOAlignment(Alignment):
    """Binary pseudo-labels from an open-vocabulary detector."""

    def __init__(self, scorer: dict | str = "grounding_dino", threshold: float = 0.3):
        self.scorer, self.threshold = SCORERS.build(scorer), threshold

    def fit(self, concepts, ctx):
        def targets(split):
            return (self.scorer.score(concepts, ctx, split) > self.threshold).float()

        return AlignedConcepts(concepts, target_type="binary", targets_fn=targets)


@ALIGNMENT.register("human")
class HumanAlignment(Alignment):
    """Binary targets from dataset annotations; concepts without annotations are dropped."""

    def fit(self, concepts, ctx):
        idx = match_names(concepts.names, ctx.dataset.concept_names)
        keep = [i for i, j in enumerate(idx) if j is not None]
        if not keep:
            raise RuntimeError("human alignment: no concept matches the dataset's annotated concepts")
        if len(keep) < len(concepts):
            warnings.warn(f"human alignment dropped {len(concepts) - len(keep)} unannotated concepts")
        cols = [idx[i] for i in keep]

        def targets(split):
            return ctx.split(split).concept_labels[:, cols]

        return AlignedConcepts(concepts.subset(keep), target_type="binary", targets_fn=targets)


@ALIGNMENT.register("segment_clip")
class SegmentCLIPAlignment(Alignment):
    """Per-instance targets from the teacher's similarity between each segment and each concept.

    ``targets: softmax`` is SEG-MIL-CBM's ``q_k(s_i) = softmax_k(cos(s_i, c_k) / temperature)``;
    ``standardized`` z-scores each concept's cosine over real training instances (LF-CBM style);
    ``cosine`` uses the raw cosine. Padding instances get all-zero targets.
    """

    def __init__(self, targets: str = "softmax", temperature: float = 0.01):
        if targets not in ("softmax", "standardized", "cosine"):
            raise ValueError(f"Unknown segment target '{targets}' (use softmax, standardized or cosine)")
        self.kind, self.temperature = targets, temperature

    def fit(self, concepts, ctx):
        if ctx.instance_source is None:
            raise RuntimeError("segment_clip alignment needs instance bags; set `instances:` in the config")
        if concepts.vectors is not None:
            raise RuntimeError("segment_clip alignment needs named concepts with text embeddings")
        text = F.normalize(ctx.encode_text(concepts.names).float(), dim=1)

        def cosine(split):
            inst = ctx.instances(split)
            feats = inst.teacher_features
            if feats is None:
                if ctx.teacher is not ctx.backbone:
                    raise RuntimeError(f"`instances: {ctx.instance_source.name}` has no teacher features; "
                                       "use a CLIP backbone or a segment source with `teacher:`")
                feats = inst.features
            return F.normalize(feats.float(), dim=-1) @ text.T, inst.mask

        stats = {}
        if self.kind == "standardized":
            sim, mask = cosine("train")
            stats = {"mean": sim[mask].mean(0), "std": sim[mask].std(0).clamp_min(1e-6)}

        def targets(split):
            sim, mask = cosine(split)
            if self.kind == "softmax":
                sim = (sim / self.temperature).softmax(-1)
            elif self.kind == "standardized":
                sim = (sim - stats["mean"]) / stats["std"]
            return sim * mask[..., None]

        return AlignedConcepts(concepts, target_type="continuous", targets_fn=targets)
