"""Instance sources turn each image into a set of instances (patch tokens or segments) for
multiple-instance CBMs. Configured by the top-level ``instances:`` key."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch

from ..structures import InstanceSplit
from ..utils import stable_hash

if TYPE_CHECKING:
    from ..context import Context


class InstanceSource(ABC):
    needs_patches: bool = False  # reads backbone patch features from the feature cache
    needs_concepts: bool = False  # depends on the filtered concept set (concept-guided segmentation)
    cacheable: bool = True  # False when the instances are derived from an existing cache

    name: str = "instances"
    kw: dict[str, Any] = {}

    def cache_key(self, ctx: "Context") -> dict[str, Any]:
        """Everything about this source that changes the instances (the dataset and encoders are added by the caller)."""
        key = {"name": self.name, **self.kw}
        if self.needs_concepts:
            key["concepts"] = stable_hash(_concepts(ctx).names)
        return key

    @abstractmethod
    def build(self, ctx: "Context", split: str) -> InstanceSplit: ...


def _concepts(ctx: "Context"):
    if ctx.concepts is None:
        raise RuntimeError("This instance source is concept-guided and needs the filtered concept set "
                           "(run discovery and filtering first)")
    return ctx.concepts


def load_instances(ctx: "Context", source: InstanceSource, split: str) -> InstanceSplit:
    """Build or load the instances of one split; features are stored in half precision."""
    if not source.cacheable:
        inst = source.build(ctx, split)
    else:
        teacher = ctx.teacher if ctx.teacher is not None and ctx.teacher is not ctx.backbone else None
        dataset, part = ctx.source(split)
        key = {"dataset": dataset.cache_key(), "backbone": ctx.backbone.cache_key(),
               "teacher": teacher.cache_key() if teacher is not None else None,
               "source": source.cache_key(ctx), "split": part}
        path = ctx.cache_dir / "instances" / dataset.name / f"{part}-{stable_hash(key)}.pt"
        if path.exists():
            inst = InstanceSplit(**torch.load(path, weights_only=False))
        else:
            inst = source.build(ctx, split)
            data = asdict(inst)
            for k in ("features", "teacher_features"):
                if data[k] is not None:
                    data[k] = data[k].half()
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, path)
    inst.features = inst.features.float()
    if inst.teacher_features is not None:
        inst.teacher_features = inst.teacher_features.float()
    return inst
