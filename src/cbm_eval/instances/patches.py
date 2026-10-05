"""Backbone patch tokens as instances: the cheapest bag (no segmentation), and the control for
whether concept-guided segments matter."""

from __future__ import annotations

import torch

from ..registry import INSTANCES
from ..structures import InstanceSplit
from .base import InstanceSource


@INSTANCES.register("patches")
class PatchInstances(InstanceSource):
    name = "patches"
    needs_patches = True
    cacheable = False  # read straight from the feature cache

    def __init__(self):
        self.kw = {}

    def build(self, ctx, split):
        fs = ctx.split(split)
        if fs.patch_features is None:
            raise RuntimeError(f"{type(ctx.backbone).__name__} gives no patch features for `instances: patches`")
        n, p, _ = fs.patch_features.shape
        return InstanceSplit(features=fs.patch_features, mask=torch.ones(n, p, dtype=torch.bool),
                             area=torch.full((n, p), 1.0 / p), grid_aligned=True)
