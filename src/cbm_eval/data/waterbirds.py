"""Waterbirds (Sagawa et al., 2020) in the standard ``metadata.csv`` layout.

Expected: ``<root>/metadata.csv`` with columns ``img_filename, y, split, place`` where split is
0/1/2 for train/val/test and ``place`` is the background (0 land, 1 water).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..registry import DATASETS
from .base import ImageDataset, Sample

_SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


@DATASETS.register("waterbirds")
class Waterbirds(ImageDataset):
    name = "waterbirds"
    n_attrs = 2

    def __init__(self, root: str = "data/waterbirds"):
        self.root = Path(root)
        meta = self.root / "metadata.csv"
        if not meta.exists():
            raise FileNotFoundError(f"Waterbirds metadata not found at {meta}")
        self.meta = pd.read_csv(meta)

    @property
    def class_names(self) -> list[str]:
        return ["landbird", "waterbird"]

    def cache_key(self):
        return {"name": self.name, "root": str(self.root.resolve())}

    def samples(self, split: str) -> list[Sample]:
        rows = self.meta[self.meta["split"] == _SPLIT_IDS[split]]
        return [
            Sample(image=str(self.root / r.img_filename), label=int(r.y), attr=int(r.place))
            for r in rows.itertuples()
        ]
