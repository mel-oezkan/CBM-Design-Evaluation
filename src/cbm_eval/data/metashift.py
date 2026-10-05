"""MetaShift spurious-correlation subset (e.g. cat vs. dog with indoor/outdoor contexts).

Expected: ``<root>/metadata.csv`` with columns ``filename, y, a, split`` where split is
0/1/2 (train/val/test) or the strings train/val/test, as produced by the SubpopBench
preprocessing scripts. ``class_names`` defaults to the cat/dog task.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..registry import DATASETS
from .base import ImageDataset, Sample

_SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


@DATASETS.register("metashift")
class MetaShift(ImageDataset):
    name = "metashift"
    n_attrs = 2

    def __init__(self, root: str = "data/metashift", class_names: list[str] | None = None):
        self.root = Path(root)
        meta = self.root / "metadata.csv"
        if not meta.exists():
            raise FileNotFoundError(f"MetaShift metadata not found at {meta}")
        self.meta = pd.read_csv(meta)
        self._class_names = class_names or ["cat", "dog"]

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    def cache_key(self):
        return {"name": self.name, "root": str(self.root.resolve())}

    def samples(self, split: str) -> list[Sample]:
        col = self.meta["split"]
        mask = (col == split) if col.dtype == object else (col == _SPLIT_IDS[split])
        return [
            Sample(image=str(self.root / r.filename), label=int(r.y), attr=int(r.a))
            for r in self.meta[mask].itertuples()
        ]
