"""CUB-200-2011 with its 312 binary attributes and 15 part keypoints.

Expected: the official ``CUB_200_2011`` directory (images.txt, classes.txt,
image_class_labels.txt, train_test_split.txt, attributes/, parts/). Validation is a
deterministic ``val_fraction`` of the official train split. Attribute annotations are
majority-voted per class by default (as in Koh et al., 2020), which removes per-image noise.
Each attribute is mapped to the body part its name refers to, which gives localization targets.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..registry import DATASETS
from .base import ImageDataset, Sample

# Attribute-name fragment -> CUB part names (from parts/parts.txt).
_ATTR_TO_PARTS = {
    "bill": ["beak"],
    "wing": ["left wing", "right wing"],
    "upperparts": ["back", "nape"],
    "underparts": ["belly", "breast"],
    "breast": ["breast"],
    "back": ["back"],
    "tail": ["tail"],
    "upper_tail": ["tail"],
    "under_tail": ["tail"],
    "head": ["crown", "forehead"],
    "throat": ["throat"],
    "eye": ["left eye", "right eye"],
    "forehead": ["forehead"],
    "nape": ["nape"],
    "belly": ["belly"],
    "leg": ["left leg", "right leg"],
    "crown": ["crown"],
}


def _read(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


@DATASETS.register("cub")
class CUB(ImageDataset):
    name = "cub"
    n_attrs = 1

    def __init__(self, root: str = "data/CUB_200_2011", class_level_concepts: bool = True,
                 min_class_count: int = 10, val_fraction: float = 0.1):
        self.root = Path(root)
        if not (self.root / "images.txt").exists():
            raise FileNotFoundError(f"CUB not found at {self.root}")
        self.class_level_concepts, self.val_fraction = class_level_concepts, val_fraction
        self.min_class_count = min_class_count
        self._load()

    def cache_key(self):
        return {"name": self.name, "root": str(self.root.resolve()), "class_level": self.class_level_concepts,
                "min_class_count": self.min_class_count, "val_fraction": self.val_fraction}

    @property
    def class_names(self) -> list[str]:
        return self._classes

    @property
    def concept_names(self) -> list[str]:
        return self._attr_names

    def _load(self) -> None:
        r = self.root
        images = {int(i): p for i, p in _read(r / "images.txt")}
        labels = {int(i): int(c) - 1 for i, c in _read(r / "image_class_labels.txt")}
        is_train = {int(i): t == "1" for i, t in _read(r / "train_test_split.txt")}
        self._classes = [n.split(".", 1)[1].replace("_", " ") for _, n in _read(r / "classes.txt")]

        attr_rows = _read(r / "attributes" / "attributes.txt") if (r / "attributes" / "attributes.txt").exists() \
            else _read(r.parent / "attributes.txt")
        all_attr_names = [name for _, name in attr_rows]
        n_attr = len(all_attr_names)
        present = np.zeros((max(images) + 1, n_attr), dtype=np.int8)
        for row in _read(r / "attributes" / "image_attribute_labels.txt"):
            img, attr, is_present = int(row[0]), int(row[1]) - 1, int(row[2])
            present[img, attr] = is_present

        if self.class_level_concepts:
            n_cls = len(self._classes)
            train_ids = [i for i in images if is_train[i]]
            cls_mean = np.zeros((n_cls, n_attr))
            for c in range(n_cls):
                ids = [i for i in train_ids if labels[i] == c]
                cls_mean[c] = present[ids].mean(0)
            cls_attr = (cls_mean >= 0.5).astype(np.int8)
            keep = np.where(cls_attr.sum(0) >= self.min_class_count)[0]
            concept_matrix = {i: cls_attr[labels[i], keep] for i in images}
        else:
            keep = np.arange(n_attr)
            concept_matrix = {i: present[i] for i in images}
        self._attr_names = [all_attr_names[k] for k in keep]

        part_names = {int(row[0]): " ".join(row[1:]) for row in _read(r / "parts" / "parts.txt")}
        parts_by_img: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
        for row in _read(r / "parts" / "part_locs.txt"):
            img, part, x, y, vis = int(row[0]), int(row[1]), float(row[2]), float(row[3]), int(row[4])
            if vis:
                parts_by_img[img][part_names[part]] = (x, y)
        sizes = self._image_sizes(images)  # header-only reads; keypoints are in pixel coordinates
        concept_parts = [self._parts_for(name) for name in self._attr_names]

        rng = np.random.default_rng(0)
        self._splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
        for i in sorted(images):
            w, h = sizes[i]
            keypoints = {}
            for k, parts in enumerate(concept_parts):
                visible = [parts_by_img[i][p] for p in parts if p in parts_by_img[i]]
                if visible and concept_matrix[i][k]:
                    x, y = visible[0]
                    keypoints[k] = (x, y, w, h)
            s = Sample(image=str(r / "images" / images[i]), label=labels[i], attr=0,
                       concepts=concept_matrix[i].tolist(), keypoints=keypoints)
            split = "test" if not is_train[i] else ("val" if rng.random() < self.val_fraction else "train")
            self._splits[split].append(s)

    def _image_sizes(self, images: dict[int, str]) -> dict[int, tuple[float, float]]:
        from PIL import Image

        return {i: Image.open(self.root / "images" / p).size for i, p in images.items()}

    @staticmethod
    def _parts_for(attr_name: str) -> list[str]:
        # attribute names look like "has_bill_shape::dagger"
        stem = attr_name.split("::")[0].removeprefix("has_")
        for key in sorted(_ATTR_TO_PARTS, key=len, reverse=True):
            if stem.startswith(key):
                return _ATTR_TO_PARTS[key]
        return []

    def samples(self, split: str) -> list[Sample]:
        return self._splits[split]
