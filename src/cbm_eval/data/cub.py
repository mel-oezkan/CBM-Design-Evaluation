"""CUB-200-2011 with its 312 binary attributes and 15 part keypoints.

Expected: the official ``CUB_200_2011`` directory (images.txt, classes.txt,
image_class_labels.txt, train_test_split.txt, attributes/, parts/). Validation is a
deterministic ``val_fraction`` of the official train split. Attribute annotations are
majority-voted per class by default (as in Koh et al., 2020), which removes per-image noise.
Each attribute is mapped to the body part its name refers to, which gives localization targets.

Part segmentations (optional) are the CUB70 masks of Behzadi et al.
(github.com/hamedbehzadi/CUB70-PartSegmentationDataset), extracted to
``<root>/part_segmentations/AnnotationMasksPerclass/<class>/<image>_<part>.png``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..registry import DATASETS
from .base import ImageDataset, PartAnnotation, PartSpec, Sample

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

# Keypoint groups: CUB parts with left and right merged.
_KEYPOINT_GROUPS = ["back", "beak", "belly", "breast", "crown", "forehead", "eye", "leg", "wing", "nape",
                    "tail", "throat"]
# CUB part -> part-segmentation group of the CUB70 masks (whose left / right masks get merged).
_PART_TO_SEG = {"back": "body", "belly": "body", "breast": "body", "throat": "body", "beak": "beak",
                "crown": "head", "forehead": "head", "nape": "neck", "tail": "tail", "eye": "eye",
                "wing": "wing", "leg": "leg"}
_SEG_GROUPS = ["body", "head", "neck", "beak", "tail", "wing", "leg", "eye"]
_SEG_FILES = {g: [f"left_{g}", f"right_{g}"] if g in ("eye", "wing", "leg") else [g] for g in _SEG_GROUPS}


def _group(part: str) -> str:
    return part.removeprefix("left ").removeprefix("right ")


def _read(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


@DATASETS.register("cub")
class CUB(ImageDataset):
    name = "cub"
    n_attrs = 1

    def __init__(self, root: str = "data/CUB_200_2011", class_level_concepts: bool = True,
                 min_class_count: int = 10, val_fraction: float = 0.1, part_segmentations: str | None = None,
                 majority_vote: str = "mean", split_dir: str | None = None):
        """``majority_vote``: ``mean`` labels a class positive when >= 50% of all its official-train
        images are (not-visible annotations count as negatives); ``koh`` follows Koh et al.'s
        ``get_class_attributes_data``: votes on the ``train`` split only, ignores "not visible"
        negatives, and breaks ties towards present (112 concepts with ``min_class_count=10``).
        ``split_dir``: a directory with Koh et al.'s ``train.pkl`` / ``val.pkl`` (``CUB_processed/
        class_attr_data_10``) to use their exact train/val split instead of ``val_fraction``."""
        if majority_vote not in ("mean", "koh"):
            raise ValueError(f"Unknown majority_vote '{majority_vote}' (use 'mean' or 'koh')")
        self.root = Path(root)
        if not (self.root / "images.txt").exists():
            raise FileNotFoundError(f"CUB not found at {self.root}")
        segs = Path(part_segmentations) if part_segmentations else \
            self.root / "part_segmentations" / "AnnotationMasksPerclass"
        self.seg_root = segs if segs.is_dir() else None
        self.class_level_concepts, self.val_fraction = class_level_concepts, val_fraction
        self.min_class_count, self.majority_vote = min_class_count, majority_vote
        self.split_dir = Path(split_dir) if split_dir else None
        self._load()

    def cache_key(self):
        key = {"name": self.name, "root": str(self.root.resolve()), "class_level": self.class_level_concepts,
               "min_class_count": self.min_class_count, "val_fraction": self.val_fraction}
        if self.majority_vote != "mean":  # omitted at the default so existing feature caches stay valid
            key["majority_vote"] = self.majority_vote
        if self.split_dir is not None:
            key["split_dir"] = str(self.split_dir.resolve())
        return key

    @property
    def class_names(self) -> list[str]:
        return self._classes

    @property
    def concept_names(self) -> list[str]:
        return self._attr_names

    def class_concepts(self) -> torch.Tensor | None:
        """The majority-voted (200, K) class attributes; None with image-level concepts."""
        return self._class_concepts

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
        not_visible = np.zeros((max(images) + 1, n_attr), dtype=bool)
        # 3.7M lines: (image, attribute, is_present, certainty, time); some carry an extra column.
        a = pd.read_csv(r / "attributes" / "image_attribute_labels.txt", sep=r"\s+", header=None,
                        usecols=[0, 1, 2, 3], dtype=np.int64).to_numpy()
        present[a[:, 0], a[:, 1] - 1] = a[:, 2]
        not_visible[a[:, 0], a[:, 1] - 1] = a[:, 3] == 1  # certainty 1 = "not visible"
        split_of = self._assign_splits(images, is_train)

        if self.class_level_concepts:
            n_cls = len(self._classes)
            if self.majority_vote == "koh":
                cls_attr = np.zeros((n_cls, n_attr), dtype=np.int8)
                for c in range(n_cls):
                    ids = [i for i in images if split_of[i] == "train" and labels[i] == c]
                    seen = ~(not_visible[ids] & (present[ids] == 0))  # a "not visible" 0 is no vote
                    pos = (present[ids] * seen).sum(0)
                    cls_attr[c] = pos >= seen.sum(0) - pos  # ties -> present
            else:
                train_ids = [i for i in images if is_train[i]]
                cls_mean = np.zeros((n_cls, n_attr))
                for c in range(n_cls):
                    ids = [i for i in train_ids if labels[i] == c]
                    cls_mean[c] = present[ids].mean(0)
                cls_attr = (cls_mean >= 0.5).astype(np.int8)
            keep = np.where(cls_attr.sum(0) >= self.min_class_count)[0]
            concept_matrix = {i: cls_attr[labels[i], keep] for i in images}
            self._class_concepts = torch.from_numpy(cls_attr[:, keep]).float()
        else:
            keep = np.arange(n_attr)
            concept_matrix = {i: present[i] for i in images}
            self._class_concepts = None
        self._attr_names = [all_attr_names[k] for k in keep]

        part_names = {int(row[0]): " ".join(row[1:]) for row in _read(r / "parts" / "parts.txt")}
        parts_by_img: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
        for row in _read(r / "parts" / "part_locs.txt"):
            img, part, x, y, vis = int(row[0]), int(row[1]), float(row[2]), float(row[3]), int(row[4])
            if vis:
                parts_by_img[img][part_names[part]] = (x, y)
        sizes = self._image_sizes(images)  # header-only reads; keypoints are in pixel coordinates
        concept_parts = [self._parts_for(name) for name in self._attr_names]
        self._concept_groups = [sorted({_group(p) for p in parts}) for parts in concept_parts]

        self._splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
        for i in sorted(images):
            w, h = sizes[i]
            keypoints = {}
            for k, parts in enumerate(concept_parts):
                visible = [parts_by_img[i][p] for p in parts if p in parts_by_img[i]]
                if visible and concept_matrix[i][k]:
                    x, y = visible[0]
                    keypoints[k] = (x, y, w, h)
            points: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for part, xy in parts_by_img[i].items():
                points[_group(part)].append(xy)
            masks = {}
            if self.seg_root is not None:
                folder, stem = self.seg_root / str(labels[i] + 1), Path(images[i]).stem
                masks = {g: [str(folder / f"{stem}_{f}.png") for f in files] for g, files in _SEG_FILES.items()}
            s = Sample(image=str(r / "images" / images[i]), label=labels[i], attr=0,
                       concepts=concept_matrix[i].tolist(), keypoints=keypoints,
                       parts=PartAnnotation(size=(int(w), int(h)), points=dict(points), masks=masks))
            self._splits[split_of[i]].append(s)

    def _assign_splits(self, images: dict[int, str], is_train: dict[int, bool]) -> dict[int, str]:
        """Image id -> split: the official test split, and train/val from ``split_dir`` or ``val_fraction``."""
        if self.split_dir is not None:
            import pickle

            val_paths = set()
            with open(self.split_dir / "val.pkl", "rb") as f:
                for d in pickle.load(f):
                    parts = d["img_path"].split("/")
                    val_paths.add("/".join(parts[parts.index("images") + 1:]))
            return {i: "test" if not is_train[i] else ("val" if images[i] in val_paths else "train")
                    for i in images}
        rng = np.random.default_rng(0)  # one draw per official-train image, in id order
        return {i: "test" if not is_train[i] else ("val" if rng.random() < self.val_fraction else "train")
                for i in sorted(images)}

    def _image_sizes(self, images: dict[int, str]) -> dict[int, tuple[float, float]]:
        from PIL import Image

        return {i: Image.open(self.root / "images" / p).size for i, p in images.items()}

    def part_spec(self) -> PartSpec:
        kp = torch.tensor([[g in groups for g in _KEYPOINT_GROUPS] for groups in self._concept_groups])
        seg = torch.tensor([[g in {_PART_TO_SEG[p] for p in groups} for g in _SEG_GROUPS]
                            for groups in self._concept_groups])
        return PartSpec(keypoint_groups=_KEYPOINT_GROUPS, seg_groups=_SEG_GROUPS if self.seg_root else [],
                        concept_keypoints=kp, concept_segs=seg if self.seg_root else seg[:, :0],
                        source=str(self.seg_root.resolve()) if self.seg_root else "")

    @staticmethod
    def _parts_for(attr_name: str) -> list[str]:
        # attribute names look like "has_bill_shape::dagger"
        if attr_name == "has_head_pattern::eyering":
            return ["left eye", "right eye"]
        stem = attr_name.split("::")[0].removeprefix("has_")
        for key in sorted(_ATTR_TO_PARTS, key=len, reverse=True):
            if stem.startswith(key):
                return _ATTR_TO_PARTS[key]
        return []

    def samples(self, split: str) -> list[Sample]:
        return self._splits[split]
