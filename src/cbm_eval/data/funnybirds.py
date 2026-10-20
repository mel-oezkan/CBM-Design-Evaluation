"""FunnyBirds (Hesse et al., "FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of
Explainable AI Methods", ICCV 2023; github.com/visinf/funnybirds): rendered 3D birds whose classes
are combinations of part variants, with a color-coded part map per image.

A class picks one variant of each of five parts (beak, eye, foot, tail, wing; ``parts.json``). The
concepts are these part variants (26 with the authors' ``parts.json``), named from the variant's
color and model, e.g. "red tail shape 2". Concept labels are per image: the authors drop a random
subset of parts from half of the train scenes (so part interventions stay in-domain), and a dropped
part has no present variant. ``class_level_concepts: true`` labels every image with its class
definition instead.

The part map colors each part instance (two eyes, feet and wings); a part's mask is the union of
its instances and its keypoints are their centroids. Background objects are no part. Part maps
must have the image's resolution (both are 256x256 in the authors' render).

Expected: a FunnyBirds directory, either the authors' render
(download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip: 50 classes, 50,000 train and 500
test images at 256x256) or one made with ``cbm_eval.data.funnybirds_render``::

    classes.json  parts.json  dataset_train.json  dataset_test.json  [meta.json]
    train/<class>/<index:06d>.png   train_part_map/<class>/<index:06d>.png   test/...   test_part_map/...

``<index>`` is the scene's position in its ``dataset_<split>.json``. Validation is a deterministic
``val_fraction`` of train. Scenes may carry a spurious attribute (``attr``, e.g. the background
color); ``meta.json`` then sets ``n_attrs``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..registry import DATASETS
from .base import ImageDataset, PartAnnotation, PartSpec, Sample

# Part instance -> its color in the part maps (from the authors' ``datasets/funny_birds.py``).
PART_COLORS = {"eye01": (255, 255, 253), "eye02": (255, 255, 254), "beak": (255, 255, 0), "foot01": (255, 0, 1),
               "foot02": (255, 0, 2), "wing01": (0, 255, 1), "wing02": (0, 255, 2), "tail": (0, 0, 255)}


def concept_name(part: str, variant: dict[str, str]) -> str:
    """Readable name of a part variant, e.g. ``{"model": "tail02.glb", "color": "red"}`` -> "red tail shape 2"."""
    shape = re.sub(r"\D", "", Path(variant["model"]).stem).lstrip("0") or "0"
    return " ".join(filter(None, [variant.get("color"), part, "shape", shape]))


@DATASETS.register("funnybirds")
class FunnyBirds(ImageDataset):
    """FunnyBirds birds with one concept per part variant and part masks / keypoints from the part maps."""

    name = "funnybirds"

    def __init__(self, root: str = "data/FunnyBirds", val_fraction: float = 0.1, class_level_concepts: bool = False):
        self.root = Path(root)
        if not (self.root / "classes.json").exists():
            raise FileNotFoundError(f"FunnyBirds not found at {self.root} (no classes.json); set dataset.root")
        self.val_fraction, self.class_level_concepts = val_fraction, class_level_concepts
        self.classes = sorted(json.loads((self.root / "classes.json").read_text()), key=lambda c: c["class_idx"])
        if [c["class_idx"] for c in self.classes] != list(range(len(self.classes))):
            raise ValueError(f"classes.json in {self.root} must number its classes 0..C-1")
        self.parts = json.loads((self.root / "parts.json").read_text())
        meta = self.root / "meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        self.n_attrs = int(self.meta.get("n_attrs", 1))
        self.part_names = list(self.parts)
        self._variants = [(p, v) for p in self.part_names for v in range(len(self.parts[p]))]
        self._concepts = [concept_name(p, self.parts[p][v]) for p, v in self._variants]
        if len(set(self._concepts)) != len(self._concepts):
            raise ValueError(f"parts.json in {self.root} gives duplicate concept names: {self._concepts}")
        self._groups = {p: [i for i in PART_COLORS if i.rstrip("0123456789") == p] for p in self.part_names}
        self._splits: dict[str, list[Sample]] = {}

    def cache_key(self):
        digest = hashlib.sha1()
        for f in ("classes.json", "parts.json", "dataset_train.json", "dataset_test.json", "meta.json"):
            if (self.root / f).exists():
                digest.update((self.root / f).read_bytes())
        return {"name": self.name, "root": str(self.root.resolve()), "val_fraction": self.val_fraction,
                "class_level_concepts": self.class_level_concepts, "scenes": digest.hexdigest()}

    @property
    def class_names(self) -> list[str]:
        return [f"funnybird {c['class_idx']:02d}" for c in self.classes]

    @property
    def concept_names(self) -> list[str]:
        return self._concepts

    def class_concepts(self) -> torch.Tensor | None:
        """(C, K) class definitions; None with image-level concepts."""
        return self._class_matrix() if self.class_level_concepts else None

    def _class_matrix(self) -> torch.Tensor:
        return torch.tensor([[float(c["parts"][p] == v) for p, v in self._variants] for c in self.classes])

    def part_spec(self) -> PartSpec | None:
        if not any((self.root / f"{s}_part_map").is_dir() for s in ("train", "test")):
            return None
        member = torch.tensor([[p == g for g in self.part_names] for p, _ in self._variants])
        colors = {p: [PART_COLORS[i] for i in insts] for p, insts in self._groups.items()}
        return PartSpec(keypoint_groups=self.part_names, seg_groups=self.part_names, concept_keypoints=member,
                        concept_segs=member, max_points=max(len(i) for i in self._groups.values()),
                        source="funnybirds part maps", seg_colors=colors)

    def samples(self, split: str) -> list[Sample]:
        if split not in self._splits:
            self._load("test" if split == "test" else "train")
        return self._splits[split]

    def _load(self, source: str) -> None:
        """Fill ``train`` and ``val`` from ``dataset_train.json``, or ``test`` from ``dataset_test.json``."""
        scenes = json.loads((self.root / f"dataset_{source}.json").read_text())
        has_maps = (self.root / f"{source}_part_map").is_dir()
        with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
            samples = list(pool.map(lambda a: self._sample(source, *a, has_maps), enumerate(scenes)))
        if source == "test":
            self._splits["test"] = samples
            return
        rng = np.random.default_rng(0)  # one draw per train scene, in order
        is_val = rng.random(len(samples)) < self.val_fraction
        self._splits["train"] = [s for s, v in zip(samples, is_val) if not v]
        self._splits["val"] = [s for s, v in zip(samples, is_val) if v]

    def _sample(self, source: str, i: int, scene: dict, has_maps: bool) -> Sample:
        c = scene["class_idx"]
        definition = self.classes[c]["parts"]
        present = {p: self.class_level_concepts or scene.get(f"{p}_model", "placeholder") != "placeholder"
                   for p in self.part_names}
        concepts = [int(present[p] and definition[p] == v) for p, v in self._variants]
        rel = Path(str(c), f"{i:06d}.png")
        parts = keypoints = None
        if has_maps:
            parts, keypoints = self._annotate(self.root / f"{source}_part_map" / rel, concepts)
        return Sample(image=str(self.root / source / rel), label=c, attr=int(scene.get("attr", 0)),
                      concepts=concepts, keypoints=keypoints, parts=parts)

    def _annotate(self, part_map: Path, concepts: list[int]):
        """Part keypoints (instance centroids) and concept keypoints from one part map."""
        rgb = np.asarray(Image.open(part_map).convert("RGB")).astype(np.int32)
        h, w = rgb.shape[:2]
        code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        points: dict[str, list[tuple[float, float]]] = {}
        for p, insts in self._groups.items():
            for inst in insts:
                r, g, b = PART_COLORS[inst]
                ys, xs = np.nonzero(code == (r << 16) | (g << 8) | b)
                if len(xs):
                    points.setdefault(p, []).append((float(xs.mean()) + 0.5, float(ys.mean()) + 0.5))
        keypoints = {k: (*points[p][0], w, h) for k, (p, _) in enumerate(self._variants)
                     if concepts[k] and p in points}
        return PartAnnotation(size=(w, h), points=points, label_map=str(part_map)), keypoints
