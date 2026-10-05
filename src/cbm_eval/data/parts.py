"""Part annotations (keypoints and segmentation masks) mapped into the frame the concept maps live in.

Concept maps from patch features cover the backbone *input* (after its resize and center crop),
so annotations are pushed through the same geometric ops. Segment instance regions are computed on
the whole image squashed to the patch grid; for those, ``geometry="full"`` skips the ops.

Keypoints come out in normalized ``[0, 1]`` coordinates of that frame (NaN when invisible or
cropped away); masks as ``(G, S, S)`` bool, area-downsampled to ``seg_size``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..structures import PartSplit
from ..utils import stable_hash
from .base import ImageDataset

Op = tuple[str, tuple[int, ...]]


def input_geometry(transform) -> list[Op]:
    """The resize / center-crop ops of a torchvision ``Compose``, in order. Anything else is ignored."""
    from torchvision import transforms as T

    ops: list[Op] = []
    for t in getattr(transform, "transforms", []):
        if isinstance(t, T.Resize):
            ops.append(("resize", tuple(t.size) if isinstance(t.size, (list, tuple)) else (t.size,)))
        elif isinstance(t, T.CenterCrop):
            ops.append(("crop", tuple(t.size)))
        elif "resize" in type(t).__name__.lower() or "crop" in type(t).__name__.lower():
            raise ValueError(f"Cannot map part annotations through {type(t).__name__}")
    return ops


def _resize_shape(size: tuple[float, float], arg: tuple[int, ...]) -> tuple[float, float]:
    """(w, h) after ``T.Resize(arg)``: an int scales the shorter side, a pair is (h, w)."""
    w, h = size
    if len(arg) == 2:
        return float(arg[1]), float(arg[0])
    s = arg[0] / min(w, h)
    return (float(arg[0]), h * s) if w <= h else (w * s, float(arg[0]))


def _crop_shape(arg: tuple[int, ...]) -> tuple[float, float]:
    return (float(arg[0]), float(arg[0])) if len(arg) == 1 else (float(arg[1]), float(arg[0]))


def map_points(points: np.ndarray, size: tuple[float, float], ops: list[Op]) -> np.ndarray:
    """(Q, 2) pixel (x, y) in an image of ``size`` (w, h) -> normalized coords in the output frame.
    Points falling outside a crop become NaN."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    w, h = size
    for kind, arg in ops:
        if kind == "resize":
            nw, nh = _resize_shape((w, h), arg)
            pts *= (nw / w, nh / h)
            w, h = nw, nh
        else:
            cw, ch = _crop_shape(arg)
            pts -= ((w - cw) / 2, (h - ch) / 2)
            w, h = cw, ch
    pts /= (w, h)
    pts[((pts < 0) | (pts > 1)).any(1)] = np.nan
    return pts


def map_mask(mask: np.ndarray, size: tuple[int, int], ops: list[Op], seg_size: int) -> torch.Tensor:
    """(H', W') bool mask (any resolution with the image's aspect) -> (seg_size, seg_size) bool."""
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    img = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    for kind, arg in ops:
        img = T.Resize(arg if len(arg) == 2 else arg[0], interpolation=InterpolationMode.NEAREST)(img) \
            if kind == "resize" else T.CenterCrop(arg)(img)
    t = torch.from_numpy(np.asarray(img) > 127).float()[None, None]
    return F.interpolate(t, size=(seg_size, seg_size), mode="area")[0, 0] >= 0.5


def _load_mask(paths: list[str]) -> np.ndarray | None:
    """Union of the mask files that exist (e.g. left and right wing), or None."""
    out = None
    for p in paths:
        if Path(p).exists():
            m = np.asarray(Image.open(p).convert("L")) > 127
            out = m if out is None else (out | m)
    return out


def load_parts(dataset: ImageDataset, split: str, cache_dir: str | Path, ops: list[Op],
               seg_size: int = 56) -> PartSplit | None:
    """Keypoints and (if the dataset has them) segmentation masks for a split, cached on disk."""
    spec = dataset.part_spec()
    if spec is None:
        return None
    key = {"dataset": dataset.cache_key(), "spec": spec.cache_key(), "split": split, "ops": ops,
           "seg_size": seg_size if spec.seg_groups else None}
    path = Path(cache_dir) / "parts" / dataset.name / f"{split}-{stable_hash(key)}.pt"
    if path.exists():
        return PartSplit(**torch.load(path, weights_only=False))

    samples = dataset.samples(split)
    Q, G = len(spec.keypoint_groups), len(spec.seg_groups)
    points = torch.full((len(samples), Q, spec.max_points, 2), math.nan)
    segs = torch.zeros(len(samples), G, seg_size, seg_size, dtype=torch.bool) if G else None
    for n, s in enumerate(samples):
        ann = s.parts
        if ann is None:
            continue
        for q, group in enumerate(spec.keypoint_groups):
            pts = ann.points.get(group)
            if pts:
                points[n, q, : len(pts)] = torch.from_numpy(map_points(np.array(pts), ann.size, ops)).float()
        for g, group in enumerate(spec.seg_groups):
            m = _load_mask(ann.masks.get(group, []))
            if m is not None and m.any():
                segs[n, g] = map_mask(m, ann.size, ops, seg_size)
    data = dict(points=points, segs=segs, keypoint_groups=list(spec.keypoint_groups),
                seg_groups=list(spec.seg_groups), concept_keypoints=spec.concept_keypoints,
                concept_segs=spec.concept_segs)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return PartSplit(**data)
