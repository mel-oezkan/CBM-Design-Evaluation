"""Segment instances.

* ``grounded_sam`` (SEG-MIL-CBM): each image's top-K concepts by teacher similarity, Grounding DINO
  boxes for them, and a SAM mask per box.
* ``grounded_boxes``: the same boxes without SAM (rectangular segments).
* ``sam_grid``: concept-agnostic SAM masks prompted from a point grid.

Segmentation runs in two cached steps. Masks are cached per (dataset, split, segmenter
settings, concept set), independent of the backbone, so backbone ablations reuse them. Features
of each masked segment are then cached per backbone/teacher by ``load_instances``.
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..registry import INSTANCES
from ..structures import InstanceSplit
from ..utils import stable_hash
from .base import InstanceSource, _concepts

log = logging.getLogger(__name__)

_MASK_RES = 64  # stored masks are cropped to their box and fit into this many pixels per side


@dataclass
class Segment:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    score: float
    concept: int = -1  # index into the concept set that proposed it
    mask: np.ndarray | None = None  # (H, W) bool; None means the whole box

    def full_mask(self, size: tuple[int, int]) -> np.ndarray:
        if self.mask is not None:
            return self.mask
        w, h = size
        x0, y0, x1, y1 = _int_box(self.box, size)
        m = np.zeros((h, w), dtype=bool)
        m[y0:y1, x0:x1] = True
        return m


def _int_box(box, size) -> tuple[int, int, int, int]:
    w, h = size
    x0, y0 = max(int(math.floor(box[0])), 0), max(int(math.floor(box[1])), 0)
    x1, y1 = min(int(math.ceil(box[2])), w), min(int(math.ceil(box[3])), h)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _bounds(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _resize_mask(mask: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(wh, Image.NEAREST)) > 127


def pack(seg: Segment, size: tuple[int, int]) -> dict[str, Any]:
    rec = {"box": list(seg.box), "score": float(seg.score), "concept": int(seg.concept)}
    if seg.mask is not None:
        x0, y0, x1, y1 = _int_box(seg.box, size)
        crop = seg.mask[y0:y1, x0:x1]
        scale = min(1.0, _MASK_RES / max(crop.shape))
        small = _resize_mask(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
        rec["mask"] = (np.packbits(small), small.shape)
    return rec


def unpack(rec: dict[str, Any], size: tuple[int, int]) -> Segment:
    mask = None
    if "mask" in rec:
        bits, shape = rec["mask"]
        small = np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool)
        x0, y0, x1, y1 = _int_box(rec["box"], size)
        mask = np.zeros((size[1], size[0]), dtype=bool)
        mask[y0:y1, x0:x1] = _resize_mask(small, (x1 - x0, y1 - y0))
    return Segment(tuple(rec["box"]), rec["score"], rec["concept"], mask)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


class SegmentSource(InstanceSource):
    """Shared post-processing and encoding for mask-based instance sources.

    ``encode: crop`` encodes each segment as its own image (the box plus ``margin``, with pixels
    outside the mask filled by the image's mean colour, or kept with ``background: keep``).
    ``encode: pool`` runs the encoder once per image and averages its patch tokens under each
    mask (much cheaper; the image is resized to a square so masks map onto the patch grid).
    """

    def __init__(self, max_segments: int = 16, min_pixels: int = 256, max_frac: float = 0.9,
                 merge_iou: float = 0.8, encode: str = "crop", margin: float = 0.1, background: str = "mean",
                 batch_size: int = 8, encode_batch_size: int = 128):
        if encode not in ("crop", "pool"):
            raise ValueError(f"Unknown segment encoding '{encode}' (use crop or pool)")
        if background not in ("mean", "keep"):
            raise ValueError(f"Unknown background '{background}' (use mean or keep)")
        self.max_segments, self.min_pixels, self.max_frac, self.merge_iou = max_segments, min_pixels, max_frac, merge_iou
        self.encode, self.margin, self.background = encode, margin, background
        self.batch_size, self.encode_batch_size = batch_size, encode_batch_size
        self.seg_kw: dict[str, Any] = {"max_segments": max_segments, "min_pixels": min_pixels,
                                       "max_frac": max_frac, "merge_iou": merge_iou}
        self.kw = {"encode": encode, "margin": margin, "background": background}

    # -- proposals ------------------------------------------------------------------------------

    def segment_key(self, ctx) -> dict[str, Any]:
        key = {"name": self.name, **self.seg_kw}
        if self.needs_concepts:
            key["concepts"] = stable_hash(_concepts(ctx).names)
        return key

    def cache_key(self, ctx):
        return {"segments": self.segment_key(ctx), **self.kw}

    def prepare(self, ctx, split: str) -> None:
        """Called once per split before ``propose`` (e.g. to rank concepts per image)."""

    @abstractmethod
    def propose(self, images: list[Image.Image], index: list[int], ctx) -> list[list[Segment]]:
        """Raw segments for a batch of images; ``index`` are their positions in the split."""

    def postprocess(self, segs: list[Segment], size: tuple[int, int]) -> list[Segment]:
        """Drop tiny / near-full segments, merge (union) overlapping ones, keep the best ``max_segments``.
        An image left without segments gets one whole-image segment."""
        w, h = size
        kept: list[tuple[Segment, np.ndarray]] = []
        for s in sorted(segs, key=lambda s: -s.score):
            m = s.full_mask(size)
            area = int(m.sum())
            if area == 0 or area < self.min_pixels or area / (w * h) > self.max_frac:
                continue
            for i, (k, km) in enumerate(kept):
                if _iou(m, km) > self.merge_iou:
                    kept[i] = (k, km | m)
                    break
            else:
                kept.append((s, m))
        out = [Segment(_bounds(m), s.score, s.concept, m) for s, m in kept[: self.max_segments]]
        return out or [Segment((0.0, 0.0, float(w), float(h)), 0.0, -1, None)]

    def _segments(self, ctx, split: str, paths: list[str]) -> list[list[dict[str, Any]]]:
        key = {"dataset": ctx.dataset.cache_key(), "split": split, "segmenter": self.segment_key(ctx)}
        path = ctx.cache_dir / "segments" / ctx.dataset.name / f"{split}-{stable_hash(key)}.pt"
        if path.exists():
            return torch.load(path, weights_only=False)
        self.prepare(ctx, split)
        records: list[list[dict[str, Any]]] = []
        for start in range(0, len(paths), self.batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start : start + self.batch_size]]
            proposals = self.propose(images, list(range(start, start + len(images))), ctx)
            for image, segs in zip(images, proposals):
                records.append([pack(s, image.size) for s in self.postprocess(segs, image.size)])
            if (start // self.batch_size) % 50 == 0:
                log.info("%s %s: segmented %d/%d images", self.name, split, len(records), len(paths))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(records, path)
        return records

    # -- encoding -------------------------------------------------------------------------------

    def build(self, ctx, split):
        paths = [s.image for s in ctx.dataset.samples(split)]
        if not paths or not isinstance(paths[0], str):
            raise RuntimeError(f"`instances: {self.name}` needs a dataset with image files")
        records = self._segments(ctx, split, paths)
        teacher = ctx.teacher if ctx.teacher is not None and ctx.teacher is not ctx.backbone else None
        encoders = [ctx.backbone] + ([teacher] if teacher is not None else [])
        for enc in encoders:
            if enc.transform() is None:
                raise RuntimeError(f"{type(enc).__name__} cannot encode image files")
        grid = ctx.backbone.patch_grid
        feats: list[list[torch.Tensor]] = [[] for _ in encoders]
        areas, regions, concepts = [], [], []
        for start in range(0, len(paths), self.batch_size):
            images, masks = [], []
            for p, recs in zip(paths[start : start + self.batch_size], records[start : start + self.batch_size]):
                image = Image.open(p).convert("RGB")
                segs = [unpack(r, image.size) for r in recs]
                ms = np.stack([s.full_mask(image.size) for s in segs])
                images.append(image)
                masks.append(ms)
                areas.append(torch.from_numpy(ms.mean((1, 2))).float())
                concepts.append(torch.tensor([s.concept for s in segs]))
                if grid:
                    regions.append(_coverage(ms, grid) > 0)
            for e, enc in enumerate(encoders):
                feats[e] += self._encode_crops(enc, images, masks) if self.encode == "crop" \
                    else _encode_pooled(enc, images, masks)
            if (start // self.batch_size) % 50 == 0:
                log.info("%s %s: encoded %d/%d images", self.name, split, len(areas), len(paths))
        mask = _pad([torch.ones(len(a), dtype=torch.bool) for a in areas], False)
        return InstanceSplit(
            features=_pad(feats[0], 0.0), mask=mask, area=_pad(areas, 0.0),
            teacher_features=_pad(feats[1], 0.0) if teacher is not None else None,
            regions=_pad(regions, False) if grid else None, concepts=_pad(concepts, -1))

    def _encode_crops(self, enc, images, masks) -> list[torch.Tensor]:
        crops = [self._crop(im, m) for im, ms in zip(images, masks) for m in ms]
        tf = enc.transform()
        out = []
        for i in range(0, len(crops), self.encode_batch_size):
            batch = torch.stack([tf(c) for c in crops[i : i + self.encode_batch_size]])
            out.append(enc.encode_images(batch).float().cpu())
        return list(torch.cat(out).split([len(ms) for ms in masks]))

    def _crop(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        x0, y0, x1, y1 = _bounds(mask)
        mx, my = self.margin * (x1 - x0), self.margin * (y1 - y0)
        x0, y0, x1, y1 = _int_box((x0 - mx, y0 - my, x1 + mx, y1 + my), image.size)
        arr = np.asarray(image)
        crop = arr[y0:y1, x0:x1].copy()
        if self.background == "mean":
            crop[~mask[y0:y1, x0:x1]] = arr.reshape(-1, 3).mean(0).astype(np.uint8)
        return Image.fromarray(crop)


def _coverage(masks: np.ndarray, grid: int) -> torch.Tensor:
    """(M, H, W) bool -> (M, grid*grid) share of each patch-grid cell covered by each mask."""
    m = torch.from_numpy(masks).float()[:, None]
    return F.adaptive_avg_pool2d(m, grid).flatten(1)


def _encode_pooled(enc, images, masks) -> list[torch.Tensor]:
    if not enc.patch_grid:
        raise RuntimeError(f"{type(enc).__name__} gives no patch tokens for `encode: pool`")
    tf = enc.transform()
    batch = torch.stack([tf(im.resize((448, 448))) for im in images])
    tokens = enc.encode_patches(batch).float().cpu()  # (B, P, D)
    out = []
    for tok, ms in zip(tokens, masks):
        w = _coverage(ms, enc.patch_grid)
        out.append((w / w.sum(1, keepdim=True).clamp_min(1e-8)) @ tok)
    return out


def _pad(rows: list[torch.Tensor], value) -> torch.Tensor:
    m = max(len(r) for r in rows)
    out = torch.full((len(rows), m, *rows[0].shape[1:]), value, dtype=rows[0].dtype)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


@INSTANCES.register("grounded_sam")
class GroundedSAM(SegmentSource):
    """SEG-MIL-CBM segments: the image's ``top_k`` concepts by teacher image-text similarity (all
    concepts when ``top_k`` is null), up to ``boxes_per_concept`` Grounding DINO boxes for each,
    and a SAM mask per box. ``detector``/``sam`` take model settings; tests pass stub objects."""

    name = "grounded_sam"
    needs_concepts = True
    use_sam = True

    def __init__(self, top_k: int | None = 10, boxes_per_concept: int = 2, detector: Any = None, sam: Any = None,
                 **kw):
        super().__init__(**kw)
        self.top_k, self.boxes_per_concept = top_k, boxes_per_concept
        self.detector = detector if hasattr(detector, "detect") else None
        self.sam = sam if hasattr(sam, "from_boxes") else None
        self.detector_kw = detector if isinstance(detector, dict) else {}
        self.sam_kw = sam if isinstance(sam, dict) else {}
        self.seg_kw |= {"top_k": top_k, "boxes_per_concept": boxes_per_concept, "detector": self.detector_kw}
        if self.use_sam:
            self.seg_kw["sam"] = self.sam_kw
        self._prompts: list[list[int]] = []

    def segment_key(self, ctx):
        key = super().segment_key(ctx)
        if self.top_k is not None and ctx.teacher is not None:  # the teacher ranks concepts per image
            key["teacher"] = ctx.teacher.cache_key()
        return key

    def prepare(self, ctx, split):
        concepts = _concepts(ctx)
        n = len(ctx.dataset.samples(split))
        if self.top_k is None or self.top_k >= len(concepts):
            self._prompts = [list(range(len(concepts)))] * n
            return
        img = F.normalize(ctx.teacher_split(split).features.float(), dim=1)
        txt = F.normalize(ctx.encode_text(concepts.names).float(), dim=1)
        self._prompts = (img @ txt.T).topk(self.top_k, dim=1).indices.tolist()

    def propose(self, images, index, ctx):
        from ..grounding import GroundingDINODetector, SAMSegmenter

        if self.detector is None:
            self.detector = GroundingDINODetector(device=ctx.device, **self.detector_kw)
        if self.use_sam and self.sam is None:
            self.sam = SAMSegmenter(device=ctx.device, **self.sam_kw)
        names = ctx.concepts.names
        prompts = [self._prompts[i] for i in index]
        detections = self.detector.detect(images, [[names[k] for k in ks] for ks in prompts])
        out = []
        for image, ks, dets in zip(images, prompts, detections):
            per_concept: dict[int, list] = defaultdict(list)
            for d in sorted(dets, key=lambda d: -d.score):
                if d.label >= 0 and len(per_concept[d.label]) < self.boxes_per_concept:
                    per_concept[d.label].append(d)
            chosen = [d for ds in per_concept.values() for d in ds]
            masks = self.sam.from_boxes(image, [d.box for d in chosen]) if self.use_sam else [None] * len(chosen)
            out.append([Segment(d.box, d.score, ks[d.label], m) for d, m in zip(chosen, masks)])
        return out


@INSTANCES.register("grounded_boxes")
class GroundedBoxes(GroundedSAM):
    """Grounding DINO boxes as rectangular segments: is SAM's mask worth its cost?"""

    name = "grounded_boxes"
    use_sam = False


@INSTANCES.register("sam_grid")
class SAMGrid(SegmentSource):
    """Concept-agnostic SAM masks from a ``points_per_side`` x ``points_per_side`` grid of point
    prompts, kept when SAM's predicted IoU is at least ``min_iou``. Does not depend on the
    concept set, so one segmentation serves every discovery/filtering ablation."""

    name = "sam_grid"

    def __init__(self, points_per_side: int = 8, min_iou: float = 0.85, sam: Any = None, **kw):
        super().__init__(**kw)
        self.points_per_side, self.min_iou = points_per_side, min_iou
        self.sam = sam if hasattr(sam, "from_points") else None
        self.sam_kw = sam if isinstance(sam, dict) else {}
        self.seg_kw |= {"points_per_side": points_per_side, "min_iou": min_iou, "sam": self.sam_kw}

    def propose(self, images, index, ctx):
        from ..grounding import SAMSegmenter

        if self.sam is None:
            self.sam = SAMSegmenter(device=ctx.device, **self.sam_kw)
        out = []
        for image in images:
            w, h = image.size
            n = self.points_per_side
            points = [((i + 0.5) * w / n, (j + 0.5) * h / n) for j in range(n) for i in range(n)]
            masks, scores = self.sam.from_points(image, points)
            out.append([Segment(_bounds(m), s, -1, m) for m, s in zip(masks, scores) if s >= self.min_iou and m.any()])
        return out
