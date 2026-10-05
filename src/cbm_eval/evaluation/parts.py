"""Part-level concept localization against CUB keypoints and part segmentations.

Re-implements the two localization metrics of ProtoCBM (github.com/pascal0012/ProtoCBM,
``localization/``) on this codebase's concept maps: the concept layer applied to patch features,
the per-patch concept scores of a patch bag, or (for segment bags) each patch's best covering
segment.

Per image and part group, the concepts that belong to the group are scored with
``select="argmax"`` (the one with the highest image-level score, as in ProtoCBM) or
``select="present"`` (every concept annotated present, averaged). Only concepts that match a
dataset concept by name are used.

* ``keypoint_distance``: distance from the map's peak (the peak patch's center) to the nearest
  visible keypoint of the group (left / right merged), in units of the image side. ``dist`` is the
  mean of the per-group means; ``pck`` is the share of (image, group) pairs within ``threshold``;
  ``dist_center`` is the same distance for a map that always points at the image center.
* ``part_iou``: IoU between the map and the group's segmentation mask. ``hard`` keeps the top
  ``keep_ratio`` of patches; soft IoU uses the min-max normalized map. ``miou`` averages over all
  (image, group) pairs, ``iou.<group>`` within a group; ``miou_center`` uses a center-prior map.
  Needs the CUB70 masks (see ``data/cub.py``).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F

from ..registry import EVALUATION
from ..stages.common import match_names
from .base import Evaluator


class _PartEvaluator(Evaluator):
    needs_patches = True
    seg_size: int = 56

    def __init__(self, split: str = "test", select: str = "argmax", batch_size: int = 64):
        if select not in ("argmax", "present"):
            raise ValueError(f"select must be 'argmax' or 'present', got {select!r}")
        self.split, self.select, self.batch_size = split, select, batch_size

    def evaluate(self, cbm, ctx):
        idx = match_names(cbm.concepts.names, ctx.dataset.concept_names)
        ks = [k for k, j in enumerate(idx) if j is not None]
        js = [idx[k] for k in ks]
        if not ks:
            return {}
        segments = ctx.instance_source is not None and not ctx.instances(self.split).grid_aligned
        geometry = "full" if segments else "input"  # segment regions cover the whole, uncropped image
        parts = ctx.parts(self.split, geometry, self.seg_size)
        if parts is None:
            return {}
        fs = ctx.split(self.split)
        scores = cbm.concept_scores.get(self.split)
        if scores is None:
            scores = cbm.image_concepts(*ctx.inputs(self.split))
        scores = scores[:, ks].float()
        present = fs.concept_labels[:, js].bool() if fs.concept_labels is not None else None
        if self.select == "present" and present is None:
            return {}
        return self.score(parts, js, scores, present, self._maps(cbm, ctx, ks))

    def score(self, parts, js, scores, present, maps) -> dict[str, float]: ...

    def weights(self, members: torch.Tensor, scores: torch.Tensor, present: torch.Tensor | None) -> torch.Tensor:
        """(B, G, K') bool: which concepts score each (image, group). ``members`` is (K', G)."""
        cand = members.T[None].expand(len(scores), -1, -1)  # (B, G, K')
        if self.select == "present":
            return cand & present[:, None, :]
        top = scores[:, None, :].masked_fill(~cand, -math.inf).argmax(-1, keepdim=True)
        return torch.zeros_like(cand).scatter(-1, top, True) & cand

    def _maps(self, cbm, ctx, ks) -> Iterator[tuple[slice, torch.Tensor]]:
        """(slice, (B, P, K') concept maps over the patch grid) in batches of images."""
        if ctx.instance_source is None:
            fs = ctx.split(self.split)
            if fs.patch_features is None:
                return
            layer = cbm.model.layer.eval()
            for start in range(0, len(fs), self.batch_size):
                sl = slice(start, start + self.batch_size)
                with torch.no_grad():
                    yield sl, layer.concept_logits(fs.patch_features[sl].float())[..., ks]
            return
        inst = ctx.instances(self.split)
        if inst.regions is None and not inst.grid_aligned:
            return
        for start in range(0, len(inst), self.batch_size):
            sl = slice(start, start + self.batch_size)
            c_hat = cbm.predict(inst.features[sl], bag=inst.bag[sl])[1][..., ks].float()  # (B, M, K')
            lo = c_hat.masked_fill(~inst.mask[sl][..., None], math.inf).amin(1, keepdim=True)  # (B, 1, K')
            c_hat = torch.where(inst.mask[sl][..., None], c_hat, lo)
            if inst.grid_aligned:
                yield sl, c_hat
                continue
            # each patch takes the best score among the segments covering it (the image minimum if none)
            cover = (inst.regions[sl] & inst.mask[sl][..., None])[..., None]  # (B, M, P, 1)
            yield sl, torch.where(cover, c_hat[:, :, None], -math.inf).amax(1).maximum(lo)


def _grid(n_patches: int) -> int:
    g = math.isqrt(n_patches)
    if g * g != n_patches:
        raise ValueError(f"Expected a square patch grid, got {n_patches} patches")
    return g


def _centers(g: int) -> torch.Tensor:
    """(g*g, 2) normalized (x, y) of each patch center, row-major."""
    c = (torch.arange(g) + 0.5) / g
    y, x = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([x.flatten(), y.flatten()], -1)


def _nan_to_inf(x: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(x.isnan(), math.inf)


def _group_means(values: torch.Tensor, valid: torch.Tensor, names: list[str], prefix: str) -> dict[str, float]:
    """Per-group mean of (N, G) values over valid entries; groups without any are left out."""
    out = {}
    for g, name in enumerate(names):
        if valid[:, g].any():
            out[f"{prefix}.{name}"] = float(values[:, g][valid[:, g]].mean())
    return out


@EVALUATION.register("keypoint_distance")
class KeypointDistanceEvaluator(_PartEvaluator):
    def __init__(self, split: str = "test", select: str = "argmax", threshold: float = 0.1,
                 batch_size: int = 64):
        super().__init__(split, select, batch_size)
        self.threshold = threshold

    def score(self, parts, js, scores, present, maps):
        members = parts.concept_keypoints[js]  # (K', Q)
        dist, center, valid = [], [], []
        for sl, m in maps:
            peaks = _centers(_grid(m.shape[1]))[m.argmax(1)]  # (B, K', 2)
            pts = parts.points[sl]  # (B, Q, R, 2)
            d = (peaks[:, None, None] - pts[:, :, :, None]).norm(dim=-1)  # (B, Q, R, K')
            d = _nan_to_inf(d).amin(2)  # nearest visible keypoint of the group
            w = self.weights(members, scores[sl], None if present is None else present[sl])  # (B, Q, K')
            ok = w & d.isfinite()
            n = ok.sum(-1)
            dist.append((d.where(ok, 0).sum(-1) / n.clamp_min(1)))
            center.append(_nan_to_inf((pts - 0.5).norm(dim=-1)).amin(-1))
            valid.append(n > 0)
        if not valid:
            return {}
        dist, center, valid = torch.cat(dist), torch.cat(center), torch.cat(valid)
        if not valid.any():
            return {}
        per_group = _group_means(dist, valid, parts.keypoint_groups, "dist")
        center_groups = _group_means(center, valid, parts.keypoint_groups, "center")
        return {"dist": sum(per_group.values()) / len(per_group),
                "dist_center": sum(center_groups.values()) / len(center_groups),
                "pck": float((dist[valid] <= self.threshold).float().mean()),
                "pck_center": float((center[valid] <= self.threshold).float().mean()),
                "n_pairs": float(valid.sum()), **per_group}


@EVALUATION.register("part_iou")
class PartIoUEvaluator(_PartEvaluator):
    def __init__(self, split: str = "test", select: str = "argmax", hard: bool = True, keep_ratio: float = 0.5,
                 seg_size: int = 56, batch_size: int = 64):
        super().__init__(split, select, batch_size)
        self.hard, self.keep_ratio, self.seg_size = hard, keep_ratio, seg_size

    def _masks(self, m: torch.Tensor) -> torch.Tensor:
        """(B, P, K') maps -> (B, K', S, S) binary (hard) or [0, 1] (soft) masks."""
        g = _grid(m.shape[1])
        m = m.transpose(1, 2)  # (B, K', P)
        if self.hard:  # keep the patches above the (1 - keep_ratio) quantile, as in ProtoCBM
            k = min(max(math.ceil((1 - self.keep_ratio) * m.shape[-1]), 1), m.shape[-1])
            m = (m > m.kthvalue(k, dim=-1, keepdim=True).values).float()
            mode = "nearest"
        else:
            lo, hi = m.amin(-1, keepdim=True), m.amax(-1, keepdim=True)
            m = (m - lo) / (hi - lo).clamp_min(1e-8)
            mode = "bilinear"
        m = m.unflatten(-1, (g, g))
        kw = {} if mode == "nearest" else {"align_corners": False}
        return F.interpolate(m, size=(self.seg_size, self.seg_size), mode=mode, **kw)

    @staticmethod
    def _iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """(B, K', S, S) x (B, G, S, S) -> (B, G, K') (soft) IoU."""
        inter = torch.einsum("bkhw,bghw->bgk", pred, gt)
        union = pred.sum((2, 3))[:, None, :] + gt.sum((2, 3))[:, :, None] - inter
        return inter / union.clamp_min(1e-8)

    def score(self, parts, js, scores, present, maps):
        if parts.segs is None or not parts.seg_groups:
            return {}
        members = parts.concept_segs[js]  # (K', G)
        iou, center, valid = [], [], []
        center_map = None
        for sl, m in maps:
            gt = parts.segs[sl].float()  # (B, G, S, S)
            has = gt.flatten(2).any(-1)  # (B, G)
            w = self.weights(members, scores[sl], None if present is None else present[sl]) & has[..., None]
            n = w.sum(-1)
            iou.append(self._iou(self._masks(m), gt).where(w, 0).sum(-1) / n.clamp_min(1))
            if center_map is None:
                c = -(_centers(_grid(m.shape[1])) - 0.5).norm(dim=-1)  # (P,) center prior
                center_map = self._masks(c[None, :, None])  # (1, 1, S, S)
            center.append(self._iou(center_map.expand(len(gt), -1, -1, -1), gt)[..., 0])
            valid.append(n > 0)
        if not valid:
            return {}
        iou, center, valid = torch.cat(iou), torch.cat(center), torch.cat(valid)
        if not valid.any():
            return {}
        return {"miou": float(iou[valid].mean()), "miou_center": float(center[valid].mean()),
                "n_pairs": float(valid.sum()), **_group_means(iou, valid, parts.seg_groups, "iou")}
