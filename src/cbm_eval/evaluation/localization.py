"""Concept localization: do concept activation maps fire where the concept actually is?

Maps are the concept layer applied to patch features. Only concepts that match an annotated
dataset concept (by name) and are present in an image are scored.

* Pointing game: the map's peak patch falls inside the annotated region.
* Locality: share of the map's (min-shifted) mass inside the region; ``locality_chance`` is the
  region's share of the image, so locality above chance means the map concentrates on the part.

With instance bags the instances themselves are the map: the pointing game asks whether the
instance scoring highest for a concept covers the part, and ``pointing_chance`` is the share of
the image's instances that do.
"""

from __future__ import annotations

import torch

from ..registry import EVALUATION
from ..stages.common import match_names
from .base import Evaluator


@EVALUATION.register("localization")
class LocalizationEvaluator(Evaluator):
    needs_patches = True

    def __init__(self, split: str = "test", batch_size: int = 256):
        self.split, self.batch_size = split, batch_size

    def evaluate(self, cbm, ctx):
        fs = ctx.split(self.split)
        if fs.part_masks is None:
            return {}
        idx = match_names(cbm.concepts.names, ctx.dataset.concept_names)
        pairs = [(k, j) for k, j in enumerate(idx) if j is not None]
        if not pairs:
            return {}
        ks, js = [k for k, _ in pairs], [j for _, j in pairs]
        if ctx.instance_source is not None:
            return self._instances(cbm, ctx, fs, ks, js)
        if fs.patch_features is None:
            return {}
        hits = mass = chance = 0.0
        n = 0
        layer = cbm.model.layer.eval()
        for start in range(0, len(fs), self.batch_size):
            sl = slice(start, start + self.batch_size)
            with torch.no_grad():
                maps = layer.concept_logits(fs.patch_features[sl].float())[..., ks]  # (B, P, K')
            masks = fs.part_masks[sl][:, js, :].transpose(1, 2).float()  # (B, P, K')
            valid = masks.sum(1) > 0  # (B, K')
            if not valid.any():
                continue
            peak = maps.argmax(1, keepdim=True)
            hit = masks.gather(1, peak).squeeze(1)
            shifted = maps - maps.min(1, keepdim=True).values
            share = (shifted * masks).sum(1) / shifted.sum(1).clamp_min(1e-8)
            area = masks.mean(1)
            hits += float(hit[valid].sum())
            mass += float(share[valid].sum())
            chance += float(area[valid].sum())
            n += int(valid.sum())
        if n == 0:
            return {}
        return {"pointing_game": hits / n, "locality": mass / n, "locality_chance": chance / n,
                "n_pairs": float(n), "n_concepts_scored": float(len(pairs))}

    def _instances(self, cbm, ctx, fs, ks, js):
        inst = ctx.instances(self.split)
        if inst.regions is None and not inst.grid_aligned:
            return {}
        c_hat = cbm.predict(inst.features, bag=inst.bag)[1][..., ks]  # (N, M, K')
        parts = fs.part_masks[:, js, :]  # (N, K', P)
        if inst.grid_aligned:  # instance i is patch i
            covers = parts.transpose(1, 2)  # (N, M=P, K')
        else:  # (N, M, P) x (N, K', P) -> does instance m overlap part k
            covers = torch.einsum("nmp,nkp->nmk", inst.regions.float(), parts.float()) > 0
        covers = covers & inst.mask[..., None]
        valid = parts.any(-1)  # (N, K') part annotated as present
        if not valid.any():
            return {}
        top = c_hat.masked_fill(~inst.mask[..., None], float("-inf")).argmax(1, keepdim=True)
        hit = covers.gather(1, top).squeeze(1).float()
        chance = covers.float().sum(1) / inst.mask.sum(1, keepdim=True).clamp_min(1)
        n = int(valid.sum())
        return {"pointing_game": float(hit[valid].sum()) / n, "pointing_chance": float(chance[valid].sum()) / n,
                "n_pairs": float(n), "n_concepts_scored": float(len(ks))}
