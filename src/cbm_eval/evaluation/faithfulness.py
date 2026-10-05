"""Segment faithfulness (SEG-MIL-CBM): does the model's own segment ranking pick the segments its
prediction actually depends on?

Segments are ranked by ``S_i = max(0, C_i,y_hat)``, the instance's contribution to the predicted
logit (by ``|C|`` for images where no contribution is positive). Segments are removed by zeroing
their embeddings (``mode: zero``, as in the paper) or dropping them from the bag (``mode: drop``).

* ``dauc`` / ``iauc``: area under the probability of the original prediction as the top-ranked
  fraction of segments is deleted / inserted (lower / higher is better). ``*_random`` uses a
  random ranking as the control.
* ``top1_insertion``: share of images whose prediction survives with only the top segment.
* ``top5_deletion``: share of images whose prediction flips when the top five are removed.

Only defined for bag-aware heads that report per-instance contributions (``predictor: mil``).
"""

from __future__ import annotations

import torch

from ..registry import EVALUATION
from ..structures import Bag
from .base import Evaluator


@EVALUATION.register("faithfulness")
class FaithfulnessEvaluator(Evaluator):
    def __init__(self, split: str = "test", n_images: int | None = 3000, steps: int = 10, mode: str = "zero"):
        if mode not in ("zero", "drop"):
            raise ValueError(f"Unknown faithfulness mode '{mode}' (use zero or drop)")
        self.split, self.n_images, self.steps, self.mode = split, n_images, steps, mode

    def evaluate(self, cbm, ctx):
        head = cbm.model.head
        x, bag = ctx.inputs(self.split)
        if bag is None or not hasattr(head, "contributions"):
            return {}
        g = torch.Generator().manual_seed(ctx.seed)
        if self.n_images is not None and self.n_images < len(x):
            sel = torch.randperm(len(x), generator=g)[: self.n_images].sort().values
            x, bag = x[sel], bag[sel]
        x = x.float()
        layer = cbm.model.layer.eval()
        with torch.no_grad():
            _, rep = layer(x)
            x_norm = layer.normalize(x)
            pred = head(rep, x_norm, bag).argmax(1)
            contrib = head.contributions(rep, x_norm, bag)[torch.arange(len(x)), :, pred]  # (N, M)
        score = contrib.clamp_min(0)
        none_positive = (score * bag.mask).sum(1, keepdim=True) == 0
        score = torch.where(none_positive, contrib.abs(), score)
        rank = _ranks(score, bag.mask)
        random_rank = _ranks(torch.rand(score.shape, generator=g), bag.mask)
        n_real = bag.mask.sum(1, keepdim=True)

        def prob(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if self.mode == "zero":
                logits = cbm.predict(x * keep[..., None], bag=bag)[0]
            else:
                logits = cbm.predict(x, bag=Bag(bag.mask & keep, bag.area))[0]
            p = logits.softmax(1)
            return p[torch.arange(len(x)), pred], p.argmax(1)

        def curves(r: torch.Tensor) -> tuple[float, float]:
            dele, ins = [], []
            for i in range(self.steps + 1):
                top = r < torch.ceil(n_real * i / self.steps)
                dele.append(prob(bag.mask & ~top)[0].mean())
                ins.append(prob(bag.mask & top)[0].mean())
            return _auc(dele), _auc(ins)

        dauc, iauc = curves(rank)
        dauc_r, iauc_r = curves(random_rank)
        top1 = (prob(bag.mask & (rank < 1))[1] == pred).float().mean()
        top5 = (prob(bag.mask & (rank >= 5))[1] != pred).float().mean()
        return {"dauc": dauc, "iauc": iauc, "dauc_random": dauc_r, "iauc_random": iauc_r,
                "top1_insertion": float(top1), "top5_deletion": float(top5), "n_images": float(len(x))}


def _ranks(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rank of each instance within its image (0 = most important); padding ranks last."""
    order = score.masked_fill(~mask, float("-inf")).argsort(1, descending=True)
    return torch.empty_like(order).scatter_(1, order, torch.arange(score.shape[1]).expand_as(order))


def _auc(values: list[torch.Tensor]) -> float:
    v = torch.stack(values)
    return float(torch.trapezoid(v, dx=1.0 / (len(v) - 1)))
