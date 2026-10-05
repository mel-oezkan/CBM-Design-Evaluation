"""Distribution-shift robustness: accuracy, per-group and worst-group accuracy, effective robustness."""

from __future__ import annotations

import math

import torch

from ..registry import EVALUATION
from .base import Evaluator


def group_accuracies(pred: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> dict[int, float]:
    return {int(g): float((pred[groups == g] == labels[groups == g]).float().mean()) for g in groups.unique()}


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def effective_robustness(id_acc: float, ood_acc: float, slope: float, intercept: float) -> float:
    """OOD accuracy above the baseline trend fitted in logit space (Taori et al., 2020)."""
    expected = 1 / (1 + math.exp(-(slope * logit(id_acc) + intercept)))
    return ood_acc - expected


@EVALUATION.register("shift")
class ShiftEvaluator(Evaluator):
    """``baseline`` = {slope, intercept} of the ID->OOD logit-linear trend; without it ER is left to analysis."""

    def __init__(self, splits: tuple[str, ...] = ("val", "test"), id_metric: str = "val.acc",
                 ood_metric: str = "test.wga", baseline: dict[str, float] | None = None):
        self.splits, self.id_metric, self.ood_metric, self.baseline = splits, id_metric, ood_metric, baseline

    def evaluate(self, cbm, ctx):
        out: dict[str, float] = {}
        for s in self.splits:
            fs = ctx.split(s)
            x, bag = ctx.inputs(s)
            pred = cbm.predict(x, bag=bag)[0].argmax(1)
            out[f"{s}.acc"] = float((pred == fs.labels).float().mean())
            if fs.groups is not None:
                accs = group_accuracies(pred, fs.labels, fs.groups)
                for g, a in accs.items():
                    out[f"{s}.group{g}_acc"] = a
                out[f"{s}.wga"] = min(accs.values())
                out[f"{s}.mean_group_acc"] = sum(accs.values()) / len(accs)
        if self.baseline is not None:
            out["effective_robustness"] = effective_robustness(
                out[self.id_metric], out[self.ood_metric], self.baseline["slope"], self.baseline["intercept"])
        return out
