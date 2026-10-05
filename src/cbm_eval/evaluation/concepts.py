"""Concept prediction quality against human annotations (Koh et al. 2020, Table 2).

``<split>.concept_error`` is 1 - accuracy of ``c_hat >= 0.5`` over all images x concepts, the
metric of Koh et al.; ``<split>.concept_f1`` is the F1 of the positive class over the same pairs
(positives are ~10% of CUB's class-level labels, so error alone flatters an all-zero predictor).
Only defined for binary concept layers whose concepts are all annotated.
"""

from __future__ import annotations

import torch

from ..registry import EVALUATION
from ..stages.common import match_names
from .base import Evaluator


@EVALUATION.register("concepts")
class ConceptEvaluator(Evaluator):
    def __init__(self, splits: tuple[str, ...] = ("test",)):
        self.splits = splits

    def evaluate(self, cbm, ctx):
        if cbm.model.layer.target_type != "binary":
            return {}
        idx = match_names(cbm.concepts.names, ctx.dataset.concept_names)
        if any(i is None for i in idx):
            return {}
        out: dict[str, float] = {}
        for s in self.splits:
            gt = ctx.split(s).concept_labels
            if gt is None:
                continue
            gt = gt[:, idx].bool()
            pred = cbm.image_concepts(*ctx.inputs(s)) >= 0.5
            tp = float((pred & gt).sum())
            out[f"{s}.concept_error"] = float((pred != gt).float().mean())
            out[f"{s}.concept_f1"] = 2 * tp / max(float(pred.sum() + gt.sum()), 1.0)
            out[f"{s}.positive_rate"] = float(gt.float().mean())
        return out
