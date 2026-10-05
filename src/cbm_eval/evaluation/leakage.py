"""Information leakage: how much the label depends on information other than concept semantics.

* Interventions: replace a growing fraction of predicted concepts with ground truth; a CBM whose
  predictor relies on concept semantics should improve steadily.
* Soft-hard gap: a probe on soft concept scores vs. one on their binarized values; the gap is
  task information carried in the scores' magnitudes rather than in concept presence.
* Spurious probe: how well the spurious attribute (e.g. background) can be decoded from concepts.
"""

from __future__ import annotations

import torch

from ..registry import EVALUATION
from ..stages.common import match_names
from .base import Evaluator
from .probes import balanced_accuracy, fit_logistic


def _ground_truth(cbm, ctx, split: str) -> torch.Tensor | None:
    """Human annotations when every concept is annotated, else the alignment's targets."""
    fs = ctx.split(split)
    idx = match_names(cbm.concepts.names, ctx.dataset.concept_names)
    if fs.concept_labels is not None and all(i is not None for i in idx):
        gt = fs.concept_labels[:, idx].float()
        if cbm.model.layer.target_type == "binary":
            return gt
        return None  # binary annotations are not in a continuous concept layer's target space
    return cbm.aligned.targets(split)


@EVALUATION.register("leakage")
class LeakageEvaluator(Evaluator):
    def __init__(self, fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0), repeats: int = 5,
                 split: str = "test", probe_split: str = "val"):
        self.fractions, self.repeats, self.split, self.probe_split = fractions, repeats, split, probe_split

    def evaluate(self, cbm, ctx):
        out: dict[str, float] = {}
        fs = ctx.split(self.split)
        (x, bag), y = ctx.inputs(self.split), fs.labels
        k = len(cbm.concepts)

        gt = _ground_truth(cbm, ctx, self.split)
        if gt is not None:
            g = torch.Generator().manual_seed(ctx.seed)
            for f in self.fractions:
                n_int = round(f * k)
                accs = []
                for _ in range(self.repeats if 0 < n_int < k else 1):
                    mask = torch.zeros(len(x), k, dtype=torch.bool)
                    if n_int:
                        cols = torch.randperm(k, generator=g)[:n_int]
                        mask[:, cols] = True
                    logits = cbm.predict(x, intervene=(mask, gt), bag=bag)[0]
                    accs.append(float((logits.argmax(1) == y).float().mean()))
                out[f"intervention.acc@{f:g}"] = sum(accs) / len(accs)
            out["intervention.gain"] = out[f"intervention.acc@{max(self.fractions):g}"] - \
                out[f"intervention.acc@{min(self.fractions):g}"]

        pfs = ctx.split(self.probe_split)
        c_probe, c_eval = cbm.image_concepts(*ctx.inputs(self.probe_split)), cbm.image_concepts(x, bag)
        binarize = cbm.model.layer.binarize
        soft = fit_logistic(c_probe, pfs.labels, ctx.num_classes)
        hard = fit_logistic(binarize(c_probe), pfs.labels, ctx.num_classes)
        out["probe.soft_acc"] = float((soft(c_eval).argmax(1) == y).float().mean())
        out["probe.hard_acc"] = float((hard(binarize(c_eval)).argmax(1) == y).float().mean())
        out["probe.soft_hard_gap"] = out["probe.soft_acc"] - out["probe.hard_acc"]

        same_attrs = ctx.source(self.split)[0].n_attrs == ctx.dataset.n_attrs  # the probe is fit on training data
        if ctx.dataset.n_attrs > 1 and fs.attrs is not None and same_attrs:
            attr_probe = fit_logistic(c_probe, pfs.attrs, ctx.dataset.n_attrs)
            out["probe.spurious_bacc"] = balanced_accuracy(attr_probe(c_eval), fs.attrs)
        return out
