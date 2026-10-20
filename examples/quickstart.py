"""Quickstart: build, train, inspect and analyze a CBM from Python instead of the CLI.

Runs offline in seconds on the synthetic dataset and the ``toy`` backbone::

    uv run python examples/quickstart.py

It writes ``.cache/``, ``runs/quickstart/`` and ``results/quickstart.jsonl`` under the current
directory. ``docs/python-api.md`` walks through it section by section; the ``--8<--`` comments
mark the sections that page includes.
"""

from __future__ import annotations

import torch

from cbm_eval import analysis
from cbm_eval.config import ExperimentConfig
from cbm_eval.pipeline import run
from cbm_eval.results import ResultsStore

# --8<-- [start:config]
# A classic CBM: the dataset's annotated concepts, trained on those annotations (`alignment: human`),
# then a sparse linear head on the predicted concepts. Same keys as an anchor YAML.
cfg = ExperimentConfig.from_dict({
    "name": "quickstart",
    "dataset": {"name": "synthetic"},
    "backbone": {"name": "toy"},
    "stages": {
        "discovery": {"name": "dataset"},
        "filtering": [{"name": "rules"}, {"name": "select", "k": 16}],
        "alignment": {"name": "human"},
        "generation": {"name": "scores"},
        "predictor": {"name": "sparse", "lam": 1e-3},
        "training": {"name": "sequential", "epochs": 50},
    },
    "evaluation": [{"name": "shift"}, {"name": "leakage"}],
    "device": "cpu",
    "paths": {"cache": ".cache", "runs": "runs/quickstart", "results": "results/quickstart.jsonl"},
})
print("run_id:", cfg.run_id)
# --8<-- [end:config]

# --8<-- [start:run]
store = ResultsStore(cfg.paths["results"])
res = run(cfg, store)        # train + evaluate; saves runs/quickstart/<run_id>-s0/ and appends one row
cbm, ctx = res.cbm, res.ctx  # the TrainedCBM and the run's Context (dataset, backbone, cached features)
print(f"{len(cbm.concepts)} concepts, test accuracy {res.metrics['shift.test.acc']:.3f}")
# --8<-- [end:run]

# --8<-- [start:predict]
x, bag = ctx.inputs("test")              # (N, D) features; bag is None without `instances:`
y = ctx.split("test").labels             # (N,)
logits, c_hat = cbm.predict(x, bag=bag)  # (N, C) class logits, (N, K) concept probabilities
print(f"accuracy {(logits.argmax(1) == y).float().mean():.3f}")
# --8<-- [end:predict]

# --8<-- [start:intervene]
# Replace the k concepts the model is least sure of by their true values. Human alignment gives
# binary concepts, so "least sure" is a probability nearest 0.5.
truth = cbm.aligned.targets("test")      # (N, K) concept labels, in the concept layer's target space
uncertain = -(c_hat - 0.5).abs()
for k in (0, 2, 4, 8):
    mask = torch.zeros_like(c_hat, dtype=torch.bool)
    mask.scatter_(1, uncertain.topk(k, dim=1).indices, True)
    logits_k, _ = cbm.predict(x, intervene=(mask, truth), bag=bag)
    print(f"intervene on {k} concepts: accuracy {(logits_k.argmax(1) == y).float().mean():.3f}")
# --8<-- [end:intervene]

# --8<-- [start:weights]
# With `generation: scores` the head reads one score per concept: its weight is (C, K).
w = cbm.model.head.linear.weight.detach()
for c, cls in enumerate(ctx.class_names):
    top = w[c].topk(3).indices.tolist()
    print(f"{cls}: " + ", ".join(f"{cbm.concepts.names[j]} ({w[c, j]:+.2f})" for j in top))
# --8<-- [end:weights]

# --8<-- [start:analyze]
# A second config that differs in one key; its run_id differs, so it gets its own row.
dense = cfg.with_overrides({"stages.predictor": {"name": "dense"}})
run(dense, store)
df = store.load()  # one row per (run_id, seed), columns like factor.predictor, metric.shift.test.acc
table = analysis.aggregate(df, ["metric.shift.test.acc", "metric.leakage.intervention.gain"])
print(table[["factor.predictor", "metric.shift.test.acc.mean", "metric.leakage.intervention.gain.mean"]]
      .to_string(index=False))
# --8<-- [end:analyze]
