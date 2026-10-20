"""Per-epoch training history (``history.jsonl``, ``ctx.history``) and live W&B tracking, with a stub
W&B client so that nothing touches the network."""

import json
import random

import pytest
import torch

from cbm_eval import cli
from cbm_eval.pipeline import run, sweep
from cbm_eval.results import ResultsStore
from cbm_eval.tracking import WandbTracker

FT = {"epochs": 2, "lr": 0.05, "batch_size": 32, "patience": 2, "num_workers": 0}


class StubRun:
    def __init__(self, kw):
        self.kw, self.logged, self.metrics, self.summary, self.exit_code = kw, [], [], {}, None

    def define_metric(self, name, step_metric=None):
        self.metrics.append((name, step_metric))

    def log(self, data):
        random.random(), torch.rand(1)  # a client that draws from the global generators must not shift the run
        self.logged.append(data)

    def finish(self, exit_code=0):
        self.exit_code = exit_code


class StubWandb:
    def __init__(self):
        self.runs = []

    def init(self, **kw):
        random.random(), torch.rand(1)
        self.runs.append(StubRun(kw))
        return self.runs[-1]


def _cfg(base_cfg, training, **over):
    return base_cfg.with_overrides({"stages.alignment": {"name": "human"}, "stages.generation": {"name": "logits"},
                                    "stages.predictor": {"name": "dense"}, "stages.training": training,
                                    "evaluation": [{"name": "shift"}], **over})


def _weights(res):
    return {k: v.clone() for k, v in res.cbm.model.state_dict().items()}


def test_history_has_one_row_per_epoch_and_marks_the_kept_epoch(base_cfg):
    res = run(_cfg(base_cfg, {"name": "sequential_weighted", "epochs": 6}))
    rows, log = res.ctx.history.rows, res.cbm.train_log
    for phase, prefix in (("concept", "concept"), ("head_phase0", "head")):
        mine = [r for r in rows if r["phase"] == phase]
        assert [r["epoch"] for r in mine] == list(range(1, len(mine) + 1))
        assert {"train_loss", "val_loss", "train_acc", "val_acc", "lr", "seconds", "best"} <= mine[0].keys()
        assert max(r["epoch"] for r in mine if r["best"]) == log[f"{prefix}_best_epoch"]
    assert [json.loads(line) for line in (res.run_dir / "history.jsonl").read_text().splitlines()] == rows


def test_history_has_no_concept_accuracy_for_continuous_targets(base_cfg):
    res = run(_cfg(base_cfg, {"name": "sequential", "epochs": 3}, **{"stages.alignment": {"name": "clip"}}),
              save=False)
    concept = [r for r in res.ctx.history.rows if r["phase"] == "concept"]
    assert concept and not any("val_acc" in r or "train_acc" in r for r in concept)


def test_finetuned_history_records_the_encoder_lr(base_cfg):
    res = run(_cfg(base_cfg, {"name": "joint", "epochs": 2, "finetune": {**FT, "lr_step": 1, "lr_gamma": 0.5}}),
              save=False)
    joint = [r for r in res.ctx.history.rows if r["phase"] == "joint"]
    assert [r["encoder_lr"] for r in joint] == pytest.approx([0.05, 0.025])
    assert all("val_acc" in r for r in joint)


def test_tracking_streams_the_history_and_changes_nothing(base_cfg):
    cfg = _cfg(base_cfg, {"name": "joint_weighted", "epochs": 4})
    plain = run(cfg, save=False)
    client = StubWandb()
    tracked = run(cfg, save=False, tracker=WandbTracker("proj", client=client))
    assert all(torch.equal(a, b) for a, b in zip(_weights(plain).values(), _weights(tracked).values()))
    assert plain.metrics == tracked.metrics
    (r,) = client.runs
    assert r.kw["name"] == f"{cfg.name}-{cfg.run_id}-s{cfg.seed}" and r.kw["group"] == cfg.name
    assert r.kw["config"]["run_id"] == cfg.run_id
    assert len(r.logged) == len(tracked.ctx.history.rows) and "joint/val_loss" in r.logged[0]
    assert ("joint/*", "joint/epoch") in r.metrics
    assert r.summary["shift.test.acc"] == tracked.metrics["shift.test.acc"] and r.exit_code == 0


def test_a_failed_run_marks_its_wandb_run_failed(base_cfg, monkeypatch):
    client = StubWandb()

    def boom(*a, **kw):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr("cbm_eval.pipeline.PipelineBuilder.evaluate", boom)
    with pytest.raises(RuntimeError):
        run(_cfg(base_cfg, {"name": "joint", "epochs": 2}), save=False, tracker=WandbTracker("proj", client=client))
    assert client.runs[0].exit_code == 1 and client.runs[0].logged


def test_sweep_and_cli_start_one_wandb_run_per_config_and_seed(base_cfg, tmp_path, monkeypatch):
    client = StubWandb()
    cfgs = [_cfg(base_cfg, {"name": "joint", "epochs": 2}).with_overrides({"seed": s}) for s in (0, 1)]
    sweep(cfgs, ResultsStore(tmp_path / "r.jsonl"), tracker=WandbTracker("proj", client=client))
    assert [r.kw["name"].rsplit("-", 1)[1] for r in client.runs] == ["s0", "s1"]

    made = []
    monkeypatch.setattr(cli, "WandbTracker", lambda *a, **kw: made.append((a, kw)) or WandbTracker(*a, **kw,
                                                                                                  client=client))
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfgs[0].to_dict()))
    cli.main(["run", str(path), "--force", "--wandb", "proj", "--wandb-group", "g", "--wandb-mode", "offline"])
    assert made == [(("proj",), {"group": "g", "mode": "offline"})] and client.runs[-1].kw["group"] == "g"
