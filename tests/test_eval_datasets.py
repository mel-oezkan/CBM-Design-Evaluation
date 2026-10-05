"""Test-only domains (``eval_datasets:``) and re-evaluating saved runs (``evaluate_saved``)."""

import math

import pytest
import torch

from cbm_eval.context import Context
from cbm_eval.data.synthetic import SyntheticDataset
from cbm_eval.pipeline import PipelineBuilder, evaluate_saved, run
from cbm_eval.registry import DATASETS
from cbm_eval.results import ResultsStore

SMALL = {"sizes": {"train": 400, "val": 200, "test": 300}}  # as conftest's SMALL_DATA
SHIFTED = {"name": "synthetic", **SMALL, "noise": 0.6, "background_strength": 1.0}
HUMAN = {"stages.alignment": {"name": "human"}}


class Reversed(SyntheticDataset):
    """The synthetic data with its classes listed in reverse order (labels follow)."""

    name = "reversed"

    @property
    def class_names(self):
        return super().class_names[::-1]

    def _generate(self, split):
        out = super()._generate(split)
        for s in out:
            s.label = self.n_classes - 1 - s.label
        return out


class ClassLevel(SyntheticDataset):
    """Synthetic data that also publishes class-level concept labels."""

    def class_concepts(self):
        return (self.class_concept_prob > 0.5).float()


class Unannotated(SyntheticDataset):
    """A test-only domain without concept annotations."""

    name = "unannotated"
    splits = ("test",)

    @property
    def concept_names(self):
        return None


def _ctx(base_cfg, dataset, **eval_datasets):
    return Context(base_cfg, dataset, PipelineBuilder(base_cfg).context().backbone, eval_datasets=eval_datasets)


def test_eval_split_is_scored_in_training_space(base_cfg):
    cfg = base_cfg.with_overrides({**HUMAN, "eval_datasets": {"shifted": SHIFTED},
                                   "evaluation": [{"name": "shift", "splits": ["test", "shifted"]},
                                                  {"name": "concepts", "splits": ["test", "shifted"]},
                                                  {"name": "leakage", "split": "shifted", "repeats": 2}]})
    res = run(cfg, save=False)
    m = res.metrics
    assert all(not math.isnan(v) for v in m.values()), m
    assert 0.25 < m["shift.shifted.acc"] < m["shift.test.acc"]  # noisier domain, still above chance
    assert "concepts.shifted.concept_error" in m and "leakage.intervention.gain" in m
    assert "shifted" in res.cbm.concept_scores


def test_eval_datasets_are_left_out_of_run_id_when_unset(base_cfg):
    assert "eval_datasets" not in base_cfg.to_dict()
    assert base_cfg.with_overrides({"eval_datasets": {"shifted": SHIFTED}}).run_id != base_cfg.run_id


def test_classes_are_matched_by_name(base_cfg):
    ctx = _ctx(base_cfg, SyntheticDataset(**SMALL), rev=Reversed(**SMALL))
    assert torch.equal(ctx.split("rev").labels, ctx.split("test").labels)
    assert torch.equal(ctx.split("rev").concept_labels, ctx.split("test").concept_labels)
    with pytest.raises(ValueError, match="Unknown split 'nope'"):
        ctx.split("nope")


def test_missing_classes_are_refused(base_cfg):
    ctx = _ctx(base_cfg, SyntheticDataset(**SMALL), more=SyntheticDataset(**{**SMALL, "n_classes": 5}))
    with pytest.raises(ValueError, match="eval_datasets.more.*lacks"):
        ctx.split("more")


def test_unannotated_eval_split_inherits_class_concepts(base_cfg):
    train = ClassLevel(**SMALL)
    fs = _ctx(base_cfg, train, plain=Unannotated(**SMALL)).split("plain")
    assert torch.equal(fs.concept_labels, train.class_concepts()[fs.labels])
    assert _ctx(base_cfg, SyntheticDataset(**SMALL), plain=Unannotated(**SMALL)).split("plain").concept_labels is None


def test_dataset_misuse_is_refused(base_cfg):
    DATASETS._items["_test_only"] = Unannotated
    try:
        with pytest.raises(ValueError, match="test-only.*eval_datasets"):
            PipelineBuilder(base_cfg.with_overrides({"dataset": {"name": "_test_only"}}))
        PipelineBuilder(base_cfg.with_overrides({"eval_datasets": {"plain": {"name": "_test_only"}}}))
    finally:
        del DATASETS._items["_test_only"]
    with pytest.raises(ValueError, match="shadows"):
        PipelineBuilder(base_cfg.with_overrides({"eval_datasets": {"test": SHIFTED}}))


@pytest.mark.parametrize("overrides", [
    {},
    {"stages.generation": {"name": "embeddings", "emb_dim": 4}},
    {"stages.predictor": {"name": "residual"}},
    {"stages.alignment": {"name": "weights"}},
    {**HUMAN, "stages.training": {"name": "joint", "epochs": 10}},
    {**HUMAN, "stages.training": {"name": "joint", "epochs": 3, "finetune": {"epochs": 2, "lr": 0.05, "batch_size": 32, "num_workers": 0}},
     "evaluation": [{"name": "shift"}, {"name": "concepts"}]},  # the fine-tuned encoder is reloaded
    {"instances": {"name": "patches"}, "stages.predictor": {"name": "mil"},
     "stages.training": {"name": "joint", "epochs": 10}},
])
def test_saved_run_evaluates_identically(base_cfg, overrides):
    cfg = base_cfg.with_overrides(overrides)
    res = run(cfg)
    again = evaluate_saved(res.run_dir)
    assert again.cfg.run_id == cfg.run_id
    for k, v in res.metrics.items():
        assert again.metrics[k] == pytest.approx(v, abs=1e-5), k


def test_saved_run_gains_eval_datasets_without_retraining(base_cfg):
    store = ResultsStore(base_cfg.paths["results"])
    res = run(base_cfg, store)
    new = {"eval_datasets": {"shifted": SHIFTED}, "evaluation": [{"name": "shift", "splits": ["test", "shifted"]}]}
    again = evaluate_saved(res.run_dir, new, store)
    assert again.metrics["shift.test.acc"] == pytest.approx(res.metrics["shift.test.acc"])
    assert "shift.shifted.acc" in again.metrics
    assert store.has(base_cfg.with_overrides(new).run_id, base_cfg.seed)
    row = store._rows()[-1]
    assert row["evaluated_from"] == str(res.run_dir)
    assert row["concept_trace"] == dict(res.concept_trace)  # the training run's trace, read from run_dir


def test_evaluate_refuses_model_changes(base_cfg):
    res = run(base_cfg.with_overrides({"evaluation": [{"name": "shift"}]}))
    with pytest.raises(ValueError, match=r"may only change.*\['stages'\]"):
        evaluate_saved(res.run_dir, {"stages.predictor.lam": 0.1})


def test_cli_evaluate_takes_eval_keys_from_a_config(base_cfg, tmp_path, capsys):
    import json

    import yaml

    from cbm_eval.cli import main

    res = run(base_cfg.with_overrides({"evaluation": [{"name": "shift"}]}))
    spec = {**base_cfg.to_dict(), "stages": {}, "eval_datasets": {"shifted": SHIFTED},
            "evaluation": [{"name": "shift", "splits": ["shifted"]}]}
    (tmp_path / "anchor.yaml").write_text(yaml.safe_dump(spec))
    main(["evaluate", str(res.run_dir), "--config", str(tmp_path / "anchor.yaml")])
    out = json.loads(capsys.readouterr().out)
    assert "shift.shifted.acc" in out and out["run_id"] != base_cfg.run_id


def test_cub_class_concepts_match_its_image_labels(cub_root):
    from cbm_eval.data.cub import CUB

    cub = CUB(root=str(cub_root), min_class_count=0)
    cc = cub.class_concepts()
    assert cc.shape == (len(cub.class_names), len(cub.concept_names))
    assert all(cc[s.label].tolist() == s.concepts for s in cub.samples("test"))
    assert CUB(root=str(cub_root), class_level_concepts=False).class_concepts() is None
