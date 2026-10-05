"""Test-only domains (``eval_datasets:``)."""

import math

import pytest
import torch

from cbm_eval.context import Context
from cbm_eval.data.synthetic import SyntheticDataset
from cbm_eval.pipeline import PipelineBuilder, run
from cbm_eval.registry import DATASETS

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


def test_cub_class_concepts_match_its_image_labels(cub_root):
    from cbm_eval.data.cub import CUB

    cub = CUB(root=str(cub_root), min_class_count=0)
    cc = cub.class_concepts()
    assert cc.shape == (len(cub.class_names), len(cub.concept_names))
    assert all(cc[s.label].tolist() == s.concepts for s in cub.samples("test"))
    assert CUB(root=str(cub_root), class_level_concepts=False).class_concepts() is None
