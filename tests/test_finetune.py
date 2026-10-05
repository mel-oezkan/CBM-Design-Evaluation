"""End-to-end backbone fine-tuning (``stages.training.finetune``) on the synthetic dataset + toy backbone."""

import math

import pytest
import torch

from cbm_eval.backbones.toy import ToyBackbone
from cbm_eval.pipeline import PipelineBuilder, run

FT = {"epochs": 3, "lr": 0.05, "batch_size": 32, "patience": 3, "num_workers": 0}
HUMAN = {"stages.alignment": {"name": "human"}}


def _cfg(base_cfg, training):
    return base_cfg.with_overrides({**HUMAN, "stages.training": training,
                                    "evaluation": [{"name": "shift"}, {"name": "concepts"}, {"name": "leakage"}]})


@pytest.mark.parametrize("training", [
    {"name": "independent", "epochs": 5, "finetune": FT},
    {"name": "sequential", "epochs": 5, "finetune": FT},
    {"name": "joint", "epochs": 5, "finetune": FT},
    {"name": "joint_weighted", "epochs": 5, "finetune": {**FT, "optimizer": "adam", "lr": 1e-3, "lr_step": 1}},
])
def test_finetune_trains_the_encoder_and_evaluates_on_its_features(base_cfg, training):
    res = run(_cfg(base_cfg, training), save=False)
    m = res.metrics
    assert all(not math.isnan(v) for v in m.values()), m
    assert m["shift.test.acc"] > 0.4 and 0 <= m["concepts.test.concept_error"] < 0.5
    assert res.cbm.train_log["finetune_epochs"] >= 1
    frozen = ToyBackbone().proj
    assert not torch.allclose(res.cbm.encoder.proj.detach().cpu(), frozen)  # the encoder moved


def test_inputs_are_the_fine_tuned_features(base_cfg):
    cfg = _cfg(base_cfg, {"name": "joint", "epochs": 5, "finetune": FT})
    builder = PipelineBuilder(cfg)
    ctx = builder.context()
    cbm = builder.train(ctx, builder.concepts(ctx))
    images = torch.stack([s.image for s in ctx.dataset.samples("test")])
    with torch.no_grad():
        expected = cbm.encoder(images)
    assert torch.allclose(ctx.inputs("test")[0], expected, atol=1e-5)
    # the frozen cache is untouched, and a new run on the same Context starts from it again
    plain = PipelineBuilder(base_cfg.with_overrides(HUMAN))
    plain.train(ctx, plain.concepts(ctx))
    assert torch.allclose(ctx.inputs("test")[0], ToyBackbone().encode_images(images), atol=1e-5)


def test_finetune_needs_a_trainable_backbone(base_cfg, monkeypatch):
    monkeypatch.setattr(ToyBackbone, "trainable", False)
    with pytest.raises(ValueError, match="trainable backbone"):
        run(_cfg(base_cfg, {"name": "joint", "finetune": FT}), save=False)


def test_finetune_refuses_patch_readers_and_bags(base_cfg):
    cfg = base_cfg.with_overrides({**HUMAN, "stages.training": {"name": "joint", "finetune": FT}})
    with pytest.raises(ValueError, match="frozen patch features"):  # base_cfg evaluates localization
        PipelineBuilder(cfg)
    with pytest.raises(ValueError, match="instances"):
        PipelineBuilder(cfg.with_overrides({"instances": {"name": "patches"}, "stages.predictor": {"name": "mil"},
                                            "evaluation": [{"name": "shift"}]}))


def test_unknown_finetune_keys_are_rejected(base_cfg):
    with pytest.raises(TypeError):
        PipelineBuilder(_cfg(base_cfg, {"name": "joint", "finetune": {"lrr": 0.1}}))
