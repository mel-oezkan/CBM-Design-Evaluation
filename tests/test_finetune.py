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



def test_a_reused_context_drops_the_previous_runs_encoder(base_cfg):  # issue #2
    res = run(_cfg(base_cfg, {"name": "joint", "epochs": 2, "finetune": FT}), save=False)
    ctx = PipelineBuilder(base_cfg.with_overrides(HUMAN)).reuse(res.ctx)  # what run(cfg, ctx=...) does first
    images = torch.stack([s.image for s in ctx.dataset.samples("train")])
    assert torch.allclose(ctx.split("train").features, ToyBackbone().encode_images(images), atol=1e-5)

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


ON_VAL = {"joint": {"select": "train_acc", "train_on_val": True},
          "sequential": {"concept_select": "train_acc", "concept_train_on_val": True},
          "independent_weighted": {"concept_select": "train_acc", "concept_train_on_val": True, "select": "val_acc"}}


@pytest.mark.parametrize("name", ON_VAL)
def test_finetune_trains_on_train_and_val(base_cfg, name):  # issue #11
    res = run(_cfg(base_cfg, {"name": name, "epochs": 5, "finetune": FT, **ON_VAL[name]}), save=False)
    m, log = res.metrics, res.cbm.train_log
    assert m["shift.test.acc"] > 0.4 and 0 <= m["concepts.test.concept_error"] < 0.5
    phase = "joint" if name == "joint" else "concept"
    assert 0 < log[f"{phase}_train_acc"] <= 1 and math.isnan(log[f"{phase}_val_loss"])  # NaN: no val split left
    if name != "joint":  # the head still validates, as it does on a frozen backbone
        assert not math.isnan(log["head_val_loss"])


def test_the_on_val_feed_serves_every_train_and_val_image_once(base_cfg):
    from cbm_eval.stages.training import _ImageFeed

    builder = PipelineBuilder(_cfg(base_cfg, {"name": "joint", "finetune": FT}))
    ctx = builder.context()
    aligned = builder.alignment.fit(builder.concepts(ctx), ctx)
    spec = builder.training.finetune.__class__(**{**FT, "batch_size": 1, "augment": False})
    feed = _ImageFeed(ctx, aligned, ctx.backbone.encoder(), spec, torch.device("cpu"), 0, on_val=True)
    with torch.no_grad():
        served = list(feed.batches("train"))
        expected = [feed.encode(s) for s in ("train", "val")]

    def rows(batches):  # (x, y, t) per image, as a sorted list: the order of the shuffled batches is irrelevant
        x, y, t = (torch.cat([getattr(b, k) for b in batches]) for k in ("x", "y", "t"))
        return sorted(map(tuple, torch.cat([x, y[:, None].float(), t], 1).round(decimals=4).tolist()))

    assert rows(served) == rows(expected)
