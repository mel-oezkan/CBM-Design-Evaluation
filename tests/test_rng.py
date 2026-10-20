"""Cache misses never advance the global RNG (issue #7): the first run of a config, on a cold cache,
equals every later run of it on the warm cache."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cbm_eval.backbones.inception import InceptionBackbone
from cbm_eval.backbones.toy import ToyBackbone
from cbm_eval.pipeline import run
from cbm_eval.stages.llm import ClaudeClient
from cbm_eval.utils import rng_preserved, rng_state, seed_everything


def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def _same_run(a, b):
    assert a.metrics == b.metrics
    sa, sb = a.cbm.model.state_dict(), b.cbm.model.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_rng_preserved_restores_every_global_generator():
    seed_everything(0)
    before = rng_state()
    with rng_preserved():
        _draw()
    assert rng_state() == before


@pytest.mark.parametrize("over", [
    {},
    {"stages.alignment": {"name": "human"}, "stages.generation": {"name": "logits"}, "stages.predictor": {"name": "dense"},
     "backbone": {"name": "toy", "aux_dim": 8},
     "stages.training": {"name": "joint", "epochs": 5, "aux": {"weight": 0.4, "init_std": 0.001}}},
    {"instances": {"name": "patches"}, "stages.predictor": {"name": "mil"}, "evaluation": [{"name": "shift"}]},
], ids=["features", "aux-features", "instances"])
def test_a_cold_cache_gives_the_same_run_as_a_warm_one(base_cfg, over):
    cfg = base_cfg.with_overrides({"evaluation": [{"name": "shift"}, {"name": "leakage", "repeats": 2}], **over})
    cold, warm = run(cfg, save=False), run(cfg, save=False)
    _same_run(cold, warm)


def test_a_backbone_that_draws_while_encoding_does_not_shift_the_run(base_cfg, monkeypatch):
    """As a backbone that builds its network on first use (Inception), on a cache miss only."""
    encode = ToyBackbone.encode_images
    monkeypatch.setattr(ToyBackbone, "encode_images", lambda self, images: _draw() and encode(self, images))
    cfg = base_cfg.with_overrides({"evaluation": [{"name": "shift"}]})
    _same_run(run(cfg, save=False), run(cfg, save=False))


def test_building_inception_draws_nothing():
    """Its layers initialize from the global RNG before the weights load; that must not shift a run,
    whether the build happens during feature extraction (cold) or in ``encoder()`` (fine-tuning)."""
    seed_everything(0)
    before = rng_state()
    InceptionBackbone(weights=None).encoder()
    assert rng_state() == before


def test_an_llm_request_draws_nothing(tmp_path):
    class Messages:
        def create(self, **kw):
            _draw()  # e.g. the SDK's retry jitter
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="- beak")])

    client = ClaudeClient(cache_dir=tmp_path, client=SimpleNamespace(beta=SimpleNamespace(messages=Messages())))
    seed_everything(0)
    before = rng_state()
    assert client.complete("concepts?") == "- beak"
    assert rng_state() == before
