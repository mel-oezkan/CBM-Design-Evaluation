"""Koh et al.'s auxiliary concept head (``stages.training.aux``) on the synthetic dataset + toy backbone
with an auxiliary tap (``backbone.aux_dim``), plus the Inception-v3 tap itself."""

import math

import pytest
import torch

from cbm_eval.backbones.inception import _TRUNK, InceptionBackbone
from cbm_eval.backbones.toy import ToyBackbone
from cbm_eval.pipeline import PipelineBuilder, run
from cbm_eval.stages.training import Batch, _AuxHead

AUX = {"weight": 0.4, "init_std": 0.001}
FT = {"epochs": 2, "lr": 0.05, "batch_size": 32, "patience": 2, "num_workers": 0}


def _cfg(base_cfg, training, **over):
    return base_cfg.with_overrides({"backbone": {"name": "toy", "aux_dim": 8},
                                    "stages.alignment": {"name": "human"},
                                    "stages.generation": {"name": "logits"}, "stages.predictor": {"name": "dense"},
                                    "stages.training": training, "evaluation": [{"name": "shift"}, {"name": "concepts"}],
                                    **over})


def _head_weight(res):
    return res.cbm.model.head.linear.weight.detach()


def test_frozen_joint_aux_trains_the_shared_head(base_cfg):
    plain = run(_cfg(base_cfg, {"name": "joint_weighted", "epochs": 5}), save=False)
    aux = run(_cfg(base_cfg, {"name": "joint_weighted", "epochs": 5, "aux": AUX}), save=False)
    assert all(not math.isnan(v) for v in aux.metrics.values()), aux.metrics
    assert aux.metrics["shift.test.acc"] > 0.4
    assert not torch.allclose(_head_weight(plain), _head_weight(aux))
    assert plain.cfg.run_id != aux.cfg.run_id


def test_frozen_joint_aux_supports_continuous_targets(base_cfg):
    res = run(_cfg(base_cfg, {"name": "joint", "epochs": 3, "aux": AUX}, **{"stages.alignment": {"name": "clip"}}),
              save=False)
    assert all(not math.isnan(v) for v in res.metrics.values()), res.metrics


def test_aux_gradient_reaches_the_head_and_aux_head_but_not_the_concept_layer(base_cfg):
    builder = PipelineBuilder(_cfg(base_cfg, {"name": "joint", "aux": AUX}))
    ctx = builder.context()
    aligned = builder.alignment.fit(builder.concepts(ctx), ctx)
    layer = builder.generation.build(ctx.feat_dim, aligned)
    head = builder.predictor.build(layer.rep_dim, ctx.num_classes, ctx.feat_dim)
    xa = ctx.aux_features("train")
    aux_head = _AuxHead(xa.shape[1], layer.n_concepts, 0.001)
    b = Batch(ctx.split("train").features, ctx.split("train").labels, aligned.targets("train"), xa=xa)
    builder.training._aux_term(aux_head, layer, head, b).backward()
    assert layer.linear.weight.grad is None
    assert head.linear.weight.grad.abs().sum() > 0 and aux_head.linear.weight.grad.abs().sum() > 0
    assert builder.training._aux_term(aux_head, layer, head, Batch(b.x, b.y, b.t)) == 0.0  # val batches: no xa


@pytest.mark.parametrize("name,generation", [("independent_weighted", "scores"), ("sequential_weighted", "logits"),
                                             ("joint_weighted", "logits")])
def test_finetuned_aux_reaches_the_encoder_and_round_trips(base_cfg, name, generation):
    gen = {"stages.generation": {"name": generation}}
    cfg = _cfg(base_cfg, {"name": name, "epochs": 3, "finetune": FT, "aux": AUX}, **gen)
    res = run(cfg, save=True)
    assert all(not math.isnan(v) for v in res.metrics.values()), res.metrics
    enc = res.cbm.encoder
    assert not torch.allclose(enc.aux_proj.detach().cpu(), ToyBackbone(aux_dim=8).aux_proj)  # the tap trained
    plain = run(_cfg(base_cfg, {"name": name, "epochs": 3, "finetune": FT}, **gen), save=False).cbm.encoder
    assert not torch.allclose(enc.proj.detach().cpu(), plain.proj.detach().cpu())  # aux changed the trunk
    builder = PipelineBuilder(cfg)
    loaded = builder.load(builder.context(), res.run_dir)  # encoder.pt holds the tap; loads strictly
    assert torch.allclose(loaded.encoder.proj, enc.proj.cpu())


@pytest.mark.parametrize("name", ["independent", "sequential_weighted"])
def test_frozen_independent_and_sequential_refuse_aux(base_cfg, name):
    with pytest.raises(ValueError, match="finetune"):
        PipelineBuilder(_cfg(base_cfg, {"name": name, "aux": AUX}))


@pytest.mark.parametrize("over,match", [
    ({"backbone": {"name": "toy"}}, "auxiliary feature tap"),
    ({"stages.alignment": {"name": "weights"}}, "targets"),
    ({"stages.generation": {"name": "embeddings", "emb_dim": 4}}, "read only concepts"),
    ({"stages.predictor": {"name": "residual"}}, "read only concepts"),
])
def test_aux_refuses_what_it_cannot_express(base_cfg, over, match):
    with pytest.raises(ValueError, match=match):
        run(_cfg(base_cfg, {"name": "joint", "epochs": 2, "aux": AUX}, **over), save=False)


def test_aux_refuses_bags_and_bad_specs(base_cfg):
    with pytest.raises(ValueError, match="instances"):
        PipelineBuilder(_cfg(base_cfg, {"name": "joint", "aux": AUX}, instances={"name": "patches"},
                             **{"stages.predictor": {"name": "mil"}}))
    with pytest.raises(ValueError, match="positive"):
        PipelineBuilder(_cfg(base_cfg, {"name": "joint", "aux": {"weight": 0}}))
    with pytest.raises(TypeError):
        PipelineBuilder(_cfg(base_cfg, {"name": "joint", "aux": {"wieght": 0.4}}))
    assert PipelineBuilder(_cfg(base_cfg, {"name": "joint", "aux": {}})).training.needs_aux  # {} = defaults
    with pytest.raises(ValueError, match="finetune"):
        PipelineBuilder(_cfg(base_cfg, {"name": "sequential", "aux": {}}))


def test_a_cold_aux_cache_does_not_shift_the_run(base_cfg):
    cfg = _cfg(base_cfg, {"name": "joint_weighted", "epochs": 3, "aux": AUX})
    run(_cfg(base_cfg, {"name": "joint_weighted", "epochs": 1}), save=False)  # warm the main feature cache only
    cold, warm = (_head_weight(run(cfg, save=False)) for _ in range(2))
    assert torch.equal(cold, warm)


def test_inception_aux_tap_matches_one_pass_and_stays_out_of_plain_encoders():
    torch.manual_seed(0)
    bb = InceptionBackbone(weights=None)
    images = torch.randn(2, 3, 299, 299)
    enc = bb.encoder(aux=True).eval()
    with torch.no_grad():
        x, xa = enc.forward_aux(images)
    assert xa.shape == (2, 768) and bb.aux_dim == 768
    assert torch.allclose(x, bb.encode_images(images), atol=1e-4)
    assert torch.allclose(xa, bb.encode_aux(images), atol=1e-4)
    assert not any(k.startswith("aux.") for k in bb.encoder().state_dict())  # plain encoder.pt is unchanged
    assert any(k.startswith("aux.") for k in enc.state_dict())


def test_inception_aux_gradient_reaches_mixed_6e_but_not_mixed_7():
    torch.manual_seed(0)
    enc = InceptionBackbone(weights=None).encoder(aux=True).train()
    _, xa = enc.forward_aux(torch.randn(2, 3, 299, 299))
    xa.sum().backward()
    grad = lambda m: any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    trunk = {name: enc.trunk[_TRUNK.index(name)] for name in ("Mixed_6e", "Mixed_7a")}
    assert grad(enc.aux) and grad(trunk["Mixed_6e"]) and not grad(trunk["Mixed_7a"])
