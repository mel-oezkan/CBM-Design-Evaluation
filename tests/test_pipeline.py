import math

import pytest
import torch

from cbm_eval.pipeline import PipelineBuilder, run, sweep
from cbm_eval.registry import TRAINING
from cbm_eval.results import ResultsStore
from cbm_eval.stages.training import Joint, Sequential


def _ok(metrics):
    return all(not math.isnan(v) for v in metrics.values())


def test_anchor_runs_end_to_end(base_cfg, tmp_path):
    store = ResultsStore(base_cfg.paths["results"])
    res = run(base_cfg, store)
    m = res.metrics
    assert _ok(m)
    assert m["shift.test.acc"] > 0.5  # 4 classes -> chance is 0.25
    assert m["shift.test.wga"] <= m["shift.test.acc"]
    assert 0 <= m["localization.pointing_game"] <= 1
    # "class_0" names a class and must be removed by the rules filter
    assert "class_0" not in res.cbm.concepts.names
    assert m["n_concepts"] == 16
    assert (res.run_dir / "model.pt").exists()
    assert store.has(base_cfg.run_id, base_cfg.seed)
    df = store.load()
    assert len(df) == 1 and "metric.shift.test.wga" in df


@pytest.mark.parametrize("stage,spec", [
    ("generation", {"name": "embeddings", "emb_dim": 4}),
    ("generation", {"name": "boc"}),
    ("alignment", {"name": "weights"}),
    ("alignment", {"name": "human"}),
    ("alignment", {"name": "dino", "scorer": "oracle"}),
    ("predictor", {"name": "dense"}),
    ("predictor", {"name": "residual"}),
    ("training", {"name": "independent", "epochs": 20}),
    ("training", {"name": "joint", "epochs": 20}),
    ("discovery", {"name": "dataset"}),
    ("discovery", {"name": "sae", "n_latents": 32, "k": 4, "epochs": 5}),
    ("filtering", [{"name": "rules"}, {"name": "clip"}, {"name": "dino", "scorer": "oracle"},
                   {"name": "select", "k": 10, "method": "discriminative"}]),
])
def test_every_stage_variant_runs(base_cfg, stage, spec):
    cfg = base_cfg.with_overrides({f"stages.{stage}": spec})
    res = run(cfg, save=False)
    assert _ok(res.metrics), res.metrics
    assert res.metrics["shift.test.acc"] > 0.4


@pytest.mark.parametrize("parent", [Sequential, Joint])
def test_task_loss_hook_is_used(base_cfg, parent):
    calls = []

    class Counted(parent):
        def _task_loss(self, logits, b):
            calls.append(len(b))
            return super()._task_loss(logits, b)

    TRAINING._items["_counted"] = Counted  # registered by hand so the name can be reused per parameter
    try:
        cfg = base_cfg.with_overrides({"stages.training": {"name": "_counted", "epochs": 2}})
        assert _ok(run(cfg, save=False).metrics)
    finally:
        del TRAINING._items["_counted"]
    assert calls


def test_human_alignment_beats_chance_on_interventions(base_cfg):
    cfg = base_cfg.with_overrides({"stages.alignment": {"name": "human"}})
    m = run(cfg, save=False).metrics
    # Replacing predicted concepts with true annotations must help a CBM on concept-driven data.
    assert m["leakage.intervention.acc@1"] > m["leakage.intervention.acc@0"]


def test_independent_needs_targets(base_cfg):
    cfg = base_cfg.with_overrides({"stages.alignment": {"name": "weights"},
                                   "stages.training": {"name": "independent"}})
    with pytest.raises(RuntimeError, match="needs concept targets"):
        run(cfg, save=False)


def test_sparse_head_is_sparse(base_cfg):
    cfg = base_cfg.with_overrides({"stages.predictor.lam": 0.05, "evaluation": [{"name": "shift"}]})
    res = run(cfg, save=False)
    assert res.cbm.train_log["head.nonzero_frac"] < 1.0


def test_intervention_with_full_ground_truth_uses_targets(base_cfg):
    builder = PipelineBuilder(base_cfg.with_overrides({"stages.alignment": {"name": "human"}}))
    ctx = builder.context()
    cbm = builder.train(ctx, builder.concepts(ctx))
    x = ctx.split("test").features
    gt = cbm.aligned.targets("test")
    mask = torch.ones_like(gt, dtype=torch.bool)
    _, c_hat = cbm.predict(x, intervene=(mask, gt))
    assert torch.equal(c_hat, gt)


def test_sweep_skips_existing_and_records_failures(base_cfg):
    store = ResultsStore(base_cfg.paths["results"])
    cheap = base_cfg.with_overrides({"evaluation": [{"name": "shift"}]})
    bad = cheap.with_overrides({"stages.filtering": [{"name": "rules", "blocklist": [f"concept_{i:02d}" for i in range(24)] + ["distractor_00"]}]})
    status = sweep([cheap, bad], store, save=False)
    assert [s["status"] for s in status][0] == "ok"
    assert status[1]["status"].startswith("error")
    assert sweep([cheap], store, save=False)[0]["status"] == "skipped"
    assert len(store.load()) == 1
    assert len(store.load(include_failed=True)) == 2


def _count_contexts(monkeypatch) -> list:
    built, context = [], PipelineBuilder.context
    monkeypatch.setattr(PipelineBuilder, "context", lambda self: built.append(context(self)) or built[-1])
    return built


def test_sweep_shares_a_context_without_changing_results(base_cfg, monkeypatch):
    cheap = base_cfg.with_overrides({"evaluation": [{"name": "shift"}, {"name": "leakage", "repeats": 2}]})
    configs = [cheap, cheap.with_overrides({"stages.predictor": {"name": "dense"}}), cheap.with_overrides({"seed": 1}),
               cheap.with_overrides({"stages.training": {"name": "joint", "epochs": 10}})]
    run(cheap, save=False)  # a cold feature cache draws from the global RNG; warm it for both sides
    alone = [run(cfg, save=False).metrics for cfg in configs]
    built = _count_contexts(monkeypatch)
    status = sweep(configs, ResultsStore(base_cfg.paths["results"]), save=False)
    assert len(built) == 1
    for s, m in zip(status, alone):
        assert {k: s[k] for k in m} == m  # equal, not close


def test_sweep_does_not_share_a_context_that_drew_random_numbers(base_cfg, monkeypatch):
    from cbm_eval.backbones.toy import ToyBackbone

    init = ToyBackbone.__init__
    monkeypatch.setattr(ToyBackbone, "__init__", lambda self, **kw: (torch.rand(1), init(self, **kw))[1])
    built = _count_contexts(monkeypatch)
    cheap = base_cfg.with_overrides({"evaluation": [{"name": "shift"}]})
    sweep([cheap, cheap.with_overrides({"seed": 1})], ResultsStore(base_cfg.paths["results"]), save=False)
    assert len(built) == 2 and all(ctx.key is None for ctx in built)


def test_context_key_separates_what_the_context_reads(base_cfg):
    key = lambda **o: PipelineBuilder(base_cfg.with_overrides(o)).context_key()
    assert key() == key(**{"stages.predictor": {"name": "dense"}}) == key(seed=3)
    assert key() != key(**{"dataset.sizes.train": 300})
    assert key() != key(evaluation=[{"name": "shift"}])  # no evaluator reads patch features
    assert key(evaluation=[], **{"stages.training.finetune": {"epochs": 1}}) is None
    ctx = PipelineBuilder(base_cfg).context()
    with pytest.raises(ValueError, match="another dataset"):
        PipelineBuilder(base_cfg.with_overrides({"dataset.sizes.train": 300})).reuse(ctx)


HUMAN = {"stages.alignment": {"name": "human"}}


@pytest.mark.parametrize("overrides", [
    {"stages.generation": {"name": "logits"}},
    {"stages.generation": {"name": "logits"}, "stages.training": {"name": "joint_weighted", "epochs": 20}},
    {"stages.training": {"name": "sequential_weighted", "epochs": 20}},
    {"stages.training": {"name": "independent_weighted", "epochs": 20}},
])
def test_koh2020_variants_run(base_cfg, overrides):
    cfg = base_cfg.with_overrides({**HUMAN, **overrides,
                                   "evaluation": [{"name": "shift"}, {"name": "concepts"}, {"name": "leakage"}]})
    res = run(cfg, save=False)
    m = res.metrics
    assert _ok(m), m
    assert m["shift.test.acc"] > 0.4
    assert 0 <= m["concepts.test.concept_error"] < 0.5 and 0 < m["concepts.test.concept_f1"] <= 1


def test_logits_generation_feeds_logits():
    from cbm_eval.stages.base import AlignedConcepts
    from cbm_eval.stages.generation import LogitScores
    from cbm_eval.structures import ConceptSet

    layer = LogitScores().build(8, AlignedConcepts(ConceptSet(["a", "b", "c"]), target_type="binary"))
    x = torch.randn(5, 8)
    c_hat, rep = layer(x)
    assert torch.allclose(rep, layer.concept_logits(x), atol=1e-4)
    assert torch.all((c_hat >= 0) & (c_hat <= 1))


def test_weighted_training_refuses_continuous_targets(base_cfg):
    cfg = base_cfg.with_overrides({"stages.training": {"name": "joint_weighted", "epochs": 2}})
    with pytest.raises(ValueError, match="binary image-level concept targets"):
        run(cfg, save=False)
