import pytest

from cbm_eval.config import Ablation, ExperimentConfig, parse_override, set_dotted
from cbm_eval.registry import Registry


def test_overrides_do_not_mutate_anchor(base_cfg):
    before = base_cfg.to_dict()
    changed = base_cfg.with_overrides({"stages.predictor": {"name": "dense"}, "stages.filtering.1.k": 8})
    assert base_cfg.to_dict() == before
    assert changed.stages["predictor"] == {"name": "dense"}
    assert changed.stages["filtering"][1]["k"] == 8


def test_run_id_ignores_seed_and_paths(base_cfg):
    other = base_cfg.with_overrides({"seed": 3, "paths.runs": "/elsewhere", "name": "renamed"})
    assert other.run_id == base_cfg.run_id
    assert base_cfg.with_overrides({"stages.predictor.lam": 0.1}).run_id != base_cfg.run_id


def test_ofat_varies_one_factor_at_a_time(base_cfg):
    ab = Ablation(base_cfg, {"stages.predictor": [{"name": "dense"}, {"name": "residual"}],
                             "stages.generation": [{"name": "boc"}]}, seeds=[0, 1])
    runs = ab.expand()
    assert len(runs) == (1 + 2 + 1) * 2
    for cfg in runs:
        changed = [s for s in ("predictor", "generation") if cfg.stages[s] != base_cfg.stages[s]]
        assert len(changed) <= 1


def test_grid_and_dedup(base_cfg):
    ab = Ablation(base_cfg, {"stages.predictor.lam": [1e-3, 1e-2], "stages.generation.name": ["scores", "boc"]},
                  seeds=[0], mode="grid")
    assert len({c.run_id for c in ab.expand()}) == 4
    # OFAT with the anchor's own value must not duplicate the anchor run
    ofat = Ablation(base_cfg, {"stages.predictor.lam": [1e-3, 1e-2]}, seeds=[0])
    assert len(ofat.expand()) == 2


def test_parse_override_yaml_values():
    assert parse_override("a.b=0.5") == ("a.b", 0.5)
    assert parse_override("a={name: x, k: 2}") == ("a", {"name": "x", "k": 2})
    d = {"a": [{"k": 1}]}
    set_dotted(d, "a.0.k", 2)
    assert d == {"a": [{"k": 2}]}


def test_missing_stage_rejected(base_cfg):
    d = base_cfg.to_dict()
    del d["stages"]["training"]
    with pytest.raises(ValueError, match="missing stages"):
        ExperimentConfig.from_dict(d)


def test_registry_errors():
    r = Registry("thing")
    r.register("a")(dict)
    with pytest.raises(KeyError, match="already registered"):
        r.register("a")(dict)
    with pytest.raises(KeyError, match="Available"):
        r.build({"name": "b"})
    assert r.build({"name": "a", "x": 1}) == {"x": 1}
