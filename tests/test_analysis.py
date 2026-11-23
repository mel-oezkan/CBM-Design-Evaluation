"""Per-arm comparison (`analysis.arms`) and its CLI/JSON output, on hand-made and synthetic results."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from cbm_eval import analysis
from cbm_eval.cli import main
from cbm_eval.pipeline import sweep
from cbm_eval.results import ResultsStore


def _rows(arms: dict[str, list[float]], metric: str = "metric.m", **extra) -> pd.DataFrame:
    rows = [{"run_id": f"r_{arm}", "seed": s, "factor.discovery": arm, "factor.predictor": "dense",
             metric: v, "metric.n_concepts": 20.0, **extra}
            for arm, vals in arms.items() for s, v in enumerate(vals)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("p,dof,expected", [(0.975, 1, 12.706205), (0.975, 2, 4.302653), (0.975, 10, 2.228139),
                                            (0.95, 2, 2.919986), (0.975, 1e5, 1.959964)])
def test_t_quantile_matches_tables(p, dof, expected):
    assert analysis.t_quantile(p, dof) == pytest.approx(expected, abs=1e-5)


def test_mean_ci_and_welch_diff():
    m, lo, hi = analysis.mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and (hi - m) == pytest.approx(4.302653 / math.sqrt(3), abs=1e-5)
    assert math.isnan(analysis.mean_ci([1.0])[1]) and math.isnan(analysis.mean_ci([])[0])
    # Welch: var/n = 1/3 and 4/3 -> dof = (5/3)^2 / ((1/3)^2/2 + (4/3)^2/2) = 50/17
    d, lo, hi = analysis.diff_ci([4.0, 5.0, 6.0], [0.0, 2.0, 4.0])
    assert d == 3.0 and (hi - d) == pytest.approx(analysis.t_quantile(0.975, 50 / 17) * math.sqrt(5 / 3), abs=1e-9)
    assert analysis.diff_ci([1.0, 1.0], [0.5, 0.5]) == (0.5, 0.5, 0.5)  # no variance: a point interval


def test_arms_reference_first_with_effects():
    df = _rows({"sae": [0.1, 0.2, 0.3], "dataset": [0.5, 0.5, 0.6], "llm": [0.6, 0.7, 0.8]})
    t = analysis.arms(df, "discovery", ["m"], reference="dataset")
    assert t["arm"].tolist() == ["dataset", "llm", "sae"]
    ref, sae = t.iloc[0], t.set_index("arm").loc["sae"]
    assert math.isnan(ref["effect"]) and ref["n_seeds"] == 3 and ref["n_concepts"] == 20
    assert sae["effect"] == pytest.approx(0.2 - 1.6 / 3)
    assert sae["effect_ci_low"] < sae["effect"] < sae["effect_ci_high"] < 0
    assert t.attrs == {"factor": "discovery", "reference": "dataset", "level": 0.95, "metrics": ["m"]}
    text = analysis.summarize_arms(t)
    assert "(reference)" in text and "95% CI" in text


def test_arms_reference_by_registered_name_and_errors():
    df = _rows({"dataset": [0.5, 0.6], "sae(k=4)": [0.1, 0.2]})
    assert analysis.arms(df, "factor.discovery", ["metric.m"], reference="sae").attrs["reference"] == "sae(k=4)"
    with pytest.raises(ValueError, match="--reference"):  # a tie: no default reference
        analysis.arms(df, "discovery", ["m"])
    with pytest.raises(ValueError, match="matches 0 arms"):
        analysis.arms(df, "discovery", ["m"], reference="kb")
    with pytest.raises(ValueError, match="not in the results"):
        analysis.arms(df, "discovery", ["nope"], reference="dataset")
    with pytest.raises(ValueError, match="No column 'factor.filtering'"):
        analysis.arms(df, "filtering", ["m"], reference="dataset")
    other = df.assign(**{"factor.predictor": ["dense", "sparse", "dense", "dense"]})
    with pytest.raises(ValueError, match="also vary.*--query"):
        analysis.arms(other, "discovery", ["m"], reference="dataset")
    chain = pd.DataFrame({"run_id": ["a", "a", "b", "b"], "seed": [0, 1, 0, 1], "metric.m": [0.5, 0.6, 0.1, 0.2],
                          "factor.filtering": ["rules>clip(dedupe_sim=0.9)"] * 2 + ["rules>select(k=20)"] * 2})
    t = analysis.arms(chain, "filtering", ["m"], reference="rules>select")
    assert t.attrs["reference"] == "rules>select(k=20)" and t["name"].tolist() == ["rules>select", "rules>clip"]
    mixed = df.assign(run_id=["a", "b", "c", "c"])
    with pytest.raises(ValueError, match="more than one run_id"):
        analysis.arms(mixed, "discovery", ["m"], reference="dataset")


def test_arms_skips_metrics_an_arm_does_not_report():
    df = _rows({"dataset": [0.5, 0.6], "sae": [0.1, 0.2]})
    df["metric.gain"] = [np.nan, np.nan, 0.01, 0.02]  # e.g. no intervention ground truth for one arm
    t = analysis.arms(df, "discovery", ["m", "gain"], reference="dataset")
    assert t[["arm", "metric"]].values.tolist() == [["dataset", "m"], ["sae", "m"], ["sae", "gain"]]
    out = analysis.arms_json(t)
    sae = out["arms"][1]["metrics"]
    assert list(out["arms"][0]["metrics"]) == ["m"]
    assert sae["gain"]["effect"] == {"diff": None, "ci_low": None, "ci_high": None}
    json.dumps(out, allow_nan=False)  # NaN is written as null
    with pytest.raises(ValueError, match="No run reports"):
        analysis.arms(df.assign(**{"metric.gain": np.nan}), "discovery", ["gain"], reference="dataset")


def test_cli_arms_table_and_json_on_synthetic_runs(base_cfg, tmp_path, capsys):
    cfg = base_cfg.with_overrides({"evaluation": [{"name": "shift"}, {"name": "leakage", "repeats": 2}],
                                   "stages.training.epochs": 10})
    arms = [{"name": "dataset"}, {"name": "sae", "n_latents": 32, "k": 4, "epochs": 5}]
    store = ResultsStore(cfg.paths["results"])
    sweep([cfg.with_overrides({"stages.discovery": a, "seed": s}) for a in arms for s in (0, 1)], store,
          continue_on_error=False)
    results = cfg.paths["results"]

    main(["analyze", "--results", results, "--metric", "shift.test.acc"])
    default = capsys.readouterr().out
    assert "metric.shift.test.acc.mean" in default and "arms vs reference" not in default

    out_json = tmp_path / "page" / "arms.json"
    main(["analyze", "--results", results, "--arms", "discovery", "--reference", "dataset",
          "--metrics", "shift.test.wga", "shift.test.acc", "leakage.probe.soft_hard_gap", "--json", str(out_json)])
    text = capsys.readouterr().out
    assert "discovery arms vs reference 'dataset'" in text and f"wrote {out_json}" in text

    page = json.loads(out_json.read_text())
    assert page["schema"] == analysis.ARMS_SCHEMA and page["factor"] == "discovery" and page["reference"] == "dataset"
    by_name = {a["name"]: a for a in page["arms"]}
    assert [a["name"] for a in page["arms"]] == ["dataset", "sae"] and by_name["dataset"]["reference"]
    for a in page["arms"]:
        assert a["n_seeds"] == 2 and a["n_concepts"] == 16 and a["n_discovered"] > 0
        m = a["metrics"]["shift.test.acc"]
        assert m["n"] == 2 and m["ci_low"] <= m["mean"] <= m["ci_high"]
    assert by_name["dataset"]["metrics"]["shift.test.acc"]["effect"] is None
    eff = by_name["sae"]["metrics"]["shift.test.acc"]["effect"]
    sae_acc, ref_acc = (by_name[n]["metrics"]["shift.test.acc"]["mean"] for n in ("sae", "dataset"))
    assert eff["diff"] == pytest.approx(sae_acc - ref_acc)

    with pytest.raises(SystemExit, match="--arms"):
        main(["analyze", "--results", results, "--json", str(out_json)])
    with pytest.raises(SystemExit, match="--reference"):
        main(["analyze", "--results", results, "--arms", "discovery"])
