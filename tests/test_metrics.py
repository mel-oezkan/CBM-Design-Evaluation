import numpy as np
import pandas as pd
import pytest
import torch

from cbm_eval.analysis import effects, fit_robustness_baseline, pareto_frontier
from cbm_eval.data.base import keypoint_mask
from cbm_eval.evaluation.shift import effective_robustness, group_accuracies
from cbm_eval.stages.llm import parse_list
from cbm_eval.stages.predictor import SparseHead


def test_group_accuracies_and_wga():
    pred = torch.tensor([0, 0, 1, 1, 0, 1])
    y = torch.tensor([0, 0, 1, 1, 1, 0])
    g = torch.tensor([0, 0, 1, 1, 2, 2])
    accs = group_accuracies(pred, y, g)
    assert accs == {0: 1.0, 1: 1.0, 2: 0.0}


def test_effective_robustness_on_trend_is_zero():
    slope, intercept = 1.2, -0.5
    id_acc = 0.8
    logit = np.log(id_acc / (1 - id_acc))
    ood = 1 / (1 + np.exp(-(slope * logit + intercept)))
    assert effective_robustness(id_acc, ood, slope, intercept) == pytest.approx(0, abs=1e-9)
    df = pd.DataFrame({"id": [0.6, 0.7, 0.8, 0.9]})
    df["ood"] = 1 / (1 + np.exp(-(slope * np.log(df.id / (1 - df.id)) + intercept)))
    s, i = fit_robustness_baseline(df, "id", "ood")
    assert (s, i) == pytest.approx((slope, intercept), abs=1e-6)


def test_pareto_frontier():
    df = pd.DataFrame({"a": [1, 2, 3, 2], "b": [3, 2, 1, 1]})
    assert sorted(pareto_frontier(df, "a", "b").index.tolist()) == [0, 1, 2]
    assert pareto_frontier(df, "a", "b", maximize=(True, False)).index.tolist() == [2]


def test_effects_recovers_additive_model():
    rng = np.random.default_rng(0)
    rows = []
    for a in ["x", "y", "z"]:
        for b in ["p", "q"]:
            for _ in range(5):
                rows.append({"factor.a": a, "factor.b": b,
                             "m": 0.5 + {"x": 0, "y": 0.1, "z": -0.2}[a] + {"p": 0, "q": 0.05}[b] + rng.normal(0, 1e-3)})
    table = effects(pd.DataFrame(rows), "m", ["factor.a", "factor.b"], reference={"factor.a": "x", "factor.b": "p"})
    coef = dict(zip(table.term, table.coef))
    assert coef["a=y"] == pytest.approx(0.1, abs=5e-3)
    assert coef["a=z"] == pytest.approx(-0.2, abs=5e-3)
    assert coef["b=q"] == pytest.approx(0.05, abs=5e-3)


def test_keypoint_mask():
    m = keypoint_mask({0: (10.0, 10.0, 100.0, 100.0)}, n_concepts=2, grid=4, radius=0)
    assert m.shape == (2, 16) and m[0, 0] and m[0].sum() == 1 and m[1].sum() == 0


def test_sparse_prox_step_zeros_small_weights():
    head = SparseHead(4, 2, lam=1.0, alpha=1.0, lr=0.1)
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[0.05, -0.5, 0.0, 2.0], [0.0, 0.0, -0.01, 0.3]]))
    head.proximal_step()
    w = head.linear.weight.detach()
    assert w[0, 0] == 0 and w[1, 2] == 0
    assert w[0, 1].item() == pytest.approx(-0.4) and w[0, 3].item() == pytest.approx(1.9)


def test_parse_list():
    text = "Here you go:\n- Long beak\n2. webbed feet.\n* \"Water\"\n\n"
    assert parse_list(text) == ["long beak", "webbed feet", "water"]
