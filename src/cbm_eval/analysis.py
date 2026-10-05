"""Analysis over the results store: seed aggregation, factor effects models, Pareto frontiers,
and effective-robustness baselines."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import STAGE_ORDER

DEFAULT_FACTORS = ["factor.instances"] + [f"factor.{s}" for s in STAGE_ORDER]


def varying_factors(df: pd.DataFrame, candidates: list[str] | None = None) -> list[str]:
    cols = candidates or [c for c in df.columns if c.startswith("factor.")]
    return [c for c in cols if c in df and df[c].nunique() > 1]


def aggregate(df: pd.DataFrame, metrics: list[str], by: list[str] | None = None) -> pd.DataFrame:
    """Mean, std and seed count of each metric per configuration."""
    by = by or ["run_id"] + varying_factors(df)
    agg = df.groupby(by, dropna=False)[metrics].agg(["mean", "std", "count"])
    agg.columns = [f"{m}.{s}" for m, s in agg.columns]
    return agg.reset_index()


def effects(df: pd.DataFrame, metric: str, factors: list[str] | None = None,
            reference: dict[str, str] | None = None, group: str | None = None) -> pd.DataFrame:
    """Additive effects model ``metric ~ factor_1 + ... + factor_n`` with treatment coding.

    Each coefficient is the change in the metric from switching one factor from its reference
    level (default: the most frequent level, i.e. the anchor in OFAT sweeps) to another level,
    holding the others fixed. With ``group`` (e.g. ``factor.dataset``), fits a random-intercept
    mixed model via statsmodels instead, so effects generalize across that grouping.
    """
    factors = varying_factors(df, factors or DEFAULT_FACTORS)
    data = df[[metric] + factors + ([group] if group else [])].dropna(subset=[metric]).copy()
    reference = dict(reference or {})
    for f in factors:
        reference.setdefault(f, data[f].value_counts().idxmax())

    if group:
        return _mixed_effects(data, metric, factors, reference, group)

    cols, names = [np.ones(len(data))], ["intercept"]
    for f in factors:
        for level in sorted(set(data[f]) - {reference[f]}):
            cols.append((data[f] == level).to_numpy(float))
            names.append(f"{f.removeprefix('factor.')}={level}")
    X, y = np.column_stack(cols), data[metric].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    sigma2 = float(((y - X @ beta) ** 2).sum()) / dof
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(X.T @ X)) * sigma2, 0, None))
    out = pd.DataFrame({"term": names, "coef": beta, "se": se})
    out["t"] = out["coef"] / out["se"].replace(0, np.nan)
    out["ci_low"], out["ci_high"] = out["coef"] - 1.96 * out["se"], out["coef"] + 1.96 * out["se"]
    out.attrs.update(reference=reference, n=len(y), dof=dof, metric=metric)
    return out


def _mixed_effects(data, metric, factors, reference, group) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    rename = {c: c.replace(".", "_") for c in [metric, group] + factors}
    data = data.rename(columns=rename)
    terms = [f"C({rename[f]}, Treatment(reference={reference[f]!r}))" for f in factors]
    model = smf.mixedlm(f"{rename[metric]} ~ " + " + ".join(terms), data, groups=data[rename[group]]).fit()
    ci = model.conf_int()
    out = pd.DataFrame({"term": model.params.index, "coef": model.params.values, "se": model.bse.values,
                        "ci_low": ci[0].values, "ci_high": ci[1].values})
    out.attrs.update(reference=reference, n=len(data), metric=metric, group=group)
    return out


def pareto_frontier(df: pd.DataFrame, x: str, y: str, maximize: tuple[bool, bool] = (True, True)) -> pd.DataFrame:
    """Rows not dominated on (x, y); e.g. accuracy vs. interpretability, WGA vs. leakage."""
    sx, sy = (1 if maximize[0] else -1), (1 if maximize[1] else -1)
    pts = df[[x, y]].to_numpy(float) * np.array([sx, sy])
    keep = []
    for i, p in enumerate(pts):
        if np.isnan(p).any():
            continue
        dominated = np.any(np.all(pts >= p, axis=1) & np.any(pts > p, axis=1))
        if not dominated:
            keep.append(i)
    return df.iloc[keep].sort_values(x)


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_robustness_baseline(df: pd.DataFrame, id_col: str, ood_col: str) -> tuple[float, float]:
    """Logit-linear fit of OOD on ID accuracy over baseline runs; returns (slope, intercept)."""
    slope, intercept = np.polyfit(_logit(df[id_col].to_numpy(float)), _logit(df[ood_col].to_numpy(float)), 1)
    return float(slope), float(intercept)


def add_effective_robustness(df: pd.DataFrame, id_col: str, ood_col: str, slope: float, intercept: float,
                             out_col: str = "effective_robustness") -> pd.DataFrame:
    expected = 1 / (1 + np.exp(-(slope * _logit(df[id_col].to_numpy(float)) + intercept)))
    return df.assign(**{out_col: df[ood_col].to_numpy(float) - expected})


def summarize_effects(table: pd.DataFrame, digits: int = 4) -> str:
    lines = [f"{table.attrs.get('metric', '')}  (n={table.attrs.get('n')}, reference levels: "
             f"{ {k.removeprefix('factor.'): v for k, v in table.attrs.get('reference', {}).items()} })"]
    for r in table.itertuples():
        lines.append(f"  {r.term:<60} {r.coef:+.{digits}f}  [{r.ci_low:+.{digits}f}, {r.ci_high:+.{digits}f}]"
                     if not math.isnan(r.se) else f"  {r.term:<60} {r.coef:+.{digits}f}")
    return "\n".join(lines)
