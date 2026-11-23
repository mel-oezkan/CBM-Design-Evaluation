"""Analysis over the results store: seed aggregation, factor effects models, per-arm comparisons
with confidence intervals, Pareto frontiers, and effective-robustness baselines."""

from __future__ import annotations

import math
from statistics import NormalDist

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
    level (default: the most frequent level, i.e. the anchor in OFAT sweeps; on a tie, the level
    seen first in the results) to another level, holding the others fixed. With ``group`` (e.g. ``factor.dataset``), fits a random-intercept
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


ARMS_SCHEMA = "cbm-eval/arms/v1"


def _t_cdf(t: float, dof: float, n: int = 2000) -> float:
    """``P(T <= t)`` for Student's t with ``dof`` degrees of freedom, ``t >= 0`` (Simpson's rule)."""
    x = np.linspace(0.0, t, n + 1)
    log_c = math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2) - 0.5 * math.log(dof * math.pi)
    f = np.exp(log_c - (dof + 1) / 2 * np.log1p(x * x / dof))
    return 0.5 + t / n / 3 * float(f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())


def t_quantile(p: float, dof: float) -> float:
    """Quantile of Student's t for ``p`` in [0.5, 1) by bisection (no scipy dependency)."""
    if dof > 1e4:
        return NormalDist().inv_cdf(p)
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, dof) < p:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _t_cdf(mid, dof) < p else (lo, mid)
    return (lo + hi) / 2


def mean_ci(values, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and t confidence interval over seeds; the interval is NaN with fewer than 2 values."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return math.nan, math.nan, math.nan
    m = float(v.mean())
    if len(v) < 2:
        return m, math.nan, math.nan
    half = t_quantile(0.5 + level / 2, len(v) - 1) * float(v.std(ddof=1)) / math.sqrt(len(v))
    return m, m - half, m + half


def diff_ci(a, b, level: float = 0.95) -> tuple[float, float, float]:
    """``mean(a) - mean(b)`` with a Welch t interval (unequal variances, unpaired seeds)."""
    a, b = (np.asarray(x, float) for x in (a, b))
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return math.nan, math.nan, math.nan
    d = float(a.mean() - b.mean())
    if len(a) < 2 or len(b) < 2:
        return d, math.nan, math.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return d, d, d
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = t_quantile(0.5 + level / 2, dof) * se
    return d, d - half, d + half


def _metric_col(m: str) -> str:
    return m if m.startswith("metric.") else f"metric.{m}"


def _pick_reference(levels: list[str], counts: pd.Series, factor: str, reference: str | None) -> str:
    if reference is None:
        top = counts[counts == counts.max()].index.tolist()
        if len(top) > 1:
            raise ValueError(f"Every arm of '{factor}' has the same number of runs, so there is no default "
                             f"reference; pass one with --reference (one of {levels})")
        return top[0]
    if reference in levels:
        return reference
    named = [lv for lv in levels if lv.split("(")[0] == reference]
    if len(named) != 1:
        raise ValueError(f"--reference '{reference}' matches {len(named)} arms of '{factor}'; "
                         f"use the full arm label, one of {levels}")
    return named[0]


def arms(df: pd.DataFrame, factor: str, metrics: list[str], reference: str | None = None,
         level: float = 0.95) -> pd.DataFrame:
    """Compare the levels ("arms") of one factor, e.g. ``discovery``, over seeds.

    Returns one row per (arm, metric) that at least one run reports: the seed count, the number of
    concepts in the trained model (``metric.n_concepts``, mean over seeds) and the number proposed by
    discovery (``concept_trace.discovery``), the metric's mean and t interval over seeds, and the
    effect ``arm - reference`` with a Welch t interval (``NaN`` for the reference itself). Unlike
    `effects`, the intervals use t quantiles and per-arm variances, which matters with 3 seeds.
    Every arm must be one configuration: if another factor or the run id varies inside the rows,
    this raises and names the columns to restrict with ``df.query`` (``--query``).
    """
    col = factor if factor.startswith("factor.") else f"factor.{factor}"
    if col not in df:
        raise ValueError(f"No column '{col}' in the results; factors: "
                         f"{[c.removeprefix('factor.') for c in df.columns if c.startswith('factor.')]}")
    others = [f for f in varying_factors(df) if f != col]
    if others:
        raise ValueError(f"Factors other than '{col}' also vary ({others}); select one comparison with "
                         f"--query, e.g. '`tags.ablation` == \"<ablation name>\"'")
    cols = [_metric_col(m) for m in metrics]
    missing = [c for c in cols if c not in df]
    if missing:
        raise ValueError(f"Metrics {missing} are not in the results; available: "
                         f"{sorted(c.removeprefix('metric.') for c in df.columns if c.startswith('metric.'))}")
    runs = df.groupby(col)["run_id"].unique()
    mixed = {lv: list(ids) for lv, ids in runs.items() if len(ids) > 1}
    if mixed:
        raise ValueError(f"Some arms hold more than one run_id (e.g. other evaluation settings): {mixed}; "
                         f"restrict the rows with --query")
    levels = sorted(runs.index)
    ref = _pick_reference(levels, df[col].value_counts(), col, reference)
    out = []
    for lv in [ref] + [lv for lv in levels if lv != ref]:
        rows = df[df[col] == lv]
        n_concepts = rows["metric.n_concepts"].mean() if "metric.n_concepts" in rows else math.nan
        n_disc = rows["concept_trace.discovery"].mean() if "concept_trace.discovery" in rows else math.nan
        for c in cols:
            if rows[c].notna().sum() == 0:
                continue
            m, lo, hi = mean_ci(rows[c], level)
            d, dlo, dhi = (math.nan,) * 3 if lv == ref else diff_ci(rows[c], df.loc[df[col] == ref, c], level)
            out.append({"arm": lv, "name": lv.split("(")[0], "run_id": runs[lv][0], "n_seeds": rows["seed"].nunique(),
                        "n_concepts": n_concepts, "n_discovered": n_disc, "metric": c.removeprefix("metric."),
                        "n": int(rows[c].notna().sum()), "mean": m, "std": float(rows[c].std(ddof=1)),
                        "ci_low": lo, "ci_high": hi, "effect": d, "effect_ci_low": dlo, "effect_ci_high": dhi})
    if not out:
        raise ValueError(f"No run reports any of {metrics}; choose other metrics with --metrics")
    table = pd.DataFrame(out)
    table.attrs.update(factor=col.removeprefix("factor."), reference=ref, level=level,
                       metrics=[c.removeprefix("metric.") for c in cols])
    return table


def _num(x) -> float | int | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    x = float(x)
    return int(x) if x.is_integer() else x


def arms_json(table: pd.DataFrame) -> dict:
    """The `arms` table as a JSON-ready object, schema ``cbm-eval/arms/v1`` (documented in
    *Designing an experiment*, "Compare the arms of one factor"); NaN becomes ``None``."""
    out = []
    for arm, rows in table.groupby("arm", sort=False):
        first = rows.iloc[0]
        ref = arm == table.attrs["reference"]
        out.append({
            "arm": arm, "name": first["name"], "run_id": first["run_id"], "reference": ref,
            "n_seeds": int(first["n_seeds"]), "n_concepts": _num(first["n_concepts"]),
            "n_discovered": _num(first["n_discovered"]),
            "metrics": {r.metric: {"n": int(r.n), "mean": _num(r.mean), "std": _num(r.std),
                                   "ci_low": _num(r.ci_low), "ci_high": _num(r.ci_high),
                                   "effect": None if ref else {"diff": _num(r.effect), "ci_low": _num(r.effect_ci_low),
                                                               "ci_high": _num(r.effect_ci_high)}}
                        for r in rows.itertuples()},
        })
    a = table.attrs
    return {"schema": ARMS_SCHEMA, "factor": a["factor"], "reference": a["reference"], "level": a["level"],
            "metrics": a["metrics"], "arms": out}


def summarize_arms(table: pd.DataFrame, digits: int = 4) -> str:
    """Plain-text per-arm table: seeds, concepts, mean [CI] per metric and the effect vs the reference."""
    a = table.attrs
    pct = f"{a['level']:.0%}"

    def iv(lo, hi):
        return "" if math.isnan(lo) else f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"

    def count(x):
        return "-" if math.isnan(x) else f"{x:g}"

    t = pd.DataFrame({
        "arm": table["arm"], "seeds": table["n_seeds"], "concepts": table["n_concepts"].map(count),
        "discovered": table["n_discovered"].map(count), "metric": table["metric"],
        "mean": table["mean"].map(lambda x: f"{x:.{digits}f}"),
        f"{pct} CI": [iv(lo, hi) for lo, hi in zip(table["ci_low"], table["ci_high"])],
        "effect": ["(reference)" if arm == a["reference"] else "-" if math.isnan(e) else f"{e:+.{digits}f}"
                   for arm, e in zip(table["arm"], table["effect"])],
        f"effect {pct} CI": [iv(lo, hi) for lo, hi in zip(table["effect_ci_low"], table["effect_ci_high"])],
    })
    head = (f"{a['factor']} arms vs reference '{a['reference']}' (mean and t interval over seeds; "
            f"effect = arm - reference, Welch t interval)")
    return head + "\n" + t.to_string(index=False)


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
