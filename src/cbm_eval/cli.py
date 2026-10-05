"""Command line: ``cbm-eval run|sweep|evaluate|analyze|list``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import analysis, registry
from .config import load_ablation, load_config, load_yaml, parse_override
from .pipeline import EVAL_KEYS, evaluate_saved, run, sweep
from .results import ResultsStore


def _overrides(items: list[str] | None) -> dict:
    return dict(parse_override(s) for s in items or [])


def cmd_run(args) -> None:
    cfg = load_config(args.config, _overrides(args.set))
    store = ResultsStore(cfg.paths["results"])
    seeds = args.seeds or [cfg.seed]
    configs = [cfg.with_overrides({"seed": s}) for s in seeds]
    status = sweep(configs, store, skip_existing=not args.force, continue_on_error=False)
    print(json.dumps(status, indent=1, default=float))


def cmd_sweep(args) -> None:
    ab = load_ablation(args.ablation)
    if args.set:
        ab.anchor = ab.anchor.with_overrides(_overrides(args.set))
    configs = ab.expand()
    if args.dry_run:
        for c in configs:
            print(c.run_id, c.seed, {"instances": c.instances or "global", **c.stages})
        print(f"{len(configs)} runs")
        return
    store = ResultsStore(ab.anchor.paths["results"])
    status = sweep(configs, store, skip_existing=not args.force)
    ok = sum(s["status"] == "ok" for s in status)
    skipped = sum(s["status"] == "skipped" for s in status)
    print(f"{ok} ran, {skipped} skipped, {len(status) - ok - skipped} failed")


def cmd_evaluate(args) -> None:
    overrides = {}
    if args.config:  # take the evaluation side of a config, e.g. an anchor that gained eval_datasets
        spec = load_yaml(args.config)
        overrides = {k: spec[k] for k in EVAL_KEYS if k in spec}
    overrides |= _overrides(args.set)
    for run_dir in args.run_dirs:
        cfg = json.loads((Path(run_dir) / "config.json").read_text())
        store = ResultsStore(args.results or cfg["paths"]["results"])
        res = evaluate_saved(run_dir, overrides, store)
        print(json.dumps({"run_dir": run_dir, "run_id": res.cfg.run_id, "seed": res.cfg.seed, **res.metrics},
                         indent=1, default=float))


def cmd_analyze(args) -> None:
    df = ResultsStore(args.results).load()
    if df.empty:
        sys.exit(f"No successful runs in {args.results}")
    if args.query:
        df = df.query(args.query)
    metric = args.metric if args.metric.startswith("metric.") else f"metric.{args.metric}"
    factors = [f if f.startswith("factor.") else f"factor.{f}" for f in args.factors] if args.factors else None
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.max_colwidth", 60):
        print(analysis.aggregate(df, [metric], analysis.varying_factors(df, factors) or ["run_id"]).to_string(index=False))
        print()
        print(analysis.summarize_effects(analysis.effects(df, metric, factors, group=args.group)))
        if args.frontier:
            other = args.frontier if args.frontier.startswith("metric.") else f"metric.{args.frontier}"
            agg = df.groupby("run_id")[[metric, other]].mean().reset_index()
            print("\nPareto frontier:")
            print(analysis.pareto_frontier(agg, metric, other).to_string(index=False))


def cmd_list(args) -> None:
    registry.load_all()
    for kind, reg in registry.ALL.items():
        print(f"{kind:<11} {', '.join(reg.names())}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cbm-eval")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one anchor config")
    r.add_argument("config")
    r.add_argument("--set", nargs="*", help="dotted overrides, e.g. stages.predictor.lam=0.01")
    r.add_argument("--seeds", nargs="*", type=int)
    r.add_argument("--force", action="store_true", help="re-run even if results exist")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="run an ablation file")
    s.add_argument("ablation")
    s.add_argument("--set", nargs="*", help="overrides applied to the anchor")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_sweep)

    e = sub.add_parser("evaluate", help="evaluate saved runs again (e.g. on new eval_datasets) without retraining")
    e.add_argument("run_dirs", nargs="+", help="run directories written by `run`/`sweep`, e.g. runs/<run_id>-s0")
    e.add_argument("--config", help="YAML whose `evaluation:` and `eval_datasets:` replace the saved ones")
    e.add_argument("--set", nargs="*", help="dotted overrides, e.g. eval_datasets.paintings={name: cub_paintings}")
    e.add_argument("--results", help="results file to append to (default: the run's paths.results)")
    e.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("analyze", help="effects model and frontier over the results store")
    a.add_argument("--results", default="results/results.jsonl")
    a.add_argument("--metric", default="shift.test.wga")
    a.add_argument("--factors", nargs="*")
    a.add_argument("--group", help="random-intercept grouping column, e.g. factor.dataset (needs statsmodels)")
    a.add_argument("--frontier", help="second metric for a Pareto frontier")
    a.add_argument("--query", help="pandas query to subset runs")
    a.set_defaults(func=cmd_analyze)

    sub.add_parser("list", help="list registered components").set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
