"""Command line: ``cbm-eval run|sweep|evaluate|analyze|list``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import analysis, cache, registry
from .config import DEFAULT_PATHS, load_ablation, load_config, load_yaml, parse_override
from .pipeline import EVAL_KEYS, evaluate_saved, run, sweep
from .results import ResultsStore
from .tracking import WandbTracker


def _overrides(items: list[str] | None) -> dict:
    return dict(parse_override(s) for s in items or [])


def _tracker(args) -> WandbTracker | None:
    return WandbTracker(args.wandb, group=args.wandb_group, mode=args.wandb_mode) if args.wandb else None


def _add_tracking(parser) -> None:
    g = parser.add_argument_group("live tracking (needs the `tracking` extra)")
    g.add_argument("--wandb", metavar="PROJECT", help="stream each run's epoch history and metrics to this W&B project")
    g.add_argument("--wandb-group", help="W&B group (default: the config's name)")
    g.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], help="W&B mode (default: W&B's)")


def cmd_run(args) -> None:
    cfg = load_config(args.config, _overrides(args.set))
    store = ResultsStore(cfg.paths["results"])
    seeds = args.seeds or [cfg.seed]
    configs = [cfg.with_overrides({"seed": s}) for s in seeds]
    status = sweep(configs, store, skip_existing=not args.force, continue_on_error=False, tracker=_tracker(args))
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
    status = sweep(configs, store, skip_existing=not args.force, tracker=_tracker(args))
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
    if args.json and not args.arms:
        sys.exit("--json writes the per-arm table: also pass --arms <factor>, e.g. --arms discovery")
    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.max_colwidth", 60):
        if args.arms:
            try:
                table = analysis.arms(df, args.arms, args.metrics or [metric], args.reference)
            except ValueError as e:
                sys.exit(str(e))
            print(analysis.summarize_arms(table))
            if args.json:
                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json).write_text(json.dumps(analysis.arms_json(table), indent=1) + "\n")
                print(f"\nwrote {args.json}")
        else:
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


def cmd_cache_prune(args) -> None:
    removed = cache.prune(args.cache, args.unused_days, dry_run=args.dry_run)
    verb = "would delete" if args.dry_run else "deleted"
    for path in removed:
        print(f"{verb} {path}")
    print(f"{verb} {len(removed)} cache file(s) unused for {args.unused_days:g} days")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cbm-eval")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one anchor config")
    r.add_argument("config")
    r.add_argument("--set", nargs="*", help="dotted overrides, e.g. stages.predictor.lam=0.01")
    r.add_argument("--seeds", nargs="*", type=int)
    r.add_argument("--force", action="store_true", help="re-run even if results exist")
    _add_tracking(r)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="run an ablation file")
    s.add_argument("ablation")
    s.add_argument("--set", nargs="*", help="overrides applied to the anchor")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--force", action="store_true")
    _add_tracking(s)
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
    g = a.add_argument_group("per-arm comparison of one factor (replaces the table and effects model above)")
    g.add_argument("--arms", metavar="FACTOR", help="factor whose levels are compared, e.g. discovery")
    g.add_argument("--reference", help="arm the others are compared against: its label or registered name, "
                                       "e.g. dataset (default: the arm with the most runs)")
    g.add_argument("--metrics", nargs="+", help="metrics in the per-arm table (default: --metric)")
    g.add_argument("--json", metavar="PATH", help="also write the per-arm table as JSON (schema cbm-eval/arms/v1)")
    a.set_defaults(func=cmd_analyze)

    sub.add_parser("list", help="list registered components").set_defaults(func=cmd_list)

    c = sub.add_parser("cache", help="manage the on-disk cache").add_subparsers(dest="cache_cmd", required=True)
    cp = c.add_parser("prune", help="delete augmented views and patch features that no run used for some days "
                                    "(clean features and LLM/VLM caches stay)")
    cp.add_argument("--cache", default=DEFAULT_PATHS["cache"], help="cache directory (paths.cache)")
    cp.add_argument("--unused-days", type=float, default=cache.DEFAULT_UNUSED_DAYS)
    cp.add_argument("--dry-run", action="store_true", help="list what would be deleted, delete nothing")
    cp.set_defaults(func=cmd_cache_prune)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
