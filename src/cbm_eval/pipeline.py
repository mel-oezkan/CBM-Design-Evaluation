"""Pipeline builder: assemble a CBM from a config via the registries, train it, evaluate it."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import registry
from .config import _NON_IDENTITY_KEYS, ExperimentConfig
from .context import Context
from .data.base import SPLITS
from .model import CBM, TrainedCBM
from .registry import (ALIGNMENT, BACKBONES, DATASETS, DISCOVERY, EVALUATION, FILTERING, GENERATION,
                       INSTANCES, PREDICTOR, TRAINING)
from .results import ResultsStore, make_row
from .structures import ConceptSet
from .utils import seed_everything

log = logging.getLogger(__name__)

# Config keys that only change how a trained CBM is evaluated; ``evaluate_saved`` may change these.
EVAL_KEYS = ("evaluation", "eval_datasets")


@dataclass
class RunResult:
    cfg: ExperimentConfig
    cbm: TrainedCBM
    metrics: dict[str, float]
    concept_trace: list[tuple[str, int]]  # (stage, n_concepts) after each step
    run_dir: Path | None = None


class PipelineBuilder:
    """Builds each stage from the config. The steps are public so notebooks can run them one by one."""

    def __init__(self, cfg: ExperimentConfig):
        registry.load_all()
        self.cfg = cfg
        st = cfg.stages
        self.discovery = DISCOVERY.build(st["discovery"])
        self.filters = [FILTERING.build(f) for f in st["filtering"]]
        self.alignment = ALIGNMENT.build(st["alignment"])
        self.generation = GENERATION.build(st["generation"])
        self.predictor = PREDICTOR.build(st["predictor"])
        self.training = TRAINING.build(st["training"])
        self.evaluators = {e["name"]: EVALUATION.build(e) for e in cfg.evaluation}
        self.instances = INSTANCES.build(cfg.instances) if cfg.instances else None
        self._check_datasets()
        if self.instances is not None and not self.predictor.supports_bags:
            raise ValueError(f"`instances: {cfg.instances['name']}` gives each image a bag of instances; "
                             f"use a bag-aware predictor such as {{name: mil}}, not '{st['predictor']['name']}'")
        if self.training.trains_backbone:
            patchy = [n for n, e in self.evaluators.items() if e.needs_patches]
            if self.instances is not None or patchy:
                raise ValueError("`stages.training.finetune` changes the backbone, but "
                                 + ("`instances:`" if self.instances is not None else f"evaluators {patchy}")
                                 + " read frozen patch features; drop them or remove `finetune:`")

    def _check_datasets(self) -> None:
        name = self.cfg.dataset["name"]
        if DATASETS.get(name).splits != SPLITS:
            raise ValueError(f"Dataset '{name}' is test-only; train on its source dataset and add it under "
                             f"`eval_datasets: {{<alias>: {{name: {name}, ...}}}}`")
        for alias, spec in (self.cfg.eval_datasets or {}).items():
            if alias in SPLITS:
                raise ValueError(f"`eval_datasets.{alias}` shadows the training dataset's '{alias}' split; rename it")
            if "test" not in DATASETS.get(spec["name"]).splits:
                raise ValueError(f"`eval_datasets.{alias}`: dataset '{spec['name']}' has no test split")

    def context(self) -> Context:
        dataset = DATASETS.build(self.cfg.dataset)
        backbone = BACKBONES.build(self.cfg.backbone)
        teacher = BACKBONES.build(self.cfg.teacher) if self.cfg.teacher else None
        eval_datasets = {alias: DATASETS.build(spec) for alias, spec in (self.cfg.eval_datasets or {}).items()}
        need_patches = any(e.needs_patches for e in self.evaluators.values()) or \
            bool(self.instances is not None and self.instances.needs_patches)
        return Context(self.cfg, dataset, backbone, teacher, need_patches=need_patches, instances=self.instances,
                       eval_datasets=eval_datasets)

    def concepts(self, ctx: Context, trace: list[tuple[str, int]] | None = None) -> ConceptSet:
        concepts = self.discovery.discover(ctx)
        if trace is not None:
            trace.append(("discovery", len(concepts)))
        for spec, f in zip(self.cfg.stages["filtering"], self.filters):
            concepts = f.filter(concepts, ctx)
            if trace is not None:
                trace.append((f"filter:{spec['name']}", len(concepts)))
            if len(concepts) == 0:
                raise RuntimeError(f"Filter '{spec['name']}' removed every concept")
        ctx.concepts = concepts
        return concepts

    def train(self, ctx: Context, concepts: ConceptSet) -> TrainedCBM:
        ctx.use_encoder(None)  # a reused Context must not carry a previous run's fine-tuned encoder
        aligned = self.alignment.fit(concepts, ctx)
        layer = self.generation.build(ctx.feat_dim, aligned)
        head = self.predictor.build(layer.rep_dim, ctx.num_classes, ctx.feat_dim)
        encoder = None
        if self.training.trains_backbone:
            if not ctx.backbone.trainable:
                raise ValueError(f"`stages.training.finetune` needs a trainable backbone; "
                                 f"'{self.cfg.backbone['name']}' is frozen-only (use e.g. inception)")
            encoder = ctx.backbone.encoder()
            train_log = self.training.fit(layer, head, aligned, ctx, encoder=encoder)
            ctx.use_encoder(encoder.eval(), num_workers=self.training.num_workers)  # inputs = fine-tuned features
        else:
            train_log = self.training.fit(layer, head, aligned, ctx)
        train_log |= {f"head.{k}": v for k, v in head.stats().items()}
        cbm = TrainedCBM(CBM(layer, head).eval(), aligned, self.cfg.to_dict(), train_log, encoder=encoder)
        cbm.cache_scores(ctx)
        return cbm

    def load(self, ctx: Context, run_dir: str | Path) -> TrainedCBM:
        """The CBM that ``run`` saved to ``run_dir``, rebuilt without retraining. Alignment is refitted
        on the saved concepts (which only rebuilds its lazy targets) and the weights are loaded."""
        run_dir = Path(run_dir)
        saved = json.loads((run_dir / "concepts.json").read_text())
        vectors = torch.load(run_dir / "concept_vectors.pt") if saved["has_vectors"] else None
        concepts = ConceptSet(saved["names"], vectors, saved["meta"])
        ctx.concepts = concepts
        ctx.use_encoder(None)
        aligned = self.alignment.fit(concepts, ctx)
        if aligned.concepts.names != concepts.names:
            raise RuntimeError(f"Alignment '{self.cfg.stages['alignment']['name']}' changed the concepts saved in "
                               f"{run_dir}; the run cannot be rebuilt from its config")
        layer = self.generation.build(ctx.feat_dim, aligned)
        head = self.predictor.build(layer.rep_dim, ctx.num_classes, ctx.feat_dim)
        model = CBM(layer, head)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        encoder = None
        if (run_dir / "encoder.pt").exists():
            encoder = ctx.backbone.encoder()
            encoder.load_state_dict(torch.load(run_dir / "encoder.pt", map_location="cpu"))
            ctx.use_encoder(encoder.to(ctx.device).eval(), num_workers=self.training.num_workers)
        cbm = TrainedCBM(model.eval(), aligned, self.cfg.to_dict(), json.loads((run_dir / "train_log.json").read_text()),
                         torch.load(run_dir / "concept_scores.pt"), encoder=encoder)
        cbm.cache_scores(ctx, tuple(s for s in ctx.splits if s not in cbm.concept_scores))
        return cbm

    def evaluate(self, cbm: TrainedCBM, ctx: Context) -> dict[str, float]:
        metrics: dict[str, float] = {"n_concepts": float(len(cbm.concepts))}
        if ctx.instance_source is not None:
            mask = ctx.instances("test").mask
            metrics["instances.per_image"] = float(mask.sum(1).float().mean())
            metrics["instances.max"] = float(mask.shape[1])
        for name, ev in self.evaluators.items():
            metrics |= {f"{name}.{k}": v for k, v in ev.evaluate(cbm, ctx).items()}
        return metrics


def run(cfg: ExperimentConfig, store: ResultsStore | None = None, save: bool = True,
        ctx: Context | None = None) -> RunResult:
    """Run one config end to end and (optionally) append one row to the results store."""
    seed_everything(cfg.seed)
    start = time.time()
    builder = PipelineBuilder(cfg)
    ctx = ctx or builder.context()
    trace: list[tuple[str, int]] = []
    concepts = builder.concepts(ctx, trace)
    cbm = builder.train(ctx, concepts)
    trace.append(("alignment", len(cbm.concepts)))
    metrics = builder.evaluate(cbm, ctx)
    run_dir = cbm.save(Path(cfg.paths["runs"]) / f"{cfg.run_id}-s{cfg.seed}") if save else None
    if run_dir is not None:
        (run_dir / "concept_trace.json").write_text(json.dumps(trace))
    if store is not None:
        store.append(make_row(cfg, metrics, cbm.train_log, trace, time.time() - start))
    log.info("run %s seed %d: %s", cfg.run_id, cfg.seed, _headline(metrics))
    return RunResult(cfg, cbm, metrics, trace, run_dir)


def evaluate_saved(run_dir: str | Path, overrides: dict[str, Any] | None = None,
                   store: ResultsStore | None = None) -> RunResult:
    """Evaluate a CBM saved by ``run`` again, with ``overrides`` applied to its saved config.

    Only ``evaluation:`` and ``eval_datasets:`` (and keys outside the run id, such as ``paths``) may
    change, because the model is not retrained. The row is the one ``run`` would write for the new
    config, under that config's run id, so a later ``sweep`` of the same config skips it.
    """
    run_dir = Path(run_dir)
    saved = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    cfg = saved.with_overrides(overrides or {})
    old, new = saved.to_dict(), cfg.to_dict()
    changed = sorted(k for k in old.keys() | new.keys()
                     if k not in EVAL_KEYS + _NON_IDENTITY_KEYS and old.get(k) != new.get(k))
    if changed:
        raise ValueError(f"Evaluating a saved run may only change {list(EVAL_KEYS)}, but {changed} differ from "
                         f"{run_dir}; train that config with `cbm-eval run` instead")
    seed_everything(cfg.seed)
    start = time.time()
    builder = PipelineBuilder(cfg)
    ctx = builder.context()
    cbm = builder.load(ctx, run_dir)
    metrics = builder.evaluate(cbm, ctx)
    trace_file = run_dir / "concept_trace.json"
    trace = [tuple(t) for t in json.loads(trace_file.read_text())] if trace_file.exists() else \
        [("alignment", len(cbm.concepts))]
    if store is not None:
        row = make_row(cfg, metrics, cbm.train_log, trace, time.time() - start)
        store.append(row | {"evaluated_from": str(run_dir)})
    log.info("evaluated %s as run %s seed %d: %s", run_dir, cfg.run_id, cfg.seed, _headline(metrics))
    return RunResult(cfg, cbm, metrics, trace, run_dir)


def sweep(configs: list[ExperimentConfig], store: ResultsStore, skip_existing: bool = True,
          save: bool = True, continue_on_error: bool = True) -> list[dict[str, Any]]:
    """Run many configs, reusing feature caches (they key on dataset+backbone, not on the run)."""
    status = []
    for i, cfg in enumerate(configs, 1):
        if skip_existing and store.has(cfg.run_id, cfg.seed):
            status.append({"run_id": cfg.run_id, "seed": cfg.seed, "status": "skipped"})
            continue
        log.info("[%d/%d] %s seed %d", i, len(configs), cfg.run_id, cfg.seed)
        try:
            res = run(cfg, store, save=save)
            status.append({"run_id": cfg.run_id, "seed": cfg.seed, "status": "ok", **res.metrics})
        except Exception as e:  # keep the sweep going; record the failure for the summary
            if not continue_on_error:
                raise
            log.exception("run %s failed", cfg.run_id)
            store.append_failure(cfg, e)
            status.append({"run_id": cfg.run_id, "seed": cfg.seed, "status": f"error: {e}"})
    return status


def _headline(metrics: dict[str, float]) -> str:
    keys = ["shift.test.acc", "shift.test.wga", "leakage.intervention.gain", "localization.pointing_game"]
    return ", ".join(f"{k}={metrics[k]:.3f}" for k in keys if k in metrics)
