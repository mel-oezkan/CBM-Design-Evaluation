"""Pipeline builder: assemble a CBM from a config via the registries, train it, evaluate it."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import registry
from .config import ExperimentConfig
from .context import Context
from .data.base import SPLITS
from .model import CBM, TrainedCBM
from .registry import (ALIGNMENT, BACKBONES, DATASETS, DISCOVERY, EVALUATION, FILTERING, GENERATION,
                       INSTANCES, PREDICTOR, TRAINING)
from .results import ResultsStore, make_row
from .structures import ConceptSet
from .utils import seed_everything

log = logging.getLogger(__name__)


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
    if store is not None:
        store.append(make_row(cfg, metrics, cbm.train_log, trace, time.time() - start))
    log.info("run %s seed %d: %s", cfg.run_id, cfg.seed, _headline(metrics))
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
