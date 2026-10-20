"""Per-epoch training history of a run, and optional live tracking of it on Weights & Biases.

Training stages report one row per epoch and phase through ``ctx.history`` (``History.phase``);
``run`` saves the rows as ``history.jsonl`` in the run directory. A ``WandbTracker`` streams the
same rows while the run trains and stores the run's metrics as the W&B run's summary. Neither draws
from the run's random generators or changes a gradient, so the run id and the results stay the same.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from .config import ExperimentConfig
from .results import factors
from .utils import rng_preserved


class History:
    """Rows ``{"phase", "epoch", "train_loss", "val_loss", ..., "best"}``, in the order they were
    recorded, each also handed to ``tracker`` (if any) as it arrives."""

    def __init__(self, tracker: "WandbTracker | None" = None):
        self.rows: list[dict[str, Any]] = []
        self.tracker = tracker

    def phase(self, name: str) -> Callable[..., None]:
        """A callback that records one epoch of phase ``name`` (``concept``, ``head_phase0``, ``joint``)."""
        return lambda **metrics: self.record(name, **metrics)

    def record(self, phase: str, **metrics: Any) -> None:
        row = {"phase": phase} | {k: v for k, v in metrics.items() if v is not None}
        self.rows.append(row)
        if self.tracker is not None:
            self.tracker.log(row)

    def save(self, path: str | Path) -> None:
        Path(path).write_text("".join(json.dumps(r, allow_nan=True) + "\n" for r in self.rows))


class WandbTracker:
    """Streams a run's ``History`` to W&B: one W&B run per ``(config, seed)``, named
    ``<name>-<run_id>-s<seed>`` in ``group`` (default: the config's name), with the run id, seed and
    factors as its config. Each phase gets its own epoch axis (``<phase>/epoch``). ``finish`` writes
    the run's metrics and training log as the summary, or marks the W&B run failed.

    ``mode`` is W&B's (``online``, ``offline``, ``disabled``; None: W&B's default). ``client`` replaces
    the ``wandb`` module (tests pass a stub); by default it is imported on first use, so install the
    ``tracking`` extra."""

    def __init__(self, project: str, group: str | None = None, entity: str | None = None,
                 mode: str | None = None, client: Any = None):
        self.project, self.group, self.entity, self.mode, self.client = project, group, entity, mode, client
        self.run, self._phases = None, set()

    def start(self, cfg: ExperimentConfig) -> None:
        if self.client is None:
            import wandb

            self.client = wandb
        self._phases = set()
        with rng_preserved():
            self.run = self.client.init(project=self.project, entity=self.entity, group=self.group or cfg.name,
                                        name=f"{cfg.name}-{cfg.run_id}-s{cfg.seed}", mode=self.mode,
                                        config={"run_id": cfg.run_id, "seed": cfg.seed, **factors(cfg)},
                                        reinit="finish_previous")

    def log(self, row: dict[str, Any]) -> None:
        phase = row["phase"]
        with rng_preserved():
            if phase not in self._phases:
                self._phases.add(phase)
                self.run.define_metric(f"{phase}/epoch")
                self.run.define_metric(f"{phase}/*", step_metric=f"{phase}/epoch")
            self.run.log({f"{phase}/{k}": float(v) for k, v in row.items() if k != "phase"})

    def finish(self, metrics: dict[str, float] | None, train_log: dict[str, float] | None = None) -> None:
        """Close the W&B run: with the run's ``metrics`` as summary, or (None) as failed."""
        if self.run is None:
            return
        with rng_preserved():
            if metrics is not None:
                summary = metrics | {f"train/{k}": v for k, v in (train_log or {}).items()}
                self.run.summary.update({k: v for k, v in summary.items()
                                         if not (isinstance(v, float) and math.isnan(v))})
            self.run.finish(exit_code=0 if metrics is not None else 1)
        self.run = None
