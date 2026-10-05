"""Results store: one JSON line per (run, seed), flattened to a DataFrame for analysis."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import STAGE_ORDER, ExperimentConfig
from .utils import stable_hash


def _value(v: Any) -> str:
    if isinstance(v, list) and len(v) > 4:
        return f"[{len(v)} items #{stable_hash(v, 6)}]"
    return json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)


def describe(spec: dict[str, Any] | list | str) -> str:
    """Compact, stable label for a stage spec, e.g. ``sparse(lam=0.001)`` or ``rules>select(k=50)``."""
    if isinstance(spec, list):
        return ">".join(describe(s) for s in spec)
    if isinstance(spec, str):
        return spec
    params = {k: v for k, v in spec.items() if k != "name"}
    if not params:
        return spec["name"]
    inner = ",".join(f"{k}={_value(v)}" for k, v in sorted(params.items()))
    return f"{spec['name']}({inner})"


def factors(cfg: ExperimentConfig) -> dict[str, str]:
    out = {"dataset": describe(cfg.dataset), "backbone": describe(cfg.backbone),
           "instances": describe(cfg.instances or "global")}
    out |= {stage: describe(cfg.stages[stage]) for stage in STAGE_ORDER}
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_row(cfg: ExperimentConfig, metrics: dict[str, float], train_log: dict[str, float],
             trace: list[tuple[str, int]], duration: float) -> dict[str, Any]:
    return {
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "name": cfg.name,
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "duration_s": round(duration, 2),
        "factor": factors(cfg),
        "tags": cfg.tags,
        "metric": metrics,
        "train": train_log,
        "concept_trace": dict(trace),
        "config": cfg.to_dict(),
    }


class ResultsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def has(self, run_id: str, seed: int) -> bool:
        return any(r["run_id"] == run_id and r["seed"] == seed and r.get("status") == "ok" for r in self._rows())

    def append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=float) + "\n")

    def append_failure(self, cfg: ExperimentConfig, error: Exception) -> None:
        self.append({"run_id": cfg.run_id, "seed": cfg.seed, "name": cfg.name, "status": "error",
                     "error": f"{type(error).__name__}: {error}", "factor": factors(cfg), "tags": cfg.tags,
                     "config": cfg.to_dict(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def load(self, include_failed: bool = False) -> pd.DataFrame:
        """Flat frame with columns like ``factor.discovery`` and ``metric.shift.test.wga``.

        The last row wins when a (run_id, seed) pair was recorded more than once.
        """
        rows = [r for r in self._rows() if include_failed or r.get("status") == "ok"]
        if not rows:
            return pd.DataFrame()
        df = pd.json_normalize([{k: v for k, v in r.items() if k != "config"} for r in rows])
        return df.drop_duplicates(["run_id", "seed"], keep="last").reset_index(drop=True)
