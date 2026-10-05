"""Experiment configs: anchors (a full CBM recipe) and ablations (sweeps over an anchor).

An anchor YAML fully specifies one run::

    name: waterbirds_lfcbm
    dataset: {name: waterbirds, root: data/waterbirds}
    backbone: {name: clip, model: ViT-B-16, pretrained: openai}
    stages:
      discovery: {name: llm, per_class: 10}
      filtering: [{name: rules}, {name: clip}, {name: select, k: 50}]
      generation: {name: scores}
      alignment: {name: clip}
      predictor: {name: sparse, lam: 0.0007}
      training: {name: sequential}
    evaluation: [{name: shift}, {name: leakage}]

Bag-based CBMs (SEG-MIL-CBM) add a top-level ``instances: {name: patches | grounded_sam | ...}``
that turns each image into a set of instances. Without it each image is one feature vector.

``eval_datasets: {alias: {name: ..., **kw}}`` adds test-only domains to a run. The model is trained
on ``dataset`` alone; each alias names that dataset's ``test`` split, with classes and concepts matched
to the training dataset's by name, and evaluators score it when listed in their splits::

    eval_datasets: {paintings: {name: cub_paintings, root: data/cub_paintings}}
    evaluation: [{name: shift, splits: [val, test, paintings]}]

An ablation YAML points at an anchor and varies dotted keys::

    anchor: ../anchors/waterbirds_lfcbm.yaml
    mode: ofat            # one-factor-at-a-time (default) or grid
    seeds: [0, 1, 2]
    factors:
      stages.discovery: [{name: llm}, {name: kb}, {name: sae, n_latents: 256}]
      stages.predictor.lam: [0.0001, 0.001]
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STAGE_ORDER = ("discovery", "filtering", "generation", "alignment", "predictor", "training")

# Keys that do not change *what* is being measured, so they are excluded from the run id.
_NON_IDENTITY_KEYS = ("name", "seed", "paths", "device", "tags")

DEFAULT_PATHS = {"cache": ".cache", "runs": "runs", "results": "results/results.jsonl"}


@dataclass
class ExperimentConfig:
    dataset: dict[str, Any]
    backbone: dict[str, Any]
    stages: dict[str, Any]
    evaluation: list[dict[str, Any]] = field(default_factory=lambda: [{"name": "shift"}])
    teacher: dict[str, Any] | None = None
    instances: dict[str, Any] | None = None
    eval_datasets: dict[str, dict[str, Any]] | None = None
    name: str = "experiment"
    seed: int = 0
    device: str = "auto"
    paths: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PATHS))
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [s for s in STAGE_ORDER if s not in self.stages]
        if missing:
            raise ValueError(f"Config '{self.name}' is missing stages: {missing}")
        unknown = set(self.stages) - set(STAGE_ORDER)
        if unknown:
            raise ValueError(f"Config '{self.name}' has unknown stages: {sorted(unknown)}")
        if isinstance(self.stages["filtering"], dict):
            self.stages["filtering"] = [self.stages["filtering"]]
        self.paths = {**DEFAULT_PATHS, **self.paths}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return cls(**copy.deepcopy(d))

    def to_dict(self) -> dict[str, Any]:
        """A deep copy, so callers can mutate it without touching this config.

        ``instances`` and ``eval_datasets`` are left out when unset so configs written before they
        existed keep their run_id.
        """
        d = {
            "name": self.name,
            "seed": self.seed,
            "device": self.device,
            "dataset": self.dataset,
            "backbone": self.backbone,
            "teacher": self.teacher,
            "stages": self.stages,
            "evaluation": self.evaluation,
            "paths": self.paths,
            "tags": self.tags,
        }
        if self.instances is not None:
            d["instances"] = self.instances
        if self.eval_datasets is not None:
            d["eval_datasets"] = self.eval_datasets
        return copy.deepcopy(d)

    @property
    def run_id(self) -> str:
        """Stable hash of everything that defines the CBM and its evaluation (seed excluded)."""
        d = {k: v for k, v in self.to_dict().items() if k not in _NON_IDENTITY_KEYS}
        blob = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def with_overrides(self, overrides: dict[str, Any]) -> "ExperimentConfig":
        d = self.to_dict()
        for key, value in overrides.items():
            set_dotted(d, key, value)
        return ExperimentConfig.from_dict(d)


def _resolve_path(container: Any, part: str) -> Any:
    return container[int(part)] if isinstance(container, list) else container[part]


def set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    """Set ``a.b.0.c`` style keys; list indices are integers, missing dict keys are created."""
    parts = key.split(".")
    node: Any = d
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node.setdefault(part, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = copy.deepcopy(value)
    else:
        node[last] = copy.deepcopy(value)


def get_dotted(d: dict[str, Any], key: str) -> Any:
    node: Any = d
    for part in key.split("."):
        node = _resolve_path(node, part)
    return node


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key=value`` with the value interpreted as YAML (so numbers/lists/dicts work)."""
    if "=" not in text:
        raise ValueError(f"Override must look like key=value, got '{text}'")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig.from_dict(load_yaml(path))
    return cfg.with_overrides(overrides) if overrides else cfg


@dataclass
class Ablation:
    anchor: ExperimentConfig
    factors: dict[str, list[Any]]
    seeds: list[int]
    mode: str = "ofat"
    name: str = "ablation"

    def expand(self) -> list[ExperimentConfig]:
        """All runs of the sweep, one per (variant, seed); the anchor itself is always included."""
        if self.mode == "grid":
            keys = list(self.factors)
            variants = [dict(zip(keys, combo)) for combo in itertools.product(*self.factors.values())]
        elif self.mode == "ofat":
            variants = [{}]
            for key, values in self.factors.items():
                variants += [{key: v} for v in values]
        else:
            raise ValueError(f"Unknown ablation mode '{self.mode}' (use 'ofat' or 'grid')")

        runs: list[ExperimentConfig] = []
        seen: set[tuple[str, int]] = set()
        for overrides in variants:
            base = self.anchor.with_overrides(overrides)
            for seed in self.seeds:
                cfg = base.with_overrides({"seed": seed, "tags.ablation": self.name})
                if (cfg.run_id, seed) not in seen:  # OFAT can revisit the anchor value
                    seen.add((cfg.run_id, seed))
                    runs.append(cfg)
        return runs


def load_ablation(path: str | Path) -> Ablation:
    path = Path(path)
    spec = load_yaml(path)
    anchor = load_config((path.parent / spec["anchor"]).resolve())
    anchor = anchor.with_overrides(spec.get("overrides", {}))
    return Ablation(
        anchor=anchor,
        factors=spec.get("factors", {}),
        seeds=spec.get("seeds", [anchor.seed]),
        mode=spec.get("mode", "ofat"),
        name=spec.get("name", path.stem),
    )
