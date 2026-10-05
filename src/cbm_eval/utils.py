from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stable_hash(obj: Any, n: int = 12) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


class JsonCache:
    """Tiny on-disk cache for expensive, deterministic-ish calls (LLM responses, KB lookups)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def get(self, key: Any) -> Any | None:
        f = self.path / f"{stable_hash(key, 20)}.json"
        return json.loads(f.read_text())["value"] if f.exists() else None

    def set(self, key: Any, value: Any) -> None:
        f = self.path / f"{stable_hash(key, 20)}.json"
        f.write_text(json.dumps({"key": key, "value": value}, default=str))


def standardize(x: torch.Tensor, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
    mean = x.mean(0) if mean is None else mean
    std = x.std(0).clamp_min(1e-6) if std is None else std
    return (x - mean) / std, mean, std
