"""Name -> implementation registries, one per pluggable component kind.

Every stage module registers its variants with ``@REGISTRY.register("name")`` so the
pipeline builder can assemble a CBM purely from a config dict ``{"name": ..., **kwargs}``.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(obj: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def build(self, spec: dict[str, Any] | str, **extra: Any) -> T:
        """Instantiate from ``{"name": ..., **kwargs}`` (or a bare name)."""
        if isinstance(spec, str):
            spec = {"name": spec}
        spec = dict(spec)
        name = spec.pop("name")
        return self.get(name)(**spec, **extra)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


DATASETS: Registry = Registry("dataset")
BACKBONES: Registry = Registry("backbone")
DISCOVERY: Registry = Registry("discovery")
FILTERING: Registry = Registry("filtering")
GENERATION: Registry = Registry("generation")
ALIGNMENT: Registry = Registry("alignment")
PREDICTOR: Registry = Registry("predictor")
TRAINING: Registry = Registry("training")
EVALUATION: Registry = Registry("evaluation")
INSTANCES: Registry = Registry("instances")

ALL = {
    r.kind: r
    for r in (DATASETS, BACKBONES, INSTANCES, DISCOVERY, FILTERING, GENERATION, ALIGNMENT, PREDICTOR, TRAINING,
              EVALUATION)
}


def load_all() -> None:
    """Import every module that registers components (registration is an import side effect)."""
    from . import backbones, data, evaluation, instances, stages  # noqa: F401
