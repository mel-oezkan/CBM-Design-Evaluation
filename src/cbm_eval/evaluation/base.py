from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import Context
    from ..model import TrainedCBM


class Evaluator(ABC):
    """Computes a flat dict of metrics for one trained CBM. Keys are prefixed by the evaluator name."""

    needs_patches: bool = False

    @abstractmethod
    def evaluate(self, cbm: "TrainedCBM", ctx: "Context") -> dict[str, float]: ...
