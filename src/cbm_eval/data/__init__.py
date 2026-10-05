from . import cub, metashift, synthetic, waterbirds  # noqa: F401  (registration side effects)
from .base import ImageDataset, Sample
from .features import load_features

__all__ = ["ImageDataset", "Sample", "load_features"]
