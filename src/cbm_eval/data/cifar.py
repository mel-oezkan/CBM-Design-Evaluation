"""CIFAR-10 / CIFAR-100 (Krizhevsky, 2009), the label-free CBM benchmarks of LF-CBM, LaBo and VLG-CBM.

Expected: the extracted python version of the dataset, as distributed at
https://www.cs.toronto.edu/~kriz/cifar.html. ``root`` is the extracted folder itself:
``cifar-10-batches-py/`` (``data_batch_1`` ... ``data_batch_5``, ``test_batch``, ``batches.meta``) or
``cifar-100-python/`` (``train``, ``test``, ``meta``; fine labels). Nothing is downloaded.

Validation is a seeded holdout of ``val_fraction`` of the official train split. There are no concept
annotations, so these datasets pair with label-free discovery (``llm``, ``kb``) and alignment (``clip``,
``dino``). Images are held in memory, not as files: stages that read image files (``vlm`` discovery,
the ``grounding_dino`` scorer, segment instance sources) refuse them.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from ..registry import DATASETS
from .base import ImageDataset, Sample

# The only globals a CIFAR pickle references (numpy arrays; ``_codecs.encode`` in python 3 pickles), so a
# tampered file cannot run code on load.
_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode")}


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"CIFAR file references {module}.{name}; is it a CIFAR batch?")
        return super().find_class(module, name)


def _load(path: Path) -> dict:
    with open(path, "rb") as f:
        return _ArrayUnpickler(f, encoding="latin1").load()


class _CIFAR(ImageDataset):
    """Shared loader; subclasses name their files and label keys."""

    n_attrs = 1
    train_files: tuple[str, ...]
    test_file: str
    meta_file: str
    label_key: str
    names_key: str

    def __init__(self, root: str, val_fraction: float = 0.1, split_seed: int = 0):
        """``val_fraction`` of the official train images, drawn with ``split_seed``, form ``val``."""
        if not 0 < val_fraction < 1:
            raise ValueError(f"`dataset.val_fraction` must be in (0, 1), got {val_fraction}")
        self.root = Path(root)
        files = [*self.train_files, self.test_file, self.meta_file]
        missing = [f for f in files if not (self.root / f).exists()]
        if missing:
            raise FileNotFoundError(f"{self.name} not found at {self.root} (missing {', '.join(missing)}); set "
                                    f"`dataset.root` to the extracted python version of the dataset")
        self.val_fraction, self.split_seed = val_fraction, split_seed
        self._classes = [c.replace("_", " ") for c in _load(self.root / self.meta_file)[self.names_key]]
        train = [_load(self.root / f) for f in self.train_files]
        test = _load(self.root / self.test_file)
        x = np.concatenate([b["data"] for b in train])
        y = np.concatenate([np.asarray(b[self.label_key]) for b in train])
        perm = np.random.default_rng(split_seed).permutation(len(y))
        n_val = round(val_fraction * len(y))
        val, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        self._arrays = {"train": (x[tr], y[tr]), "val": (x[val], y[val]),
                        "test": (test["data"], np.asarray(test[self.label_key]))}
        self._cache: dict[str, list[Sample]] = {}

    @property
    def class_names(self) -> list[str]:
        return self._classes

    def cache_key(self):
        return {"name": self.name, "root": str(self.root.resolve()), "val_fraction": self.val_fraction,
                "split_seed": self.split_seed}

    def samples(self, split: str) -> list[Sample]:
        if split not in self._cache:
            x, y = self._arrays[split]
            images = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # rows are channel-major: R, G, B planes
            self._cache[split] = [Sample(image=Image.fromarray(im), label=int(c)) for im, c in zip(images, y)]
        return self._cache[split]


@DATASETS.register("cifar10")
class CIFAR10(_CIFAR):
    """CIFAR-10: 10 classes, 50k train / 10k test images of 32x32 pixels."""

    name = "cifar10"
    train_files = tuple(f"data_batch_{i}" for i in range(1, 6))
    test_file, meta_file, label_key, names_key = "test_batch", "batches.meta", "labels", "label_names"

    def __init__(self, root: str = "data/cifar-10-batches-py", **kw):
        super().__init__(root, **kw)


@DATASETS.register("cifar100")
class CIFAR100(_CIFAR):
    """CIFAR-100 with its 100 fine labels (underscores in class names become spaces)."""

    name = "cifar100"
    train_files = ("train",)
    test_file, meta_file, label_key, names_key = "test", "meta", "fine_labels", "fine_label_names"

    def __init__(self, root: str = "data/cifar-100-python", **kw):
        super().__init__(root, **kw)
