"""Extract and cache backbone features per (dataset, backbone, split), augmented views of them, and
features of its auxiliary tap."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from ..structures import FeatureSplit
from ..utils import rng_preserved, stable_hash
from .base import ImageDataset

log = logging.getLogger(__name__)


def load_features(dataset: ImageDataset, backbone, split: str, cache_dir: str | Path,
                  patches: bool = False, batch_size: int = 64, num_workers: int = 0) -> FeatureSplit:
    key = {"dataset": dataset.cache_key(), "backbone": backbone.cache_key(), "split": split, "patches": patches}
    path = Path(cache_dir) / "features" / dataset.name / f"{split}-{stable_hash(key)}.pt"
    if path.exists():
        return FeatureSplit(**torch.load(path, weights_only=False))
    with rng_preserved():  # a cold cache must not shift the run's random draws (see ``rng_preserved``)
        data = _extract(dataset, backbone, split, patches, batch_size, num_workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return FeatureSplit(**data)


def _extract(dataset: ImageDataset, backbone, split: str, patches: bool, batch_size: int,
             num_workers: int) -> dict:
    grid = backbone.patch_grid if patches else None
    loader = DataLoader(dataset.torch_split(split, backbone.transform(), grid),
                        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    feats, patch_feats, labels, attrs, concepts, parts = [], [], [], [], [], []
    for batch in loader:
        if patches:
            p = backbone.encode_patches(batch["image"])
            patch_feats.append(p.half().cpu())
        feats.append(backbone.encode_images(batch["image"]).cpu())
        labels.append(batch["label"])
        attrs.append(batch["attr"])
        if "concepts" in batch:
            concepts.append(batch["concepts"])
        if "parts" in batch:
            parts.append(batch["parts"])

    labels_t, attrs_t = torch.cat(labels).long(), torch.cat(attrs).long()
    data = dict(
        features=torch.cat(feats).float(),
        labels=labels_t,
        attrs=attrs_t,
        groups=labels_t * dataset.n_attrs + attrs_t,
        concept_labels=torch.cat(concepts) if concepts else None,
        patch_features=torch.cat(patch_feats) if patch_feats else None,
        part_masks=torch.cat(parts) if parts else None,
        paths=[s.image if isinstance(s.image, str) else None for s in dataset.samples(split)],
    )
    return data


def load_aux_features(dataset: ImageDataset, backbone, split: str, cache_dir: str | Path,
                      batch_size: int = 64, num_workers: int = 0) -> torch.Tensor:
    """(N, aux_dim) features of the backbone's auxiliary tap (``Backbone.encode_aux``), rows aligned with
    ``load_features``; cached per dataset, backbone and split. ``batch_size`` / ``num_workers`` only batch."""
    key = {"dataset": dataset.cache_key(), "backbone": backbone.cache_key(), "split": split, "tap": "aux"}
    path = Path(cache_dir) / "features" / dataset.name / f"{split}-aux-{stable_hash(key)}.pt"
    if path.exists():
        return torch.load(path)
    # its own generator: a DataLoader draws a base seed even unshuffled; ``rng_preserved`` also covers a
    # backbone that builds its network on first use
    with rng_preserved():
        loader = DataLoader(dataset.torch_split(split, backbone.transform()), batch_size=batch_size,
                            num_workers=num_workers, shuffle=False, generator=torch.Generator())
        out = torch.cat([backbone.encode_aux(batch["image"]).float().cpu() for batch in loader])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


@torch.no_grad()
def encode_split(images, encoder: torch.nn.Module, device, batch_size: int = 128,
                 num_workers: int = 0) -> torch.Tensor:
    """(N, D) features of an image dataset (``Context.image_split``) under ``encoder`` in eval mode;
    used for a fine-tuned encoder, whose features are run-specific and so are not cached on disk."""
    was_training = encoder.training
    encoder.eval()
    loader = DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = [encoder(batch["image"].to(device)).float().cpu() for batch in loader]
    encoder.train(was_training)
    return torch.cat(feats)


class _SeededViews(TorchDataset):
    """Item ``v * N + i`` is view ``v`` of image ``i``: ``images[i]`` (whose transform augments) under a
    torch RNG seeded by ``(seed, v, i)``, so a view never depends on the worker count, the batch order,
    the global RNG or how many views are drawn."""

    def __init__(self, images: TorchDataset, views: int, seed: int):
        self.images, self.views, self.seed, self.n = images, views, seed, len(images)

    def __len__(self) -> int:
        return self.views * self.n

    def __getitem__(self, j: int):
        v, i = divmod(j, self.n)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(((self.seed * 1_000_003 + v) * 1_000_003 + i) % 2**63)
            return self.images[i]


def load_augmented_features(dataset: ImageDataset, backbone, split: str, cache_dir: str | Path, views: int,
                            seed: int = 0, batch_size: int = 128, num_workers: int = 0) -> torch.Tensor:
    """(N, V, D) float16 features of ``views`` augmented copies of every image of a split, under the
    backbone's ``train_transform`` and its frozen encoder. Cached per dataset, backbone, transform,
    split, views and seed; ``batch_size`` / ``num_workers`` only batch the encoding."""
    key = {"dataset": dataset.cache_key(), "backbone": backbone.cache_key(), "split": split, "views": views,
           "seed": seed, "transform": repr(backbone.train_transform())}
    path = Path(cache_dir) / "features" / dataset.name / f"{split}-views-{stable_hash(key)}.pt"
    if path.exists():
        return torch.load(path)
    images = _SeededViews(dataset.torch_split(split, backbone.train_transform()), views, seed)
    feats = []
    with rng_preserved():  # also covers a backbone that builds its network on first use
        # its own generator: a DataLoader draws a base seed even unshuffled, which would shift the global RNG
        loader = DataLoader(images, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                            generator=torch.Generator())
        for b, batch in enumerate(loader):
            feats.append(backbone.encode_images(batch["image"]).half().cpu())
            if b % 100 == 0:
                log.info("augmented %s features of %s: %d / %d", split, dataset.name, b * batch_size, len(images))
    out = torch.cat(feats).view(views, images.n, -1).transpose(0, 1).contiguous()
    if not torch.isfinite(out).all():
        raise ValueError(f"Augmented features of '{backbone.cache_key()['name']}' overflow float16 storage; "
                         "the backbone's features are out of range (are its weights loaded?)")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out
