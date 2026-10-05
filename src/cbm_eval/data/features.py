"""Extract and cache backbone features per (dataset, backbone, split)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..structures import FeatureSplit
from ..utils import stable_hash
from .base import ImageDataset


def load_features(dataset: ImageDataset, backbone, split: str, cache_dir: str | Path,
                  patches: bool = False, batch_size: int = 64, num_workers: int = 0) -> FeatureSplit:
    key = {"dataset": dataset.cache_key(), "backbone": backbone.cache_key(), "split": split, "patches": patches}
    path = Path(cache_dir) / "features" / dataset.name / f"{split}-{stable_hash(key)}.pt"
    if path.exists():
        return FeatureSplit(**torch.load(path, weights_only=False))

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
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return FeatureSplit(**data)
