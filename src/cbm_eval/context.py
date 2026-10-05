"""Shared run state handed to every stage: dataset, backbones, cached features, text embeddings,
and (for bag-based CBMs) per-image instance sets. After a run fine-tunes the backbone,
``PipelineBuilder`` installs the fine-tuned encoder and the CBM's inputs become its features."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset

from .backbones.base import Backbone
from .config import ExperimentConfig
from .data.base import ImageDataset
from .data.features import encode_split, load_features
from .data.parts import input_geometry, load_parts
from .structures import Bag, ConceptSet, FeatureSplit, InstanceSplit, PartSplit
from .utils import resolve_device


class Context:
    def __init__(self, cfg: ExperimentConfig, dataset: ImageDataset, backbone: Backbone,
                 teacher: Backbone | None = None, need_patches: bool = False, instances=None):
        self.cfg, self.dataset, self.backbone = cfg, dataset, backbone
        self.teacher = teacher if teacher is not None else (backbone if backbone.has_text else None)
        self.need_patches = need_patches
        self.device = resolve_device(cfg.device)
        self.cache_dir = Path(cfg.paths["cache"])
        self.seed = cfg.seed
        self._splits: dict[tuple[str, str], FeatureSplit] = {}
        self._text: dict[tuple[str, str], torch.Tensor] = {}
        self.instance_source = instances
        self.concepts: ConceptSet | None = None  # the filtered set; concept-guided instance sources read it
        self._instances: dict[tuple, InstanceSplit] = {}
        self._parts: dict[tuple, PartSplit | None] = {}
        self._encoder: nn.Module | None = None  # fine-tuned image encoder of the current run, if any
        self._images: dict[tuple[str, bool], TorchDataset] = {}
        self._workers = 0  # dataloader workers for encoding images with the fine-tuned encoder
        for bb in {id(b): b for b in (self.backbone, self.teacher) if b is not None}.values():
            bb.to(self.device)

    @property
    def class_names(self) -> list[str]:
        return self.dataset.class_names

    @property
    def num_classes(self) -> int:
        return len(self.dataset.class_names)

    @property
    def feat_dim(self) -> int:
        return self.backbone.dim

    def split(self, name: str) -> FeatureSplit:
        """The CBM's input features for a split: cached frozen-backbone features, or, once a
        fine-tuned encoder is installed (``use_encoder``), that encoder's features (in memory)."""
        key = ("backbone", name)
        if key not in self._splits:
            self._splits[key] = load_features(self.dataset, self.backbone, name, self.cache_dir,
                                              patches=self.need_patches)
        if self._encoder is None:
            return self._splits[key]
        tuned = ("encoder", name)
        if tuned not in self._splits:
            feats = encode_split(self.image_split(name), self._encoder, self.device, num_workers=self._workers)
            self._splits[tuned] = dataclasses.replace(self._splits[key], features=feats, patch_features=None)
        return self._splits[tuned]

    def image_split(self, name: str, augment: bool = False) -> TorchDataset:
        """Images of a split as a torch dataset of dicts (``image``, ``label``, ``index``, ...), with
        the backbone's eval transform or, with ``augment``, its fine-tuning transform."""
        key = (name, augment)
        if key not in self._images:
            tf = self.backbone.train_transform() if augment else self.backbone.transform()
            self._images[key] = self.dataset.torch_split(name, tf)
        return self._images[key]

    def use_encoder(self, encoder: nn.Module | None, num_workers: int = 0) -> None:
        """Install (or, with None, remove) a fine-tuned encoder. Only ``PipelineBuilder`` calls this."""
        self._encoder, self._workers = encoder, num_workers
        for key in [k for k in self._splits if k[0] == "encoder"]:
            del self._splits[key]

    def instances(self, name: str) -> InstanceSplit:
        """Per-image instance sets (patches or segments) for a split, from the ``instances:`` source."""
        from .instances.base import load_instances

        if self.instance_source is None:
            raise RuntimeError("No instance source configured; set `instances:` in the config")
        concepts = tuple(self.concepts.names) if self.instance_source.needs_concepts and self.concepts else None
        key = (name, concepts)
        if key not in self._instances:
            self._instances[key] = load_instances(self, self.instance_source, name)
        return self._instances[key]

    def inputs(self, name: str) -> tuple[torch.Tensor, Bag | None]:
        """What the CBM consumes: (N, D) image features, or an (N, M, D) instance bag plus its mask."""
        if self.instance_source is None:
            return self.split(name).features, None
        inst = self.instances(name)
        return inst.features, inst.bag

    def parts(self, name: str, geometry: str = "input", seg_size: int = 56) -> PartSplit | None:
        """Part keypoints / masks for a split, or None if the dataset has none. ``geometry="input"``
        maps them through the backbone's resize and crop (patch maps); ``"full"`` keeps the whole
        image (segment instance regions)."""
        key = (name, geometry, seg_size)
        if key not in self._parts:
            ops = input_geometry(self.backbone.transform()) if geometry == "input" else []
            self._parts[key] = load_parts(self.dataset, name, self.cache_dir, ops, seg_size)
        return self._parts[key]

    def teacher_split(self, name: str) -> FeatureSplit:
        """Features from the vision-language teacher used for pseudo-labels and text filters."""
        if self.teacher is None:
            raise RuntimeError("This stage needs a text-capable teacher; set `teacher:` or use a CLIP backbone")
        if self.teacher is self.backbone:
            return self.split(name)
        key = ("teacher", name)
        if key not in self._splits:
            self._splits[key] = load_features(self.dataset, self.teacher, name, self.cache_dir)
        return self._splits[key]

    def encode_text(self, texts: list[str], which: str = "teacher") -> torch.Tensor:
        model = self.teacher if which == "teacher" else self.backbone
        if model is None or not model.has_text:
            raise RuntimeError(f"No text encoder available for '{which}'")
        missing = [t for t in dict.fromkeys(texts) if (which, t) not in self._text]
        if missing:
            emb = model.encode_text(missing).float().cpu()
            for t, e in zip(missing, emb):
                self._text[(which, t)] = e
        return torch.stack([self._text[(which, t)] for t in texts])
