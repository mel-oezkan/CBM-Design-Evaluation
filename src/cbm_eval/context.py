"""Shared run state handed to every stage: dataset, backbones, cached features, text embeddings,
and (for bag-based CBMs) per-image instance sets. After a run fine-tunes the backbone,
``PipelineBuilder`` installs the fine-tuned encoder and the CBM's inputs become its features.

A split name is ``train``/``val``/``test`` of the training dataset or a key of ``eval_datasets:``,
which names that dataset's ``test`` split. Eval splits are served in the training dataset's label
and concept space, so evaluators treat every split alike."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset

from .backbones.base import Backbone
from .config import ExperimentConfig
from .data.base import SPLITS, ImageDataset
from .data.features import encode_split, load_features
from .data.parts import input_geometry, load_parts
from .stages.common import match_names
from .structures import Bag, ConceptSet, FeatureSplit, InstanceSplit, PartSplit
from .utils import resolve_device


class Context:
    def __init__(self, cfg: ExperimentConfig, dataset: ImageDataset, backbone: Backbone,
                 teacher: Backbone | None = None, need_patches: bool = False, instances=None,
                 eval_datasets: dict[str, ImageDataset] | None = None):
        self.cfg, self.dataset, self.backbone = cfg, dataset, backbone
        self.eval_datasets = eval_datasets or {}
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

    @property
    def splits(self) -> tuple[str, ...]:
        """Every split name of the run: the training dataset's, then the ``eval_datasets`` keys."""
        return (*SPLITS, *self.eval_datasets)

    def source(self, name: str) -> tuple[ImageDataset, str]:
        """The dataset a split name refers to and that dataset's own split name."""
        if name in SPLITS:
            return self.dataset, name
        if name in self.eval_datasets:
            return self.eval_datasets[name], "test"
        raise ValueError(f"Unknown split '{name}': use one of {list(self.splits)} "
                         f"(add test-only datasets under `eval_datasets:`)")

    def samples(self, name: str) -> list:
        """Raw samples of a split (``Dataset.samples`` of the dataset it refers to)."""
        dataset, split = self.source(name)
        return dataset.samples(split)

    def split(self, name: str) -> FeatureSplit:
        """The CBM's input features for a split: cached frozen-backbone features, or, once a
        fine-tuned encoder is installed (``use_encoder``), that encoder's features (in memory)."""
        key = ("backbone", name)
        if key not in self._splits:
            self._splits[key] = self._features(self.backbone, name, patches=self.need_patches)
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
            dataset, split = self.source(name)
            self._images[key] = dataset.torch_split(split, tf)
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
            dataset, split = self.source(name)
            parts = load_parts(dataset, split, self.cache_dir, ops, seg_size)
            if parts is not None and dataset is not self.dataset:
                idx = match_names(self.dataset.concept_names or [], dataset.concept_names)
                parts = None if not idx or None in idx else dataclasses.replace(
                    parts, concept_keypoints=parts.concept_keypoints[idx],
                    concept_segs=None if parts.concept_segs is None else parts.concept_segs[idx])
            self._parts[key] = parts
        return self._parts[key]

    def teacher_split(self, name: str) -> FeatureSplit:
        """Features from the vision-language teacher used for pseudo-labels and text filters."""
        if self.teacher is None:
            raise RuntimeError("This stage needs a text-capable teacher; set `teacher:` or use a CLIP backbone")
        if self.teacher is self.backbone:
            return self.split(name)
        key = ("teacher", name)
        if key not in self._splits:
            self._splits[key] = self._features(self.teacher, name)
        return self._splits[key]

    def _features(self, backbone: Backbone, name: str, patches: bool = False) -> FeatureSplit:
        dataset, split = self.source(name)
        fs = load_features(dataset, backbone, split, self.cache_dir, patches=patches)
        return fs if dataset is self.dataset else self._in_training_space(name, dataset, fs)

    def _in_training_space(self, name: str, dataset: ImageDataset, fs: FeatureSplit) -> FeatureSplit:
        """An eval split with labels as training-class indices and concept labels in the order of
        the training dataset's ``concept_names``: its own annotations when it has all of them, else
        the training dataset's ``class_concepts`` of each image's class, else none."""
        cls = match_names(dataset.class_names, self.dataset.class_names)
        missing = [c for c, i in zip(dataset.class_names, cls) if i is None]
        if missing:
            raise ValueError(f"`eval_datasets.{name}` ({dataset.name}) has classes that the training dataset "
                             f"'{self.dataset.name}' lacks, e.g. {missing[:3]}; train on a dataset with them")
        labels = torch.tensor(cls)[fs.labels]
        groups = None if fs.attrs is None else labels * dataset.n_attrs + fs.attrs
        concepts = masks = None
        idx = match_names(self.dataset.concept_names or [], dataset.concept_names)
        if fs.concept_labels is not None and idx and None not in idx:
            concepts = fs.concept_labels[:, idx]
            masks = None if fs.part_masks is None else fs.part_masks[:, idx]
        elif (class_concepts := self.dataset.class_concepts()) is not None:
            concepts = class_concepts[labels].float()
        return dataclasses.replace(fs, labels=labels, groups=groups, concept_labels=concepts, part_masks=masks)

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
