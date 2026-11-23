"""Readable concept text (issue #79): text encoders and detectors read ``ConceptSet.texts``, which
``discovery: {name: dataset, text: readable}`` fills from ``ImageDataset.concept_text()``. Names, and so
matching to annotations, stay unchanged; without the option every text is its name."""

from types import SimpleNamespace

import pytest
import torch

from cbm_eval.data.cub import CUB, readable_attribute
from cbm_eval.data.synthetic import SyntheticDataset
from cbm_eval.stages.common import concept_embeddings
from cbm_eval.stages.discovery import DatasetDiscovery
from cbm_eval.structures import ConceptSet


@pytest.mark.parametrize("name, text", [
    ("has_bill_shape::curved_(up_or_down)", "curved (up or down) bill shape"),
    ("has_wing_color::blue", "blue wing color"),
    ("has_size::small_(5_-_9_in)", "small (5 - 9 in) size"),
    ("has_bill_length::about_the_same_as_head", "about the same as head bill length"),
    ("has_shape::duck-like", "duck-like shape"),
    ("wing", "wing"),  # not a CUB key: kept
])
def test_readable_cub_attribute(name, text):
    assert readable_attribute(name) == text


def test_cub_concept_text_follows_concept_names(cub_root):
    ds = CUB(root=str(cub_root), min_class_count=0)
    assert ds.concept_text() == [readable_attribute(n) for n in ds.concept_names]
    assert "dagger bill shape" in ds.concept_text()


def test_default_concept_text_is_the_names():
    ds = SyntheticDataset(sizes={"train": 4, "val": 4, "test": 4})
    assert ds.concept_text() == ds.concept_names


def test_dataset_discovery_text_is_opt_in(cub_root):
    ctx = SimpleNamespace(dataset=CUB(root=str(cub_root), min_class_count=0))
    names = ctx.dataset.concept_names
    plain = DatasetDiscovery().discover(ctx)
    assert plain.names == names and plain.texts == names
    assert all("text" not in m for m in plain.meta)  # default: meta exactly as before the option existed
    assert plain.text_key() == names

    readable = DatasetDiscovery(text="readable", extra=["red eye"]).discover(ctx)
    assert readable.names == names + ["red eye"]  # names stay the matching key
    assert readable.texts == ctx.dataset.concept_text() + ["red eye"]
    assert readable.subset([0]).texts == readable.texts[:1]
    assert readable.text_key() == [readable.names, readable.texts]
    with pytest.raises(ValueError, match="name or readable"):
        DatasetDiscovery(text="pretty")


def test_text_encoders_read_texts():
    seen = []
    ctx = SimpleNamespace(encode_text=lambda texts: seen.append(texts) or torch.ones(len(texts), 3))
    concepts = ConceptSet(["has_wing_color::blue", "beak"], meta=[{"text": "blue wing color"}, {}])
    concept_embeddings(concepts, ctx)
    assert seen == [["blue wing color", "beak"]]
