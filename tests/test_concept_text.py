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


READABLE = ConceptSet(["has_wing_color::blue", "beak"], meta=[{"text": "blue wing color"}, {}])


def _recording_ctx(seen, **kw):
    def encode_text(texts, which="teacher"):
        seen.append(list(texts))
        return torch.ones(len(texts), 3)
    return SimpleNamespace(encode_text=encode_text, concepts=READABLE, **kw)


def test_alignments_read_texts():
    from cbm_eval.stages.alignment import SegmentCLIPAlignment, WeightAlignment

    seen = []
    WeightAlignment().fit(READABLE, _recording_ctx(seen, backbone=SimpleNamespace(has_text=True)))
    SegmentCLIPAlignment().fit(READABLE, _recording_ctx(seen, instance_source=object()))
    assert seen == [READABLE.texts] * 2


class _RecordingDetector:
    def __init__(self):
        self.prompts = []

    def detect(self, images, prompts):
        self.prompts += prompts
        return [[] for _ in images]


def test_detectors_read_texts(tmp_path):
    from PIL import Image

    from cbm_eval.instances.segments import GroundedBoxes
    from cbm_eval.stages.scorers import GroundingDINOScorer

    image = Image.new("RGB", (8, 8))
    seen, det = [], _RecordingDetector()
    teacher = SimpleNamespace(features=torch.ones(2, 3))
    ctx = _recording_ctx(seen, samples=lambda split: [0, 0], teacher_split=lambda split: teacher, device="cpu")
    GroundedBoxes(top_k=1, detector=det).prepare(ctx, "train")  # ranks concepts per image by the teacher's text
    assert seen == [READABLE.texts]
    src = GroundedBoxes(top_k=None, detector=det)  # prompts every concept
    src.prepare(ctx, "train")
    src.propose([image, image], [0, 1], ctx)
    assert det.prompts == [READABLE.texts] * 2

    image.save(tmp_path / "a.png")
    scorer = GroundingDINOScorer()
    scorer.detector = det = _RecordingDetector()
    ctx = SimpleNamespace(split=lambda split: SimpleNamespace(paths=[str(tmp_path / "a.png")]), device="cpu")
    scorer._detect(READABLE, ctx, "val")
    assert det.prompts == [READABLE.texts]


def test_readable_text_needs_one_text_per_concept(cub_root):
    ds = CUB(root=str(cub_root), min_class_count=0)
    ds.concept_text = lambda: ["only one"]
    with pytest.raises(ValueError, match="stages.discovery.text: name"):
        DatasetDiscovery(text="readable").discover(SimpleNamespace(dataset=ds))
