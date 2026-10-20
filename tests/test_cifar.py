"""CIFAR-10 / CIFAR-100 on tiny fake pickles: splits, pixel layout, refusals of label-dependent stages,
and the LF-CBM anchors end to end with a stub LLM and the ``test_pixels`` backbone. Fully offline."""

import pickle

import numpy as np
import pytest

from cbm_eval.config import load_config
from cbm_eval.data.cifar import CIFAR10, CIFAR100
from cbm_eval.pipeline import PipelineBuilder, run


class StubLLM:
    NAMES = ("metal wings", "four wheels", "feathers", "whiskers", "antlers", "wet nose", "green skin", "mane",
             "sails", "cargo bed", "red petals", "fins", "tree bark", "round fruit", "blue sky", "grass")

    def complete(self, prompt, images=None, system=None):
        return "\n".join(f"- {n}" for n in self.NAMES)


def _raw(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _cfg(tmp_path, anchor: str, root, **overrides):
    return load_config(f"configs/anchors/{anchor}.yaml").with_overrides({
        "dataset.root": str(root),
        "backbone": {"name": "test_pixels"}, "teacher": {"name": "test_pixels"},  # CLIP without weights
        "stages.training": {"name": "sequential", "epochs": 5},
        "evaluation": [{"name": "shift"}, {"name": "leakage", "repeats": 1}],
        "device": "cpu",
        "paths": {"cache": str(tmp_path / "cache"), "runs": str(tmp_path / "runs"),
                  "results": str(tmp_path / "r.jsonl")},
        **overrides,
    })


def test_cifar10_splits_and_pixels(cifar10_root):
    ds = CIFAR10(str(cifar10_root))
    train, val, test = (ds.samples(s) for s in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (36, 4, 10)
    assert ds.class_names[0] == "airplane" and ds.concept_names is None
    raw = _raw(cifar10_root / "test_batch")
    image = test[0].image
    assert image.size == (32, 32) and image.mode == "RGB"
    # rows hold the R, G and B planes one after the other, each in row-major order
    assert image.getpixel((1, 0)) == tuple(int(raw["data"][0, 1 + c * 1024]) for c in range(3))
    assert [s.label for s in test] == raw["labels"]
    # val is a holdout of train: together they are the official train images, each once
    pixels = lambda samples: sorted(np.asarray(s.image).tobytes() for s in samples)
    batches = [_raw(cifar10_root / f"data_batch_{i}") for i in range(1, 6)]
    official = np.concatenate([b["data"] for b in batches]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    assert pixels(train + val) == sorted(im.tobytes() for im in official)


def test_val_holdout_is_seeded(cifar10_root):
    val = lambda **kw: [np.asarray(s.image).tobytes() for s in CIFAR10(str(cifar10_root), **kw).samples("val")]
    assert val() == val()
    assert val(split_seed=1) != val()
    assert len(val(val_fraction=0.25)) == 10
    with pytest.raises(ValueError, match="val_fraction"):
        CIFAR10(str(cifar10_root), val_fraction=0.0)


def test_cifar100_fine_labels(cifar100_root):
    ds = CIFAR100(str(cifar100_root))
    assert ds.class_names == ["apple", "aquarium fish", "maple tree", "rose"]
    assert (len(ds.samples("train")), len(ds.samples("val")), len(ds.samples("test"))) == (36, 4, 10)
    assert {s.label for s in ds.samples("test")} <= set(range(4))


def test_missing_files_and_foreign_pickles_are_refused(cifar10_root, tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset.root"):
        CIFAR10(str(tmp_path / "nowhere"))
    with open(cifar10_root / "batches.meta", "wb") as f:
        pickle.dump({"label_names": print}, f, protocol=2)  # a pickle that references a callable
    with pytest.raises(pickle.UnpicklingError, match="print; is it a CIFAR batch"):
        CIFAR10(str(cifar10_root))


@pytest.mark.parametrize("overrides, error, match", [
    ({"stages.alignment": {"name": "human"}}, RuntimeError, r"no concept annotations.*\{name: clip\}"),
    ({"stages.discovery": {"name": "dataset"}}, RuntimeError, r"stages.discovery.*\{name: llm\}"),
    ({"evaluation": [{"name": "shift"}, {"name": "concepts"}]}, ValueError, r"\['concepts'\].*evaluation:"),
])
def test_label_dependent_stages_are_refused(cifar10_root, tmp_path, monkeypatch, overrides, error, match):
    from cbm_eval.stages import discovery

    monkeypatch.setattr(discovery, "ClaudeClient", lambda **kw: StubLLM())
    with pytest.raises(error, match=match):
        run(_cfg(tmp_path, "cifar10_lfcbm", cifar10_root, **overrides), save=False)


@pytest.mark.parametrize("anchor, root", [("cifar10_lfcbm", "cifar10_root"), ("cifar100_lfcbm", "cifar100_root")])
def test_lfcbm_anchors_run_on_fake_cifar(tmp_path, monkeypatch, request, anchor, root):
    from cbm_eval.stages import discovery

    monkeypatch.setattr(discovery, "ClaudeClient", lambda **kw: StubLLM())
    cfg = _cfg(tmp_path, anchor, request.getfixturevalue(root))
    res = run(cfg, save=False)
    assert 0 <= res.metrics["shift.test.acc"] <= 1
    assert 0 < res.metrics["n_concepts"] <= len(StubLLM.NAMES)
    ctx = PipelineBuilder(cfg).context()
    assert ctx.split("train").paths[0] is None  # in-memory images: stages that read files refuse them
