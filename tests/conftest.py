import numpy as np
import pytest
from PIL import Image

from cbm_eval.config import ExperimentConfig

SMALL_DATA = {"name": "synthetic", "sizes": {"train": 400, "val": 200, "test": 300}}


@pytest.fixture
def base_cfg(tmp_path):
    names = [f"concept_{i:02d}" for i in range(24)] + ["distractor_00", "class_0"]
    return ExperimentConfig.from_dict({
        "name": "test",
        "dataset": SMALL_DATA,
        "backbone": {"name": "toy"},
        "stages": {
            "discovery": {"name": "static", "names": names},
            "filtering": [{"name": "rules"}, {"name": "select", "k": 16}],
            "generation": {"name": "scores"},
            "alignment": {"name": "clip"},
            "predictor": {"name": "sparse", "lam": 1e-3},
            "training": {"name": "sequential", "epochs": 30},
        },
        "evaluation": [{"name": "shift"}, {"name": "leakage", "repeats": 2}, {"name": "localization"}],
        "device": "cpu",
        "paths": {"cache": str(tmp_path / "cache"), "runs": str(tmp_path / "runs"),
                  "results": str(tmp_path / "results.jsonl")},
    })


# -- a tiny fake CUB_200_2011 folder, shared by the parts and cache-key tests -------------------

_PARTS = ["back", "beak", "belly", "breast", "crown", "forehead", "left eye", "left leg", "left wing", "nape",
          "right eye", "right leg", "right wing", "tail", "throat"]
_ATTRS = ["has_bill_shape::dagger", "has_wing_color::blue", "has_breast_color::red", "has_size::small"]


@pytest.fixture
def cub_root(tmp_path):
    root = tmp_path / "CUB_200_2011"
    for d in ("images/001.A_bird", "images/002.B_bird", "attributes", "parts"):
        (root / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    images, labels, split, attrs, locs = [], [], [], [], []
    seg = root / "part_segmentations" / "AnnotationMasksPerclass"
    for i in range(1, 25):
        c = i % 2
        rel = f"{c + 1:03d}.{'AB'[c]}_bird/{'AB'[c]}_bird_{i:04d}.jpg"
        Image.fromarray(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)).save(root / "images" / rel)
        images.append(f"{i} {rel}")
        labels.append(f"{i} {c + 1}")
        split.append(f"{i} {int(i % 3 != 0)}")
        attrs += [f"{i} {a + 1} {int(rng.random() < 0.6)} 3 1.0" for a in range(len(_ATTRS))]
        locs += [f"{i} {p + 1} {rng.uniform(0, 60):.1f} {rng.uniform(0, 40):.1f} {int(rng.random() < 0.8)}"
                 for p in range(len(_PARTS))]
        (seg / str(c + 1)).mkdir(parents=True, exist_ok=True)
        stem = rel.split("/")[1].removesuffix(".jpg")
        for part in ("beak", "body", "left_wing"):
            mask = np.zeros((20, 30), np.uint8)  # masks are stored at a lower resolution
            y, x = rng.integers(0, 15), rng.integers(0, 25)
            mask[y : y + 5, x : x + 5] = 255
            Image.fromarray(np.stack([mask] * 3, -1)).save(seg / str(c + 1) / f"{stem}_{part}.png")
    (root / "images.txt").write_text("\n".join(images))
    (root / "image_class_labels.txt").write_text("\n".join(labels))
    (root / "train_test_split.txt").write_text("\n".join(split))
    (root / "classes.txt").write_text("1 001.A_bird\n2 002.B_bird")
    (root / "attributes" / "attributes.txt").write_text("\n".join(f"{i + 1} {a}" for i, a in enumerate(_ATTRS)))
    (root / "attributes" / "image_attribute_labels.txt").write_text("\n".join(attrs))
    (root / "parts" / "parts.txt").write_text("\n".join(f"{i + 1} {p}" for i, p in enumerate(_PARTS)))
    (root / "parts" / "part_locs.txt").write_text("\n".join(locs))
    return root
