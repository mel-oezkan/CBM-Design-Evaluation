"""CUB annotation parsing and Koh et al. (2020) options on a tiny fake CUB (offline): the ``koh``
majority vote, the authors' train/val split via ``split_dir``, and the ``cub_koh2020`` anchor's
components end to end."""

import pickle

import pytest

from cbm_eval.config import load_config
from cbm_eval.data.cub import CUB
from cbm_eval.pipeline import run


def _write_labels(root, label_fn):
    """Rewrite image_attribute_labels.txt: ``label_fn(img, attr) -> (present, certainty)``."""
    ids = [int(line.split()[0]) for line in (root / "images.txt").read_text().splitlines()]
    n_attr = len((root / "attributes" / "attributes.txt").read_text().splitlines())
    rows = [f"{i} {a + 1} {p} {c} 1.0" for i in ids for a in range(n_attr) for p, c in [label_fn(i, a)]]
    (root / "attributes" / "image_attribute_labels.txt").write_text("\n".join(rows))


def test_attribute_labels_tolerate_lines_with_an_extra_column(cub_root):
    _write_labels(cub_root, lambda i, a: ((i + a) % 2, 3))
    path = cub_root / "attributes" / "image_attribute_labels.txt"
    lines = path.read_text().splitlines()
    lines[5] += " 0"  # the official file has 606 such lines
    path.write_text("\n".join(lines))
    ds = CUB(str(cub_root), class_level_concepts=False, val_fraction=0.0)
    ids = {line.split()[1]: int(line.split()[0]) for line in (cub_root / "images.txt").read_text().splitlines()}
    for s in ds.samples("train") + ds.samples("test"):
        i = ids["/".join(s.image.split("/")[-2:])]
        assert s.concepts == [(i + a) % 2 for a in range(len(ds.concept_names))]


def test_koh_vote_ignores_not_visible_negatives_and_breaks_ties_to_present(cub_root):
    # fixture: image i has class i % 2 and is in the official train split iff i % 3 != 0
    present = {1, 2, 13, 14}  # 2 of the 8 training images of each class
    # attr 0: present on those, every other image "not visible" (certainty 1) -> mean says 0, koh says 1
    # attr 1: present on those, every other image visibly absent -> 0 under both rules
    def label(i, a):
        if a in (0, 1) and i in present:
            return 1, 3
        return (0, 1) if a == 0 else (0, 3)

    _write_labels(cub_root, label)
    mean = CUB(str(cub_root), min_class_count=1, val_fraction=0.0)
    koh = CUB(str(cub_root), min_class_count=1, val_fraction=0.0, majority_vote="koh")
    assert "has_bill_shape::dagger" not in mean.concept_names
    assert koh.concept_names == ["has_bill_shape::dagger"]
    assert all(s.concepts == [1] for s in koh.samples("test"))
    assert mean.cache_key() != koh.cache_key()


def test_split_dir_uses_the_authors_val_images(cub_root, tmp_path):
    lines = (cub_root / "images.txt").read_text().splitlines()
    train_paths = [p for line, split in zip(lines, (cub_root / "train_test_split.txt").read_text().splitlines())
                   for p in [line.split()[1]] if split.split()[1] == "1"]
    val = train_paths[:3]
    split_dir = tmp_path / "class_attr_data_10"
    split_dir.mkdir()
    with open(split_dir / "val.pkl", "wb") as f:  # Koh et al.'s pickles store absolute image paths
        pickle.dump([{"img_path": f"/somewhere/CUB_200_2011/images/{p}"} for p in val], f)
    ds = CUB(str(cub_root), split_dir=str(split_dir))
    rel = lambda s: "/".join(s.image.split("/")[-2:])
    assert sorted(map(rel, ds.samples("val"))) == sorted(val)
    assert len(ds.samples("train")) == len(train_paths) - 3
    assert len(ds.samples("test")) == len(lines) - len(train_paths)
    assert "split_dir" in ds.cache_key() and "split_dir" not in CUB(str(cub_root)).cache_key()


def test_unknown_majority_vote_is_refused(cub_root):
    with pytest.raises(ValueError, match="majority_vote"):
        CUB(str(cub_root), majority_vote="median")


@pytest.mark.parametrize("training,generation", [
    ({"name": "independent_weighted", "epochs": 2}, {"name": "scores"}),
    ({"name": "sequential_weighted", "epochs": 2}, {"name": "logits"}),
    ({"name": "joint_weighted", "concept_weight": 1.12, "epochs": 2}, {"name": "logits"}),
])
def test_koh2020_anchor_components_run_on_fake_cub(cub_root, tmp_path, training, generation):
    cfg = load_config("configs/anchors/cub_koh2020.yaml").with_overrides({
        "dataset": {"name": "cub", "root": str(cub_root), "majority_vote": "koh", "min_class_count": 1,
                    "val_fraction": 0.3},
        "backbone": {"name": "inception", "weights": None},  # random init: no download
        "stages.training": training, "stages.generation": generation, "device": "cpu",
        "paths": {"cache": str(tmp_path / "cache"), "runs": str(tmp_path / "runs"),
                  "results": str(tmp_path / "r.jsonl")},
    })
    m = run(cfg, save=False).metrics
    assert 0 <= m["shift.test.acc"] <= 1
    assert 0 <= m["concepts.test.concept_error"] <= 1
