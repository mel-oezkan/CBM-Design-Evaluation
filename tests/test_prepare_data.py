"""The dataset folder checks of ``cbm_eval.data.prepare`` on small fake folders (offline: no downloads)."""

import pickle
import re
import shutil
import urllib.request

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cbm_eval.data.prepare import RECIPES, check, prepare

# The `cub_root` fixture (conftest.py): 24 images, 2 classes, 4 attributes, 15 parts, 72 CUB70 masks.
CUB = {"images": 24, "classes": 2, "attributes": 4, "parts": 15, "train": 16, "test": 8}
PARTS = {"classes": 2, "masks": 72}


def _img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)


@pytest.fixture
def waterbirds(tmp_path):
    folder = tmp_path / "waterbirds"
    rows = []
    for i in range(12):
        rel = f"{i % 2}/img_{i}.jpg"
        _img(folder / rel)
        rows.append({"img_filename": rel, "y": i % 2, "split": [0, 0, 1, 2][i % 4], "place": (i // 2) % 2})
    pd.DataFrame(rows).to_csv(folder / "metadata.csv", index=False)
    return tmp_path, {"images": 12, "train": 6, "val": 3, "test": 3}


@pytest.fixture
def subpopbench_metashift(tmp_path):
    """The folder SubpopBench leaves: ``metadata_metashift.csv`` with paths that start at its ``--data_path``."""
    folder, rows = tmp_path / "metashift", []
    for i in range(8):
        rel = f"MetaShift-Cat-Dog-indoor-outdoor/train/cat/cat(indoor)/{i}.jpg"
        _img(folder / rel)
        rows.append({"filename": f"some/data_path/metashift/{rel}", "y": i % 2, "a": 1, "split": [0, 0, 1, 2][i % 4]})
    pd.DataFrame(rows).to_csv(folder / "metadata_metashift.csv", index=False)
    return tmp_path, {"images": 8, "train": 4, "val": 2, "test": 2}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("tests must not download")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_recipes_pin_https_urls_and_sha256():
    for r in RECIPES.values():
        assert r.terms and (r.archives or r.manual), r.name
        for a in r.archives:
            assert a.url.startswith("https://") and re.fullmatch(r"[0-9a-f]{64}", a.sha256), r.name


def test_cub_ok_and_loader_runs(cub_root):
    assert check("cub", cub_root.parent, CUB) == []


def test_cub_reports_truncated_file_and_missing_images(cub_root):
    labels = cub_root / "attributes" / "image_attribute_labels.txt"
    labels.write_text("\n".join(labels.read_text().splitlines()[:-5]))
    next((cub_root / "images").rglob("*.jpg")).unlink()
    problems = check("cub", cub_root.parent, CUB)
    assert any("image_attribute_labels.txt has 91 rows, expected 96" in p for p in problems)
    assert any("1 of 24 image files are missing" in p for p in problems)


def test_cub_attributes_next_to_folder_as_in_the_archive(cub_root):
    shutil.move(cub_root / "attributes" / "attributes.txt", cub_root.parent / "attributes.txt")
    assert check("cub", cub_root.parent, CUB) == []
    (cub_root.parent / "attributes.txt").unlink()
    assert any("attributes.txt not found" in p for p in check("cub", cub_root.parent, CUB))


def test_cub_wrong_split_counts(cub_root):
    assert check("cub", cub_root.parent, {**CUB, "train": 20, "test": 4}) == [
        "train_test_split.txt marks 16 images as train, expected 20",
        "train_test_split.txt marks 8 images as test, expected 4"]


def test_nested_folder_gets_a_hint(cub_root, tmp_path):
    data = tmp_path / "data"
    (data / "CUB_200_2011").mkdir(parents=True)
    shutil.move(cub_root, data / "CUB_200_2011" / "CUB_200_2011")
    [msg] = check("cub", data, CUB)
    assert "nested one level too deep" in msg


def test_cub_parts_ok_and_detects_wrong_layout(cub_root):
    assert check("cub-parts", cub_root.parent, PARTS) == []
    masks = cub_root / "part_segmentations" / "AnnotationMasksPerclass"
    shutil.move(masks / "1", masks / "2" / "nested")  # class-1 masks moved into class 2's folder
    problems = check("cub-parts", cub_root.parent, PARTS)
    assert "has 1 class folders, expected 2" in problems[0] and "has 36 mask files" in problems[1]
    (masks / "2" / "nested").rename(masks / "001.A_bird")
    assert "one folder per CUB class id" in check("cub-parts", cub_root.parent, PARTS)[0]


def test_cub_parts_masks_must_match_images_of_their_class(cub_root):
    masks = cub_root / "part_segmentations" / "AnnotationMasksPerclass"
    one = next((masks / "1").glob("*_beak.png"))
    one.rename(masks / "2" / one.name)
    assert any("masks match no image of their class" in p for p in check("cub-parts", cub_root.parent, PARTS))


def test_cub_koh_split(cub_root):
    folder = cub_root.parent / "CUB_processed" / "class_attr_data_10"
    folder.mkdir(parents=True)
    paths = [p.split(maxsplit=1)[1] for p in (cub_root / "images.txt").read_text().splitlines()]
    train = [p for p, t in zip(paths, (cub_root / "train_test_split.txt").read_text().splitlines()) if t.endswith("1")]
    test = [p for p in paths if p not in train]
    for name, rows in [("train", train[4:]), ("val", train[:4]), ("test", test)]:
        with open(folder / f"{name}.pkl", "wb") as f:
            pickle.dump([{"img_path": f"/juice/CUB_200_2011/images/{p}"} for p in rows], f)
    sizes = {"train": 12, "val": 4, "test": 8}
    assert check("cub-koh-split", cub_root.parent, sizes) == []
    assert check("cub-koh-split", cub_root.parent, {**sizes, "val": 5}) == [
        f"{folder / 'val.pkl'} has 4 images, expected 5"]


def test_waterbirds_ok_and_problems(waterbirds):
    root, expect = waterbirds
    assert check("waterbirds", root, expect) == []
    meta = pd.read_csv(root / "waterbirds" / "metadata.csv")
    meta.loc[0, "place"] = 2
    meta.to_csv(root / "waterbirds" / "metadata.csv", index=False)
    (root / "waterbirds" / meta.img_filename[1]).unlink()
    problems = check("waterbirds", root, {**expect, "val": 4})
    assert problems == [
        f"{root / 'waterbirds' / 'metadata.csv'} has 3 val rows (split == 1), expected 4",
        f"{root / 'waterbirds' / 'metadata.csv'}: column 'place' has values [2] outside 0/1",
        f"1 of 12 image files are missing under {root / 'waterbirds'}, e.g. {meta.img_filename[1]}"]
    meta.drop(columns="place").to_csv(root / "waterbirds" / "metadata.csv", index=False)
    assert check("waterbirds", root, expect) == [f"{root / 'waterbirds' / 'metadata.csv'} lacks the columns ['place']"]


def test_waterbirds_under_its_archive_name_gets_a_hint(waterbirds):
    root, expect = waterbirds
    (root / "waterbirds").rename(root / "waterbird_complete95_forest2water2")
    [msg] = check("waterbirds", root, expect)
    assert "rename waterbird_complete95_forest2water2 to waterbirds" in msg


def test_prepare_refuses_to_overwrite_a_broken_folder(waterbirds):
    root, expect = waterbirds
    (root / "waterbirds" / "0" / "img_0.jpg").unlink()
    problems = prepare("waterbirds", root, expect)  # would download if it went past the refusal
    assert "prepare never overwrites it" in problems[0] and "image files are missing" in problems[1]


def test_prepare_skips_a_folder_that_passes(waterbirds, capsys):
    root, expect = waterbirds
    assert prepare("waterbirds", root, expect) == []
    assert "already prepared" in capsys.readouterr().out


def test_metashift_converts_subpopbench_metadata(subpopbench_metashift):
    root, expect = subpopbench_metashift
    assert "run the prepare step to convert it" in check("metashift", root, expect)[0]
    assert prepare("metashift", root, expect) == []
    meta = pd.read_csv(root / "metashift" / "metadata.csv")
    assert list(meta.columns) == ["filename", "y", "a", "split"]
    assert meta.filename[0] == "MetaShift-Cat-Dog-indoor-outdoor/train/cat/cat(indoor)/0.jpg"


def test_metashift_string_splits_and_manual_steps(subpopbench_metashift, capsys):
    root, expect = subpopbench_metashift
    (root / "metashift" / "metadata_metashift.csv").unlink()
    assert prepare("metashift", root, expect)[0].endswith("run the SubpopBench steps above first")
    assert "subpopbench/scripts/download.py metashift" in capsys.readouterr().out
    rows = [{"filename": f"MetaShift-Cat-Dog-indoor-outdoor/train/cat/cat(indoor)/{i}.jpg", "y": i % 2, "a": 1,
             "split": ["train", "train", "val", "test"][i % 4]} for i in range(8)]
    pd.DataFrame(rows).to_csv(root / "metashift" / "metadata.csv", index=False)
    assert check("metashift", root, expect) == []
