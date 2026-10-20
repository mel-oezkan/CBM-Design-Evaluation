"""FunnyBirds: scene sampling, rendering through a stub server, the loader's concepts and part
annotations, and an end-to-end run with the part evaluators (offline; no node server)."""

import math
import os
import random
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import torch
from PIL import Image

from cbm_eval.config import ExperimentConfig
from cbm_eval.data.funnybirds import PART_COLORS, FunnyBirds, concept_name
from cbm_eval.data.funnybirds_render import (PARTS, render, sample_classes, sample_scene, scene_url, serve,
                                             write_scenes)
from cbm_eval.data.parts import load_parts
from cbm_eval.pipeline import run

# Stub part map layout in the 512x512 screenshot: part instance -> top-left corner of a 32 px square.
FB_BOXES = {"beak": (64, 64), "eye01": (160, 64), "eye02": (224, 64), "foot01": (64, 384), "foot02": (128, 384),
            "wing01": (256, 256), "wing02": (352, 256), "tail": (416, 416)}


def _png(img):
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def fb_stub_fetch(url):
    """Stand-in for the render server: a gray body plus each kept part's instances at ``FB_BOXES``."""
    q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    part_map = q["render_mode"] == "part_map"
    img = np.zeros((512, 512, 4), np.uint8) if part_map else np.full((512, 512, 4), [85, 85, 153, 255], np.uint8)
    img[..., 3] = 255
    img[160:240, 160:240, :3] = 170  # the body, which has no part color
    for inst, (x, y) in FB_BOXES.items():
        if q.get(f"{inst.rstrip('0123456789')}_model", "placeholder") != "placeholder":
            img[y : y + 32, x : x + 32, :3] = PART_COLORS[inst] if part_map else (200, 200, 40)
    return _png(img)


@pytest.fixture
def funnybirds_root(tmp_path):
    root = write_scenes(tmp_path / "FunnyBirds", n_classes=3, train_per_class=8, test_per_class=4, seed=0,
                        spurious_corr=0.9, bg_objects=(1, 3))
    for split in ("train", "test"):
        render(root, split, fetch=fb_stub_fetch, workers=2)
    return root


def test_sample_classes_are_distinct():
    classes = sample_classes(50, random.Random(0))
    assert len({tuple(c["parts"].values()) for c in classes}) == 50
    assert [c["class_idx"] for c in classes] == list(range(50))
    with pytest.raises(ValueError, match="part combinations"):
        sample_classes(4 * 3 * 4 * 9 * 6 + 1, random.Random(0))


def test_scene_parts_drop_and_spurious_attribute():
    rng = random.Random(0)
    cls = sample_classes(5, rng)[3]
    kept = [sample_scene(cls, rng, train=True, drop_parts=0.0) for _ in range(20)]
    assert all("placeholder" not in s.values() for s in kept)
    dropped = [sample_scene(cls, rng, train=True, drop_parts=1.0) for _ in range(50)]
    assert any("placeholder" in s.values() for s in dropped)
    test = [sample_scene(cls, rng, train=False, drop_parts=1.0) for _ in range(20)]
    assert all("placeholder" not in s.values() for s in test)  # test birds are always complete

    train = [sample_scene(cls, rng, train=True, spurious_corr=1.0, bg_objects=(2, 4)) for _ in range(20)]
    assert {s["attr"] for s in train} == {3 % 4}
    assert all(set(s["bg_color"].split(",")) == {"yellow"} for s in train)  # BG_COLORS[3]
    assert len({sample_scene(cls, rng, train=False, spurious_corr=1.0)["attr"] for _ in range(50)}) > 1
    url = scene_url(train[0], "http://localhost:8081/", "part_map")
    assert url.startswith("http://localhost:8081/render?render_mode=part_map&")
    assert "attr=" not in url and "class_idx" not in url and "bg_color=yellow,yellow" in url


def test_write_scenes_refuses_to_overwrite(tmp_path):
    write_scenes(tmp_path, n_classes=2, train_per_class=1, test_per_class=1)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_scenes(tmp_path, n_classes=2, train_per_class=1, test_per_class=1)
    write_scenes(tmp_path, n_classes=2, train_per_class=1, test_per_class=1, overwrite=True)


def test_render_resumes_and_retries_blank(funnybirds_root):
    assert render(funnybirds_root, "test", fetch=fb_stub_fetch) == 0  # every file exists
    assert len(list((funnybirds_root / "test_part_map").rglob("*.png"))) == 12

    def blank(url):  # the page had not drawn the scene yet
        return _png(np.full((512, 512, 4), 255, np.uint8))

    for f in (funnybirds_root / "test").rglob("000000.png"):
        f.unlink()
    with pytest.raises(RuntimeError, match="blank"):
        render(funnybirds_root, "test", stop=1, fetch=blank)


def test_serve_needs_an_installed_renderer(tmp_path):
    (tmp_path / "page.html").write_text("")  # but no node_modules: `npm install` was not run
    with pytest.raises(FileNotFoundError, match="npm install"):
        with serve(tmp_path):
            pass


@pytest.mark.skipif(not os.environ.get("FUNNYBIRDS_RENDERER"),
                    reason="set FUNNYBIRDS_RENDERER to the funnybirds render/ folder (after npm install)")
def test_serve_renders_real_scenes(tmp_path):
    """Opt-in: runs node and headless Chrome (no network). The part map must match the image."""
    root = write_scenes(tmp_path, n_classes=2, train_per_class=1, test_per_class=2)
    with serve(os.environ["FUNNYBIRDS_RENDERER"], port=8093, pages=2) as url:
        assert render(root, "test", url, workers=2) == 8
    for s in FunnyBirds(str(root)).samples("test"):
        img = np.asarray(Image.open(s.image).convert("RGB")).astype(int)
        drawn = np.abs(img - [85, 85, 153]).sum(-1) > 30  # not the scene background (0x555599)
        fg = np.asarray(Image.open(s.parts.label_map).convert("RGB")).sum(-1) > 0
        assert s.parts.points and (fg & drawn).sum() / fg.sum() > 0.99  # the part map is this image's


def test_concepts_follow_class_definition_and_dropped_parts(funnybirds_root):
    ds = FunnyBirds(str(funnybirds_root), val_fraction=0.25)
    assert len(ds.concept_names) == sum(len(v) for v in PARTS.values()) == 26
    assert ds.concept_names[0] == "yellow beak shape 1" and "red tail shape 2" in ds.concept_names
    assert concept_name("eye", {"model": "eye03.glb"}) == "eye shape 3"
    assert ds.n_attrs == 4 and ds.class_concepts() is None
    assert len(ds.samples("train")) + len(ds.samples("val")) == 24 and len(ds.samples("test")) == 12
    defs = ds._class_matrix()
    part_of = [n.split(" shape")[0].split()[-1] for n in ds.concept_names]
    for s in ds.samples("train") + ds.samples("test"):
        assert (torch.tensor(s.concepts, dtype=torch.float) <= defs[s.label]).all()  # only the class's variants
        assert set(s.parts.points) == {part_of[k] for k, on in enumerate(s.concepts) if on}  # kept parts are drawn
    assert all(sum(s.concepts) == 5 for s in ds.samples("test"))
    assert any(sum(s.concepts) < 5 for s in ds.samples("train"))  # half the train scenes drop parts
    assert {s.attr for s in ds.samples("test")} <= set(range(4))
    full = FunnyBirds(str(funnybirds_root), class_level_concepts=True)
    assert torch.equal(full.class_concepts(), defs)
    assert all(sum(s.concepts) == 5 for s in full.samples("train"))


def test_part_points_and_masks_come_from_the_part_map(funnybirds_root, tmp_path):
    ds = FunnyBirds(str(funnybirds_root))
    s = next(s for s in ds.samples("test"))
    # 512 px screenshot -> 256 px image: a 32 px square at (x, y) has its center at (x / 2 + 8, y / 2 + 8)
    assert s.parts.size == (256, 256)
    for inst, (x, y) in FB_BOXES.items():
        part = inst.rstrip("0123456789")
        assert (x / 2 + 8, y / 2 + 8) in s.parts.points[part]
    beak = next(k for k, n in enumerate(ds.concept_names) if "beak" in n and s.concepts[k])
    assert s.keypoints[beak] == (40.0, 40.0, 256, 256)

    spec = ds.part_spec()
    assert spec.seg_groups == list(PARTS) and spec.max_points == 2 and spec.concept_segs.sum() == 26
    parts = load_parts(ds, "test", tmp_path / "cache", ops=[], seg_size=16)
    eye, tail = parts.seg_groups.index("eye"), parts.seg_groups.index("tail")
    # 16x16 grid over the 512 px screenshot: one cell is 32 px, so each square fills one cell
    assert parts.segs[0, eye].sum() == 2 and parts.segs[0, eye, 2, 5] and parts.segs[0, eye, 2, 7]
    assert parts.segs[0, tail].sum() == 1 and parts.segs[0, tail, 13, 13]
    assert math.isclose(parts.points[0, eye, 0, 1].item(), 40 / 256)
    assert torch.isnan(parts.points[0, list(PARTS).index("beak"), 1]).all()  # one beak: second slot empty


@pytest.mark.parametrize("bags", [False, True])
def test_funnybirds_end_to_end(tmp_path, funnybirds_root, bags):
    cfg = ExperimentConfig.from_dict({
        "dataset": {"name": "funnybirds", "root": str(funnybirds_root), "val_fraction": 0.25},
        "backbone": {"name": "resnet", "model": "resnet18", "weights": None, "image_size": 64},
        "stages": {"discovery": {"name": "dataset"}, "filtering": [{"name": "rules", "remove_class_names": False}],
                   "generation": {"name": "scores"}, "alignment": {"name": "human"},
                   "predictor": {"name": "dense"}, "training": {"name": "sequential", "epochs": 2}},
        "evaluation": [{"name": "shift"}, {"name": "localization"}, {"name": "keypoint_distance"},
                       {"name": "part_iou"}],
        "device": "cpu",
        "paths": {"cache": str(tmp_path / "cache"), "runs": str(tmp_path / "runs"),
                  "results": str(tmp_path / "r.jsonl")},
    })
    if bags:
        cfg = cfg.with_overrides({"instances": {"name": "patches"}, "stages.predictor": {"name": "mil"},
                                  "stages.training": {"name": "joint", "epochs": 2}})
    m = run(cfg, save=False).metrics
    assert m["n_concepts"] == 26 and 0 <= m["shift.test.wga"] <= 1
    assert {f"part_iou.iou.{p}" for p in PARTS} <= set(m) and 0 <= m["part_iou.miou"] <= 1
    assert {f"keypoint_distance.dist.{p}" for p in PARTS} <= set(m)
