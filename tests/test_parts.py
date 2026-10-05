"""Part localization against keypoints and part segmentations: annotation geometry, the metrics on
hand-made maps, and an end-to-end run on a tiny fake CUB (offline)."""

import math

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from cbm_eval.config import ExperimentConfig
from cbm_eval.data.parts import input_geometry, map_mask, map_points
from cbm_eval.evaluation.parts import KeypointDistanceEvaluator, PartIoUEvaluator
from cbm_eval.pipeline import run
from cbm_eval.structures import PartSplit


def test_map_points_resize_and_center_crop():
    ops = input_geometry(T.Compose([T.Resize(32), T.CenterCrop(32), T.ToTensor()]))
    assert ops == [("resize", (32,)), ("crop", (32, 32))]
    # 200x100 -> 64x32 -> crop x in [16, 48)
    pts = map_points(np.array([[100.0, 50.0], [10.0, 50.0], [150.0, 25.0]]), (200, 100), ops)
    assert pts[0] == pytest.approx([0.5, 0.5])
    assert np.isnan(pts[1]).all()  # cropped away
    assert pts[2] == pytest.approx([(48 - 16) / 32, 0.25])
    assert map_points(np.array([[50.0, 25.0]]), (200, 100), [])[0] == pytest.approx([0.25, 0.25])


def test_map_mask_follows_crop_and_mask_resolution():
    ops = [("resize", (32,)), ("crop", (32,))]
    mask = np.zeros((50, 100), bool)  # half-resolution mask of a 200x100 image
    mask[:, 50:75] = True  # x in [100, 150) of the image -> right half of the crop
    out = map_mask(mask, (200, 100), ops, seg_size=8)
    assert out.shape == (8, 8) and out[:, 4:].all() and not out[:, :4].any()


def _parts(points, segs=None):
    n = len(points)
    return PartSplit(points=torch.tensor(points, dtype=torch.float).reshape(n, 1, 1, 2), keypoint_groups=["beak"],
                     concept_keypoints=torch.tensor([[True], [True]]), segs=segs, seg_groups=["beak"] if segs is not None else [],
                     concept_segs=torch.tensor([[True], [True]]))


def test_keypoint_distance_uses_argmax_concept_peak():
    # 2x2 grid; concept 0 peaks at the top-left patch, concept 1 at the bottom-right
    maps = torch.tensor([[[5.0, 0], [0, 0], [0, 0], [0, 5.0]]])  # (1, P=4, K=2)
    parts = _parts([[0.25, 0.25]])
    ev = KeypointDistanceEvaluator(threshold=0.1)
    m = ev.score(parts, [0, 1], torch.tensor([[1.0, 0.0]]), None, iter([(slice(0, 1), maps)]))
    assert m["dist"] == pytest.approx(0) and m["pck"] == 1 and m["dist.beak"] == pytest.approx(0)
    assert m["dist_center"] == pytest.approx(math.hypot(0.25, 0.25))
    m = ev.score(parts, [0, 1], torch.tensor([[0.0, 1.0]]), None, iter([(slice(0, 1), maps)]))
    assert m["dist"] == pytest.approx(math.hypot(0.5, 0.5)) and m["pck"] == 0
    # "present" averages over every concept annotated present
    ev = KeypointDistanceEvaluator(select="present")
    m = ev.score(parts, [0, 1], torch.tensor([[1.0, 0.0]]), torch.tensor([[True, True]]),
                 iter([(slice(0, 1), maps)]))
    assert m["dist"] == pytest.approx(math.hypot(0.5, 0.5) / 2)


def test_invisible_keypoints_are_skipped():
    maps = torch.zeros(1, 4, 2)
    ev = KeypointDistanceEvaluator()
    assert ev.score(_parts([[math.nan, math.nan]]), [0, 1], torch.zeros(1, 2), None,
                    iter([(slice(0, 1), maps)])) == {}


@pytest.mark.parametrize("hard", [True, False])
def test_part_iou(hard):
    segs = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    segs[0, 0, :2, :2] = True  # top-left quarter
    maps = torch.tensor([[[5.0, 0], [0, 1], [0, 1], [0, 1]]])  # concept 0 fires on the top-left patch only
    ev = PartIoUEvaluator(hard=hard, keep_ratio=0.25, seg_size=4)
    right = ev.score(_parts([[0.25, 0.25]], segs), [0, 1], torch.tensor([[1.0, 0.0]]), None,
                     iter([(slice(0, 1), maps)]))
    wrong = ev.score(_parts([[0.25, 0.25]], segs), [0, 1], torch.tensor([[0.0, 1.0]]), None,
                     iter([(slice(0, 1), maps)]))
    assert right["miou"] == right["iou.beak"] and right["miou"] > wrong["miou"]
    if hard:  # the top patch upsamples to exactly the masked quarter
        assert right["miou"] == pytest.approx(1) and wrong["miou"] == pytest.approx(0)


# -- end to end on a fake CUB ---------------------------------------------------------------------

# the `cub_root` fixture (a tiny fake CUB_200_2011 folder) lives in conftest.py


def test_cub_part_spec(cub_root):
    from cbm_eval.data.cub import CUB

    ds = CUB(str(cub_root), class_level_concepts=False, val_fraction=0.0)
    spec = ds.part_spec()
    kp, seg = spec.keypoint_groups, spec.seg_groups
    assert spec.concept_keypoints[0, kp.index("beak")] and spec.concept_keypoints[1, kp.index("wing")]
    assert spec.concept_segs[2, seg.index("body")] and not spec.concept_segs[3].any()
    s = ds.samples("test")[0]
    assert s.parts.size == (60, 40) and set(s.parts.masks["wing"][0].split("_")[-2:]) == {"left", "wing.png"}


@pytest.mark.parametrize("variant,bags", [({}, False), ({"select": "present"}, False), ({}, True)])
def test_cub_end_to_end(tmp_path, cub_root, variant, bags):
    cfg = ExperimentConfig.from_dict({
        "dataset": {"name": "cub", "root": str(cub_root), "class_level_concepts": False, "val_fraction": 0.3},
        "backbone": {"name": "resnet", "model": "resnet18", "weights": None, "image_size": 64},
        "stages": {"discovery": {"name": "dataset"}, "filtering": [{"name": "rules", "remove_class_names": False}],
                   "generation": {"name": "scores"}, "alignment": {"name": "human"},
                   "predictor": {"name": "dense"}, "training": {"name": "sequential", "epochs": 2}},
        "evaluation": [{"name": "localization"}, {"name": "keypoint_distance", **variant},
                       {"name": "part_iou", "hard": not variant, **variant}],
        "device": "cpu",
        "paths": {"cache": str(tmp_path / "cache"), "runs": str(tmp_path / "runs"),
                  "results": str(tmp_path / "r.jsonl")},
    })
    if bags:  # patches as instances: the maps are the per-patch concept scores
        cfg = cfg.with_overrides({"instances": {"name": "patches"}, "stages.predictor": {"name": "mil"},
                                  "stages.training": {"name": "joint", "epochs": 2}})
    m = run(cfg, save=False).metrics
    assert 0 <= m["keypoint_distance.dist"] <= math.sqrt(2) and 0 <= m["keypoint_distance.pck"] <= 1
    assert "keypoint_distance.dist.beak" in m and m["keypoint_distance.n_pairs"] > 0
    assert 0 <= m["part_iou.miou"] <= 1 and 0 <= m["part_iou.miou_center"] <= 1
    assert set(k for k in m if k.startswith("part_iou.iou.")) <= {"part_iou.iou.beak", "part_iou.iou.body",
                                                                  "part_iou.iou.wing"}
    assert list((tmp_path / "cache" / "parts" / "cub").glob("*.pt"))
