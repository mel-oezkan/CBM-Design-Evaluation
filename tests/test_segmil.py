"""SEG-MIL-CBM components: instance bags, the MIL head, segment-level alignment, segment sources
(with stub Grounding DINO / SAM), and the faithfulness evaluator. Fully offline."""

import math

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from cbm_eval.backbones.base import Backbone
from cbm_eval.data.synthetic import toy_latent
from cbm_eval.grounding import Detection, match_phrase
from cbm_eval.instances.segments import GroundedSAM, Segment, pack, unpack
from cbm_eval.pipeline import PipelineBuilder, run
from cbm_eval.registry import BACKBONES
from cbm_eval.stages.predictor import LinearHead, MILHead
from cbm_eval.stages.training import concept_loss
from cbm_eval.structures import Bag

SEGMIL = {
    "instances": {"name": "patches"},
    "stages.alignment": {"name": "segment_clip"},
    "stages.predictor": {"name": "mil"},
    "stages.training": {"name": "joint", "concept_weight": 0.1, "concept_loss": "cosine", "epochs": 30},
    "evaluation": [{"name": "shift"}, {"name": "leakage", "repeats": 1}, {"name": "localization"},
                   {"name": "faithfulness"}],
}


def _ok(metrics):
    return all(not math.isnan(v) for v in metrics.values())


# -- bags on the synthetic data (patches as instances) ---------------------------------------------


def test_patch_mil_end_to_end(base_cfg):
    m = run(base_cfg.with_overrides(SEGMIL), save=False).metrics
    assert _ok(m)
    assert m["shift.test.acc"] > 0.5
    assert m["instances.per_image"] == 16
    assert m["faithfulness.dauc"] < m["faithfulness.dauc_random"]
    assert m["faithfulness.iauc"] > m["faithfulness.iauc_random"]
    assert m["localization.pointing_game"] > m["localization.pointing_chance"]


@pytest.mark.parametrize("override", [
    {"stages.predictor": {"name": "mil", "pooling": "mean"}},
    {"stages.predictor": {"name": "mil", "pooling": "max"}},
    {"stages.predictor": {"name": "mil", "area_gamma": 0.25}},
    {"stages.alignment": {"name": "segment_clip", "targets": "standardized"}},
    {"stages.alignment": {"name": "human"}},  # image-level targets supervise the max over patches
    {"stages.alignment": {"name": "weights"}},
    {"stages.training": {"name": "sequential", "epochs": 100}},
    {"stages.generation": {"name": "embeddings", "emb_dim": 4}},
])
def test_bag_variants_run(base_cfg, override):
    cfg = base_cfg.with_overrides(SEGMIL).with_overrides(override)
    m = run(cfg, save=False).metrics
    assert _ok(m), m
    assert m["shift.test.acc"] > 0.4


def test_mil_on_global_features_equals_dense(base_cfg):
    # a bag of one gets attention weight 1, so the MIL head is exactly a linear head
    cheap = base_cfg.with_overrides({"evaluation": [{"name": "shift"}]})
    mil = run(cheap.with_overrides({"stages.predictor": {"name": "mil"}}), save=False).metrics
    dense = run(cheap.with_overrides({"stages.predictor": {"name": "dense"}}), save=False).metrics
    assert mil["shift.test.acc"] == pytest.approx(dense["shift.test.acc"], abs=0.02)  # float drift only
    head, linear = _head(), LinearHead(5, 3)
    linear.linear.load_state_dict(head.cls.state_dict())
    rep, xn = torch.randn(4, 5), torch.randn(4, 7)
    assert torch.allclose(head(rep, xn), linear(rep, xn), atol=1e-6)


def test_bag_needs_a_bag_aware_predictor(base_cfg):
    cfg = base_cfg.with_overrides({"instances": {"name": "patches"}})
    with pytest.raises(ValueError, match="bag-aware predictor"):
        PipelineBuilder(cfg)


def test_segment_clip_needs_instances(base_cfg):
    with pytest.raises(RuntimeError, match="needs instance bags"):
        run(base_cfg.with_overrides({"stages.alignment": {"name": "segment_clip"}}), save=False)


def test_instances_key_only_changes_run_id_when_set(base_cfg):
    assert "instances" not in base_cfg.to_dict()
    assert base_cfg.with_overrides({"instances": None}).run_id == base_cfg.run_id
    assert base_cfg.with_overrides({"instances": {"name": "patches"}}).run_id != base_cfg.run_id


def test_image_level_intervention_broadcasts_over_instances(base_cfg):
    cfg = base_cfg.with_overrides(SEGMIL).with_overrides({"stages.alignment": {"name": "human"}})
    builder = PipelineBuilder(cfg)
    ctx = builder.context()
    cbm = builder.train(ctx, builder.concepts(ctx))
    x, bag = ctx.inputs("test")
    gt = cbm.aligned.targets("test")  # (N, K) image level
    _, c_hat = cbm.predict(x, intervene=(torch.ones_like(gt, dtype=torch.bool), gt), bag=bag)
    assert c_hat.shape == (*bag.mask.shape, gt.shape[1])
    assert torch.equal(c_hat, gt[:, None].expand_as(c_hat))


# -- unit tests ------------------------------------------------------------------------------------


def _head(pooling="attention", area_gamma=0.0):
    torch.manual_seed(0)
    return MILHead(5, 3, 7, pooling, attn_dim=4, tau=1.0, area_gamma=area_gamma, area_min=0.01)


@pytest.mark.parametrize("pooling", ["attention", "mean", "max"])
def test_mil_head_ignores_padding(pooling):
    head = _head(pooling, area_gamma=0.25)
    rep, xn = torch.randn(2, 3, 5), torch.randn(2, 3, 7)
    area = torch.rand(2, 3) * 0.5 + 0.05
    base = head(rep, xn, Bag(torch.ones(2, 3, dtype=torch.bool), area))
    # append a padded instance with extreme values: the output must not move
    rep_p = torch.cat([rep, 100 * torch.ones(2, 1, 5)], 1)
    xn_p = torch.cat([xn, 100 * torch.ones(2, 1, 7)], 1)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    padded = head(rep_p, xn_p, Bag(mask, torch.cat([area, torch.full((2, 1), 1e-4)], 1)))
    assert torch.allclose(base, padded, atol=1e-5)


def test_mil_contributions_sum_to_logits_minus_bias():
    head = _head()
    rep, xn = torch.randn(4, 6, 5), torch.randn(4, 6, 7)
    bag = Bag(torch.rand(4, 6) > 0.3)
    bag.mask[:, 0] = True
    alpha, eta, _ = head.evidence(rep, xn, bag)
    bias = (alpha * eta[..., None]).sum(1) * head.cls.bias
    assert torch.allclose(head.contributions(rep, xn, bag).sum(1) + bias, head(rep, xn, bag), atol=1e-5)


def test_area_weights_average_to_one():
    head = _head(area_gamma=0.25)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    eta = head._area_weights(Bag(mask, torch.tensor([[0.5, 0.02, 0.9]])), mask)
    assert eta[0, :2].mean().item() == pytest.approx(1.0) and eta[0, 2] == 0
    assert eta[0, 1] > eta[0, 0]  # small segments are up-weighted


def test_concept_loss_masks_padding():
    class Layer:
        target_type = "continuous"

        def concept_logits(self, x):
            return x

    x = torch.tensor([[[1.0, 0.0], [5.0, 5.0]]])
    t = torch.tensor([[[1.0, 0.0], [-5.0, 5.0]]])
    bag = Bag(torch.tensor([[True, False]]))
    assert concept_loss(Layer(), x, t, bag, "cosine").item() == pytest.approx(0.0, abs=1e-6)
    assert concept_loss(Layer(), x, t, None, "cosine").item() > 0.1


def test_pack_roundtrip_and_match_phrase():
    mask = np.zeros((90, 120), dtype=bool)
    mask[20:60, 30:100] = True
    seg = Segment((30, 20, 100, 60), 0.9, 3, mask)
    back = unpack(pack(seg, (120, 90)), (120, 90))
    assert back.concept == 3 and (back.mask == mask).mean() > 0.99
    assert match_phrase("long beak", ["wing", "long beak"]) == 1
    assert match_phrase("red wing long beak", ["wing", "long beak"]) == 1  # merged phrase -> longest
    assert match_phrase("tree", ["wing", "long beak"]) == -1


def test_postprocess_filters_merges_and_falls_back():
    src = GroundedSAM(max_segments=2, min_pixels=10, max_frac=0.8, merge_iou=0.5)
    size = (20, 20)

    def box(x0, y0, x1, y1, s):
        return Segment((x0, y0, x1, y1), s)

    out = src.postprocess([box(0, 0, 2, 2, 0.9),  # too small
                           box(0, 0, 20, 19, 0.8),  # covers 95% of the image
                           box(2, 2, 10, 10, 0.7), box(2, 2, 10, 11, 0.6),  # merged into one
                           box(12, 12, 18, 18, 0.5), box(14, 2, 18, 6, 0.4)], size)
    assert len(out) == 2 and out[0].box == (2.0, 2.0, 10.0, 11.0) and out[1].score == 0.5
    fallback = src.postprocess([], size)
    assert len(fallback) == 1 and fallback[0].concept == -1


# -- segment sources on image files, with stub detector / SAM --------------------------------------


@BACKBONES.register("test_pixels")
class PixelBackbone(Backbone):
    """Tiny image encoder with a text tower, so concept-guided segmentation runs without CLIP."""

    has_text = True
    dim = 8

    def __init__(self):
        super().__init__()
        self.proj = torch.randn(3 * 4 * 4, self.dim, generator=torch.Generator().manual_seed(0))
        self.patch_grid = 4

    def cache_key(self):
        return {"name": "test_pixels"}

    def transform(self):
        return T.Compose([T.Resize((16, 16)), T.ToTensor()])

    def encode_patches(self, images):
        p = images.unfold(2, 4, 4).unfold(3, 4, 4)  # (B, 3, 4, 4, 4, 4)
        return p.permute(0, 2, 3, 1, 4, 5).flatten(3).flatten(1, 2) @ self.proj

    def encode_images(self, images):
        return self.encode_patches(images).mean(1)

    def encode_text(self, texts):
        return torch.stack([toy_latent(t, self.dim) for t in texts])


class StubDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, images, prompts):
        self.calls += 1
        out = []
        for i, (im, names) in enumerate(zip(images, prompts)):
            w, h = im.size
            if i == 1:  # one image per batch without detections -> whole-image fallback
                out.append([])
                continue
            out.append([Detection((2 + 8 * j, 2, 18 + 8 * j, 20), 0.9 - 0.1 * j, j) for j in range(len(names))]
                       + [Detection((0, 0, 5, 5), 0.99, -1)])  # unmatched phrase is ignored
        return out


class StubSAM:
    def from_boxes(self, image, boxes):
        w, h = image.size
        masks = []
        for x0, y0, x1, y1 in boxes:
            m = np.zeros((h, w), dtype=bool)
            m[int(y0) + 1 : int(y1), int(x0) + 1 : int(x1)] = True
            masks.append(m)
        return masks


@pytest.fixture
def image_root(tmp_path):
    root = tmp_path / "waterbirds"
    rows = ["img_filename,y,split,place"]
    rng = np.random.default_rng(0)
    for i in range(24):
        y, place, split = i % 2, (i // 2) % 2, [0, 0, 1, 2][i % 4]
        rel = f"{y}/img_{i}.jpg"
        (root / str(y)).mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(root / rel)
        rows.append(f"{rel},{y},{split},{place}")
    (root / "metadata.csv").write_text("\n".join(rows))
    return root


def _image_cfg(base_cfg, root, tmp_path, instances):
    return base_cfg.with_overrides(SEGMIL).with_overrides({
        "dataset": {"name": "waterbirds", "root": str(root)},
        "backbone": {"name": "test_pixels"},
        "instances": instances,
        "stages.discovery": {"name": "static", "names": ["feathers", "water", "beak", "branch", "sky"]},
        "stages.filtering": [{"name": "rules"}],
        "evaluation": [{"name": "shift"}, {"name": "leakage", "repeats": 1}, {"name": "faithfulness"}],
        "paths.cache": str(tmp_path / "cache"),
    })


@pytest.mark.parametrize("encode", ["crop", "pool"])
def test_grounded_sam_with_stubs(base_cfg, image_root, tmp_path, encode):
    cfg = _image_cfg(base_cfg, image_root, tmp_path,
                     {"name": "grounded_sam", "top_k": 3, "min_pixels": 4, "batch_size": 4, "encode": encode})
    builder = PipelineBuilder(cfg)
    ctx = builder.context()
    det = StubDetector()
    ctx.instance_source.detector, ctx.instance_source.sam = det, StubSAM()
    res = run(cfg, save=False, ctx=ctx)
    assert _ok(res.metrics)

    inst = ctx.instances("train")
    n_real = inst.mask.sum(1)
    assert inst.features.shape == (12, 3, 8) and inst.regions.shape == (12, 3, 16)
    assert (n_real[1::4] == 1).all() and (inst.concepts[1::4, 0] == -1).all()  # fallback images
    assert (n_real[0::4] == 3).all() and (inst.concepts[0][inst.mask[0]] >= 0).all()
    assert inst.area[inst.mask].max() <= 1 and (inst.area[~inst.mask] == 0).all()
    assert list((tmp_path / "cache" / "segments" / "waterbirds").glob("*.pt"))
    assert list((tmp_path / "cache" / "instances" / "waterbirds").glob("*.pt"))

    # a second context loads both caches: the detector is not called again
    ctx2 = PipelineBuilder(cfg).context()
    ctx2.instance_source.detector = det
    ctx2.concepts = ctx.concepts
    calls = det.calls
    assert torch.equal(ctx2.instances("train").mask, inst.mask)
    assert det.calls == calls


def test_grounded_boxes_and_backbone_swap_reuse_segments(base_cfg, image_root, tmp_path):
    cfg = _image_cfg(base_cfg, image_root, tmp_path, {"name": "grounded_boxes", "top_k": None, "min_pixels": 4})
    builder = PipelineBuilder(cfg)
    ctx = builder.context()
    det = StubDetector()
    ctx.instance_source.detector = det
    assert _ok(run(cfg, save=False, ctx=ctx).metrics)
    calls = det.calls
    # a different encoding re-encodes features but reuses the cached masks
    cfg2 = cfg.with_overrides({"instances.encode": "pool"})
    ctx2 = PipelineBuilder(cfg2).context()
    ctx2.instance_source.detector = det
    assert _ok(run(cfg2, save=False, ctx=ctx2).metrics)
    assert det.calls == calls
