"""Fine-tuning speed options (``finetune: {input: decoded, precision: bf16, channels_last: true, compile: true}``),
offline: the batch augmentation, the decoded-image accessor and a fine-tuned run on the fake CUB."""

import pytest
import torch

from cbm_eval.backbones.augment import batch_augment
from cbm_eval.config import load_config
from cbm_eval.pipeline import PipelineBuilder, run
from cbm_eval.stages.training import Finetune
from cbm_eval.utils import rng_state, seed_everything


def _images():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(0, 256, (3, h, w), dtype=torch.uint8, generator=g) for h, w in ((40, 60), (70, 30))]


def test_batch_augmentation_is_seeded_and_shaped():
    out = lambda seed: batch_augment(_images(), 32, [0.5] * 3, [2.0] * 3, torch.Generator().manual_seed(seed))
    assert out(0).shape == (2, 3, 32, 32)
    assert torch.equal(out(0), out(0)) and not torch.equal(out(0), out(1))


def test_batch_augmentation_crops_never_read_the_padding():
    """Images of different sizes share one padded batch; a constant image must stay constant."""
    ims = [torch.full((3, 40, 60), 200, dtype=torch.uint8), torch.full((3, 70, 30), 200, dtype=torch.uint8)]
    for seed in range(5):
        x = batch_augment(ims, 32, [0.0] * 3, [1.0] * 3, torch.Generator().manual_seed(seed), brightness=0.0,
                        saturation=(1.0, 1.0))
        assert torch.allclose(x, torch.full_like(x, 200 / 255), atol=1e-5)


def test_unknown_speed_options_are_refused():
    with pytest.raises(ValueError, match="input"):
        Finetune(input="gpu")
    with pytest.raises(ValueError, match="precision"):
        Finetune(precision="fp16")
    with pytest.raises(ValueError, match="augment"):
        Finetune(input="decoded", augment=False)


def _fake_cub(cfg, cub_root, tmp_path, finetune: dict):
    finetune = cfg.stages["training"]["finetune"] | {"epochs": 2, "patience": 1, "num_workers": 0, "batch_size": 4,
                                                     **finetune}
    return cfg.with_overrides({
        "dataset": {"name": "cub", "root": str(cub_root), "majority_vote": "koh", "min_class_count": 1,
                    "val_fraction": 0.3},
        "backbone.weights": None, "backbone.image_size": 96,  # random init: no download; small for speed
        "stages.training.aux": None,  # the aux tap needs 299 px; see test_decoded_batches_feed_the_aux_tap
        "stages.training.finetune": finetune, "stages.training.device": "cpu", "device": "cpu",  # issue #61
        "paths": {"cache": str(tmp_path / "cache"), "runs": str(tmp_path / "runs"),
                  "results": str(tmp_path / "r.jsonl")},
    })


@pytest.mark.parametrize("anchor, train_on_val", [("cub_koh2020", True), ("cub_koh2020", False),
                                                   ("cub_koh2020_independent", False)])
def test_decoded_bf16_channels_last_fine_tuning_runs_and_repeats(cub_root, tmp_path, anchor, train_on_val):
    cfg = _fake_cub(load_config(f"configs/anchors/{anchor}.yaml"), cub_root, tmp_path,
                    {"input": "decoded", "precision": "bf16", "channels_last": True, "epochs": 1})
    if anchor == "cub_koh2020" and not train_on_val:
        cfg = cfg.with_overrides({"stages.training.train_on_val": False})
    first, second = run(cfg, save=False), run(cfg, save=False)
    assert first.cbm.train_log["finetune_epochs"] >= 1
    assert 0 <= first.metrics["shift.test.acc"] <= 1
    assert first.metrics == second.metrics  # seeded augmentation: the same run twice


def test_the_speed_options_change_nothing_by_default():
    cfg = load_config("configs/anchors/cub_koh2020.yaml")
    assert cfg.run_id == "3cde53c008"  # the run id of the results made before these options existed
    assert Finetune().input == "pil" and Finetune().precision == "fp32" and not Finetune().channels_last


def test_decoding_draws_no_random_numbers(cub_root, tmp_path):
    cfg = _fake_cub(load_config("configs/anchors/cub_koh2020.yaml"), cub_root, tmp_path, {})
    ctx = PipelineBuilder(cfg).context()
    seed_everything(0)
    before = rng_state()
    images = ctx.decoded("train")
    assert rng_state() == before
    assert len(images) == len(ctx.split("train").labels) and images[0].dtype == torch.uint8


def test_decoded_input_needs_a_backbone_that_augments_batches(base_cfg):
    cfg = base_cfg.with_overrides({"stages.training": {"name": "joint", "epochs": 1, "device": "cpu", "finetune": {
        "epochs": 1, "num_workers": 0, "batch_size": 4, "input": "decoded"}}, "evaluation": [{"name": "shift"}], "device": "cpu"})
    with pytest.raises(ValueError, match="input: pil"):
        run(cfg, save=False)


def test_batch_augmentation_flips_about_half_the_images():
    im = torch.zeros(3, 40, 40, dtype=torch.uint8)
    im[:, :, 20:] = 255  # dark left half, bright right half
    x = batch_augment([im] * 64, 16, [0.0] * 3, [1.0] * 3, torch.Generator().manual_seed(0), brightness=0.0,
                    saturation=(1.0, 1.0), scale=(1.0, 1.0), ratio=(1.0, 1.0))
    flipped = x[:, 0, 8, 0] > x[:, 0, 8, -1]  # bright on the left
    assert 16 < int(flipped.sum()) < 48


def test_the_padded_store_gives_the_crops_of_per_batch_padding():
    from cbm_eval.backbones.augment import PaddedImages, pad_images

    g = torch.Generator().manual_seed(3)
    ims = [torch.randint(0, 256, (3, int(h), int(w)), dtype=torch.uint8, generator=g)
           for h, w in torch.randint(20, 80, (12, 2), generator=g)]
    store, idx = pad_images(ims), torch.tensor([5, 0, 11, 7])
    batch = PaddedImages(store.pixels[idx], store.sizes[idx])  # padded to the whole set's largest size
    a = batch_augment(batch, 24, [0.5] * 3, [0.25] * 3, torch.Generator().manual_seed(0))
    b = batch_augment([ims[i] for i in idx], 24, [0.5] * 3, [0.25] * 3, torch.Generator().manual_seed(0))
    assert torch.equal(a, b)


def test_very_uneven_images_are_kept_as_a_list():
    from cbm_eval.backbones.augment import PaddedImages
    from cbm_eval.stages.training import _decoded_store

    even = [torch.zeros(3, 40, 50, dtype=torch.uint8)] * 4
    uneven = even + [torch.zeros(3, 400, 500, dtype=torch.uint8)]
    assert isinstance(_decoded_store(even, torch.device("cpu")), PaddedImages)
    assert isinstance(_decoded_store(uneven, torch.device("cpu")), list)


def test_compile_wraps_the_training_forward_pass(cub_root, tmp_path, monkeypatch):
    compiled = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: compiled.append(fn) or fn)  # no real compile offline
    cfg = _fake_cub(load_config("configs/anchors/cub_koh2020.yaml"), cub_root, tmp_path,
                    {"input": "decoded", "compile": True, "epochs": 1})
    run(cfg, save=False)
    assert len(compiled) == 1


def test_decoded_batches_feed_the_aux_tap():
    """Inception's augment_batch gives inputs that the encoder and its aux tap (299 px only) accept."""
    from cbm_eval.backbones.inception import InceptionBackbone

    bb = InceptionBackbone(weights=None, mean=[0.5] * 3, std=[2.0] * 3)
    x = bb.augment_batch(_images(), torch.Generator().manual_seed(0))
    assert x.shape == (2, 3, 299, 299)
    with torch.no_grad():
        final, aux = bb.encoder(aux=True).eval().forward_aux(x.contiguous(memory_format=torch.channels_last))
    assert final.shape == (2, bb.dim) and aux.shape == (2, bb.aux_dim)
