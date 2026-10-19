"""Training options that reproduce a paper's recipe on frozen features: model selection, training on
train + val, per-phase optimizers, ``drop_last``, augmented views, loss normalization, and the
concept layer's dropout / standardization / init. Each is off by default (see ``test_training.py``
for the defaults staying bit-identical)."""

import pytest
import torch

from cbm_eval.backbones.toy import ToyBackbone
from cbm_eval.data.features import load_augmented_features
from cbm_eval.data.synthetic import SyntheticDataset
from cbm_eval.pipeline import PipelineBuilder, run
from cbm_eval.stages.base import AlignedConcepts
from cbm_eval.stages.generation import Scores
from cbm_eval.stages.training import Joint, Sequential, optimize
from cbm_eval.structures import ConceptSet
from cbm_eval.utils import seed_everything

HUMAN = {"stages.alignment": {"name": "human"}, "stages.predictor": {"name": "dense"}, "evaluation": []}


class Noise:
    """A stochastic ``train_transform`` for the synthetic dataset's tensor images."""

    def __call__(self, image):
        return image + 0.5 * torch.randn_like(image)

    def __repr__(self):
        return "Noise(0.5)"


@pytest.fixture
def noisy_views(monkeypatch):
    monkeypatch.setattr(ToyBackbone, "train_transform", lambda self: Noise())


def _cfg(base_cfg, training, **overrides):
    return base_cfg.with_overrides({**HUMAN, "stages.training": training, **overrides})


def _state(cfg):
    builder = PipelineBuilder(cfg)
    ctx = builder.context()
    concepts = builder.concepts(ctx)
    for split in ("train", "val"):
        ctx.inputs(split)
    seed_everything(cfg.seed)
    cbm = builder.train(ctx, concepts)
    return cbm, ctx


def test_selection_by_validation_accuracy_keeps_the_best_epoch(base_cfg):
    cbm, ctx = _state(_cfg(base_cfg, {"name": "independent", "epochs": 30, "select": "val_acc",
                                      "optimizer": "sgd", "lr": 0.05}))
    log = cbm.train_log
    assert 1 <= log["head_best_epoch"] <= log["head_phase0_epochs"]
    # the restored head is the one whose validation accuracy was logged (independent: on true concepts)
    targets = cbm.aligned.targets("val")
    acc = (cbm.model.head(cbm.model.layer.represent(targets, None), None).argmax(1) == ctx.split("val").labels)
    assert acc.float().mean().item() == pytest.approx(log["head_val_acc"])


def test_train_on_val_needs_training_accuracy_and_trains_on_both_splits(base_cfg):
    with pytest.raises(ValueError, match="select: train_acc"):
        Joint(train_on_val=True)
    with pytest.raises(ValueError, match="concept_select: train_acc"):
        Sequential(concept_train_on_val=True)
    cbm, _ = _state(_cfg(base_cfg, {"name": "sequential", "epochs": 5, "concept_select": "train_acc",
                                    "concept_train_on_val": True}))
    log = cbm.train_log
    assert 0.5 < log["concept_train_acc"] <= 1 and log["concept_val_loss"] != log["concept_val_loss"]  # NaN: no val
    assert "head_val_loss" in log and log["head_val_loss"] == log["head_val_loss"]  # the head still validates


def test_drop_last_skips_the_incomplete_minibatch():
    sizes = []
    w = torch.nn.Parameter(torch.zeros(1))

    def step(idx):
        sizes.append(len(idx))
        return (w - 1).pow(2).sum(), None

    from cbm_eval.stages.training import OptimizerSpec
    for drop in (False, True):
        sizes.clear()
        optimize([([w], OptimizerSpec())], [], 10, step, lambda: (torch.tensor(0.0), None), 1, 4, 1, 0,
                 drop_last=drop)
        assert sizes == ([4, 4, 2] if not drop else [4, 4])


def test_concept_phase_has_its_own_optimizer_settings():
    stage = Sequential(optimizer="sgd", lr=0.001, weight_decay=5e-5, concept_lr=0.01, concept_weight_decay=4e-5)
    assert (stage._optimizer().lr, stage._optimizer().weight_decay) == (0.001, 5e-5)
    spec = stage._optimizer(stage.concept_lr, stage.concept_weight_decay)
    assert (spec.lr, spec.weight_decay) == (0.01, 4e-5)


def test_normalize_loss_equals_scaling_lr_and_weight_decay(base_cfg):
    """(task + w * concept) / (1 + w) under SGD = the unnormalized loss at lr / (1 + w), wd * (1 + w)."""
    w, lr, wd = 1.5, 0.05, 1e-3
    common = {"name": "joint", "epochs": 4, "concept_weight": w, "optimizer": "sgd", "momentum": 0.9, "patience": 100}
    a, _ = _state(_cfg(base_cfg, {**common, "normalize_loss": True, "lr": lr, "weight_decay": wd}))
    b, _ = _state(_cfg(base_cfg, {**common, "lr": lr / (1 + w), "weight_decay": wd * (1 + w)}))
    for (name, p), q in zip(a.model.state_dict().items(), b.model.state_dict().values()):
        assert torch.allclose(p, q, atol=1e-5), name


def test_concept_layer_options():
    aligned = AlignedConcepts(ConceptSet(["a", "b", "c"]), target_type="binary")
    torch.manual_seed(0)
    layer = Scores(dropout=0.5, standardize=False, init_std=0.1).build(64, aligned)
    assert layer.linear.weight.abs().max() <= 0.2 and layer.linear.weight.std() > 0.05  # truncated at 2 std
    x = torch.randn(32, 64) * 3 + 1
    layer.fit_input_stats(x)
    assert torch.equal(layer.normalize(x), x)  # raw features
    layer.eval()
    assert torch.equal(layer.concept_logits(x), layer.concept_logits(x))
    layer.train()
    assert not torch.equal(layer.concept_logits(x), layer.concept_logits(x))  # dropout only while training
    assert Scores().build(64, aligned).dropout is None  # no dropout module, so no RNG draws, by default


def test_augmented_views_are_cached_seeded_and_independent_of_the_view_count(tmp_path, noisy_views):
    ds = SyntheticDataset(sizes={"train": 20, "val": 10, "test": 10})
    bb = ToyBackbone()
    before = torch.random.get_rng_state()
    three = load_augmented_features(ds, bb, "train", tmp_path, views=3, batch_size=7)
    assert torch.equal(torch.random.get_rng_state(), before)  # the global RNG is untouched
    two = load_augmented_features(ds, bb, "train", tmp_path, views=2, batch_size=4)
    assert three.shape == (20, 3, bb.dim) and three.dtype == torch.float16
    assert torch.equal(three[:, :2], two)  # view v of image i is fixed by (seed, v, i) alone
    assert not torch.equal(three[:, 0], three[:, 1])
    other = load_augmented_features(ds, bb, "train", tmp_path, views=2, seed=1)
    assert not torch.equal(other, two)
    assert len(list((tmp_path / "features").rglob("train-views-*.pt"))) == 3


@pytest.mark.parametrize("training", [
    {"name": "independent_weighted", "concept_select": "train_acc", "concept_train_on_val": True},
    {"name": "sequential", "head_on_augmented": True},
    {"name": "joint", "select": "train_acc", "train_on_val": True, "concept_weight": 0.5, "normalize_loss": True},
])
def test_training_on_augmented_views_runs(base_cfg, noisy_views, training):
    cfg = _cfg(base_cfg, {**training, "epochs": 20, "lr": 0.01, "augment": {"views": 3, "num_workers": 0}},
               **{"stages.generation": {"name": "logits", "dropout": 0.5, "standardize": False, "init_std": 0.1},
                  "evaluation": [{"name": "shift"}, {"name": "concepts"}]})
    m = run(cfg, save=False).metrics
    assert m["shift.test.acc"] > 0.3 and m["concepts.test.concept_error"] < 0.5


def test_augmented_views_change_training(base_cfg, noisy_views):
    spec = {"name": "sequential", "epochs": 3}
    clean, _ = _state(_cfg(base_cfg, spec))
    augmented, _ = _state(_cfg(base_cfg, {**spec, "augment": {"views": 2, "num_workers": 0}}))
    assert not torch.equal(clean.model.layer.linear.weight, augmented.model.layer.linear.weight)


def test_invalid_combinations_are_refused(base_cfg):
    with pytest.raises(ValueError, match="needs `stages.training.augment`"):
        Sequential(head_on_augmented=True)
    with pytest.raises(ValueError, match="finetune"):
        Joint(finetune={"epochs": 1}, augment={"views": 2})
    with pytest.raises(ValueError, match="Unknown `stages.training.select`"):
        Joint(select="val_f1")
    cfg = base_cfg.with_overrides({"instances": {"name": "patches"}, "stages.predictor": {"name": "mil"},
                                   "stages.training": {"name": "sequential", "augment": {"views": 2}},
                                   "evaluation": []})
    with pytest.raises(ValueError, match="instance bags"):
        run(cfg, save=False)
    cfg = base_cfg.with_overrides({"stages.training": {"name": "sequential", "concept_select": "val_acc"},
                                   "evaluation": []})  # clip alignment: continuous targets
    with pytest.raises(ValueError, match="binary image-level"):
        run(cfg, save=False)
