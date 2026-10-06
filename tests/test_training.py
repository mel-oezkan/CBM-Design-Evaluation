"""Training on frozen features normalizes them once, computes a frozen concept layer's predictions
once, and drops features no component reads. None of that may change a single trained weight."""

import itertools

import pytest
import torch

from cbm_eval.pipeline import PipelineBuilder
from cbm_eval.stages.training import (Independent, IndependentWeighted, Joint, JointWeighted, Sequential,
                                      SequentialWeighted)
from cbm_eval.utils import seed_everything


class _PerStep:
    """Every minibatch normalizes its features and recomputes frozen concept predictions, the head
    always gets ``x_norm`` and features are never dropped: training as it was before the precomputation."""

    @staticmethod
    def _normalized(layer, *batches):
        return list(batches)

    def _fit(self, layer, head, *args, **kw):
        layer.represent_uses_x = head.uses_features = True
        return super()._fit(layer, head, *args, **kw)

    def _fit_head(self, layer, head, tr, va, rep_fn, seed):
        if isinstance(self, Sequential):
            def rep_fn(b):
                with torch.no_grad():
                    c_hat = layer.activate(layer.concept_logits(b.x))
                return layer.represent(c_hat, b.x)
        return super()._fit_head(layer, head, tr, va, rep_fn, seed)


PER_STEP = {cls: type(f"PerStep{cls.__name__}", (_PerStep, cls), {})
            for cls in (Sequential, Independent, Joint, SequentialWeighted, IndependentWeighted, JointWeighted)}

GENERATIONS = [{"name": "scores"}, {"name": "logits"}, {"name": "boc"}, {"name": "embeddings", "emb_dim": 4}]
PREDICTORS = [{"name": "dense"}, {"name": "sparse"}, {"name": "residual"}]
HUMAN = [{"stages.alignment": {"name": "human"}, "stages.generation": g, "stages.predictor": p,
          "stages.training": {"name": t}}
         for g, p, t in itertools.product(GENERATIONS, PREDICTORS, ["sequential", "independent", "joint"])
         if p["name"] == "dense" or g["name"] == "scores"]
OTHER = [
    {"stages.alignment": {"name": "human"}, "stages.training": {"name": "sequential_weighted"}},
    {"stages.alignment": {"name": "human"}, "stages.training": {"name": "joint_weighted"}},
    {"stages.alignment": {"name": "weights"}, "stages.training": {"name": "sequential"}},
    {"stages.alignment": {"name": "weights"}, "stages.training": {"name": "joint"}},
    {"stages.training": {"name": "sequential"}},  # clip alignment: continuous targets
    {"instances": {"name": "patches"}, "stages.predictor": {"name": "mil"}, "stages.training": {"name": "sequential"}},
    {"instances": {"name": "patches"}, "stages.predictor": {"name": "mil"}, "stages.training": {"name": "joint"}},
]


def _train(cfg, per_step: bool) -> dict[str, torch.Tensor]:
    builder = PipelineBuilder(cfg)
    if per_step:
        spec = {k: v for k, v in cfg.stages["training"].items() if k != "name"}
        builder.training = PER_STEP[type(builder.training)](**spec)
    ctx = builder.context()
    concepts = builder.concepts(ctx)
    for split in ("train", "val"):
        ctx.inputs(split)  # extracting features on a cold cache draws from the global RNG
    seed_everything(cfg.seed)
    return builder.train(ctx, concepts).model.state_dict()


@pytest.mark.parametrize("overrides", HUMAN + OTHER, ids=lambda o: "-".join(
    v["name"] if isinstance(v, dict) else k for k, v in o.items()))
def test_precomputed_training_matches_per_step_training(base_cfg, overrides):
    cfg = base_cfg.with_overrides({"evaluation": [], **overrides})
    cfg = cfg.with_overrides({"stages.training.epochs": 5})
    fast, slow = _train(cfg, per_step=False), _train(cfg, per_step=True)
    assert fast.keys() == slow.keys()
    for name in fast:
        assert torch.equal(fast[name], slow[name]), name


PADDED_BAGS = [
    {"stages.training": {"name": "sequential"}},
    {"stages.training": {"name": "joint"}},
    {"stages.generation": {"name": "embeddings", "emb_dim": 4}, "stages.training": {"name": "joint"}},
    {"stages.alignment": {"name": "segment_clip"},  # per-instance targets
     "stages.training": {"name": "joint", "concept_weight": 0.1, "concept_loss": "cosine"}},
]


@pytest.mark.parametrize("overrides", PADDED_BAGS, ids=lambda o: "-".join(v["name"] for v in o.values()))
def test_padded_bags_train_on_real_instances_only(base_cfg, overrides, monkeypatch):
    """Concept logits of zero-padded bags are computed on the real instances only; the padding is
    masked everywhere, so the trained weights must not change (up to float summation order)."""
    from cbm_eval.context import Context
    from cbm_eval.structures import Bag

    inputs = Context.inputs

    def padded(self, name):  # image i keeps its first 1 + i % M patches, as segment bags are padded
        x, bag = inputs(self, name)
        n = 1 + torch.arange(len(x)) % x.shape[1]
        mask = torch.arange(x.shape[1])[None] < n[:, None]
        return x * mask[..., None], Bag(mask, bag.area)

    monkeypatch.setattr(Context, "inputs", padded)
    cfg = base_cfg.with_overrides({"evaluation": [], "instances": {"name": "patches"},
                                   "stages.predictor": {"name": "mil"}, **overrides})
    cfg = cfg.with_overrides({"stages.training.epochs": 5})
    packed = _train(cfg, per_step=False)
    monkeypatch.setattr(Bag, "on_rows", lambda self, fn, x: fn(x))  # every padded slot computed
    full = _train(cfg, per_step=False)
    # The attention's last bias adds the same constant to every instance before the softmax, so its
    # gradient is rounding noise that Adam scales up to lr-sized steps; it cannot change an output.
    for name in packed.keys() - {"head.attn.2.bias"}:
        assert torch.allclose(packed[name], full[name], rtol=1e-4, atol=1e-6), name


def test_training_optimizer_applies_unless_the_head_brings_its_own():
    from cbm_eval.stages.predictor import Dense, Sparse

    stage = Sequential(optimizer="sgd", momentum=0.9, weight_decay=1e-4, lr=0.05)
    _, dense = stage._head_group(Dense().build(4, 3, 8), [])
    _, sparse = stage._head_group(Sparse(lr=0.1).build(4, 3, 8), [])
    assert (dense.kind, dense.lr, dense.momentum, dense.weight_decay) == ("sgd", 0.05, 0.9, 1e-4)
    assert (sparse.kind, sparse.lr, sparse.momentum, sparse.weight_decay) == ("sgd", 0.1, 0.0, 0.0)
    assert Sequential()._optimizer().kind == "adam"
    with pytest.raises(ValueError, match="stages.training.optimizer"):
        Joint(optimizer="lbfgs")


def test_sgd_training_runs(base_cfg):
    cfg = base_cfg.with_overrides({"evaluation": [], "stages.alignment": {"name": "human"},
                                   "stages.predictor": {"name": "dense"},
                                   "stages.training": {"name": "joint", "epochs": 5, "optimizer": "sgd",
                                                       "momentum": 0.9, "weight_decay": 1e-4, "lr": 0.01}})
    state = _train(cfg, per_step=False)
    assert all(torch.isfinite(v).all() for v in state.values())


def test_momentum_needs_sgd():
    with pytest.raises(ValueError, match="optimizer: sgd"):
        Joint(momentum=0.9)
