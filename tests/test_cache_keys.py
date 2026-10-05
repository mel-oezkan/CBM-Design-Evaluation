"""Every constructor argument of a disk-cached component must either change its cache key or be
listed as exempt with a reason. A new argument fails this test until that choice is made, so a
cache can't silently serve results computed under different settings."""

import inspect
import pickle
import shutil
from types import SimpleNamespace

import pytest

from cbm_eval.backbones.inception import InceptionBackbone
from cbm_eval.backbones.toy import ToyBackbone
from cbm_eval.data.cub import CUB
from cbm_eval.data.metashift import MetaShift
from cbm_eval.data.synthetic import SyntheticDataset
from cbm_eval.data.waterbirds import Waterbirds
from cbm_eval.instances.segments import GroundedBoxes, GroundedSAM, SAMGrid
from cbm_eval.stages.llm import ClaudeClient
from cbm_eval.stages.scorers import GroundingDINOScorer
from cbm_eval.structures import ConceptSet

SEGMENT_PARAMS = {"max_segments": 4, "min_pixels": 10, "max_frac": 0.5, "merge_iou": 0.5, "encode": "pool",
                  "margin": 0.2, "background": "keep"}
BATCHING = {"batch_size": "only groups images", "encode_batch_size": "only groups crops"}
CTX = SimpleNamespace(concepts=ConceptSet(["wing", "beak"]), teacher=None)


def init_params(cls) -> set[str]:
    """Named __init__ parameters, following ``**kw`` up to the parent constructors."""
    names: set[str] = set()
    for klass in cls.__mro__:
        if "__init__" not in vars(klass):
            continue
        params = inspect.signature(klass.__init__).parameters.values()
        names |= {p.name for p in params if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        if not any(p.kind is p.VAR_KEYWORD for p in params):
            break
    return names


def _metadata(root, header: str):
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.csv").write_text(header + "\n")
    return str(root)


# (class, build(**overrides) -> instance, key(instance) -> dict, perturbed values, exempt params with reasons)
CASES = {
    "synthetic": (SyntheticDataset, lambda **kw: SyntheticDataset(**kw), lambda o: o.cache_key(),
                  {"n_classes": 3, "n_concepts": 12, "n_distractors": 2, "latent_dim": 32, "grid": 2,
                   "sizes": {"train": 10, "val": 10, "test": 10}, "spurious_corr": 0.5, "p_on": 0.7, "p_off": 0.2,
                   "noise": 0.1, "concept_strength": 1.0, "background_strength": 0.5, "residual_strength": 0.1,
                   "data_seed": 1}, {}),
    "toy": (ToyBackbone, lambda **kw: ToyBackbone(**kw), lambda o: o.cache_key(),
            {"latent_dim": 32, "dim": 64, "seed": 1},
            {"grid": "patch_grid metadata only; patch features come from the synthetic dataset"}),
    "inception": (InceptionBackbone, lambda **kw: InceptionBackbone(**kw), lambda o: o.cache_key(),
                  {"weights": None, "image_size": 331}, {}),  # weights load lazily, so nothing is downloaded
    "grounding_dino": (GroundingDINOScorer, lambda **kw: GroundingDINOScorer(**kw), lambda o: o.cache_key(),
                       {"model": "IDEA-Research/grounding-dino-base", "box_threshold": 0.4, "concepts_per_prompt": 4},
                       {"batch_size": BATCHING["batch_size"]}),
    "grounded_sam": (GroundedSAM, lambda **kw: GroundedSAM(**kw), lambda o: o.cache_key(CTX),
                     {**SEGMENT_PARAMS, "top_k": 5, "boxes_per_concept": 3, "detector": {"box_threshold": 0.4},
                      "sam": {"model": "facebook/sam-vit-large"}}, BATCHING),
    "grounded_boxes": (GroundedBoxes, lambda **kw: GroundedBoxes(**kw), lambda o: o.cache_key(CTX),
                       {**SEGMENT_PARAMS, "top_k": 5, "boxes_per_concept": 3, "detector": {"box_threshold": 0.4}},
                       {**BATCHING, "sam": "boxes are used as segments; SAM never runs"}),
    "sam_grid": (SAMGrid, lambda **kw: SAMGrid(**kw), lambda o: o.cache_key(CTX),
                 {**SEGMENT_PARAMS, "points_per_side": 4, "min_iou": 0.5, "sam": {"model": "facebook/sam-vit-large"}},
                 BATCHING),
    "claude": (ClaudeClient, lambda **kw: ClaudeClient(**{"client": object(), **kw}), lambda o: o.cache_key("p"),
               {"model": "claude-sonnet-5-5", "effort": "high"},
               {"max_tokens": "cut-off responses raise and are never cached, so cached ones are complete",
                "cache_dir": "where the cache lives, not what it holds", "client": "transport only"}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_init_arg_is_keyed_or_exempt(case):
    cls, build, key, perturbed, exempt = CASES[case]
    assert init_params(cls) == set(perturbed) | set(exempt), \
        f"{cls.__name__}: put each new constructor arg in its cache key (and here) or list it as exempt"
    base = key(build())
    for name, value in perturbed.items():
        assert key(build(**{name: value})) != base, f"{cls.__name__}({name}=...) does not change the cache key"


@pytest.mark.parametrize("cls, header, exempt", [
    (Waterbirds, "img_filename,y,split,place", {}),
    (MetaShift, "filename,y,a,split", {"class_names": "labels only; samples and features are unchanged"}),
])
def test_dataset_root_is_keyed(tmp_path, cls, header, exempt):
    assert init_params(cls) == {"root"} | set(exempt)
    a = cls(root=_metadata(tmp_path / "a", header)).cache_key()
    b = cls(root=_metadata(tmp_path / "b", header)).cache_key()
    assert a != b


def test_cub_keys_cover_features_and_parts(tmp_path, cub_root):
    """CUB feeds two caches: features (``cache_key``) and part annotations (``part_spec().cache_key``)."""
    segs = shutil.copytree(cub_root / "part_segmentations" / "AnnotationMasksPerclass", tmp_path / "segs")
    split_dir = tmp_path / "koh_split"
    split_dir.mkdir()
    with open(split_dir / "val.pkl", "wb") as f:  # Koh et al.'s format; image 1 is in the official train split
        pickle.dump([{"img_path": "CUB_200_2011/images/002.B_bird/B_bird_0001.jpg"}], f)
    perturbed = {"root": str(shutil.copytree(cub_root, tmp_path / "copy" / "CUB_200_2011")),
                 "class_level_concepts": False, "min_class_count": 2, "val_fraction": 0.3,
                 "part_segmentations": str(segs), "majority_vote": "koh", "split_dir": str(split_dir)}
    assert init_params(CUB) == set(perturbed)

    def key(**kw):
        ds = CUB(**{"root": str(cub_root), "min_class_count": 1, **kw})
        return {"data": ds.cache_key(), "parts": ds.part_spec().cache_key()}

    base = key()
    for name, value in perturbed.items():
        assert key(**{name: value}) != base, f"CUB({name}=...) does not change a cache key"
