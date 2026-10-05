"""Exercise the image-file code paths offline: metadata loaders, PIL loading, a torchvision backbone,
and text-based discovery with stub clients (no network, no pretrained weights)."""

import json

import numpy as np
import pytest
from PIL import Image

from cbm_eval.config import ExperimentConfig
from cbm_eval.pipeline import PipelineBuilder, run


@pytest.fixture
def waterbirds_root(tmp_path):
    root = tmp_path / "waterbirds"
    rows = ["img_filename,y,split,place"]
    rng = np.random.default_rng(0)
    for i in range(36):
        y, place, split = i % 2, (i // 2) % 2, [0, 0, 1, 2][i % 4]
        rel = f"{y}/img_{i}.jpg"
        (root / str(y)).mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(root / rel)
        rows.append(f"{rel},{y},{split},{place}")
    (root / "metadata.csv").write_text("\n".join(rows))
    return root


def _cfg(tmp_path, root, **stages):
    base = {
        "discovery": {"name": "sae", "n_latents": 16, "k": 2, "epochs": 2, "min_frequency": 0.0},
        "filtering": [{"name": "rules"}],
        "generation": {"name": "scores"},
        "alignment": {"name": "weights"},
        "predictor": {"name": "dense"},
        "training": {"name": "sequential", "epochs": 3},
    }
    return ExperimentConfig.from_dict({
        "dataset": {"name": "waterbirds", "root": str(root)},
        "backbone": {"name": "resnet", "model": "resnet18", "weights": None, "image_size": 64},
        "stages": base | stages,
        "evaluation": [{"name": "shift"}, {"name": "leakage", "repeats": 1}],
        "device": "cpu",
        "paths": {"cache": str(tmp_path / "cache"), "runs": str(tmp_path / "runs"),
                  "results": str(tmp_path / "r.jsonl")},
    })


def test_waterbirds_resnet_end_to_end(tmp_path, waterbirds_root):
    res = run(_cfg(tmp_path, waterbirds_root), save=False)
    assert 0 <= res.metrics["shift.test.wga"] <= 1
    # second run hits the feature cache
    assert list((tmp_path / "cache" / "features" / "waterbirds").glob("*.pt"))


class StubLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, images=None, system=None):
        self.prompts.append((prompt, images))
        return "- feathers\n- water\n- 3. long beak\nlandbird"


def test_llm_and_vlm_discovery_with_stub(tmp_path, waterbirds_root):
    from cbm_eval.stages.discovery import LLMDiscovery, VLMDiscovery

    ctx = PipelineBuilder(_cfg(tmp_path, waterbirds_root)).context()
    stub = StubLLM()
    cs = LLMDiscovery(prompts=["features"], client=stub).discover(ctx)
    assert set(cs.names) == {"feathers", "water", "long beak", "landbird"}
    assert len(stub.prompts) == 2 and "waterbird" in stub.prompts[1][0]
    # the rules filter should then drop the concept that names a class
    from cbm_eval.stages.filtering import RuleFilter
    assert "landbird" not in RuleFilter().filter(cs, ctx).names

    vstub = StubLLM()
    VLMDiscovery(images_per_class=2, client=vstub).discover(ctx)
    assert all(len(images) == 2 and images[0].endswith(".jpg") for _, images in vstub.prompts)


def test_kb_discovery_from_file(tmp_path, waterbirds_root):
    from cbm_eval.stages.discovery import KnowledgeBaseDiscovery

    kb = tmp_path / "kb.json"
    kb.write_text(json.dumps({"landbird": ["branch", "feathers"], "waterbird": ["lake", "feathers"]}))
    ctx = PipelineBuilder(_cfg(tmp_path, waterbirds_root)).context()
    cs = KnowledgeBaseDiscovery(path=str(kb)).discover(ctx)
    assert cs.names == ["branch", "feathers", "lake"]
    assert cs.meta[1]["classes"] == ["landbird", "waterbird"]
