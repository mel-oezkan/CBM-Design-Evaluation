import pytest

from cbm_eval.config import ExperimentConfig

SMALL_DATA = {"name": "synthetic", "sizes": {"train": 400, "val": 200, "test": 300}}


@pytest.fixture
def base_cfg(tmp_path):
    names = [f"concept_{i:02d}" for i in range(24)] + ["distractor_00", "class_0"]
    return ExperimentConfig.from_dict({
        "name": "test",
        "dataset": SMALL_DATA,
        "backbone": {"name": "toy"},
        "stages": {
            "discovery": {"name": "static", "names": names},
            "filtering": [{"name": "rules"}, {"name": "select", "k": 16}],
            "generation": {"name": "scores"},
            "alignment": {"name": "clip"},
            "predictor": {"name": "sparse", "lam": 1e-3},
            "training": {"name": "sequential", "epochs": 30},
        },
        "evaluation": [{"name": "shift"}, {"name": "leakage", "repeats": 2}, {"name": "localization"}],
        "device": "cpu",
        "paths": {"cache": str(tmp_path / "cache"), "runs": str(tmp_path / "runs"),
                  "results": str(tmp_path / "results.jsonl")},
    })
