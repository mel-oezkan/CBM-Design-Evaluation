"""Every script in ``examples/`` runs offline, top to bottom, as a user would run it."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


def test_examples_exist():
    assert EXAMPLES


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # caches, run folders and results files land in tmp_path
    runpy.run_path(str(script), run_name="__main__")
