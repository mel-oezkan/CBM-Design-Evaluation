"""``cache.prune`` deletes large feature caches (views, patches) unused for some days, and nothing else."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import torch

from cbm_eval.backbones.toy import ToyBackbone
from cbm_eval.cache import MARKER, mark_used, prune
from cbm_eval.cli import main
from cbm_eval.data.features import load_augmented_features, load_features
from cbm_eval.data.synthetic import SyntheticDataset

NOW = datetime(2026, 11, 23, tzinfo=timezone.utc)


def _file(root, name, used_days_ago=None, now=NOW):
    path = root / "features" / "ds" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if used_days_ago is not None:
        mark_used(path, now - timedelta(days=used_days_ago))
    return path


def test_prune_deletes_only_marked_files_unused_for_longer_than_the_limit(tmp_path):
    stale = _file(tmp_path, "train-views-a.pt", used_days_ago=6)
    fresh = _file(tmp_path, "train-views-b.pt", used_days_ago=4)
    patches = _file(tmp_path, "train-c.pt", used_days_ago=10)  # patch features carry a marker too
    clean = _file(tmp_path, "train-d.pt")  # clean features: no marker, never deleted
    llm = tmp_path / "llm" / "answers.json"
    llm.parent.mkdir()
    llm.write_text("{}")

    assert prune(tmp_path, 5, now=NOW, dry_run=True) == [patches, stale]
    assert stale.exists() and patches.exists()  # dry run deletes nothing

    assert prune(tmp_path, 5, now=NOW) == [patches, stale]
    assert not stale.exists() and not patches.exists()
    assert not (stale.parent / (stale.name + MARKER)).exists()
    assert fresh.exists() and clean.exists() and llm.exists()


def test_unmarked_views_start_counting_and_orphan_markers_go(tmp_path):
    legacy = _file(tmp_path, "train-views-old.pt")  # written before markers existed
    orphan = _file(tmp_path, "val-views-gone.pt", used_days_ago=1)
    orphan.unlink()

    assert prune(tmp_path, 5, now=NOW) == []
    assert legacy.exists() and (legacy.parent / (legacy.name + MARKER)).exists()
    assert not (orphan.parent / (orphan.name + MARKER)).exists()
    assert prune(tmp_path, 5, now=NOW + timedelta(days=6)) == [legacy]


def test_loaders_mark_views_and_patches_but_not_clean_features(tmp_path):
    ds, bb = SyntheticDataset(sizes={"train": 8, "val": 4, "test": 4}), ToyBackbone()
    load_features(ds, bb, "train", tmp_path)
    load_features(ds, bb, "train", tmp_path, patches=True)
    load_augmented_features(ds, bb, "train", tmp_path, views=2)
    marked = {p.name.removesuffix(MARKER) for p in (tmp_path / "features").rglob(f"*{MARKER}")}
    data = {p.name for p in (tmp_path / "features").rglob("*.pt")}
    assert len(data) == 3 and len(marked) == 2 and any("-views-" in n for n in marked)

    # a cache hit refreshes the marker, so a file in use is never pruned
    for m in (tmp_path / "features").rglob(f"*{MARKER}"):
        m.write_text((NOW - timedelta(days=30)).isoformat())
    before = load_augmented_features(ds, bb, "train", tmp_path, views=2)
    load_features(ds, bb, "train", tmp_path, patches=True)
    assert prune(tmp_path, 5) == []
    assert torch.equal(load_augmented_features(ds, bb, "train", tmp_path, views=2), before)


def test_cli_cache_prune(tmp_path, capsys):
    stale = _file(tmp_path, "train-views-a.pt", used_days_ago=6, now=datetime.now(timezone.utc))  # CLI: real clock
    main(["cache", "prune", "--cache", str(tmp_path), "--dry-run"])
    assert "would delete 1" in capsys.readouterr().out and stale.exists()
    main(["cache", "prune", "--cache", str(tmp_path), "--unused-days", "10"])
    assert stale.exists()
    main(["cache", "prune", "--cache", str(tmp_path)])
    assert not stale.exists()
