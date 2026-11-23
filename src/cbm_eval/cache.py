"""Delete large feature caches that no run has used for some days (``cbm-eval cache prune``).

Augmented views (``*-views-*.pt``) and patch features are large (GBs per dataset and backbone) and
only some configs read them. Each time a run loads or writes one, ``mark_used`` records the time in a
marker file next to it (``<file>.last_used``). ``prune`` deletes every marked file whose last use is
older than ``unused_days``. Files without a marker (clean features, LLM/VLM answers, instances) are
never deleted. Deleting a cache never changes results: the next run that needs it extracts it again.

Standard library only, so a minimal container (``scripts/cache_prune_modal.py``) can run it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

MARKER = ".last_used"
DEFAULT_UNUSED_DAYS = 5.0


def _marker(path: Path) -> Path:
    return path.with_name(path.name + MARKER)


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def mark_used(path: str | Path, now: datetime | None = None) -> None:
    """Record that a run used the cache file ``path``, which makes it prunable. Never fails a run: a
    read-only cache only logs."""
    try:
        _marker(Path(path)).write_text(_now(now).isoformat())
    except OSError as e:
        log.debug("cannot mark %s as used: %s", path, e)


def _last_used(marker: Path) -> datetime | None:
    try:
        return datetime.fromisoformat(marker.read_text().strip())
    except (OSError, ValueError):
        return None


def prune(cache_dir: str | Path, unused_days: float = DEFAULT_UNUSED_DAYS, now: datetime | None = None,
          dry_run: bool = False) -> list[Path]:
    """Delete the marked cache files under ``cache_dir/features`` not used for ``unused_days``; return them.

    A views file without a marker (written before markers existed) gets one now, so it goes after
    ``unused_days`` more without use. A marker whose file is gone is removed. With ``dry_run``
    nothing changes on disk."""
    root, now = Path(cache_dir) / "features", _now(now)
    cutoff = now - timedelta(days=unused_days)
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for marker in sorted(root.rglob(f"*{MARKER}")):
        data = marker.with_name(marker.name.removesuffix(MARKER))
        if not data.exists():
            if not dry_run:
                marker.unlink(missing_ok=True)
            continue
        last = _last_used(marker)
        if last is None:  # unreadable: start counting now rather than delete what may be in use
            if not dry_run:
                mark_used(data, now)
            continue
        if last < cutoff:
            removed.append(data)
            if not dry_run:
                data.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
    if not dry_run:
        for data in root.rglob("*-views-*.pt"):
            if not _marker(data).exists():
                mark_used(data, now)
    for data in removed:
        log.info("%s %s (unused for %g days)", "would delete" if dry_run else "deleted", data, unused_days)
    return removed
