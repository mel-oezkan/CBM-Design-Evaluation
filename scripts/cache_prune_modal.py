"""Delete large feature caches on the Modal volume that no run has used for ``UNUSED_DAYS`` days.

Once a day it runs ``cbm_eval.cache.prune`` on the volume's ``cache/``: augmented views and patch
features go, clean features and the LLM/VLM caches stay (see ``src/cbm_eval/cache.py``). The schedule
only runs while the app is deployed:

    uv run modal deploy scripts/cache_prune_modal.py          # start (or update) the daily job
    uv run modal run scripts/cache_prune_modal.py --dry-run   # list what would go now, delete nothing
    uv run modal app stop cbm-cache-prune                     # stop the daily job

Cost: one container with a quarter CPU for a few seconds a day. The volume is the one that
``koh2020_modal.py`` fills; its entrypoints also prune once per launch, before they spawn work.
"""

from __future__ import annotations

from pathlib import Path

import modal

VOL = "/vol"
UNUSED_DAYS = 5

app = modal.App("cbm-cache-prune")
vol = modal.Volume.from_name("cbm-koh2020")
image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("cbm_eval")


@app.function(image=image, volumes={VOL: vol}, cpu=0.25, memory=256, timeout=600, schedule=modal.Period(days=1))
def prune(dry_run: bool = False) -> list[str]:
    from cbm_eval.cache import prune as prune_cache

    removed = prune_cache(Path(VOL, "cache"), UNUSED_DAYS, dry_run=dry_run)
    vol.commit()  # also persists the markers it adds to unmarked views files
    for path in removed:
        print(f"{'would delete' if dry_run else 'deleted'} {path}", flush=True)
    print(f"{len(removed)} file(s) unused for {UNUSED_DAYS} days", flush=True)
    return [str(p) for p in removed]


@app.local_entrypoint()
def main(dry_run: bool = False):
    prune.remote(dry_run)
