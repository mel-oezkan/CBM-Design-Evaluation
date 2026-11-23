"""The Modal side of ``scripts/modal_ops.py``: one function, ``sh``, that runs a shell command on this
checkout's code in a container on Modal.

Do not ``modal run`` this file yourself: ``modal_ops.py launch`` prices the worst case, checks it against
the approved dollar cap and the free credit, sets the container's resources and timeout with
``with_options`` and records the launch in the ledger. The timeout is the hard cost cap: Modal kills
the container when it ends, even if the Mac that launched it is asleep.

The image installs the locked dependencies (``uv.lock``, core + extras + dev) on Modal, so nothing is
downloaded on the launching machine. The working tree (not just the last commit) is copied to
``/root/repo``, so a launch from a git worktree runs that worktree's code. Inside the container:

- ``data/`` and ``.cache/`` are the project Volume's ``data/`` and ``cache/`` (CUB and the feature cache
  that ``koh2020_modal.py`` filled live there too);
- ``runs/`` and ``results/`` are ``ops/<launch_id>/`` on the Volume, next to ``log.txt`` (the command's
  combined stdout/stderr) and ``launch.json`` (what was launched, from which commit);
- ``cbm-eval`` is on the PATH, as after ``uv sync`` locally.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"
VOLUME = "cbm-koh2020"  # the project volume: CUB, the authors' split and the feature cache
WORKDIR = "/root/repo"
_IGNORE = [".git", ".venv", "data", ".cache", "runs", "results", "site", "papers", ".claude", ".pytest_cache",
           "**/__pycache__", "**/.DS_Store", "reports"]

app = modal.App("cbm-eval-ops")
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(str(REPO), extras=["clip", "hf", "llm", "analysis", "tracking"], groups=["dev"])
    # the project itself is not installed (uv_sync is --no-install-project): src/ on PYTHONPATH and a
    # `cbm-eval` shim give the same entry points as a local `uv sync`
    .run_commands("printf '#!/bin/sh\\nexec python -c \"import sys; from cbm_eval.cli import main; "
                  "sys.exit(main())\" \"$@\"\\n' > /usr/local/bin/cbm-eval && chmod +x /usr/local/bin/cbm-eval")
    .env({"PYTHONPATH": f"{WORKDIR}/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO, WORKDIR, ignore=_IGNORE)
)


def _link(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(target)


# Conservative defaults; `modal_ops.py launch` replaces gpu/cpu/memory/timeout per launch.
@app.function(image=image, volumes={VOL: vol}, cpu=1, memory=1024, timeout=300, retries=0, max_containers=1)
def sh(command: str, launch_id: str, launch: dict) -> dict:
    """Run ``command`` with bash in ``/root/repo``; stream its output to the logs and to
    ``ops/<launch_id>/log.txt`` on the Volume. Returns the exit code, the wall time and the log tail."""
    out = Path(VOL, "ops", launch_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "launch.json").write_text(json.dumps(launch, indent=2))
    repo = Path(WORKDIR)
    for link, target in {repo / "data": Path(VOL, "data"), repo / ".cache": Path(VOL, "cache"),
                         repo / "runs": out / "runs", repo / "results": out / "results"}.items():
        _link(link, target)
    start, tail = time.time(), []
    with open(out / "log.txt", "w") as log:
        proc = subprocess.Popen(["bash", "-c", command], cwd=repo, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=os.environ | {"CBM_LAUNCH_ID": launch_id})
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            tail = (tail + [line.rstrip("\n")])[-40:]
        code = proc.wait()
    result = {"exit_code": code, "seconds": round(time.time() - start, 1), "log_tail": tail}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    vol.commit()
    return result
