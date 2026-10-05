"""Run the Koh et al. 2020 reproduction (``configs/anchors/cub_koh2020*.yaml`` and
``configs/ablations/cub_koh2020*.yaml``) on Modal.

Thin glue only: unpack CUB next to the authors' split, call ``cbm-eval`` exactly as locally, and
mirror new result rows to W&B. Caches, runs and results live on the Volume ``cbm-koh2020``.

- ``anchors``: the fine-tuned reproduction, one GPU container per (anchor, seed) in parallel. Each
  writes its own results file (concurrent appends to one volume file can lose rows).
- ``ablations``: the frozen-backbone fidelity ablations, sequentially in one container
  (``sweep`` skips finished ``(run_id, seed)``, so a restart resumes).

    uv run modal run scripts/koh2020_modal.py::anchors [--anchors cub_koh2020 --seeds 1 --overrides 'k=v ...']
    uv run modal run scripts/koh2020_modal.py::ablations
    uv run modal run scripts/koh2020_modal.py::fetch          # merged -> results/cub_koh2020.jsonl
    uv run cbm-eval analyze --results results/cub_koh2020.jsonl --metric concepts.test.concept_error
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"
WORK = "/root/work"  # cwd of cbm-eval: data/, .cache, runs/ and results/ as the configs expect
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_TGZ = f"{VOL}/data/CUB_200_2011.tgz"
SPLIT_DIR = f"{VOL}/data/CUB_processed"  # Koh et al.'s Codalab bundle 0x5b9d528d2101418b87212db92fea6683
WANDB_PROJECT = "cbm-koh2020-modular"
ANCHORS = ("cub_koh2020_independent", "cub_koh2020_sequential", "cub_koh2020")
FINETUNE_GPU = "A100-40GB"

app = modal.App("cbm-koh2020-modular")
vol = modal.Volume.from_name("cbm-koh2020", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "numpy", "pandas", "pyyaml", "pillow", "wandb")
    .add_local_python_source("cbm_eval")
    .add_local_dir(REPO / "configs", f"{WORK}/configs")
)
secrets = [modal.Secret.from_name("wandb")]


def _prepare_workdir() -> None:
    """CUB unpacked on local disk (the loader reads ~12k image headers and fine-tuning reads every
    image each epoch: slow on a volume); the authors' split and persistent dirs from the volume."""
    import shutil
    import tarfile
    import urllib.request

    data = Path(WORK, "data")
    data.mkdir(parents=True, exist_ok=True)
    if not Path(CUB_TGZ).exists():
        Path(CUB_TGZ).parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(CUB_URL, headers={"User-Agent": "Mozilla/5.0"})  # urllib's UA gets a 403
        with urllib.request.urlopen(req) as r, open(CUB_TGZ, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
        vol.commit()
    if not (data / "CUB_200_2011" / "images.txt").exists():
        with tarfile.open(CUB_TGZ) as t:
            t.extractall(data)
    links = {data / "CUB_processed": Path(SPLIT_DIR), Path(WORK, ".cache"): Path(VOL, "cache"),
             Path(WORK, "runs"): Path(VOL, "runs"), Path(WORK, "results"): Path(VOL, "results")}
    for link, target in links.items():
        target.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(target)
    os.chdir(WORK)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _mirror_to_wandb(rows: list[dict], group: str) -> list[dict]:
    """One W&B run per result row: factors as config, metrics and training log as summary."""
    import wandb

    for row in rows:
        run = wandb.init(project=WANDB_PROJECT, group=group, name=f"{row['name']}-{row['run_id']}-s{row['seed']}",
                         config={"run_id": row["run_id"], "seed": row["seed"], **row.get("factor", {})},
                         reinit="finish_previous")
        run.summary.update({"status": row.get("status", "ok"), **row.get("metric", {}),
                            **{f"train/{k}": v for k, v in (row.get("train") or {}).items()}})
        run.finish()
    return [{k: row.get(k) for k in ("name", "run_id", "seed", "status")} |
            {m: row.get("metric", {}).get(m) for m in ("shift.test.acc", "concepts.test.concept_error")}
            for row in rows]


@app.function(image=image, volumes={VOL: vol}, gpu=FINETUNE_GPU, cpu=16, memory=65536, timeout=24 * 3600,
              secrets=secrets)
def run_anchor(anchor: str, seed: int, overrides: list[str]) -> list[dict]:
    from cbm_eval.cli import main as cbm_eval

    _prepare_workdir()
    tag = "-".join([anchor, f"s{seed}"] + [o.replace("/", "_") for o in overrides])
    results = Path(f"results/finetune/{tag}.jsonl")
    results.parent.mkdir(parents=True, exist_ok=True)
    try:
        cbm_eval(["-v", "run", f"configs/anchors/{anchor}.yaml", "--seeds", str(seed),
                  "--set", f"paths.results={results}", *overrides])
    finally:
        vol.commit()
    return _mirror_to_wandb(_rows(results), anchor)


@app.function(image=image, volumes={VOL: vol}, gpu="L4", cpu=8, memory=32768, timeout=6 * 3600, secrets=secrets)
def run_ablations(ablations: list[str]) -> list[dict]:
    from cbm_eval.cli import main as cbm_eval

    _prepare_workdir()
    results = Path("results/cub_koh2020.jsonl")
    before = len(_rows(results))
    try:
        for ablation in ablations:
            cbm_eval(["-v", "sweep", f"configs/ablations/{ablation}.yaml"])
            vol.commit()
    finally:
        vol.commit()
    return _mirror_to_wandb(_rows(results)[before:], "frozen-ablations")


@app.local_entrypoint()
def anchors(anchors: str = ",".join(ANCHORS), seeds: str = "1,2,3", overrides: str = ""):
    """``--overrides`` takes space-separated dotted overrides, e.g. ``stages.training.finetune.epochs=2``."""
    overrides = overrides.split()
    jobs = [(a, int(s), overrides) for a in anchors.split(",") for s in seeds.split(",")]
    for rows in run_anchor.starmap(jobs, return_exceptions=True):
        print(rows if isinstance(rows, Exception) else "\n".join(json.dumps(r) for r in rows), flush=True)


@app.local_entrypoint()
def ablations(ablations: str = ",".join(ANCHORS)):
    for r in run_ablations.remote(ablations.split(",")):
        print(json.dumps(r))


@app.local_entrypoint()
def fetch():
    """Merge the frozen-ablation results and every fine-tuning run into results/cub_koh2020.jsonl."""
    lines = []
    for entry in [*vol.listdir("results", recursive=True)]:
        if entry.path.endswith(".jsonl") and "old_driver" not in entry.path:
            lines += [line for line in b"".join(vol.read_file(entry.path)).decode().splitlines() if line.strip()]
    local = REPO / "results" / "cub_koh2020.jsonl"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} rows -> {local}")
