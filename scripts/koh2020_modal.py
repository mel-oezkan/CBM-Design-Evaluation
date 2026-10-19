"""Run the Koh et al. 2020 reproduction (``configs/anchors/cub_koh2020*.yaml`` and
``configs/ablations/cub_koh2020_frozen*.yaml``) on Modal.

Thin glue only: unpack CUB next to the authors' split, call the pipeline exactly as locally, and
mirror new result rows to W&B. Caches, runs and results live on the Volume ``cbm-koh2020``.

- ``frozen``: the frozen-backbone arm (anchors and ablations). First one GPU container per distinct
  feature set extracts the clean and augmented-view features into the cache, then every
  ``(config, seed)`` trains on a CPU container of its own, writing its own results file (concurrent
  appends to one volume file can lose rows). Runs whose results file exists are skipped.
- ``anchors``: the fine-tuned reproduction, one GPU container per (anchor, seed) in parallel.

    uv run modal run scripts/koh2020_modal.py::frozen [--only-anchors] [--seeds 1,2,3]
    uv run modal run scripts/koh2020_modal.py::anchors [--anchors cub_koh2020 --seeds 1 --overrides 'k=v ...']
    uv run modal run scripts/koh2020_modal.py::fetch --arm frozen     # merged -> results/cub_koh2020_frozen.jsonl
    uv run cbm-eval analyze --results results/cub_koh2020_frozen.jsonl --metric concepts.test.concept_error
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
FROZEN = ("cub_koh2020_frozen_independent", "cub_koh2020_frozen_sequential", "cub_koh2020_frozen")
FINETUNE_GPU = "A100-40GB"
FEATURE_GPU = "A100-40GB"  # feature extraction is bound by JPEG decoding on its 16 CPUs

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


@app.function(image=image, volumes={VOL: vol}, gpu=FINETUNE_GPU, cpu=16, memory=32768, timeout=24 * 3600,
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


def _frozen_configs(only_anchors: bool, seeds: list[int]) -> list[dict]:
    """Every (config, seed) of the frozen arm as config dicts: the anchors, plus their ablations."""
    from cbm_eval.config import load_ablation, load_config

    cfgs = [load_config(REPO / "configs" / "anchors" / f"{a}.yaml").with_overrides({"seed": s})
            for a in FROZEN for s in seeds]
    if not only_anchors:
        cfgs += [c for a in FROZEN for c in load_ablation(REPO / "configs" / "ablations" / f"{a}.yaml").expand()
                 if c.seed in seeds]
    unique = {(c.run_id, c.seed): c for c in cfgs}
    return [c.to_dict() for c in unique.values()]


def _feature_groups(cfgs: list[dict]) -> list[dict]:
    """One config per distinct (dataset, backbone, augmented views) set: what ``frozen_features`` warms."""
    import hashlib

    groups = {}
    for d in cfgs:
        aug = d["stages"]["training"].get("augment")
        key = json.dumps([d["dataset"], d["backbone"], aug and {k: aug.get(k) for k in ("views", "seed")}],
                         sort_keys=True)
        groups.setdefault(hashlib.sha1(key.encode()).hexdigest(), d)
    return list(groups.values())


@app.function(image=image, volumes={VOL: vol}, gpu=FEATURE_GPU, cpu=16, memory=65536, timeout=6 * 3600)
def frozen_features(cfg: dict) -> str:
    """Extract and cache the clean features of every split and the augmented views the training reads."""
    from cbm_eval.config import ExperimentConfig
    from cbm_eval.pipeline import PipelineBuilder

    _prepare_workdir()
    builder = PipelineBuilder(ExperimentConfig.from_dict(cfg))
    ctx = builder.context()
    for split in ("train", "val", "test"):
        ctx.split(split)
    vol.commit()
    training = builder.training
    if training.augment is not None:
        for split in ("train", "val"):
            training._views(ctx, split)  # what fit() reads: train, and val when a phase trains on it
            vol.commit()
    return f"{cfg['dataset']} {cfg['backbone']} {training.augment}"


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=24576, timeout=6 * 3600, secrets=secrets,
              retries=1)
def frozen_run(cfg: dict) -> list[dict]:
    from cbm_eval.config import ExperimentConfig
    from cbm_eval.pipeline import run
    from cbm_eval.results import ResultsStore

    _prepare_workdir()
    c = ExperimentConfig.from_dict(cfg)
    results = Path(f"results/frozen/{c.run_id}-s{c.seed}.jsonl")
    if results.exists():
        return []
    results.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"/tmp/{results.name}")  # moved onto the volume only once the run is complete
    try:
        run(c, ResultsStore(tmp))
        results.write_text(tmp.read_text())
    finally:
        vol.commit()
    return _mirror_to_wandb(_rows(results), c.tags.get("ablation", c.name))


@app.local_entrypoint()
def anchors(anchors: str = ",".join(ANCHORS), seeds: str = "1,2,3", overrides: str = ""):
    """``--overrides`` takes space-separated dotted overrides, e.g. ``stages.training.finetune.epochs=2``.
    Each (anchor, seed) is spawned, so with ``modal run --detach`` the launching machine may disconnect;
    follow with ``modal app logs`` and collect with ``fetch --arm finetune``."""
    overrides = overrides.split()
    for a in anchors.split(","):
        for s in seeds.split(","):
            call = run_anchor.spawn(a, int(s), overrides)
            print(f"spawned {a} seed {s}: {call.object_id}", flush=True)


@app.local_entrypoint()
def frozen(only_anchors: bool = False, seeds: str = "1,2,3"):
    """The frozen-backbone arm: features on GPU first, then every (config, seed) on its own CPU container.
    The configs are expanded here and ``frozen_driver`` is spawned on Modal. Spawned, not called with
    ``.remote()``: a remote call is cancelled when the launching client goes away, a spawned one in a
    ``modal run --detach`` app is not. The command returns at once; follow with ``modal app logs``."""
    cfgs = _frozen_configs(only_anchors, [int(s) for s in seeds.split(",")])
    groups = _feature_groups(cfgs)
    print(f"{len(cfgs)} runs, {len(groups)} feature sets", flush=True)
    call = frozen_driver.spawn(cfgs, groups)
    print(f"spawned frozen_driver: {call.object_id}", flush=True)


@app.local_entrypoint()
def fetch(arm: str = "frozen"):
    """Merge one arm's results into results/cub_koh2020_<arm>.jsonl: ``frozen`` (results/frozen/) or
    ``finetune`` (results/finetune/ and the older frozen ablations in results/cub_koh2020.jsonl)."""
    lines = []
    for entry in vol.listdir("results", recursive=True):
        path = entry.path
        if not path.endswith(".jsonl") or "old_driver" in path or (arm == "frozen") != path.startswith("results/frozen/"):
            continue
        lines += [line for line in b"".join(vol.read_file(path)).decode().splitlines() if line.strip()]
    local = REPO / "results" / ("cub_koh2020_frozen.jsonl" if arm == "frozen" else "cub_koh2020.jsonl")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} rows -> {local}")
