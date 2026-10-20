"""FunnyBirds on Modal: download the authors' render or render a custom variant onto the Volume
``cbm-funnybirds``, then run ``cbm-eval`` on it (``configs/anchors/funnybirds_human.yaml``).

Thin glue only. Rendering is ``cbm_eval.data.funnybirds_render``: each CPU container runs
``funnybirds_server.js`` (one Chrome, ``PAGES`` tabs) on the authors' ``render/`` folder
(github.com/visinf/funnybirds, pinned to ``FUNNYBIRDS_COMMIT``). One container renders one shard of
scenes and writes it as one zip, so containers never write the same Volume file and an interrupted
render resumes per shard.

    uv run modal run scripts/funnybirds_modal.py::download            # the authors' 50 classes, 1.6 GB
    uv run modal run scripts/funnybirds_modal.py::render --name FunnyBirdsSpurious --spurious-corr 0.95 \\
        --bg-objects 5,34                                             # scenes + images + part maps
    uv run modal run scripts/funnybirds_modal.py::run --config configs/anchors/funnybirds_human.yaml \\
        --data FunnyBirdsSpurious --overrides 'dataset.root=data/FunnyBirdsSpurious'
    uv run modal volume get cbm-funnybirds results/ results/funnybirds/   # result rows

Volume layout: ``data/FunnyBirds.zip`` (download) or ``data/<name>/*.json`` plus
``data/<name>/shards/<split>-<start>.zip`` (render); ``cache/``, ``runs/`` and ``results/`` as in
``scripts/koh2020_modal.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"
WORK = "/root/work"  # cwd of cbm-eval: data/, .cache, runs/ and results/ as the configs expect
FUNNYBIRDS_URL = "https://download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip"
FUNNYBIRDS_COMMIT = "91b4b4628ffa962148144ee6bb5af5f022cac2f8"
SERVER = "/funnybirds/render"
SHARD = 250  # scenes per render container: 500 screenshots, ~7 min
PAGES = 8  # Chrome tabs per container; measured best on 8 CPUs (0.86 s per screenshot; 16 tabs: 1.33 s)

app = modal.App("cbm-funnybirds")
vol = modal.Volume.from_name("cbm-funnybirds", create_if_missing=True)
# Chrome's runtime libraries for the Chromium that puppeteer downloads during `npm install`.
CHROME_LIBS = ["libnss3", "libatk1.0-0", "libatk-bridge2.0-0", "libcups2", "libdrm2", "libxkbcommon0",
               "libxcomposite1", "libxdamage1", "libxfixes3", "libxrandr2", "libgbm1", "libasound2",
               "libpango-1.0-0", "libcairo2", "libx11-xcb1", "libxshmfence1", "libxss1", "libgtk-3-0",
               "fonts-liberation"]
render_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "nodejs", "npm", *CHROME_LIBS)
    .run_commands(f"git clone https://github.com/visinf/funnybirds /funnybirds && cd /funnybirds && "
                  f"git checkout {FUNNYBIRDS_COMMIT} && cd render && npm install --no-audit --no-fund")
    .uv_pip_install("numpy", "pillow")
    # The generator needs only numpy and PIL; importing it as a top-level module skips cbm_eval's torch imports.
    .add_local_file(REPO / "src/cbm_eval/data/funnybirds_render.py", "/root/funnybirds_render.py")
    .add_local_file(REPO / "src/cbm_eval/data/funnybirds_server.js", "/root/funnybirds_server.js")
)
run_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "numpy", "pandas", "pyyaml", "pillow",
                    "open_clip_torch>=2.24")
    .add_local_python_source("cbm_eval")
    .add_local_dir(REPO / "configs", f"{WORK}/configs")
)


@app.function(image=render_image, volumes={VOL: vol}, timeout=3600)
def download() -> None:
    """The authors' render (50 classes, 50,000 train / 500 test) -> ``data/FunnyBirds.zip``."""
    import shutil
    import urllib.request

    dst = Path(VOL, "data", "FunnyBirds.zip")
    if dst.exists():
        print(f"{dst} exists")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(FUNNYBIRDS_URL) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(dst)
    vol.commit()


@app.function(image=render_image, volumes={VOL: vol}, timeout=600)
def make_scenes(name: str, kw: dict) -> dict[str, int]:
    """Sample scenes into ``data/<name>`` (keeps existing scenes, so a re-run resumes); scenes per split."""
    import json

    import funnybirds_render as fr

    root = Path(VOL, "data", name)
    if not (root / "dataset_train.json").exists():
        fr.write_scenes(root, **kw)
        vol.commit()
    else:
        print(f"{root} already has scenes; rendering those (args: {json.loads((root / 'meta.json').read_text())})")
    return {s: len(json.loads((root / f"dataset_{s}.json").read_text())) for s in ("train", "test")}


@app.function(image=render_image, volumes={VOL: vol}, cpu=8, memory=8192, timeout=4 * 3600, retries=2)
def render_shard(name: str, split: str, start: int, stop: int, pages: int = PAGES) -> int:
    """Render scenes ``[start, stop)`` on local disk with a local server, then zip them to the Volume."""
    import shutil
    import tempfile
    import zipfile

    import funnybirds_render as fr

    shard = Path(VOL, "data", name, "shards", f"{split}-{start:06d}.zip")
    if shard.exists():
        return 0
    local = Path(tempfile.mkdtemp())
    for f in Path(VOL, "data", name).glob("*.json"):
        shutil.copy(f, local)
    with fr.serve(SERVER, pages=pages) as url:
        n = fr.render(local, split, url, start=start, stop=stop, workers=pages)
    tmp = Path(tempfile.mkdtemp(), shard.name)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # PNGs are already compressed
        for f in local.rglob("*.png"):
            z.write(f, Path(name) / f.relative_to(local))
    shard.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp, shard)
    vol.commit()
    return n


@app.local_entrypoint()
def render(name: str = "FunnyBirdsCustom", n_classes: int = 50, train_per_class: int = 1000,
           test_per_class: int = 10, seed: int = 0, drop_parts: float = 0.5, spurious_corr: float = -1.0,
           bg_objects: str = "0,34"):
    """``--spurious-corr`` < 0 (the default) means no spurious attribute."""
    lo, hi = (int(x) for x in bg_objects.split(","))
    kw = {"n_classes": n_classes, "train_per_class": train_per_class, "test_per_class": test_per_class,
          "seed": seed, "drop_parts": drop_parts, "spurious_corr": spurious_corr if spurious_corr >= 0 else None,
          "bg_objects": (lo, hi)}
    sizes = make_scenes.remote(name, kw)
    jobs = [(name, split, a, min(a + SHARD, n)) for split, n in sizes.items() for a in range(0, n, SHARD)]
    print(f"{sizes} scenes -> {len(jobs)} shards")
    done = list(render_shard.starmap(jobs))
    print(f"{sum(done)} images rendered into data/{name}")


def _prepare_workdir(names: list[str]) -> None:
    """Each dataset unpacked on local disk (fast image reads), persistent dirs from the Volume."""
    import shutil
    import zipfile

    data = Path(WORK, "data")
    data.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (data / name / "classes.json").exists():
            continue
        zips = [Path(VOL, "data", f"{name}.zip")]
        if not zips[0].exists():
            if not Path(VOL, "data", name, "classes.json").exists():
                raise FileNotFoundError(f"data/{name} is not on the Volume; run ::download or ::render --name {name}")
            (data / name).mkdir()
            for f in Path(VOL, "data", name).glob("*.json"):
                shutil.copy(f, data / name)
            zips = sorted(Path(VOL, "data", name, "shards").glob("*.zip"))
        for z in zips:
            with zipfile.ZipFile(z) as f:
                f.extractall(data)
        if not (data / name / "classes.json").exists():
            raise FileNotFoundError(f"Unpacking data/{name} gave no {name}/classes.json; found {os.listdir(data)}")
    for link, target in {Path(WORK, ".cache"): Path(VOL, "cache"), Path(WORK, "runs"): Path(VOL, "runs"),
                         Path(WORK, "results"): Path(VOL, "results")}.items():
        target.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(target)
    os.chdir(WORK)


@app.function(image=run_image, volumes={VOL: vol}, gpu="L4", cpu=8, memory=32768, timeout=12 * 3600)
def run_config(config: str, data: list[str], seeds: list[int], overrides: list[str], sweep: bool) -> None:
    from cbm_eval.cli import main as cbm_eval

    _prepare_workdir(data)
    try:
        if sweep:
            cbm_eval(["-v", "sweep", config, *(["--set", *overrides] if overrides else [])])
        else:
            cbm_eval(["-v", "run", config, "--seeds", *map(str, seeds), *(["--set", *overrides] if overrides else [])])
    finally:
        vol.commit()


@app.local_entrypoint()
def run(config: str = "configs/anchors/funnybirds_human.yaml", data: str = "FunnyBirds", seeds: str = "0",
        overrides: str = "", sweep: bool = False):
    """``--config`` is an anchor (``run``) or, with ``--sweep``, an ablation; ``--data`` names the
    Volume datasets to unpack; ``--overrides`` takes space-separated dotted overrides."""
    run_config.remote(config, data.split(","), [int(s) for s in seeds.split(",")], overrides.split(), sweep)
