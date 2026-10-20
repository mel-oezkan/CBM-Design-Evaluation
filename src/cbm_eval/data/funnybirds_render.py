"""Make a FunnyBirds-format dataset (read by ``data/funnybirds.py``): sample classes and scenes,
then render them with the authors' three.js scene (github.com/visinf/funnybirds, ``render/``).

Scene sampling follows the authors' ``render/create_dataset.py``: camera, light and 0-34 random
background objects per scene; in train, half of the scenes drop a random subset of parts. It differs
in three ways. Sampling uses one seeded generator (so it does not reproduce the authors' JSON files).
``drop_parts`` sets the share of train scenes that drop parts. ``spurious_corr`` adds a spurious
attribute: one ``BG_COLORS`` entry colors every background object of a scene. On train it is the
class's color (``class % 4``) with that probability, otherwise uniform; on test it is always uniform.

Rendering goes through ``funnybirds_server.js`` (started by ``serve``): the authors' ``page.html``
and models in one persistent headless Chrome with a pool of tabs. Its images are pixel-identical
to the authors' ``server.js`` and it is ~3x faster per tab. The authors' server speaks the same API
(``render(server=...)``), but it is safe only with ``workers=1``: it keeps the screenshot in an
undeclared global, so concurrent requests can return each other's images. It also leaves a ~5 MB
Chrome profile in the temp dir per screenshot.

Each scene gives ``<split>/<class>/<index>.png`` and ``<split>_part_map/<class>/<index>.png``, both
resized to 256x256 like the authors' images. Files that exist are skipped, so an interrupted render
resumes. ``--renderer`` is the authors' ``render/`` folder after ``npm install``::

    python -m cbm_eval.data.funnybirds_render scenes --root data/FunnyBirdsSpurious --spurious-corr 0.95
    python -m cbm_eval.data.funnybirds_render render --root data/FunnyBirdsSpurious --split test \
        --renderer ../funnybirds/render

``scripts/funnybirds_modal.py`` runs both on Modal, sharded over containers.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

# The authors' render/parts.json: the variants of each part (the server loads these models).
PARTS = {
    "beak": [{"model": f"beak0{i}.glb", "color": "yellow"} for i in range(1, 5)],
    "eye": [{"model": f"eye0{i}.glb"} for i in range(1, 4)],
    "foot": [{"model": f"foot0{i}.glb"} for i in range(1, 5)],
    "tail": [{"model": f"tail0{i}.glb", "color": c} for c in ("red", "green", "blue") for i in range(1, 4)],
    "wing": [{"model": f"wing0{i}.glb", "color": c} for c in ("red", "green", "blue") for i in range(1, 3)],
}
BG_COLORS = ["red", "green", "blue", "yellow"]
BG_SHAPES = 5  # box, cone, sphere, cylinder, capsule
SIZE = 256
SERVER_JS = Path(__file__).with_name("funnybirds_server.js")


def sample_classes(n_classes: int, rng: random.Random, parts: dict = PARTS) -> list[dict]:
    """``n_classes`` distinct part combinations: ``[{"class_idx": c, "parts": {part: variant}}]``."""
    total = math.prod(len(v) for v in parts.values())
    if n_classes > total:
        raise ValueError(f"n_classes={n_classes}, but only {total} part combinations exist")
    combos: list[tuple[int, ...]] = []
    while len(combos) < n_classes:
        combo = tuple(rng.randrange(len(parts[p])) for p in parts)
        if combo not in combos:
            combos.append(combo)
    return [{"class_idx": c, "parts": dict(zip(parts, combo))} for c, combo in enumerate(combos)]


def sample_scene(cls: dict, rng: random.Random, train: bool, parts: dict = PARTS, drop_parts: float = 0.5,
                 spurious_corr: float | None = None, bg_objects: tuple[int, int] = (0, 34)) -> dict:
    """One scene's render parameters (the server's query string) plus ``class_idx`` and, with
    ``spurious_corr``, the background-color attribute ``attr``."""
    tau = 2 * math.pi
    scene = {"class_idx": cls["class_idx"], "camera_distance": rng.randint(200, 400),
             "camera_pitch": rng.uniform(0, tau), "camera_roll": rng.uniform(0, tau), "light_distance": 300,
             "light_pitch": rng.uniform(0, tau), "light_roll": rng.uniform(0, tau)}
    keep = list(parts)
    if train and rng.random() < drop_parts:
        keep = rng.sample(keep, len(keep) - rng.randint(0, len(keep)))
    for p in parts:
        for key, value in parts[p][cls["parts"][p]].items():
            scene[f"{p}_{key}"] = value if p in keep else "placeholder"

    n = rng.randint(*bg_objects)
    if spurious_corr is not None:
        aligned = train and rng.random() < spurious_corr
        attr = cls["class_idx"] % len(BG_COLORS) if aligned else rng.randrange(len(BG_COLORS))
        colors = [BG_COLORS[attr]] * n
        scene["attr"] = attr
    else:
        colors = [rng.choice(BG_COLORS) for _ in range(n)]
    draws = {"bg_objects": lambda: rng.randrange(BG_SHAPES), "bg_radius": lambda: rng.randint(100, 200),
             "bg_pitch": lambda: rng.uniform(0, tau), "bg_roll": lambda: rng.uniform(0, tau),
             "bg_scale_x": lambda: rng.randint(5, 20), "bg_scale_y": lambda: rng.randint(5, 20),
             "bg_scale_z": lambda: rng.randint(5, 20), "bg_rot_x": lambda: rng.uniform(0, tau),
             "bg_rot_y": lambda: rng.uniform(0, tau), "bg_rot_z": lambda: rng.uniform(0, tau)}
    for key, draw in draws.items():
        scene[key] = ",".join(str(draw()) for _ in range(n))
    scene["bg_color"] = ",".join(colors)
    return scene


def write_scenes(root: str | Path, n_classes: int = 50, train_per_class: int = 1000, test_per_class: int = 10,
                 seed: int = 0, drop_parts: float = 0.5, spurious_corr: float | None = None,
                 bg_objects: tuple[int, int] = (0, 34), overwrite: bool = False) -> Path:
    """Write ``classes.json``, ``parts.json``, ``dataset_{train,test}.json`` and ``meta.json`` (the
    arguments and ``n_attrs``) to ``root``. Refuses to replace existing scenes unless ``overwrite``."""
    root = Path(root)
    if (root / "dataset_train.json").exists() and not overwrite:
        raise FileExistsError(f"{root} already has scenes; pass overwrite=True (--overwrite) to replace them")
    args = {"n_classes": n_classes, "train_per_class": train_per_class, "test_per_class": test_per_class,
            "seed": seed, "drop_parts": drop_parts, "spurious_corr": spurious_corr, "bg_objects": list(bg_objects)}
    rng = random.Random(seed)
    classes = sample_classes(n_classes, rng)
    kw = {"drop_parts": drop_parts, "spurious_corr": spurious_corr, "bg_objects": tuple(bg_objects)}
    splits = {split: [sample_scene(c, rng, split == "train", **kw) for c in classes for _ in range(n)]
              for split, n in (("train", train_per_class), ("test", test_per_class))}
    root.mkdir(parents=True, exist_ok=True)
    meta = {"generator": "cbm_eval.data.funnybirds_render", "args": args,
            "n_attrs": len(BG_COLORS) if spurious_corr is not None else 1}
    files = {"classes.json": classes, "parts.json": PARTS, "meta.json": meta,
             **{f"dataset_{s}.json": scenes for s, scenes in splits.items()}}
    for name, data in files.items():
        (root / name).write_text(json.dumps(data))
    return root


def scene_url(scene: dict, server: str, render_mode: str) -> str:
    query = {"render_mode": render_mode, **{k: v for k, v in scene.items() if k not in ("class_idx", "attr")}}
    return f"{server.rstrip('/')}/render?{urllib.parse.urlencode(query, safe=',')}"


def fetch_png(url: str, timeout: float = 300) -> bytes:
    """The server answers with a base64-encoded PNG screenshot."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return base64.b64decode(r.read())


@contextmanager
def serve(renderer: str | Path, port: int = 8081, pages: int | None = None,
          startup_timeout: float = 120) -> Iterator[str]:
    """Run ``funnybirds_server.js`` on the authors' ``render/`` folder (after ``npm install``) with
    ``pages`` tabs (default: one per CPU); yields its URL and stops it on exit."""
    renderer = Path(renderer)
    if not (renderer / "page.html").exists() or not (renderer / "node_modules").is_dir():
        raise FileNotFoundError(f"{renderer} is not the funnybirds render/ folder after `npm install` "
                                "(git clone https://github.com/visinf/funnybirds; cd funnybirds/render; npm install)")
    with socket.socket() as s:
        if s.connect_ex(("localhost", port)) == 0:
            raise RuntimeError(f"Port {port} is in use; stop that server or pass another port (--port)")
    log = tempfile.TemporaryFile()
    cmd = ["node", str(SERVER_JS), str(renderer), "--port", str(port), "--pages", str(pages or os.cpu_count() or 1)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + startup_timeout
        while proc.poll() is None:
            with socket.socket() as s:
                if s.connect_ex(("localhost", port)) == 0:
                    break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Render server did not start within {startup_timeout:.0f} s")
            time.sleep(0.5)
        if proc.poll() is not None:
            log.seek(0)
            raise RuntimeError(f"Render server exited:\n{log.read().decode(errors='replace')[-2000:]}")
        yield f"http://localhost:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log.close()


def render(root: str | Path, split: str, server: str = "http://localhost:8081", start: int = 0,
           stop: int | None = None, workers: int = 1, retries: int = 5,
           fetch: Callable[[str], bytes] = fetch_png) -> int:
    """Render scenes ``[start, stop)`` of ``dataset_<split>.json`` (image and part map) with
    ``workers`` requests in flight; returns how many files were written. A blank screenshot (the
    scene had not loaded) is retried."""
    root = Path(root)
    scenes = json.loads((root / f"dataset_{split}.json").read_text())
    jobs = [(i, mode, folder) for i in range(start, len(scenes) if stop is None else min(stop, len(scenes)))
            for mode, folder in (("default", split), ("part_map", f"{split}_part_map"))]

    def one(job) -> int:
        i, mode, folder = job
        path = root / folder / str(scenes[i]["class_idx"]) / f"{i:06d}.png"
        if path.exists():
            return 0
        for _ in range(retries):
            img = Image.open(io.BytesIO(fetch(scene_url(scenes[i], server, mode))))
            red = np.asarray(img)[..., 0]
            if not (red == red.flat[0]).all():
                break
        else:
            raise RuntimeError(f"Scene {i} of {split} rendered blank {retries} times; is the server healthy?")
        resample = Image.NEAREST if mode == "part_map" else Image.BICUBIC  # keep part colors exact
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.png")
        img.resize((SIZE, SIZE), resample).save(tmp)
        tmp.rename(path)  # never leave a half-written image that a resumed render would skip
        return 1

    with ThreadPoolExecutor(workers) as pool:
        return sum(pool.map(one, jobs))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m cbm_eval.data.funnybirds_render", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenes", help="sample classes and scenes into --root")
    s.add_argument("--root", required=True)
    s.add_argument("--classes", type=int, default=50)
    s.add_argument("--train-per-class", type=int, default=1000)
    s.add_argument("--test-per-class", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--drop-parts", type=float, default=0.5)
    s.add_argument("--spurious-corr", type=float, default=None)
    s.add_argument("--bg-objects", type=int, nargs=2, default=(0, 34), metavar=("MIN", "MAX"))
    s.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("render", help="render the scenes of one split")
    r.add_argument("--root", required=True)
    r.add_argument("--split", choices=("train", "test"), required=True)
    where = r.add_mutually_exclusive_group(required=True)
    where.add_argument("--renderer", help="the funnybirds render/ folder: start funnybirds_server.js on it")
    where.add_argument("--server", help="URL of a running server (the authors' server.js: keep --workers 1)")
    r.add_argument("--pages", type=int, default=None, help="tabs (and parallel requests) with --renderer; default: CPUs")
    r.add_argument("--port", type=int, default=8081)
    r.add_argument("--workers", type=int, default=1, help="parallel requests with --server")
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--stop", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "scenes":
        root = write_scenes(a.root, a.classes, a.train_per_class, a.test_per_class, a.seed, a.drop_parts,
                            a.spurious_corr, tuple(a.bg_objects), a.overwrite)
        print(f"scenes -> {root}")
    else:
        if a.renderer:
            pages = a.pages or os.cpu_count() or 1
            with serve(a.renderer, a.port, pages) as url:
                n = render(a.root, a.split, url, a.start, a.stop, workers=pages)
        else:
            n = render(a.root, a.split, a.server, a.start, a.stop, workers=a.workers)
        print(f"{n} images rendered into {a.root}")


if __name__ == "__main__":
    main()
