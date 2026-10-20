"""Download, verify and check the folders of the real datasets (``scripts/prepare_data.py``).

A :class:`Recipe` per target names the folder under ``--root`` that the loader reads, the archives
that fill it (URL and SHA-256), the terms of use, and a check. :func:`check` compares a folder
against expected counts (files, rows, split sizes) and then builds the registered loader on it;
each problem comes back as one message that says what is wrong. :func:`prepare` downloads only
what is missing and never overwrites an existing folder. Tests run the checks on small fake
folders with their own ``expect`` counts; the download code is not tested (no network in tests).
"""

from __future__ import annotations

import hashlib
import pickle
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..registry import DATASETS

SCRIPT = "uv run python scripts/prepare_data.py"
_UA = {"User-Agent": "Mozilla/5.0"}  # data.caltech.edu answers urllib's default agent with 403


@dataclass(frozen=True)
class Archive:
    """A download, saved as ``<root>/<filename>`` (reused while its SHA-256 matches) and extracted
    into ``<root>/<extract_to>``. ``rename`` moves the extracted top folder to the loader's name."""

    url: str
    sha256: str
    filename: str
    extract_to: str = "."
    rename: tuple[str, str] | None = None


@dataclass(frozen=True)
class Recipe:
    """How to get one target into ``<root>/<folder>`` and how to check it.

    ``check(folder, expect)`` returns problems (empty = the loader accepts the folder). A target
    without ``archives`` is prepared by hand from ``manual``; ``convert`` then turns that result
    into the loader's layout. ``needs`` are targets that must be prepared first.
    """

    name: str
    folder: str
    terms: str
    check: Callable[[Path, dict[str, Any]], list[str]]
    expect: dict[str, Any] = field(default_factory=dict)
    archives: tuple[Archive, ...] = ()
    manual: str = ""
    convert: Callable[[Path], list[str]] | None = None
    needs: tuple[str, ...] = ()


# ---------------------------------------------------------------------------------------- helpers

def _read(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def _count_lines(path: Path) -> int:
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            n, last = n + chunk.count(b"\n"), chunk[-1:]
    return n + (last != b"\n")


def _rows(path: Path, want: int) -> list[str]:
    n = _count_lines(path)
    return [] if n == want else [f"{path} has {n} rows, expected {want} (truncated or wrong file)"]


def _absent(folder: Path, marker: str, aliases: tuple[str, ...] = ()) -> list[str]:
    """[] if ``folder/marker`` exists, else one message with a hint when the files sit one level
    too deep (archive extracted into its own folder) or under the archive's original folder name."""
    if (folder / marker).exists():
        return []
    msg = f"{folder / marker} not found"
    nested = sorted(folder.glob(f"*/{marker}")) if folder.is_dir() else []
    renamed = [folder.parent / a / marker for a in aliases if (folder.parent / a / marker).exists()]
    if nested:
        msg += f"; found {nested[0]}: the folder is nested one level too deep, move its content up"
    elif renamed:
        msg += f"; found {renamed[0]}: rename {renamed[0].parent.name} to {folder.name}"
    return [msg]


def _missing_files(base: Path, names: list[str]) -> list[str]:
    missing = [n for n in names if not (base / n).is_file()]
    if not missing:
        return []
    return [f"{len(missing)} of {len(names)} image files are missing under {base}, e.g. "
            + ", ".join(missing[:3])]


def _binary(meta: pd.DataFrame, cols: list[str], path: Path) -> list[str]:
    return [f"{path}: column '{c}' has values {sorted(set(meta[c]) - {0, 1})} outside 0/1"
            for c in cols if not set(meta[c]) <= {0, 1}]


def _load(spec: dict[str, Any], sizes: dict[str, int]) -> tuple[Any, list[str]]:
    """Build the registered loader; compare ``len(samples(split))`` with ``sizes`` (and refuse empty splits)."""
    try:
        ds = DATASETS.build(spec)
        got = {s: len(ds.samples(s)) for s in ds.splits}
    except Exception as e:  # noqa: BLE001 - any loader failure is a finding, reported verbatim
        return None, [f"the '{spec['name']}' loader fails on this folder: {type(e).__name__}: {e}"]
    problems = [f"the loader finds no {s} samples" for s, n in got.items() if n == 0]
    problems += [f"the loader finds {got[s]} {s} samples, expected {n}" for s, n in sizes.items() if got[s] != n]
    return ds, problems


# ---------------------------------------------------------------------------------------- checks

_CUB_FILES = ["images.txt", "classes.txt", "image_class_labels.txt", "train_test_split.txt",
              "attributes/image_attribute_labels.txt", "parts/parts.txt", "parts/part_locs.txt"]


def _cub_attributes(folder: Path) -> Path | None:
    """``attributes.txt``: inside ``attributes/`` or, as the archive ships it, next to the folder."""
    return next((p for p in (folder / "attributes" / "attributes.txt", folder.parent / "attributes.txt")
                 if p.is_file()), None)


def check_cub(folder: Path, expect: dict[str, Any]) -> list[str]:
    """The official ``CUB_200_2011`` folder (``attributes.txt`` may sit next to it)."""
    if problems := _absent(folder, "images.txt"):
        return problems
    problems = [f"{folder / f} not found" for f in _CUB_FILES if not (folder / f).is_file()]
    attrs = _cub_attributes(folder)
    if attrs is None:
        problems.append(f"attributes.txt not found in {folder / 'attributes'} or {folder.parent} "
                        "(the archive puts it next to CUB_200_2011/)")
    if problems:
        return problems
    n, k, q = expect["images"], expect["attributes"], expect["parts"]
    for path, want in [(folder / "images.txt", n), (folder / "image_class_labels.txt", n),
                       (folder / "train_test_split.txt", n), (folder / "classes.txt", expect["classes"]),
                       (attrs, k), (folder / "parts" / "parts.txt", q), (folder / "parts" / "part_locs.txt", n * q),
                       (folder / "attributes" / "image_attribute_labels.txt", n * k)]:
        problems += _rows(path, want)
    split = Counter(t for _, t in _read(folder / "train_test_split.txt"))
    problems += [f"train_test_split.txt marks {split[flag]} images as {s}, expected {expect[s]}"
                 for s, flag in (("train", "1"), ("test", "0")) if split[flag] != expect[s]]
    problems += _missing_files(folder / "images", [p for _, p in _read(folder / "images.txt")])
    if not problems:
        _, problems = _load({"name": "cub", "root": str(folder)}, {"test": expect["test"]})
    return problems


def check_cub_parts(folder: Path, expect: dict[str, Any]) -> list[str]:
    """CUB70 masks in ``<cub>/part_segmentations/AnnotationMasksPerclass/<class id>/<image>_<part>.png``."""
    from .cub import _SEG_FILES

    if problems := _absent(folder, "", ("AnnotationMasksPerclass",)):
        return problems
    cub = folder.parent.parent
    if not (cub / "images.txt").is_file():
        return [f"{cub} is not a CUB folder: the masks must sit under <CUB_200_2011>/part_segmentations/"]
    dirs = [d for d in folder.iterdir() if d.is_dir()]
    bad = [d.name for d in dirs if not d.name.isdigit()]
    if bad:
        return [f"{folder} must hold one folder per CUB class id (1..200); found {bad[:3]}"]
    masks = [p for d in dirs for p in d.glob("*.png")]
    problems = [] if len(dirs) == expect["classes"] else \
        [f"{folder} has {len(dirs)} class folders, expected {expect['classes']}"]
    if len(masks) != expect["masks"]:
        problems.append(f"{folder} has {len(masks)} mask files, expected {expect['masks']}")
    # The loader finds masks by class id and image stem: every known part file must match one image.
    images = dict(_read(cub / "images.txt"))
    stems: dict[str, set[str]] = defaultdict(set)
    for i, c in _read(cub / "image_class_labels.txt"):
        stems[c].add(Path(images[i]).stem)
    parts = [f for files in _SEG_FILES.values() for f in files]
    orphans = [p for p in masks for f in parts
               if p.name.endswith(f"_{f}.png") and p.name[: -len(f) - 5] not in stems[p.parent.name]]
    if orphans:
        problems.append(f"{len(orphans)} masks match no image of their class in {cub / 'images.txt'}, "
                        f"e.g. {orphans[0].relative_to(folder)}")
    if not problems:
        ds, problems = _load({"name": "cub", "root": str(cub)}, {})
        if ds is not None and not ds.part_spec().seg_groups:
            problems.append("the cub loader does not pick up the masks")
    return problems


def check_cub_koh_split(folder: Path, expect: dict[str, Any]) -> list[str]:
    """Koh et al.'s ``class_attr_data_10`` folder with ``train.pkl`` / ``val.pkl`` / ``test.pkl``."""
    if problems := _absent(folder, "train.pkl"):
        return problems
    problems, sizes = [], {}
    for s in ("train", "val", "test"):
        path = folder / f"{s}.pkl"
        if not path.is_file():
            problems.append(f"{path} not found")
            continue
        with open(path, "rb") as f:
            rows = pickle.load(f)
        sizes[s] = len(rows)
        if len(rows) != expect[s]:
            problems.append(f"{path} has {len(rows)} images, expected {expect[s]}")
        if any("images" not in r["img_path"].split("/") for r in rows):
            problems.append(f"{path}: some img_path values have no 'images/' component")
    cub = folder.parent.parent / "CUB_200_2011"
    if not problems and not (cub / "images.txt").is_file():
        problems.append(f"{cub} not found: prepare 'cub' first; the split only selects CUB images")
    if not problems:
        _, problems = _load({"name": "cub", "root": str(cub), "majority_vote": "koh",
                              "split_dir": str(folder)}, sizes)
    return problems


def check_waterbirds(folder: Path, expect: dict[str, Any]) -> list[str]:
    """``metadata.csv`` (``img_filename, y, split, place``; split 0/1/2) and the images it lists."""
    if problems := _absent(folder, "metadata.csv", ("waterbird_complete95_forest2water2",)):
        return problems
    path = folder / "metadata.csv"
    meta = pd.read_csv(path)
    if missing := {"img_filename", "y", "split", "place"} - set(meta.columns):
        return [f"{path} lacks the columns {sorted(missing)}"]
    problems = [] if len(meta) == expect["images"] else [f"{path} has {len(meta)} rows, expected {expect['images']}"]
    split = Counter(meta["split"])
    problems += [f"{path} has {split[i]} {s} rows (split == {i}), expected {expect[s]}"
                 for i, s in enumerate(("train", "val", "test")) if split[i] != expect[s]]
    problems += _binary(meta, ["y", "place"], path)
    problems += _missing_files(folder, list(meta["img_filename"]))
    if not problems:
        _, problems = _load({"name": "waterbirds", "root": str(folder)},
                             {s: expect[s] for s in ("train", "val", "test")})
    return problems


_MS_SPLITS = {0: "train", 1: "val", 2: "test", "train": "train", "val": "val", "test": "test"}


def check_metashift(folder: Path, expect: dict[str, Any]) -> list[str]:
    """``metadata.csv`` (``filename, y, a, split``; split 0/1/2 or train/val/test), paths relative to the folder."""
    if not (folder / "metadata.csv").exists() and (folder / "metadata_metashift.csv").exists():
        return [(f"{folder} has SubpopBench's metadata_metashift.csv but no metadata.csv; "
                 "run the prepare step to convert it")]
    if problems := _absent(folder, "metadata.csv"):
        return problems
    path = folder / "metadata.csv"
    meta = pd.read_csv(path)
    if missing := {"filename", "y", "a", "split"} - set(meta.columns):
        return [f"{path} lacks the columns {sorted(missing)}"]
    if unknown := set(meta["split"]) - set(_MS_SPLITS):
        return [f"{path}: unknown split values {sorted(map(str, unknown))} (use 0/1/2 or train/val/test)"]
    problems = [] if len(meta) == expect["images"] else [f"{path} has {len(meta)} rows, expected {expect['images']}"]
    split = Counter(_MS_SPLITS[v] for v in meta["split"])
    problems += [f"{path} has {split[s]} {s} rows, expected {expect[s]}"
                 for s in ("train", "val", "test") if split[s] != expect[s]]
    problems += _binary(meta, ["y", "a"], path)
    problems += _missing_files(folder, list(meta["filename"]))
    if not problems:
        _, problems = _load({"name": "metashift", "root": str(folder)},
                             {s: expect[s] for s in ("train", "val", "test")})
    return problems


def convert_metashift(folder: Path) -> list[str]:
    """SubpopBench's ``metadata_metashift.csv`` -> ``metadata.csv`` with paths relative to ``folder``.

    SubpopBench stores ``<data_path>/metashift/MetaShift-Cat-Dog-indoor-outdoor/...``; the loader
    joins ``filename`` to its root, so keep the part from the image folder on.
    """
    src = folder / "metadata_metashift.csv"
    if not src.is_file():
        return [f"{src} not found: run the SubpopBench steps above first"]
    meta = pd.read_csv(src)
    marker = "MetaShift-Cat-Dog-indoor-outdoor"
    rel = [Path(*p.parts[p.parts.index(marker):]).as_posix() if marker in p.parts else None
           for p in map(Path, meta["filename"])]
    if None in rel:
        return [f"{src}: some filenames do not contain '{marker}/', e.g. {meta['filename'][rel.index(None)]}"]
    meta.assign(filename=rel)[["filename", "y", "a", "split"]].to_csv(folder / "metadata.csv", index=False)
    print(f"  wrote {folder / 'metadata.csv'} from {src.name}")
    return []


# ---------------------------------------------------------------------------------------- recipes

RECIPES: dict[str, Recipe] = {r.name: r for r in [
    Recipe(
        "cub", "CUB_200_2011", check=check_cub,
        expect={"images": 11788, "classes": 200, "attributes": 312, "parts": 15, "train": 5994, "test": 5794},
        archives=(Archive("https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
                          "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081",
                          "CUB_200_2011.tgz"),),
        terms="CUB-200-2011 (Wah et al., 2011). The Caltech data record lists the license as CC BY; the "
              "images come from Flickr. Cite the dataset as the record asks: "
              "https://data.caltech.edu/records/65de6-vp158"),
    Recipe(
        "cub-parts", "CUB_200_2011/part_segmentations/AnnotationMasksPerclass", check=check_cub_parts,
        expect={"classes": 67, "masks": 14245}, needs=("cub",),
        archives=(Archive("https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset/raw/main/"
                          "AnnotationMasksPerclass.tar.xz",
                          "2d5d4f59fe8c4debd94e2272ad5cec65bfca3a0412952ffb5bfef87f29fdbd73",
                          "AnnotationMasksPerclass.tar.xz", extract_to="CUB_200_2011/part_segmentations"),),
        terms="CUB70 part segmentations (Behzadi et al.), MIT license: "
              "https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset. The masks are drawn on CUB "
              "images, so CUB's terms apply too."),
    Recipe(
        "cub-koh-split", "CUB_processed/class_attr_data_10", check=check_cub_koh_split,
        expect={"train": 4796, "val": 1198, "test": 5794}, needs=("cub",),
        manual="Download Koh et al.'s CUB_processed bundle from Codalab (worksheet "
               "0x362911581fcd4e048ddfd84f47203fd2, bundle 0x5b9d528d2101418b87212db92fea6683), through the web "
               "page or `cl download 0x5b9d528d2101418b87212db92fea6683 -o {root}/CUB_processed`, so that "
               "{root}/CUB_processed/class_attr_data_10/{{train,val,test}}.pkl exist.",
        terms="Koh et al. (2020) CUB train/val split, from https://github.com/yewsiang/ConceptBottleneck (MIT). "
              "It only selects CUB images, so CUB's terms apply."),
    Recipe(
        "waterbirds", "waterbirds", check=check_waterbirds,
        expect={"images": 11788, "train": 4795, "val": 1199, "test": 5794},
        archives=(Archive("https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz",
                          "56c51b77f7d17283ac97003d973e9107e8af9ceebffcb0c8be14d024f846f951",
                          "waterbird_complete95_forest2water2.tar.gz",
                          rename=("waterbird_complete95_forest2water2", "waterbirds")),),
        terms="Waterbirds (Sagawa et al., 2020), the group DRO release (https://github.com/kohpangwei/group_DRO, "
              "MIT). Birds come from CUB-200-2011 and backgrounds from Places; Places images are for "
              "non-commercial research and education only (http://places2.csail.mit.edu)."),
    Recipe(
        "metashift", "metashift", check=check_metashift, convert=convert_metashift,
        expect={"images": 3499, "train": 2276, "val": 349, "test": 874},
        manual="Generate the cat/dog subset with SubpopBench (https://github.com/YyzHarry/SubpopBench):\n"
               "  git clone https://github.com/YyzHarry/SubpopBench\n"
               "  pip install gdown numpy pandas scikit-learn\n"
               "  python SubpopBench/subpopbench/scripts/download.py metashift --data_path {root} --download\n"
               "This writes {root}/metashift/metadata_metashift.csv; then run this command again to convert it.",
        terms="MetaShift (Liang & Zou, 2022; https://github.com/Weixin-Liang/MetaShift, MIT). The images come "
              "from Visual Genome (CC BY 4.0); the cat/dog subset and its split come from SubpopBench (MIT)."),
]}


# ---------------------------------------------------------------------------------------- actions

def check(name: str, root: str | Path, expect: dict[str, Any] | None = None) -> list[str]:
    """Problems with ``<root>/<recipe.folder>`` for target ``name``; [] when the loader accepts it."""
    r = RECIPES[name]
    return r.check(Path(root) / r.folder, {**r.expect, **(expect or {})})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def fetch(archive: Archive, root: Path) -> Path:
    """Download ``archive`` to ``root`` (or reuse a verified copy); raise on a checksum mismatch."""
    path = root / archive.filename
    if path.is_file():
        if _sha256(path) == archive.sha256:
            print(f"  reusing {path} (checksum ok)")
            return path
        print(f"  {path} has a different checksum; downloading again")
    root.mkdir(parents=True, exist_ok=True)
    tmp, h, done = path.with_name(path.name + ".part"), hashlib.sha256(), 0
    print(f"  downloading {archive.url}")
    with urllib.request.urlopen(urllib.request.Request(archive.url, headers=_UA)) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while chunk := r.read(1 << 22):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total and sys.stderr.isatty():
                print(f"\r  {done / 2**20:.0f} / {total / 2**20:.0f} MiB", end="", file=sys.stderr)
    if total and sys.stderr.isatty():
        print(file=sys.stderr)
    if h.hexdigest() != archive.sha256:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {archive.url}: got {h.hexdigest()}, expected "
                           f"{archive.sha256}. Retry; if it persists, the file changed upstream: open an issue.")
    tmp.replace(path)
    return path


def extract(archive: Archive, path: Path, root: Path) -> None:
    dest = root / archive.extract_to
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {path.name} into {dest}")
    with tarfile.open(path) as t:
        if hasattr(tarfile, "data_filter"):  # refuses absolute paths and links out of dest
            t.extractall(dest, filter="data")
        else:
            t.extractall(dest)
    if archive.rename:
        (dest / archive.rename[0]).rename(dest / archive.rename[1])


def prepare(name: str, root: str | Path, expect: dict[str, Any] | None = None) -> list[str]:
    """Bring ``<root>/<recipe.folder>`` into the loader's layout; return the problems that remain.

    Skips a folder that already passes the check and refuses to touch one that exists but fails it.
    """
    r, root = RECIPES[name], Path(root)
    folder = root / r.folder
    print(f"{name}: {folder}\n  terms: {r.terms}")
    for need in r.needs:
        if problems := check(need, root):
            return [f"prepare '{need}' first: {problems[0]}"]
    if not (problems := check(name, root, expect)):
        print("  already prepared")
        return []
    if not r.archives:
        if r.convert is not None and (problems := r.convert(folder)):
            print("  " + r.manual.format(root=root).replace("\n", "\n  "))
            return problems
        if r.convert is None and r.manual:
            print("  " + r.manual.format(root=root).replace("\n", "\n  "))
        return check(name, root, expect)
    if folder.exists():
        return [(f"{folder} exists but fails the check; prepare never overwrites it. Fix it with the "
                 "messages below, or move it away and run prepare again."), *problems]
    for a in r.archives:
        extract(a, fetch(a, root), root)
    return check(name, root, expect)
