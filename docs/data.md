# Data

Point `dataset.root` at:

- **Waterbirds**: the standard folder with `metadata.csv` (`img_filename, y, split, place`), as
  released with group DRO (`waterbird_complete95_forest2water2`).
- **CUB**: the official `CUB_200_2011` directory
  ([Caltech data record](https://data.caltech.edu/records/65de6-vp158)). Attributes are class-level majority-voted by
  default (`majority_vote: mean`: >= 50% of the class's official-train images, counting "not
  visible" as absent; 93 concepts). `majority_vote: koh` follows Koh et al.'s code (train split only,
  "not visible" negatives ignored, ties -> present; 112 concepts), and `split_dir:` takes their
  `CUB_processed/class_attr_data_10` folder (Codalab worksheet 0x362911581fcd4e048ddfd84f47203fd2)
  to use their train/val split. Each attribute is mapped to its body part's keypoint for localization.
  Optional part segmentations for `part_iou` (CUB70, 67 classes):
  `wget https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset/raw/main/AnnotationMasksPerclass.tar.xz`,
  then extract to `<root>/part_segmentations/` (or set `dataset.part_segmentations`).
- **FunnyBirds** ([Hesse et al., ICCV 2023](https://github.com/visinf/funnybirds)): rendered 3D
  birds. A class is one variant of each of five parts (beak, eye, foot, tail, wing), so the
  concepts are the 26 part variants (e.g. "red tail shape 2"). Labels are per image: half of the
  train scenes drop random parts, and a dropped part has no present variant
  (`class_level_concepts: true` labels every image with its class instead). Masks and keypoints
  for `part_iou` / `keypoint_distance` come from the color-coded part map of each image. Validation
  is a `val_fraction` of train. See [FunnyBirds](#funnybirds) below for how to get or make the data.
- **MetaShift**: a folder with `metadata.csv` (`filename, y, a, split`), as produced by the
  [SubpopBench](https://github.com/YyzHarry/SubpopBench) preprocessing scripts.
- **Synthetic**: generated in memory; no `root` needed. Pair it with the `toy` backbone for fully
  offline runs.

Backbone features are cached under `paths.cache/features/` and keyed by dataset and backbone. All
runs sharing a dataset and backbone reuse them, so training a CBM runs only on cached tensors.
LLM, VLM, ConceptNet and Grounding DINO outputs are cached under the same directory.

`llm` and `vlm` discovery call Claude (`claude-opus-5-5` by default; override with `model:`). Server-side
refusal fallback is enabled. Credentials come from `ANTHROPIC_API_KEY` or an `ant auth login` profile.
A response cut off at `max_tokens` (default 4000) raises instead of being cached; raise it with
`{name: llm, max_tokens: 8000}`.

## FunnyBirds

| Source | Classes | Train / test | Spurious attribute | How |
|---|---|---|---|---|
| The authors' render | 50 | 50,000 / 500 | none | download `FunnyBirds.zip` (1.6 GB) |
| Your own render | any (≤ 15,552 part combinations) | any | optional background color | `cbm_eval.data.funnybirds_render` |

```text
classes.json  parts.json  dataset_train.json  dataset_test.json  [meta.json]
train/<class>/<index>.png    train_part_map/<class>/<index>.png    test/...    test_part_map/...
```

**Download.** `wget download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip` and unzip into
`data/`, or on Modal `uv run modal run scripts/funnybirds_modal.py::download`.

**Render a variant.** Rendering has two steps. Step 1 samples the scenes (camera, light, background
objects, dropped parts) into JSON. It is seeded and needs no renderer:

```bash
uv run python -m cbm_eval.data.funnybirds_render scenes --root data/FunnyBirdsSpurious \
    --classes 50 --train-per-class 1000 --test-per-class 10 --spurious-corr 0.95 --bg-objects 5 34
```

`--spurious-corr` gives each scene an attribute: one color for all its background objects. On
train the color is the class's color (`class % 4`) with that probability; on test it is uniform.
The loader reads it as `Sample.attr` (`n_attrs` comes from `meta.json`), so `shift` reports
worst-group accuracy as for Waterbirds. `--drop-parts` sets the share of train scenes that drop
parts (0.5, as the authors do).

Step 2 renders each scene twice (image and part map) with the authors' three.js scene.
`funnybirds_server.js` serves their `page.html` and models in one headless Chrome with a pool of
tabs. It takes each screenshot as soon as every model has loaded. Its images are pixel-identical to
the authors' `server.js`.

| Renderer (one 8-CPU Modal container) | Seconds per screenshot |
|---|---|
| The authors' `server.js` (one request at a time) | 2.25 |
| `funnybirds_server.js`, 8 tabs | 0.86 |

The rest of the time is the software renderer (SwiftShader) drawing 1256x1256 frames on the CPU, so
more tabs than CPUs do not help. Render the full dataset on Modal. Each container renders 250
scenes with 8 tabs, and you can run as many containers as you like:

```bash
uv run modal run scripts/funnybirds_modal.py::render --name FunnyBirdsSpurious --spurious-corr 0.95 --bg-objects 5,34
uv run modal run scripts/funnybirds_modal.py::run --data FunnyBirdsSpurious \
    --overrides 'dataset.root=data/FunnyBirdsSpurious'
```

To render locally, install the authors' renderer once
(`git clone https://github.com/visinf/funnybirds && cd funnybirds/render && npm install`), then:

```bash
uv run python -m cbm_eval.data.funnybirds_render render --root data/FunnyBirdsSpurious --split test \
    --renderer ../funnybirds/render
```

This starts and stops the server itself, with one tab per CPU (`--pages`). Images that exist are
skipped, so an interrupted render resumes.
