# CBM-Design-Evaluation
<!-- --8<-- [start:intro] -->

Research code for a controlled design study of concept bottleneck models (CBMs). A CBM is assembled
from six interchangeable stages, trained on cached backbone features, evaluated for robustness,
leakage and localization, and logged as one row per run and seed for factor-level analysis.
<!-- --8<-- [end:intro] -->

![Data flow of one run: config, pipeline builder, stages, trained CBM, evaluators, results](docs/assets/architecture.svg)

The full documentation (getting started, architecture, component catalogue, API reference) is a
docs site built from `docs/`: run `uv run --group docs properdocs serve` and open
http://127.0.0.1:8000.

## Setup

```bash
uv sync                                          # core + dev group (pytest)
uv sync --all-extras                             # + CLIP, Grounding DINO, Claude, statsmodels, W&B, Modal
uv run pytest -q                                 # fully offline, ~5 s
uv run --group docs properdocs serve             # docs site at http://127.0.0.1:8000
```

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). The extras are `clip` (OpenCLIP),
`hf` (Grounding DINO and SAM via transformers), `llm` (Claude concept discovery), `analysis`
(mixed-effects models via statsmodels), `tracking` (W&B) and `modal` (the Modal runner in
`scripts/`). Pretrained weights (CLIP, DINOv2,
ResNet, Inception) download on first use.

## Quickstart

```bash
uv run cbm-eval list                                               # registered components per stage
uv run cbm-eval run configs/anchors/synthetic.yaml --seeds 0 1 2   # one config, three seeds (offline)
uv run cbm-eval run configs/anchors/synthetic.yaml --set stages.predictor.lam=0.01   # override a key
uv run cbm-eval sweep configs/ablations/synthetic_stages.yaml --dry-run              # list the runs
uv run cbm-eval sweep configs/ablations/synthetic_stages.yaml                        # run them
uv run cbm-eval analyze --results results/synthetic.jsonl --metric shift.test.wga --frontier leakage.intervention.gain
uv run cbm-eval evaluate runs/synthetic/<run_id>-s0 --config configs/anchors/synthetic.yaml  # re-score, no retraining
```

The synthetic anchor needs no data, weights or network access and finishes in about a second;
the full synthetic sweep (45 runs) takes under a minute on a laptop CPU. Results are appended
to `results/synthetic.jsonl`, and each run's weights and concept scores are saved under
`runs/synthetic/`.

<!-- --8<-- [start:layout] -->
## Layout

| Part | Code (under `src/cbm_eval/`) |
|---|---|
| Experiment configs | `configs/anchors/*.yaml`, `configs/ablations/*.yaml`, `src/cbm_eval/config.py` |
| Datasets | `data/` — `waterbirds`, `cub`, `metashift`, `synthetic` |
| Backbones | `backbones/` — `clip` (OpenCLIP), `dinov2`, `resnet`, `inception` (Inception-v3), `toy` |
| Pipeline builder | `pipeline.py` (+ `registry.py`, `context.py`) |
| Stage modules | `stages/` — one ABC per stage in `stages/base.py` |
| Trained CBM | `model.py` — weights + cached concept scores, saved to `runs/<run_id>-s<seed>/` |
| Evaluation suite | `evaluation/` — `shift`, `concepts`, `leakage`, `localization`, `keypoint_distance`, `part_iou`, `faithfulness` |
| Results store | `results.py` — JSONL, one row per (run_id, seed) |
| Analysis | `analysis.py` — seed aggregation, effects models, Pareto frontiers, effective robustness |

### Stages

| Stage | Variants | Interface |
|---|---|---|
| Discovery | `llm`, `vlm`, `kb`, `sae`, `dataset`, `static` | `discover(ctx) -> ConceptSet` |
| Filtering (chained) | `rules`, `clip`, `dino`, `select` | `filter(concepts, ctx) -> ConceptSet` |
| Generation | `scores`, `logits`, `embeddings` (CEM), `boc` | `build(in_dim, aligned) -> ConceptLayer` |
| Alignment | `clip`, `weights`, `dino`, `human` | `fit(concepts, ctx) -> AlignedConcepts` |
| Predictor | `sparse`, `dense`, `residual` | `build(rep_dim, n_classes, feat_dim) -> PredictorHead` |
| Training | `independent`, `sequential`, `joint` (+ `*_weighted`: imbalance-weighted concept BCE) | `fit(layer, head, aligned, ctx) -> log` |

How the stages divide the work:

- **Alignment** decides how concept neurons get their meaning. `clip` uses continuous pseudo-labels
  (teacher image–text similarity, as in Label-free CBM); `dino` uses binary pseudo-labels from an
  open-vocabulary detector; `human` uses dataset annotations; `weights` fixes each neuron to the
  concept's text embedding (or SAE direction) with no targets.
- **Generation** decides what the predictor sees: the concept score (`scores`: probabilities for
  binary concepts; `logits`: their logits, as Koh et al. connect f), a CEM-style embedding mixed by
  concept probability, or a hard 0/1 bit (straight-through in joint training).
- **Training** fits on cached frozen features by default. With `finetune: {epochs, lr, optimizer,
  momentum, weight_decay, lr_step, lr_gamma, batch_size, patience, augment, num_workers}` it also
  trains the backbone end to end (backbones with `trainable = True`: `inception`, `toy`): images go
  through a trainable copy of the encoder, optimized with the concept layer (x → c phase of
  `independent`/`sequential`) or with layer and head (`joint`). Afterwards the run's inputs, and so
  every evaluator, use the fine-tuned encoder's features; it is saved as `runs/<run>/encoder.pt`.
  Not combinable with `instances:` or patch-reading evaluators (`localization`, `keypoint_distance`,
  `part_iou`).
- Non-CLIP backbones need a text-capable `teacher:` for `clip` alignment and the `clip`/`select`
  filters (see `configs/ablations/waterbirds_backbones.yaml`).
<!-- --8<-- [end:layout] -->

## Instance bags (SEG-MIL-CBM)
<!-- --8<-- [start:instances] -->

[arXiv 2510.04180](https://arxiv.org/abs/2510.04180) classifies an image from a *bag* of
concept-guided segments. A top-level `instances:` key turns on bags, and every stage runs per
instance: inputs become `(N, M, D)` with a validity mask (`structures.Bag`).

| Paper component | Config |
|---|---|
| Top-K concepts per image → Grounding DINO boxes → SAM masks | `instances: {name: grounded_sam, top_k: 10}` |
| Segment features `h_i` (masked crop through the frozen backbone) | `instances.encode: crop` (or `pool`: patch tokens averaged under the mask) |
| Frozen CLIP targets `q_k(s_i)` = softmax of cosines | `alignment: {name: segment_clip, targets: softmax, temperature: 0.01}` |
| `L = CE + 0.1 · (−cos)` | `training: {name: joint, concept_weight: 0.1, concept_loss: cosine}` |
| Class-conditioned attention MIL with area weights | `predictor: {name: mil, pooling: attention, area_gamma: 0.25}` |
| Deletion / insertion faithfulness | `evaluation: [{name: faithfulness}]` |

Controls: `instances: patches | grounded_boxes | sam_grid`, `predictor.pooling: mean | max`. Image-level
alignments (`clip`, `human`, `dino`) also work on bags; their targets supervise the max over instances.
With bags, `predictor` must be bag-aware (`mil`). Without `instances:`, `mil` is a plain linear head.

Caching: masks go to `.cache/segments/` and depend on the dataset, segmenter and (for grounded
sources) the filtered concept set, but not on the backbone. Segment features go to
`.cache/instances/`, one entry per backbone. Any change to discovery or filtering re-segments
the dataset. `sam_grid` does not depend on concepts.

Differences from the paper: the backbone stays frozen (no warm-up). Faithfulness zeroes cached
segment embeddings rather than re-encoding images. Many of the paper's hyperparameters are only
in its appendix, so `top_k`, `boxes_per_concept`, thresholds and `tau` are our defaults. With the
scale-free cosine loss, intervention metrics that write targets into the concept layer are not
meaningful.

Training stays on CPU unless `training.device` is set; the SEG-MIL anchors use `device: auto`
(GPU when present). Segmentation and encoding run on the top-level `device`.
<!-- --8<-- [end:instances] -->

## Part localization (CUB)
<!-- --8<-- [start:parts] -->

`keypoint_distance` and `part_iou` re-implement the ProtoCBM localization metrics
([pascal0012/ProtoCBM](https://github.com/pascal0012/ProtoCBM), `localization/`). Concept maps are
the concept layer on patch features, a patch bag's per-patch scores, or, for segment bags, the
best covering segment per patch. For each image and part group, `select: argmax` (the default)
scores the group's concept with the highest image-level score. `select: present` averages over
every concept annotated present. Annotations go through the backbone's resize and center crop and
are cached under `paths.cache/parts/`.

| Evaluator | Metrics | Options |
|---|---|---|
| `keypoint_distance` | `dist` (map peak → nearest visible keypoint, in image sides; mean over the 12 groups), `dist.<group>`, `pck` (share within `threshold`), `dist_center` / `pck_center` (a map that points at the center) | `select`, `threshold: 0.1` |
| `part_iou` | `miou` (over all image–group pairs), `iou.<group>` for the 8 mask groups, `miou_center` (center prior) | `select`, `hard: true`, `keep_ratio: 0.5`, `seg_size: 56` |

Differences from ProtoCBM: the peak is the center of the top patch rather than the argmax of a
bilinear upsample (the same up to sub-patch rounding, but without the corner bias of clamped
borders). Distances are normalized by the image side rather than measured in pixels. IoU is
computed at `seg_size` rather than at full image resolution.
<!-- --8<-- [end:parts] -->

## Configs
<!-- --8<-- [start:configs] -->

An anchor fully specifies one run. Each stage is `{name: <variant>, **kwargs}`, and the kwargs go
straight to the variant's constructor. An ablation names an anchor and a set of factors:

```yaml
anchor: ../anchors/waterbirds_lfcbm.yaml
mode: ofat            # one-factor-at-a-time around the anchor (default), or grid
seeds: [0, 1, 2]
overrides: {teacher: {name: clip}}     # optional, applied to the anchor first
factors:
  stages.predictor: [{name: dense}, {name: residual}]
  stages.predictor.lam: [0.0001, 0.001]   # dotted keys reach into specs; list indices work too
```

A factor value **replaces** whatever sits at that key. For example, `stages.training: [{name: joint}]`
drops any `epochs` the anchor set, so constructor defaults apply. To vary one parameter, use a
dotted key instead.

In `grid` mode, a dotted key is applied to every combination. With both `stages.predictor: [{name:
sparse}, {name: dense}]` and `stages.predictor.lam: [...]`, `lam` is also written into the `dense`
spec, which rejects it. Put the parameter inside the spec (`{name: sparse, lam: 0.001}`) or split
the sweep into two ablations.

The `run_id` is a hash of everything that defines the model and its evaluation. Seed, name, paths
and tags are excluded. Sweeps skip `(run_id, seed)` pairs that already succeeded, so an interrupted
sweep can simply be restarted. The hash covers the config, not the code: after changing how a
variant behaves, re-run with `--force` or write to a new `paths.results` file, or the sweep will
skip those runs and keep the old numbers. Each row records `git_commit` to tell versions apart.

### Test-only domains

`eval_datasets:` adds shifted test sets, such as CUB as paintings, to a run. The CBM is trained on
`dataset` alone. Each alias names the eval dataset's `test` split, and an evaluator scores it only
when the alias is in its `splits`:

```yaml
dataset: {name: cub, root: data/CUB_200_2011}
eval_datasets:
  paintings: {name: cub_paintings, root: data/cub_paintings}
evaluation:
  - {name: shift, splits: [val, test, paintings]}
  - {name: concepts, splits: [test, paintings]}
```

Classes and concepts are matched to the training dataset's by name, and every eval class must exist
in the training dataset. An eval dataset without its own concept annotations inherits the training
dataset's class-level ones (`class_concepts()`, e.g. CUB's majority-voted attributes). A test-only
dataset cannot be used as `dataset:`.

`eval_datasets` is part of the `run_id`. To score runs that are already trained on a new domain, add
it to the anchor and re-evaluate the saved runs instead of retraining:

```bash
uv run cbm-eval evaluate runs/<run_id>-s0 runs/<run_id>-s1 --config configs/anchors/cub_human.yaml
```

`evaluate` takes `evaluation:` and `eval_datasets:` from `--config` (plus any `--set` overrides) and
refuses any change that would alter the model. It appends the row that `run` would write for the new
config, marked with `evaluated_from`, so a later `sweep` of that config skips it.
<!-- --8<-- [end:configs] -->

## Adding a component
<!-- --8<-- [start:adding] -->

```python
from cbm_eval.registry import FILTERING
from cbm_eval.stages.base import Filter

@FILTERING.register("my_filter")
class MyFilter(Filter):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def filter(self, concepts, ctx):
        ...
        return concepts.subset(keep)
```

Import the module from its package `__init__.py`. It then becomes available as `{name: my_filter, threshold: 0.3}`.

A new loss is a training subclass that overrides one hook, `_task_loss(logits, batch)` or
`_concept_loss(layer, batch)`, registered under its own name:

```python
import torch.nn.functional as F

from cbm_eval.registry import TRAINING
from cbm_eval.stages.training import Joint

@TRAINING.register("joint_smooth")
class JointSmooth(Joint):
    def __init__(self, smoothing: float = 0.1, **kw):
        super().__init__(**kw)
        self.smoothing = smoothing

    def _task_loss(self, logits, b):
        return F.cross_entropy(logits, b.y, label_smoothing=self.smoothing)
```

<!-- --8<-- [end:adding] -->

See the *Contributing* page of the docs site (or [`CLAUDE.md`](CLAUDE.md), which holds the same
rules) for the full contract per component kind.

## Data
<!-- --8<-- [start:data] -->

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
<!-- --8<-- [end:data] -->

## Metrics
<!-- --8<-- [start:metrics] -->

All metrics are prefixed by their evaluator name, e.g. `shift.test.wga`.

- `shift`: per-split accuracy, per-group accuracy, worst-group accuracy (`wga`) and mean group
  accuracy. Effective robustness is reported when a `baseline: {slope, intercept}` is given.
  Otherwise, fit one over baseline runs with `analysis.fit_robustness_baseline`.
- `concepts`: `<split>.concept_error` (1 - accuracy of `c_hat >= 0.5` vs. annotations, as in
  Koh et al.'s Table 2), `<split>.concept_f1` and the annotations' `positive_rate`. Binary
  concept layers whose concepts are all annotated only.
- `leakage`:
  - `intervention.acc@f` / `intervention.gain`: accuracy after replacing a random fraction `f` of
    concept predictions with ground truth. Ground truth is human annotations when every concept is
    annotated and the layer predicts binary concepts. Otherwise it is the alignment's own targets,
    and gains are then expected to be near zero for pseudo-label alignment.
  - `probe.soft_hard_gap`: accuracy of a probe on soft scores minus one on binarized scores.
  - `probe.spurious_bacc`: how well the spurious attribute can be decoded from the concepts.
- `localization`: pointing game and locality (map mass inside the annotated region, with
  `locality_chance` for reference). Only concepts that match an annotated dataset concept by name
  are scored.
- `keypoint_distance`, `part_iou`: CUB part localization against keypoints and part masks (see
  *Part localization*).
- `faithfulness`: deletion / insertion AUC (`dauc`, `iauc`, with `*_random` controls),
  `top1_insertion` and `top5_deletion` over segments ranked by their contribution to the
  prediction. Bag-aware heads (`predictor: mil`) only.
<!-- --8<-- [end:metrics] -->

<!-- --8<-- [start:status] -->
## Status

This is research code under active development. The whole pipeline and every stage variant are
tested end-to-end on the synthetic dataset and toy backbone. The tests also cover the image-file
path: a fake Waterbirds folder, an untrained ResNet-18, and stub LLM/VLM clients.

The CUB loader and the Inception-v3 backbone have been run on the real dataset (frozen-backbone
runs of the Koh et al. 2020 reproduction). **Not yet run against real data or pretrained
weights:** OpenCLIP and DINOv2 feature extraction (including ViT patch tokens), the Waterbirds and
MetaShift loaders, live Claude/ConceptNet calls, Grounding DINO scoring, and end-to-end backbone
fine-tuning. Expect small fixes on first contact, and please open an issue when you hit one.
<!-- --8<-- [end:status] -->
