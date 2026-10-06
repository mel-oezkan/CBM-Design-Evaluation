# Getting started

This page takes you from a fresh clone to a small design study: one trained CBM, a sweep that
varies every stage, and an effects table that says which choices mattered. Everything here runs
offline on the synthetic dataset in under a minute on a laptop CPU.

## Install

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mel-oezkan/CBM-Design-Evaluation
cd CBM-Design-Evaluation
uv sync                  # core + pytest
uv run pytest -q         # ~5 s, fully offline
```

`uv sync --all-extras` adds the optional dependencies for real-data runs: OpenCLIP (`clip`),
Grounding DINO and SAM via transformers (`hf`), Claude concept discovery (`llm`), mixed-effects
models (`analysis`), W&B (`tracking`) and the Modal client for cloud runs (`modal`). You can
install them later, e.g. `uv sync --extra clip --extra llm`.

To see what you can plug into each stage:

```bash
uv run cbm-eval list
```

```text
dataset     cub, metashift, synthetic, waterbirds
backbone    clip, dinov2, inception, resnet, toy
instances   grounded_boxes, grounded_sam, patches, sam_grid
discovery   dataset, kb, llm, sae, static, vlm
filtering   clip, dino, rules, select
generation  boc, embeddings, logits, scores
alignment   clip, dino, human, segment_clip, weights
predictor   dense, mil, residual, sparse
training    independent, independent_weighted, joint, joint_weighted, sequential, sequential_weighted
evaluation  concepts, faithfulness, keypoint_distance, leakage, localization, part_iou, shift
```

The [component catalogue](components.md) documents each of these and the kwargs it accepts.

## Train one CBM

A run is described by one YAML file, an *anchor*. The synthetic anchor builds a label-free-style
CBM: a fixed list of concept names, filtered down to 20, aligned to images by CLIP-style
similarity, with a sparse linear head on top.

```yaml title="configs/anchors/synthetic.yaml (abridged)"
name: synthetic
dataset: {name: synthetic}
backbone: {name: toy}
stages:
  discovery: {name: static, names: [concept_00, ..., distractor_03, class_0]}
  filtering:                       # applied in order
    - {name: rules}
    - {name: clip, dedupe_sim: 0.9}
    - {name: select, k: 20, method: submodular}
  generation: {name: scores}
  alignment: {name: clip}
  predictor: {name: sparse, lam: 0.001}
  training: {name: sequential, epochs: 100}
evaluation:
  - {name: shift}
  - {name: leakage}
  - {name: localization}
paths: {cache: .cache, runs: runs/synthetic, results: results/synthetic.jsonl}
```

Each stage is `{name: <variant>, **kwargs}`; the kwargs go straight to that variant's constructor.
Run it:

```bash
uv run cbm-eval run configs/anchors/synthetic.yaml --seeds 0
```

The command prints the metrics of each run. They are prefixed with the evaluator that produced
them (see [Metrics](metrics.md)):

```text
  "n_concepts": 20.0,
  "shift.test.acc": 0.7887,
  "shift.test.wga": 0.5185,                  # worst-group accuracy
  "leakage.intervention.gain": -0.0050,      # accuracy gain from fixing concepts to ground truth
  "leakage.probe.spurious_bacc": 0.8524,     # how decodable the spurious attribute is
  "localization.pointing_game": 0.9938,
  ...
```

Two things are written to disk:

- **One row in `results/synthetic.jsonl`** with the `run_id`, `seed`, `status`, `git_commit`, a
  `factor` dict naming the variant chosen at every stage, and a `metric` dict. This file is what
  the analysis reads.
- **A run folder `runs/synthetic/<run_id>-s0/`** with the config, the selected concepts, the
  model weights, the cached concept scores and the training log.

The `run_id` is a hash of the config, excluding seed, name, paths and tags. Running the same
command again skips the run because `(run_id, seed)` is already in the results file; pass
`--force` to re-run it.

## Change one setting

Use `--set` with a dotted key to override any part of the config without editing the file:

```bash
uv run cbm-eval run configs/anchors/synthetic.yaml --seeds 0 1 2 --set stages.predictor.lam=0.01
```

This is a different config, so it gets a different `run_id` and its own rows.

## Run a sweep

An *ablation* file names an anchor and the factors to vary. The synthetic one changes each stage
in turn, one factor at a time (OFAT), over three seeds:

```yaml title="configs/ablations/synthetic_stages.yaml (abridged)"
anchor: ../anchors/synthetic.yaml
mode: ofat
seeds: [0, 1, 2]
factors:
  stages.discovery: [{name: dataset}, {name: sae, n_latents: 64, k: 6, epochs: 30}]
  stages.generation: [{name: embeddings}, {name: boc}]
  stages.alignment: [{name: weights}, {name: human}, {name: dino, scorer: oracle}]
  stages.predictor: [{name: dense}, {name: residual}]
  stages.training: [{name: independent}, {name: joint}]
  ...
```

List the runs first, then launch them:

```bash
uv run cbm-eval sweep configs/ablations/synthetic_stages.yaml --dry-run   # 45 runs
uv run cbm-eval sweep configs/ablations/synthetic_stages.yaml
```

```text
44 ran, 1 skipped, 0 failed
```

The skipped run is the anchor with seed 0, which already ran above. An interrupted sweep can be
restarted with the same command; finished runs are skipped. [Designing an experiment](experiments.md)
has more sweep designs, and the [config reference](configs.md) covers how factor values replace keys.

## Analyze the results

```bash
uv run cbm-eval analyze --results results/synthetic.jsonl \
    --metric shift.test.wga --frontier leakage.intervention.gain
```

The output has three parts:

1. **A table per configuration** with the mean, standard deviation and seed count of the metric.
2. **An additive effects model.** Each coefficient is the change in the metric from switching one
   factor away from its reference level (the anchor's choice), with a 95% confidence interval:

    ```text
    metric.shift.test.wga  (n=45, reference levels: {...})
      intercept                                       +0.5218  [+0.4988, +0.5447]
      discovery=sae(epochs=30,k=6,n_latents=64)       -0.3983  [-0.4308, -0.3658]
      generation=boc                                  -0.2193  [-0.2518, -0.1868]
      predictor=dense                                 -0.1259  [-0.1584, -0.0935]
      alignment=human                                 +0.0255  [-0.0070, +0.0580]
      ...
    ```

3. **A Pareto frontier** over the two metrics, here worst-group accuracy against intervention gain.

Your numbers can differ slightly with hardware and library versions. For custom analyses, load
the results as a flat DataFrame (columns like `factor.discovery`, `metric.shift.test.wga`) and use
the functions in [`cbm_eval.analysis`][cbm_eval.analysis]:

```python
from cbm_eval.results import ResultsStore
from cbm_eval import analysis

df = ResultsStore("results/synthetic.jsonl").load()
print(analysis.aggregate(df, ["metric.shift.test.wga"]))
```

## Move to real data

The other anchors in `configs/anchors/` target real datasets:

| Anchor | What it is | Needs |
|---|---|---|
| `waterbirds_lfcbm.yaml` | Label-free-CBM style on Waterbirds: Claude-generated concepts, CLIP alignment | Waterbirds, `clip` + `llm` extras, Claude credentials |
| `metashift_lfcbm.yaml` | The same design on MetaShift | MetaShift, `clip` + `llm` extras, Claude credentials |
| `cub_human.yaml` | Classic CBM on CUB: annotated attributes, human supervision, keypoint localization | CUB, `clip` extra |
| `waterbirds_segmil.yaml`, `cub_segmil.yaml` | SEG-MIL-CBM: concept-guided segments as instance bags | The dataset, `clip` + `hf` + `llm` extras, Claude credentials; a GPU helps |
| `cub_koh2020*.yaml` | Reproduction of Koh et al. 2020 with a fine-tuned Inception-v3 | CUB and the authors' split, a GPU |

To use one:

1. Download the dataset and point `dataset.root` at it (the anchors expect it under `data/`, e.g.
   `data/waterbirds`, `data/CUB_200_2011`). The expected folder layouts are on the
   [Data](data.md) page.
2. Install the extras the anchor needs, e.g. `uv sync --extra clip --extra llm`.
3. For `llm`/`vlm` discovery, set `ANTHROPIC_API_KEY`. Responses are cached under
   `.cache/`, so each prompt is only sent once.
4. Run it like the synthetic anchor. The first run extracts and caches backbone features under
   `.cache/features/`; later runs on the same dataset and backbone only train on cached tensors.

Training runs on the CPU unless the training stage sets `device` (e.g. `device: auto`).

## Next steps

- [How a run works](how-it-works.md): what each stage does in the run you just trained, and
  [how training works](training.md) in detail.
- [Designing an experiment](experiments.md): writing your own anchors and ablations, and adding
  a paper's method.
- [Contributing](contributing.md): adding a new discovery method, loss, predictor or evaluator.
