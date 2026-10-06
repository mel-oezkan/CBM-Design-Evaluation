# Pipeline map

This page maps the code: which module does what and which objects the stages pass to each other.
For a step-by-step walk through one run, start with [How a run works](how-it-works.md).

--8<-- "CLAUDE.md:architecture"

## The pipeline map

Each box links to its entry in the [component catalogue](components.md), which lists the interface
and every registered variant with its kwargs. Stage numbers follow execution order: alignment runs
before generation because the concept layer is built from the aligned concepts.

<div class="cbm-map" markdown>
<div class="cbm-row cbm-inputs" markdown>
[**Experiment configs**<span>Anchors and ablations</span>](configs.md){.cbm-node}
[**Datasets**<span>Waterbirds, CUB, MetaShift, synthetic</span>](components.md#datasets){.cbm-node}
[**Backbones**<span>CLIP, DINOv2, ResNet, Inception, toy</span>](components.md#backbones){.cbm-node}
[**Instances**<span>Optional: image → bag of segments</span>](components.md#instance-sources){.cbm-node .cbm-optional}
</div>
<div class="cbm-arrow"></div>
<div class="cbm-row" markdown>
[**Pipeline builder**<span>Looks up each `name` in its registry · builds the shared `Context`</span>](api/core.md){.cbm-node .cbm-center}
</div>
<div class="cbm-arrow"></div>
<div class="cbm-band" markdown>
<div class="cbm-band-head"><b>Stage modules</b><span>one abstract base class each · <code>stages/base.py</code></span></div>
<div class="cbm-row cbm-stages" markdown>
[**Discovery**<span>LLM, VLM, ConceptNet, SAE, dataset, static</span><span class="cbm-io">ctx → ConceptSet</span>](components.md#discovery){.cbm-node .cbm-stage}
[**Filtering**<span>rules, CLIP, DINO, select (chained)</span><span class="cbm-io">ConceptSet → ConceptSet</span>](components.md#filtering){.cbm-node .cbm-stage}
[**Alignment**<span>CLIP, weights, DINO, human, segment</span><span class="cbm-io">ConceptSet → AlignedConcepts</span>](components.md#alignment){.cbm-node .cbm-stage}
[**Generation**<span>scores, logits, embeddings (CEM), BoC</span><span class="cbm-io">AlignedConcepts → ConceptLayer</span>](components.md#generation){.cbm-node .cbm-stage}
[**Predictor**<span>sparse, dense, residual, MIL</span><span class="cbm-io">dims → PredictorHead</span>](components.md#predictor){.cbm-node .cbm-stage}
[**Training**<span>independent, sequential, joint (+ weighted)</span><span class="cbm-io">layer + head → log</span>](components.md#training){.cbm-node .cbm-stage}
</div>
</div>
<div class="cbm-arrow"></div>
<div class="cbm-row" markdown>
[**Trained CBM**<span>Weights plus cached concept scores · `runs/<run_id>-s<seed>/`</span>](api/core.md){.cbm-node .cbm-center}
</div>
<div class="cbm-arrow"></div>
<div class="cbm-band" markdown>
<div class="cbm-band-head"><b>Evaluation suite</b><span>metrics prefixed by evaluator name</span></div>
<div class="cbm-row cbm-evals" markdown>
[**Shift**<span>WGA, effective robustness</span>](components.md#evaluation){.cbm-node .cbm-eval}
[**Leakage**<span>Interventions, probes</span>](components.md#evaluation){.cbm-node .cbm-eval}
[**Concepts**<span>Concept error, F1</span>](components.md#evaluation){.cbm-node .cbm-eval}
[**Localization**<span>Pointing game, part IoU, keypoints</span>](components.md#evaluation){.cbm-node .cbm-eval}
[**Faithfulness**<span>Deletion, insertion</span>](components.md#evaluation){.cbm-node .cbm-eval}
</div>
</div>
<div class="cbm-arrow"></div>
<div class="cbm-row cbm-tail" markdown>
[**Results store**<span>One JSONL row per run and seed</span>](api/evaluation.md){.cbm-node}
<div class="cbm-h-arrow">→</div>
[**Analysis**<span>Effects models, Pareto frontiers</span>](api/evaluation.md){.cbm-node}
</div>
<p class="cbm-map-note" markdown>Dashed borders mark optional parts. Without `instances:`, each image is one feature vector of shape (N, D). With it, every stage runs per instance on (N, M, D) plus a validity mask; see the [SEG-MIL-CBM example](instance-bags.md).</p>
</div>

## What each stage hands to the next

The map shows which parts exist. This view adds the typed objects passed between the stages (arrow labels),
the config key or module behind each step (left), and which cached `Context` accessor each stage
reads (dashed). The steps are `PipelineBuilder.concepts()`, `.train()` and `.evaluate()` in
`pipeline.py`. Discovery and filtering only handle concept names (plus optional vectors); nothing is
trainable until alignment decides how each concept neuron gets its meaning.

<div class="cbm-figure">
--8<-- "docs/assets/architecture.svg"
</div>

[Open the data-flow diagram full size](assets/architecture.svg)

## Inside the trained CBM

The stages split the model as well as the pipeline. `ConceptLayer` in `stages/base.py` holds the
shared part (normalize, linear, activate, intervene); each generation variant only overrides
`represent()`. The head also receives the normalized features, so `residual` can bypass the
bottleneck and `mil` can compute attention over instances. Concept predictions stay in target
space, which is what lets interventions and the localization evaluators reuse the same layer on
patch features.

<div class="cbm-figure">
--8<-- "docs/assets/forward-pass.svg"
</div>

[Open the forward-pass diagram full size](assets/forward-pass.svg)

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
| Alignment | `clip`, `weights`, `dino`, `human`, `segment_clip` (instance bags) | `fit(concepts, ctx) -> AlignedConcepts` |
| Predictor | `sparse`, `dense`, `residual`, `mil` (instance bags) | `build(rep_dim, n_classes, feat_dim) -> PredictorHead` |
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
