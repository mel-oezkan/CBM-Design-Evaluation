# How a run works

This page follows one run from config to results row, using the offline synthetic anchor, so every
number below is one you can reproduce with
`uv run cbm-eval run configs/anchors/synthetic.yaml --seeds 0`. The anchor builds a
Label-free-CBM-style model:

```yaml title="configs/anchors/synthetic.yaml (abridged)"
dataset: {name: synthetic}       # 4 classes + a spurious attribute
backbone: {name: toy}            # frozen; 128-d features
stages:
  discovery:                     # 24 concepts, 4 distractors, 1 class name
    name: static
    names: [concept_00, ..., distractor_03, class_0]
  filtering:
    - {name: rules}
    - {name: clip, dedupe_sim: 0.9}
    - {name: select, k: 20, method: submodular}
  alignment: {name: clip}
  generation: {name: scores}
  predictor: {name: sparse, lam: 0.001}
  training: {name: sequential, epochs: 100}
evaluation: [{name: shift}, {name: leakage}, {name: localization}]
```

`cbm-eval run` calls `pipeline.run()`, which goes through the steps below in order. Each step is a
public method of `PipelineBuilder`, so you can also run them one at a time in a notebook.

## 1. Build every component

`PipelineBuilder(cfg)` looks up each `{name: ...}` in its registry and calls the class with the
remaining keys as constructor arguments: `{name: sparse, lam: 0.001}` becomes `Sparse(lam=0.001)`.
Nothing is loaded yet. A misspelled key or an incompatible combination (for example instance bags
with a head that can't pool them) fails here, before any data is touched.

## 2. Encode every image once

`builder.context()` creates the dataset, the backbone and a `Context`, the shared cache that every
stage reads from. The first time a split is needed, the frozen backbone encodes every image of it
into one feature vector, and the result is saved under `.cache/features/`:

| Split | Images | Features |
|---|---|---|
| train | 1200 | `(1200, 128)` |
| val | 400 | `(400, 128)` |
| test | 800 | `(800, 128)` |

From here on, **no stage looks at pixels.** Training a CBM is training two small layers on top of
these cached tensors, which is why a run takes seconds and a sweep of hundreds of runs is cheap.
A sweep also keeps the dataset, backbone and loaded features in memory across consecutive configs
that share them, with the same results as running each config on its own.
(The exception is `training.finetune`, which trains the backbone end to end; see
[Training](training.md#fine-tuning-the-backbone).)

The synthetic data is built so that the background agrees with the label for 90% of training and
validation images, but is random on test. A model that leans on the background loses accuracy on
test, mostly on the groups where background and label disagree.

## 3. Find concepts: discovery and filtering

`builder.concepts(ctx)` produces the list of concept names the bottleneck will have. Discovery
proposes candidates, and each filter in the chain removes some. The run records how many survive
each step (`concept_trace.json`):

| Step | Concepts | What happened |
|---|---|---|
| `discovery: static` | 29 | the names from the config |
| `filter: rules` | 28 | `class_0` names a class, so it would leak the label; removed |
| `filter: clip` | 28 | no two names are near-duplicates in the teacher's text space |
| `filter: select` | 20 | submodular selection keeps 20 that cover the rest and separate the classes |

With `discovery: llm` the candidates would come from Claude, prompted per class. With
`discovery: dataset` they are the dataset's annotated attributes. With `sae` they are directions
learned from the features, not names. Up to this point, nothing has been trained, and a concept is
just a string (or a vector).

## 4. Give each concept a meaning: alignment

`alignment.fit(concepts, ctx)` decides what each of the 20 concept neurons *should* output. Most
alignments do this by producing a target per image and concept:

- `clip` (this anchor): the cosine similarity between each image and each concept's text embedding
  under the teacher model, standardized per concept on the training split. That gives continuous
  targets of shape `(N, 20)`, the pseudo-labels of Label-free CBM.
- `human`: the dataset's 0/1 annotations (concepts without annotations are dropped).
- `dino`: 0/1 labels from an open-vocabulary detector, thresholded.
- `weights`: no targets at all. Each neuron's weight vector *is* the concept's text embedding,
  frozen. The neuron then outputs that similarity directly, without being trained.

The result is an `AlignedConcepts`: the concepts, the target type (`binary`, `continuous` or none)
and a function that produces the targets of any split. The target type matters later: it picks the
concept loss (BCE or MSE) and whether a sigmoid is applied to concept outputs.

## 5. Build the model: generation and predictor

The trainable model has two parts, and both act on the cached features `x`:

```text
x       (N, 128)   cached features
  │  concept layer: linear, 128 → 20
  ▼
c_hat   (N, 20)    concept predictions
  │  represent: set by the generation stage
  ▼
rep     (N, 20)    what the head sees
  │  head: set by the predictor stage
  ▼
logits  (N, 4)     class scores
```

- The **concept layer** is one linear layer from features to concepts: `c_hat = W·x_n + b`, where
  `x_n` is `x` standardized with training-set statistics. A sigmoid follows for binary targets. With
  `clip` targets it stays linear, so `c_hat` is directly comparable to the standardized
  similarities.
- The **generation** stage decides what the head gets from `c_hat`. With `scores` it gets `c_hat`
  itself; `logits` passes the logit of a probability, `boc` a hard 0/1 per concept, and
  `embeddings` a learned 16-d vector per concept, mixed by the concept's probability (CEM).
- The **predictor** is the head. `sparse` (this anchor) is a linear layer with an elastic-net
  penalty that can set weights to exactly zero, so each class is explained by fewer concepts;
  `lam` sets how strongly. At `lam: 0.001` on this easy data, all 20 concepts stay in use.
  `dense` has no penalty, `residual` adds a second linear path from the raw features that
  bypasses the concepts, and `mil` pools over instance bags.

Because `c_hat` is always in the same space as the targets, it can be overwritten with
ground truth at test time. That is what an intervention is, and it is the reason the bottleneck is
interpretable.

## 6. Train

`training.fit(layer, head, aligned, ctx)` fits the two parts. This anchor uses `sequential`:

1. Fit the concept layer alone so that `c_hat` matches the CLIP targets (MSE), with Adam and early
   stopping on the validation concept loss.
2. Freeze it. Fit the sparse head on the *predicted* concepts with cross-entropy, using proximal
   SGD, until the validation task loss stops improving.

`independent` differs in step 2: the head is fitted on the *true* targets instead of the
predictions. `joint` fits both parts at once on the task loss plus a weighted concept loss.
[Training](training.md) gives the exact losses, optimizers, defaults and stopping rules for every
scheme.

The run logs what happened (`train_log.json`):

```json
{"concept_epochs": 100, "concept_val_loss": 0.0123,
 "head_phase0_epochs": 100, "head_val_loss": 0.3637,
 "head.nonzero_frac": 1.0, "head.nonzero_per_class": 20.0}
```

## 7. Evaluate

The trained model is wrapped as a `TrainedCBM`, which computes the image-level concept predictions
of every split once and caches them. Each evaluator then reads predictions from it; none of them
retrains the CBM:

- `shift`: accuracy per split and per (class, background) group; `wga` is the accuracy of the worst
  group.
- `leakage`: replaces a growing fraction of predicted concepts with ground truth and measures the
  accuracy gain (`intervention.gain`), and trains small probes, e.g. to decode the background from
  the concepts (`probe.spurious_bacc`).
- `localization`: applies the concept layer to each patch and asks whether a concept peaks where
  that concept actually is (`pointing_game`).

Each evaluator returns its numbers without a prefix, and the pipeline adds the evaluator's name:
`shift.test.wga`, `leakage.intervention.gain`, ... (see [Metrics](metrics.md)).

## 8. Record

The run writes two things:

- a folder `runs/synthetic/<run_id>-s0/` with the config, the concept list, the weights, the
  cached concept predictions, the concept trace and the training log, so that `cbm-eval evaluate`
  can score it again later without retraining;
- one row in `results/synthetic.jsonl` with the `run_id` (a hash of the config without the seed),
  the seed, the git commit, a `factor` entry naming the variant of every stage, and every metric.

The `factor` entries are what make analysis possible. A sweep produces many rows that differ in
their factors, and `cbm-eval analyze` fits the metric against them to estimate the effect of each
choice (see [Designing an experiment](experiments.md)).

## How the stages constrain each other

Most combinations of variants are valid, but a few pairs depend on each other. The pipeline
checks them and fails with a message naming the config key to change:

| If you choose | Then |
|---|---|
| an alignment without targets (`weights`) | `independent` training is impossible (there is nothing to train the head on), and the leakage evaluator skips interventions |
| `human` alignment | only concepts that match the dataset's annotated names are kept, so pair it with `discovery: dataset`; if none match, the run fails |
| `clip` alignment, `clip`/`select` filters with a non-CLIP backbone | set a text-capable `teacher:` (e.g. CLIP) to compute the similarities |
| `instances:` (bags of segments or patches) | the predictor must pool over them (`mil`); see the [SEG-MIL-CBM example](instance-bags.md) |
| `training.finetune` | the backbone must be trainable (`inception`, `toy`), and patch-based evaluators and `instances:` are unavailable |
| `*_weighted` training | binary image-level targets (`human`, no bags) |
