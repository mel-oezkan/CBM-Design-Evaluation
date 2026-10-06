# Training

This page describes exactly what the `stages.training` step does: what is trained, on which loss,
with which optimizer, and when it stops. The code is `stages/training.py`. For where training sits
in a run, see [How a run works](how-it-works.md).

By default nothing here touches images. The backbone is frozen, every image is already a cached
feature vector `x`, and training fits two small modules on top of those vectors: the concept layer
and the predictor head. Both are fitted on the `train` split, and the `val` split is used only for
early stopping.

## The model being trained

For a batch of features `x` of shape `(N, D)` and `K` concepts:

```text
x_n    = (x − μ) / σ
z      = (W · x_n + b − m) / s
c_hat  = sigmoid(z) if the targets are binary, else z
rep    = represent(c_hat, x)
logits = head(rep, x_n)
```

- `μ`, `σ`: per-feature mean and standard deviation of the training split.
- `z`: the concept logits, `(N, K)`. `m` and `s` are 0 and 1 unless the layer was calibrated (see
  below).
- `represent` comes from the generation stage, and `head` from the predictor stage. `logits` has
  shape `(N, C)` for `C` classes.

There are three groups of trainable parameters:

| Group | What | Trained unless |
|---|---|---|
| Concept parameters | `W`, `b` of the concept layer | the alignment froze them (`weights`, the default `frozen: true`) |
| Representation parameters | the concept embeddings of `generation: embeddings` (CEM); empty otherwise | |
| Head parameters | the predictor's layers | |

Two things happen before any optimization step:

- `μ` and `σ` are computed on the training features and stored in the layer.
- If the alignment provides no targets (`weights`), the concept outputs are standardized with
  training statistics (`m`, `s`), so that frozen text-embedding neurons output z-scores.

## Losses

**Task loss**: cross-entropy between `logits` and the class label.

**Concept loss**: compares `z` with the alignment's targets, averaged over concepts and images.
`concept_loss:` picks it:

| `concept_loss` | Loss | Used for |
|---|---|---|
| `auto` (default) | BCE with logits for binary targets, MSE for continuous ones | `human`, `dino` → BCE; `clip` → MSE |
| `bce`, `mse` | as named | |
| `cosine` | 1 − cosine between each image's (or segment's) concept vector and its target vector | SEG-MIL-CBM |

The `*_weighted` variants (Koh et al.) scale concept `k`'s BCE by `(1 − p_k) / p_k`, where `p_k` is
the share of training images where it is present, so that rare concepts count more.

**Head penalties**: `sparse` adds an L2 term and, after each optimizer step, soft-thresholds the
weights toward zero (the L1 part of the elastic net, done as a proximal step). `residual` adds
`residual_lam · ‖W_residual‖²`.

## The three schemes

The schemes differ in what the head is trained on, and in whether task gradients reach the concept
layer:

| Scheme | Steps | Head is trained on | Task loss reaches concepts |
|---|---|---|---|
| `independent` | 1. concept layer, 2. head | the **true** targets | no |
| `sequential` | 1. concept layer, 2. head (layer frozen) | the **predicted** concepts | no |
| `joint` | one step: both together | the predicted concepts | **yes** |

Step 1 always minimizes the concept loss, and step 2 the task loss. `joint` minimizes task loss +
`concept_weight` × concept loss in a single optimization. At test time the head always sees the
predicted concepts.

- **`independent`**: the head never sees the concept layer's mistakes during training, so it
  learns to use concepts the way they are defined. It needs an alignment with targets.
- **`sequential`**: the head adapts to the concept layer's actual outputs, mistakes included. The
  concept layer is frozen during step 2, so it stays purely concept-driven.
- **`joint`**: the task loss also shapes the concept layer. This usually improves accuracy, but
  the concept outputs can start to carry task information that isn't concept meaning ("leakage").
  The `leakage` evaluator measures this. `concept_weight` (default 1.0) sets the trade-off. Without
  targets, or with `concept_weight: 0`, only the task loss is used, and the concept neurons are no
  longer tied to their concepts unless the alignment froze them.

In `independent` and `sequential`, the representation parameters (CEM embeddings) are trained in
step 2 together with the head. Alignments without targets skip step 1 entirely.

## Optimization

Every step above runs the same loop (`optimize` in `stages/training.py`):

1. Each epoch shuffles the training set with a generator seeded from the run's seed, and takes one
   optimizer step per minibatch of `batch_size`.
2. After each epoch, it computes the loss on the whole validation split: the concept loss for step
   1, the task loss for the head (without the penalty), and the total loss for `joint`.
3. It stops after `patience` epochs without an improvement of more than 1e-5, or after `epochs`
   epochs (`concept_epochs` for step 1), and **restores the weights of the best epoch**.

Concept and representation parameters use the stage's `optimizer` (Adam by default, or SGD with
`momentum`; both with `weight_decay`) at its `lr`. The head uses its own optimizer if it declares
one: `sparse` uses plain SGD at its own `lr` (default 0.1), so that the soft-thresholding step is a
true proximal-gradient update; every other head uses the stage's optimizer and `lr`.

A head can split its parameters into phases that are fitted one after the other, each with its own
early stopping. `residual` (with the default `sequential: true`, as in post-hoc CBMs) first fits the
concept head, then fits the residual path on what the concept head gets wrong. All other heads
have a single phase.

On cached features, work that cannot change during a step is done once, not per minibatch: the
features are normalized once (instance bags excepted, to avoid a second copy of the bag), a frozen
concept layer's predictions are computed once for the head, and features are left out of the
head's minibatches when neither the concept layer nor the head reads them (`ConceptLayer.represent_uses_x`,
`PredictorHead.uses_features`). On CPU the trained weights are bit-for-bit those of the per-minibatch
computation; `tests/test_training.py` checks every scheme against it. On zero-padded instance bags
(segments), concept logits are computed on the real instances only and scattered back with zeros in
the padding, which every bag reader masks; this saves the padding share of the concept layer's work
and matches the padded computation up to float summation order.

The log (`train_log.json`, and the `train.*` fields of the results row) records the epochs and the
best validation loss of each step, e.g. `concept_epochs`, `concept_val_loss`,
`head_phase0_epochs`, `head_val_loss`, plus head statistics such as `head.nonzero_frac`.

### Options

These keys apply to every scheme:

| Key | Default | Meaning |
|---|---|---|
| `epochs` | 200 | maximum epochs per head phase, or for the whole `joint` fit |
| `concept_epochs` | = `epochs` | maximum epochs for step 1 |
| `batch_size` | 256 | minibatch size |
| `lr` | 0.001 | learning rate (and the head's, unless it sets its own) |
| `optimizer` | `adam` | `adam` or `sgd`, for the concept layer and every head without its own |
| `momentum` | 0.0 | SGD momentum (needs `optimizer: sgd`) |
| `weight_decay` | 0.0 | L2 weight decay of that optimizer |
| `patience` | 10 | epochs without improvement before stopping |
| `concept_loss` | `auto` | see [Losses](#losses) |
| `device` | `cpu` | `auto` uses a GPU when there is one |
| `finetune` | none | train the backbone too; see below |
| `concept_weight` | 1.0 | `joint` only: weight of the concept loss |

Change them in the config, never in the code: `{name: joint, concept_weight: 0.1, epochs: 100}`.

## Instance bags

With `instances:`, inputs are `(N, M, D)`: M segments or patches per image, plus a mask that marks
the real ones. The concept layer runs on each instance, and the `mil` head pools them into one
prediction per image. Concept losses are averaged over real instances. If the targets are per image
(`human`, `clip`, `dino`), they supervise the maximum over each image's instances, the standard
multiple-instance assumption. See the [SEG-MIL-CBM example](instance-bags.md).

## Fine-tuning the backbone

With `finetune: {...}`, the backbone is trained too, as in Koh et al. Images go through a
trainable copy of the encoder instead of the feature cache:

- `independent` / `sequential`: step 1 trains the encoder and the concept layer together on the
  concept loss (this needs targets). The training and validation images are then re-encoded with
  the fine-tuned encoder, and step 2 fits the head on those features as usual.
- `joint`: the encoder, concept layer and head are trained together on images.

The encoder gets its own optimizer (`optimizer: sgd` with `momentum`, or `adam`, both with
`weight_decay`), learning rate schedule (`lr_step`, `lr_gamma`), `batch_size`, `epochs` and
`patience`, and training images are augmented unless `augment: false`. The concept layer and head
keep the optimizers described above. Afterwards, every evaluator reads the fine-tuned encoder's
features, and the encoder is saved to `runs/<run>/encoder.pt`.

Only backbones that support it (`inception`, `toy`) can be fine-tuned. It can't be combined with
`instances:` or with evaluators that read patch features (`localization`, `keypoint_distance`,
`part_iou`). See `configs/anchors/cub_koh2020.yaml` for a full example.

## Changing the training

Every option above is a config key, so most changes need no code. For a new loss, subclass a scheme
and override one hook, `_task_loss(logits, batch)` or `_concept_loss(layer, batch)`, and register
it under a new name (see [Contributing](contributing.md)). A concept loss reads the features as
`batch.concept_logits(layer)`, which uses the precomputed normalization (and skips the padding of instance bags). A head that needs special treatment
declares it through `penalty()`, `proximal_step()`, `phases()` or its own `optimizer` / `lr`, so the
training schemes never need to know which head they train.
