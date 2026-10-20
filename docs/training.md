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
z      = (W · dropout(x_n) + b − m) / s
c_hat  = sigmoid(z) if the targets are binary, else z
rep    = represent(c_hat, x)
logits = head(rep, x_n)
```

- `μ`, `σ`: per-feature mean and standard deviation of the training split, or 0 and 1 with
  `generation: {standardize: false}` (raw features, as Koh et al.).
- `dropout`: the identity unless `generation: {dropout: p}` (active only while training).
  `generation: {init_std: s}` draws `W` from a normal truncated at ±2s instead of PyTorch's default.
- `z`: the concept logits, `(N, K)`. `m` and `s` are 0 and 1 unless the layer was calibrated (see
  below).
- `represent` comes from the generation stage, and `head` from the predictor stage. `logits` has
  shape `(N, C)` for `C` classes.
- One generation reads `z` instead of `c_hat`: `logits` with `raw: true` gives the head `z` itself.
  The default `logits` gives it `logit(c_hat)` clamped at ±13.8, so a concept logit past ±13.8 gets
  no task-loss gradient. The code composes these steps in `ConceptLayer.from_logits`.

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

What "best" means is `select` (`concept_select` for step 1):

| `select` | The best epoch has |
|---|---|
| `val_loss` (default) | the lowest validation loss, as above |
| `val_acc` | the highest validation accuracy: task accuracy for heads and `joint`, concept accuracy (`sigmoid ≥ 0.5` against the binary targets) for step 1 |
| `train_acc` | the highest running training accuracy, averaged over the epoch's minibatches as they are trained (with dropout and augmentation) |

`train_on_val: true` (`concept_train_on_val` for step 1) trains on the training and validation
splits together, which leaves only `train_acc` to select on. This is how Koh et al. retrain their
final models. Accuracies improve only when strictly higher; `drop_last: true` skips each epoch's
last incomplete minibatch.

Concept and representation parameters use the stage's `optimizer` (Adam by default, or SGD with
`momentum`; both with `weight_decay`) at its `lr`; step 1 of `independent` / `sequential` can use
its own `concept_lr` and `concept_weight_decay`. The head uses its own optimizer if it declares
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
`head_phase0_epochs`, `head_val_loss`, plus head statistics such as `head.nonzero_frac`. It also
records each step's best epoch (`concept_best_epoch`, ...) and, when selecting on accuracy, the
selected accuracy (`concept_train_acc`, `head_val_acc`, `joint_train_acc`, ...).

### Options

These keys apply to every scheme:

| Key | Default | Meaning |
|---|---|---|
| `epochs` | 200 | maximum epochs per head phase, or for the whole `joint` fit |
| `concept_epochs` | = `epochs` | maximum epochs for step 1 |
| `batch_size` | 256 | minibatch size |
| `drop_last` | false | skip each epoch's last incomplete minibatch |
| `lr` | 0.001 | learning rate (and the head's, unless it sets its own) |
| `optimizer` | `adam` | `adam` or `sgd`, for the concept layer and every head without its own |
| `momentum` | 0.0 | SGD momentum (needs `optimizer: sgd`) |
| `weight_decay` | 0.0 | L2 weight decay of that optimizer |
| `concept_lr`, `concept_weight_decay` | = `lr`, `weight_decay` | step 1's learning rate and weight decay |
| `patience` | 10 | epochs without improvement before stopping |
| `select`, `concept_select` | `val_loss` | how the best epoch is chosen; see [Optimization](#optimization) |
| `train_on_val`, `concept_train_on_val` | false | also train on the validation split (needs `train_acc`) |
| `concept_loss` | `auto` | see [Losses](#losses) |
| `device` | `cpu` | `auto` uses a GPU when there is one |
| `finetune` | none | train the backbone too; see below |
| `augment` | none | train on augmented views of the images, backbone frozen; see below |
| `aux` | none | Koh et al.'s auxiliary concept head; see below |
| `concept_weight` | 1.0 | `joint` only: weight of the concept loss |
| `normalize_loss` | false | `joint` only: divide the loss by `1 + concept_weight` (Koh et al.'s `-normalize_loss`) |
| `head_on_augmented` | false | `sequential` only: fit the head on predictions for augmented views (needs `augment`) |

Change them in the config, never in the code: `{name: joint, concept_weight: 0.1, epochs: 100}`.

## Instance bags

With `instances:`, inputs are `(N, M, D)`: M segments or patches per image, plus a mask that marks
the real ones. The concept layer runs on each instance, and the `mil` head pools them into one
prediction per image. Concept losses are averaged over real instances. If the targets are per image
(`human`, `clip`, `dino`), they supervise the maximum over each image's instances, the standard
multiple-instance assumption. See the [SEG-MIL-CBM example](instance-bags.md).

## Augmentation with a frozen backbone

With `augment: {views: V}`, the backbone stays frozen, but step 1 and `joint` train on augmented
images, as fine-tuning does. `V` copies of every training image under the backbone's
`train_transform` (for `inception`, Koh et al.'s color jitter, random resized crop and flip) are
encoded once and cached as `(N, V, D)` float16 features (`Context.augmented`). Each optimizer step
reads one randomly drawn view per image. Validation, the head of `independent` / `sequential` and
evaluation use the clean features.

View `v` of image `i` is seeded by `(seed, v, i)` alone (`augment.seed`, default 0), so the views do
not depend on the run's seed, the worker count or how many views are drawn, and every run with the
same dataset and backbone shares them. A finite pool of views stands in for fresh augmentation
every epoch; the `augment.views` ablation measures how much the pool size matters. `num_workers` and
`batch_size` only batch the encoding. Augmentation can't be combined with `finetune:` (which
augments on its own) or with `instances:`.

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
`part_iou`). See `configs/anchors/cub_koh2020.yaml` for a full example. Fine-tuning selects on the
validation split, so `train_on_val` / `concept_train_on_val` are not available with it.

## Auxiliary concept head

With `aux: {weight: 0.4, init_std: 0.001}`, training adds Koh et al.'s auxiliary Inception head
(their `-use_aux`). A second concept layer reads the backbone's auxiliary tap: for `inception`, the
pooled 768-d output of the `AuxLogits` convolutions on `Mixed_6e`. On every training step, the loss
on its concept logits is added, multiplied by `weight`. It is the same loss as the main branch's:

- `independent` / `sequential`: the concept loss of step 1.
- `joint`: the concept loss, plus the task loss of the auxiliary concepts passed through the same
  head. So the head also learns from them.

Validation and model selection use the main branch only. The auxiliary layer is discarded after
training, so evaluation never sees it. Its weights start from a normal distribution truncated at
±2 `init_std` (`null`: PyTorch's default init).

What it changes depends on the backbone:

| | frozen backbone | `finetune:` |
|---|---|---|
| `independent` / `sequential` | nothing, so the stage refuses it | trains the encoder up to `Mixed_6e` |
| `joint` | trains the head on the auxiliary concepts | both |

It needs a backbone with an auxiliary tap (`inception`, or `toy` with `aux_dim`) and concept
targets. It can't be combined with `instances:`, nor with `augment:`: the cached auxiliary features
come from the clean images, not from the augmented views. In `joint` training, it needs a generation
and head that read only concepts (not `embeddings` or `residual`). With `train_on_val`, the
auxiliary head trains on train + val as well, and `normalize_loss` divides its terms too.

## Epoch history and live tracking

Every training step writes one row per epoch to `history.jsonl` in the run folder. The training log
keeps only the kept epoch; the history shows how losses and accuracies moved before and after it.
One row of a fine-tuned joint run (illustrative values):

```json
{"phase": "joint", "epoch": 18, "train_loss": 0.41, "train_acc": 0.93, "val_loss": 1.02, "val_acc": 0.71,
 "best": true, "lr": 0.001, "encoder_lr": 0.00047, "seconds": 21.3}
```

| Field | Meaning |
|---|---|
| `phase` | `concept` (x → c), `head_phase0` / `head_phase1` (c → y), or `joint` |
| `epoch` | epoch number within the phase, from 1 |
| `train_loss` | mean loss over the epoch's minibatches: the loss the optimizer saw, with penalty and auxiliary terms |
| `train_acc` | running accuracy over the epoch's minibatches: task accuracy, or concept accuracy in `concept` |
| `val_loss`, `val_acc` | on the validation split; missing when the phase trains on it (`train_on_val`) |
| `best` | true when the epoch improved the `select` score; the last `best` row is the kept epoch |
| `lr`, `encoder_lr` | learning rate of the first optimizer and, with `finetune:`, of the encoder |
| `seconds` | wall time of the epoch, validation included |

Concept accuracy is only recorded for binary image-level targets. Recording the history draws no
random numbers and changes no gradient: weights, metrics and run ids are the same with or without
it, and with or without W&B.

To follow runs while they train, stream the same rows to Weights & Biases (`uv sync --extra tracking`):

```bash
uv run cbm-eval run configs/anchors/cub_koh2020.yaml --seeds 1 --wandb cbm-koh2020-modular
uv run cbm-eval sweep configs/ablations/cub_koh2020_frozen.yaml --wandb cbm-koh2020-modular --wandb-group frozen
```

Each `(config, seed)` becomes one W&B run named `<name>-<run_id>-s<seed>`, grouped by the config's
name unless `--wandb-group` is set, with the run id, seed and factors as its config. Each phase has
its own x axis (`joint/epoch`, `concept/epoch`, ...). When the run ends, its metrics and training log
become the W&B summary; a run that fails is marked failed. `--wandb-mode offline` writes locally for
a later `wandb sync`. The W&B options are command-line flags, not config keys, so they don't change
the run id. In Python, pass `tracker=WandbTracker("project")` to `run` or `sweep`.

## Changing the training

Every option above is a config key, so most changes need no code. For a new loss, subclass a scheme
and override one hook, `_task_loss(logits, batch)` or `_concept_loss(layer, batch, logits=None)`, and
register it under a new name (see [Contributing](contributing.md)). A concept loss uses `logits` when
the caller passes them (so that, under dropout, it sees the same mask as the rest of the step) and
otherwise reads `batch.concept_logits(layer)`, which uses the precomputed normalization (and skips
the padding of instance bags). A head that needs special treatment
declares it through `penalty()`, `proximal_step()`, `phases()` or its own `optimizer` / `lr`, so the
training schemes never need to know which head they train.
