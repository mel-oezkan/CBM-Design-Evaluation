# Reproduction: Koh et al. 2020

This page describes how to reproduce the CUB results of *Concept Bottleneck Models* (Koh et al.,
ICML 2020, [arXiv:2007.04612](https://arxiv.org/abs/2007.04612)) for the independent, sequential
and joint bottlenecks. The reproduction uses the regular pipeline: it is a set of anchors and
ablations built from registered components, with no separate training code.

There are two arms:

- **Frozen backbone** (`cub_koh2020_frozen*.yaml`): everything as in the authors' code except that
  Inception-v3 is not fine-tuned. Features are extracted once, so only that step needs a GPU.
- **Fine-tuned backbone** (`cub_koh2020*.yaml`): Inception-v3 trained end to end, as in the paper.
  It needs GPU-days and has not been run yet.

!!! info "Status"
    The data path is verified against the authors' processed data (same 112 concepts, identical
    labels and splits), and their Inception weights equal torchvision's `IMAGENET1K_V1`. Results
    of the frozen arm are below.

## Targets

CUB, mean ± 2 SD over 3 seeds, from the paper (fine-tuned backbone):

| Model | Task error (Table 1) | Concept error (Table 2) |
|---|---|---|
| Independent | 0.240 ± 0.012 | 0.034 ± 0.002 |
| Sequential | 0.243 ± 0.006 | 0.034 ± 0.002 |
| Joint (λ = 0.01) | 0.199 ± 0.006 | 0.031 ± 0.000 |

## Results: frozen backbone

<!-- results:frozen -->
*Pending: the frozen arm is running.*
<!-- /results:frozen -->

## Configs

| File | Purpose |
|---|---|
| `configs/anchors/cub_koh2020_frozen_independent.yaml` | Independent bottleneck, frozen Inception-v3, the authors' recipe otherwise |
| `configs/anchors/cub_koh2020_frozen_sequential.yaml` | Sequential bottleneck, same |
| `configs/anchors/cub_koh2020_frozen.yaml` | Joint bottleneck (λ = 0.01), same |
| `configs/ablations/cub_koh2020_frozen*.yaml` | Undo one detail of the authors' code at a time |
| `configs/anchors/cub_koh2020{,_independent,_sequential}.yaml` | Fine-tuned arm |

Each anchor's header comment lists what matches the authors' code and what does not.

## How the authors' code maps onto the pipeline

The reference is [yewsiang/ConceptBottleneck](https://github.com/yewsiang/ConceptBottleneck): the
commands in `CUB/README.md`, plus `CUB/train.py`, `CUB/dataset.py` and `CUB/template_model.py`.

| Authors' code | Config (frozen arm) |
|---|---|
| `CUB_processed/class_attr_data_10`: their train/val split, 112 class-level concepts | `dataset: {name: cub, majority_vote: koh, split_dir: data/CUB_processed/class_attr_data_10}` |
| `inception_v3_google-1a9a5a14.pth`, `transform_input=True` | `backbone: inception` (identical weights) |
| `Normalize(mean=0.5, std=2)` before `transform_input` | `backbone: {mean: [0.5, 0.5, 0.5], std: [2, 2, 2]}` |
| Eval: `CenterCrop(299)` without resizing | `backbone: {eval_resize: false}` |
| Train: `ColorJitter`, `RandomResizedCrop(299)`, `RandomHorizontalFlip` | `training.augment: {views: 64}`: cached augmented views (the backbone is frozen) |
| `F.dropout(p=0.5)` on the pooled features before the concept FCs | `generation: {dropout: 0.5}` |
| One `FC(2048, 1)` per concept on the raw pooled features, truncated-normal init (std 0.1) | `generation: {standardize: false, init_std: 0.1}` |
| `BCEWithLogitsLoss(weight=n/n_pos − 1)` per concept, summed and divided by K (x → c) | `training.name: *_weighted` |
| x → c: SGD lr 0.01, momentum 0.9, weight decay 4e-5, batch 64, `drop_last` | `optimizer: sgd, momentum: 0.9, concept_lr: 0.01, concept_weight_decay: 0.00004, batch_size: 64, drop_last: true` |
| x → c with `-ckpt 1`: train + val, best running training accuracy, ≤ 1000 epochs, stop after 100 without improvement | `concept_train_on_val: true, concept_select: train_acc, concept_epochs: 1000, patience: 100` |
| Independent c → y: trained on true concepts, tested on `sigmoid`; SGD lr 0.001, wd 5e-5, ≤ 500 epochs, best val accuracy | `generation: scores`, `lr: 0.001, weight_decay: 0.00005, epochs: 500, select: val_acc` |
| Sequential c → y: on predicted logits under the train-time transform (`ExtractConcepts`); SGD lr 0.001, wd 4e-5, ≤ 1000 epochs | `generation: logits`, `head_on_augmented: true`, `weight_decay: 0.00004, epochs: 1000, select: val_acc` |
| Joint: one SGD (lr 0.001, wd 4e-4), loss `(CE + λ Σ_k w_k BCE_k) / (1 + λK)`, train + val, best training accuracy | `joint_weighted, concept_weight: 1.12, normalize_loss: true, train_on_val: true, select: train_acc` |
| `-use_aux`: + 0.4 × every loss term on `AuxLogits`; in joint, its concepts also go through f | `training.aux: {weight: 0.4, init_std: 0.001}` (fine-tuned anchors only; it refuses `augment:`) |
| Linear c → y (`MLP` with `expand_dim=0`), default init | `predictor: dense` |
| Task error / concept error (`inference.py`) | `1 - shift.test.acc` / `concepts.test.concept_error` |

`λ = 0.01` on a sum over 112 concepts is `concept_weight: 1.12` on our concept-averaged loss. Their
`StepLR(step=1000)` never decays within these runs, so there is no schedule to reproduce.

### Remaining differences: frozen arm

- **Frozen backbone** (by design): the encoder, including its BatchNorm statistics, keeps its
  ImageNet weights. The authors fine-tune it with BatchNorm in train mode.
- **Augmentation:** 64 cached views per image, shared by every run, instead of a fresh random
  transform every epoch. The `augment.views: 8` ablation measures how much the pool size matters.
- **Auxiliary head:** not used. `training.aux` reads the auxiliary tap of the clean images, so it
  refuses `augment:`. With a frozen backbone it cannot change the x → c model anyway. In the joint
  model, though, the authors also pass the auxiliary concept logits through the shared c → y layer
  (with 0.4 × that loss), which trains the layer; the frozen joint anchor misses that term.
- **Interventions:** the `leakage` evaluator replaces random single concepts, not the paper's
  visibility-aware concept groups. With `generation: logits`, an intervened 0/1 value becomes a
  saturated logit (±13.8) rather than the 5th/95th-percentile logits the paper uses.
- **Logit clamp** ([#47](https://github.com/mel-oezkan/CBM-Design-Evaluation/issues/47)): the
  sequential and joint anchors use `generation: logits`, which clamps the concept logits at ±13.8
  before the head. Beyond ±13.8 the task loss has no gradient. The authors pass the raw logits;
  `generation: {raw: true}` does that.

### Remaining differences: fine-tuned arm

The fine-tuned sequential and joint anchors pass the raw concept logits to the head
(`generation: {name: logits, raw: true}`, [#47](https://github.com/mel-oezkan/CBM-Design-Evaluation/issues/47)).
They also use the frozen arm's SGD settings for the concept layer and head (`optimizer: sgd`,
`concept_lr`, `normalize_loss`, as in the table above). Runs made before that used the clamped
logits and Adam, and keep their old run ids. Otherwise the fine-tuned anchors do not use the frozen
arm's options yet
([issue #11](https://github.com/mel-oezkan/CBM-Design-Evaluation/issues/11)):

- **Model selection:** early stopping on the validation loss, versus up to 1000 epochs retrained
  on train+val and selected by training accuracy. `train_on_val` does not support `finetune:` yet.
- **Inputs:** ImageNet normalization instead of mean 0.5 / std 2; evaluation images are resized
  to 299 rather than center-cropped. The auxiliary concept head reads standardized features; the
  paper's reads them raw.

## Data

You need two downloads:

1. The official `CUB_200_2011` directory ([Caltech data record](https://data.caltech.edu/records/65de6-vp158)),
   placed at `data/CUB_200_2011`.
2. The authors' processed split, `CUB_processed/class_attr_data_10`, from their
   [Codalab worksheet](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2),
   placed at `data/CUB_processed/class_attr_data_10`.

With `majority_vote: koh` the CUB loader votes attributes as the authors' code does: training
split only, "not visible" negatives ignored, ties counted as present. The default
(`majority_vote: mean`) yields 93 concepts with slightly different labels; see [Data](data.md).

## Running

Every anchor runs like any other config. The first run extracts the clean features and the 64
augmented views, about 400k images through Inception-v3, which needs a GPU. Later runs read them
from the cache and train on a CPU.

```bash
uv run cbm-eval run configs/anchors/cub_koh2020_frozen.yaml --seeds 1 2 3
uv run cbm-eval sweep configs/ablations/cub_koh2020_frozen.yaml
uv run cbm-eval analyze --results results/cub_koh2020_frozen.jsonl --metric concepts.test.concept_error
```

`scripts/koh2020_modal.py` runs the same configs on [Modal](https://modal.com). It downloads and
unpacks CUB on the worker and expects the authors' split on the Modal volume `cbm-koh2020` under
`data/CUB_processed`, plus a Modal secret named `wandb` (every run streams its epoch history and metrics to W&B). Install
the client with `uv sync --extra modal` and authenticate once with `uv run modal setup`.

```bash
uv run modal run scripts/koh2020_modal.py::frozen             # frozen arm: anchors and ablations
uv run modal run scripts/koh2020_modal.py::fetch --arm frozen # -> results/cub_koh2020_frozen.jsonl
uv run modal run scripts/koh2020_modal.py::anchors            # fine-tuned arm, all seeds
```

`frozen` first extracts every distinct feature set on a GPU container, then trains each
`(config, seed)` on its own CPU container, skipping runs that already have results.

## Ablations

Each ablation undoes one detail of the authors' code; the alternative is the pipeline's default.

| Factor | Alternative |
|---|---|
| `dataset.majority_vote` | `mean`: the pipeline's vote (93 concepts) |
| `dataset.split_dir` | `null`: the pipeline's own train/val split |
| `backbone` | ImageNet input normalization |
| `backbone.eval_resize` | `true`: eval images resized to 299 × 299 instead of center-cropped |
| `stages.training.augment` | `null`: clean features only |
| `stages.training.augment.views` | `8`: a smaller pool of augmented views |
| `stages.generation.dropout` | `0.0` |
| `stages.generation.standardize` | `true`: z-scored features |
| `stages.generation.init_std` | `null`: PyTorch's default init |
| `stages.training.(concept_)train_on_val` | `false`: train split only, still selected by training accuracy |
| `stages.generation` | `scores`: probabilities instead of logits into the head (sequential, joint) |
| `stages.training.head_on_augmented` | `false`: sequential head on clean predictions |
| `stages.training.normalize_loss` | `false`: joint loss not divided by `1 + λK` |
| `stages.training.name` | the unweighted training variant |

## Notes

- **Label-only dataset options re-extract features.** The feature cache is keyed on the full
  dataset config, so changing `majority_vote` re-extracts identical image features (and views) once.
- **Constructing the CUB loader reads every image header**, which takes minutes on a network
  file system. Unpack the dataset to local disk (the Modal runner does this).
