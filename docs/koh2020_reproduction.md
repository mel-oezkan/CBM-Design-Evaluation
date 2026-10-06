# Reproduction: Koh et al. 2020

This page describes how to reproduce the CUB results of *Concept Bottleneck Models* (Koh et al.,
ICML 2020, [arXiv:2007.04612](https://arxiv.org/abs/2007.04612)) for the independent, sequential
and joint bottlenecks. The reproduction uses the regular pipeline: it is a set of anchors and
ablations built from registered components, with no separate training code.

!!! info "Status"
    The configs and the Modal runner are complete, and the data path is verified against the
    authors' processed data (same 112 concepts, identical labels and splits). The fine-tuned runs
    have not been completed yet, so there are no reproduced numbers on this page.

## Targets

CUB, mean ± 2 SD over 3 seeds, from the paper:

| Model | Task error (Table 1) | Concept error (Table 2) |
|---|---|---|
| Independent | 0.240 ± 0.012 | 0.034 ± 0.002 |
| Sequential | 0.243 ± 0.006 | 0.034 ± 0.002 |
| Joint (λ = 0.01) | 0.199 ± 0.006 | 0.031 ± 0.000 |

## Configs

| File | Purpose |
|---|---|
| `configs/anchors/cub_koh2020_independent.yaml` | Independent bottleneck, fine-tuned Inception-v3 |
| `configs/anchors/cub_koh2020_sequential.yaml` | Sequential bottleneck, fine-tuned Inception-v3 |
| `configs/anchors/cub_koh2020.yaml` | Joint bottleneck (λ = 0.01), fine-tuned Inception-v3 |
| `configs/ablations/cub_koh2020*.yaml` | Frozen backbone; removes one Koh-specific option at a time |

Each anchor's header comment lists what matches the paper and what does not.

## How the paper maps onto the pipeline

| Koh et al. | Config |
|---|---|
| CUB, 112 class-level concepts, authors' 80/20 train/val split | `dataset: {name: cub, majority_vote: koh, split_dir: data/CUB_processed/class_attr_data_10}` |
| Inception-v3 (ImageNet), fine-tuned end to end with SGD and augmentation | `backbone: inception` + `training.finetune: {optimizer: sgd, lr, momentum: 0.9, weight_decay, batch_size: 64}` |
| x → c: lr 0.01, weight decay 4e-5 | independent and sequential anchors |
| Joint: lr 0.001, weight decay 4e-4, loss divided by 1 + λ·112 | joint anchor: lr 0.00047 (= 0.001 / 2.12, since our loss is not divided) |
| Concepts = annotated attributes | `discovery: dataset`, `filtering: [{name: rules, max_words: 10, remove_class_names: false}]` (keeps all 112) |
| Concept supervision | `alignment: human` |
| Linear c → y | `predictor: dense` |
| Independent: f trained on true c, tested on σ(ĝ(x)) | `training: independent_weighted`, `generation: scores` |
| Sequential and joint: f on concept logits | `generation: logits` + `sequential_weighted` / `joint_weighted` |
| Concept BCE weighted by the negative/positive ratio | the `*_weighted` training variants |
| Joint λ = 0.01 on a sum over 112 concepts | `concept_weight: 1.12` (our concept loss averages) |
| Task error / concept error | `1 - shift.test.acc` / `concepts.test.concept_error` |
| Test-time intervention | `leakage.intervention.acc@f` (random single concepts; approximate) |

### Known differences

- **Model selection:** early stopping on the validation loss, versus up to 1000 epochs retrained
  on train+val and selected by training accuracy.
- **Optimizers:** the concept layer and head use Adam; the paper uses SGD throughout. The encoder
  uses SGD as in the paper. `stages.training.optimizer: sgd` (with `momentum`, `weight_decay`)
  expresses the paper's choice; the anchors don't set it yet, so their run ids and results stay as they are.
- **Inputs:** no auxiliary Inception head (the paper adds 0.4 × the auxiliary loss); ImageNet
  normalization instead of mean 0.5 / std 2; evaluation images are resized to 299 rather than
  center-cropped.
- **Interventions:** the `leakage` evaluator replaces random single concepts, not the paper's
  visibility-aware concept groups. With `generation: logits`, an intervened 0/1 value becomes a
  saturated logit (±13.8) rather than the 5th/95th-percentile logits the paper uses.

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

Locally, every anchor runs like any other config (a GPU is strongly recommended for fine-tuning):

```bash
uv run cbm-eval run configs/anchors/cub_koh2020.yaml --seeds 1 2 3
uv run cbm-eval sweep configs/ablations/cub_koh2020.yaml
uv run cbm-eval analyze --results results/cub_koh2020.jsonl --metric concepts.test.concept_error
```

`scripts/koh2020_modal.py` runs the same commands on [Modal](https://modal.com). It downloads and
unpacks CUB on the worker, runs one GPU container per (anchor, seed) in parallel, and mirrors
result rows to W&B. It expects the authors' split on the Modal volume `cbm-koh2020` under
`data/CUB_processed` and a Modal secret named `wandb`. Install the client with
`uv sync --extra modal` and authenticate once with `uv run modal setup`.

```bash
uv run modal run scripts/koh2020_modal.py::anchors      # fine-tuned anchors, all seeds
uv run modal run scripts/koh2020_modal.py::ablations    # frozen-backbone ablations
uv run modal run scripts/koh2020_modal.py::fetch        # merge results into results/cub_koh2020.jsonl
```

## Ablations

The ablations run with a frozen backbone (`overrides: {stages.training.finetune: null}`) and
remove one Koh-specific option at a time, so each one measures how much that option matters:

| Factor | Alternative |
|---|---|
| `dataset.majority_vote` | `mean`: the pipeline's default vote (93 concepts) |
| `dataset.split_dir` | `null`: the pipeline's own train/val split |
| `stages.generation` | `scores`: probabilities instead of logits into the head (sequential, joint) |
| `stages.training.name` | the unweighted training variant |

## Notes

- **Fine-tuning is required to approach the paper's numbers.** With a frozen ImageNet
  Inception-v3, even a linear probe on the features stays near 0.37 task error and about 0.10
  concept error. See `training.finetune` in the [component catalogue](components.md#training).
- **Imbalance weighting barely matters with a frozen backbone.** For a frozen linear concept layer
  trained with Adam, the per-concept loss scale largely cancels; the weighted-training ablation
  measures this.
- **Label-only dataset options re-extract features.** The feature cache is keyed on the full
  dataset config, so changing `majority_vote` or `split_dir` re-extracts identical image features
  once.
- **Constructing the CUB loader reads every image header**, which takes minutes on a network
  file system. Unpack the dataset to local disk (the Modal runner does this).
