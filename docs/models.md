# Planned models

These are the published CBMs the codebase should be able to express. Each one is added as an anchor
under `configs/anchors/`, built from registered components, plus ablations that remove one of its
choices at a time. It is not a separate pipeline (see [Contributing](contributing.md)). For each
model the tables below go through the paper's steps, name the stage that owns each step, and record
whether a variant for it exists yet. The [SEG-MIL-CBM example](instance-bags.md) shows how a
finished entry turns into code.

To propose a model, add a row to the overview and a section like the ones below. "Yes" means that
`uv run cbm-eval list` already has a variant for the step. "Partly" means a variant exists but
needs an opt-in kwarg to match the paper. "No" means a new registered variant is needed. "Design"
means the step doesn't fit the current interfaces, so the design has to be discussed in an issue
before anyone builds it.

## Overview

| Model | Venue | Concepts come from | Spatial? | Status |
|---|---|---|---|---|
| [VLG-CBM](#vlg-cbm) | NeurIPS 2024 | LLM, labelled by Grounding DINO | No | Planned: mostly existing stages |
| [SALF-CBM](#salf-cbm-show-and-tell) | CVPR 2025 | LLM, CLIP on visually prompted grid | Concept maps | Planned |
| [DCBM](#dcbm) | ICML 2025 | Clustered image regions (SAM / Grounding DINO) | No | Planned: needs a design decision |
| [SL-CBM](#sl-cbm) | AAAI 2026 | Dataset annotations | Concept maps | Planned: needs a design decision |

Implemented reference points for comparison: Koh et al. 2020 (`cub_koh2020*.yaml`), a
Label-free-CBM-style anchor (`waterbirds_lfcbm.yaml`, `metashift_lfcbm.yaml`) and SEG-MIL-CBM
(`*_segmil.yaml`).

## VLG-CBM

*Training Concept Bottleneck Models with Vision-Language Guidance.* Srivastava, Yan, Weng, NeurIPS
2024. [arXiv 2408.01432](https://arxiv.org/abs/2408.01432).

The CBL is trained against binary concept labels from Grounding DINO detections instead of CLIP
similarities. The sparsity of the final layer is set to a target *number of effective concepts*
(NEC), which caps how much information can leak through the bottleneck.

| Paper step | Stage | Exists? |
|---|---|---|
| LLM concepts per class (LF-CBM prompts) | discovery `llm` | Yes |
| Drop concepts Grounding DINO never detects above `T` | filtering `dino` | Yes |
| Binary labels: box confidence > `T = 0.15` | alignment `dino` (`threshold: 0.15`) | Yes |
| Only concepts from the image's own class count | alignment `dino` | Partly: opt-in `class_conditional` kwarg, reading the proposing classes from `ConceptSet.meta` |
| Crop augmentation: crop to a detected box, the target is that box's concept | none | Design: extra training samples per box, while the backbone features are cached per image |
| Linear CBL trained with BCE, then a sparse layer on train-standardized concept logits | generation `logits`, training `sequential` | Yes; check that the standardization matches `ConceptLayer.c_mean/c_std` |
| GLM-SAGA elastic net, λ picked on the regularization path to hit a target NEC, then pruning | predictor `sparse` | Partly: fixed `lam` only; needs an opt-in `nec` target |
| Random-CBL baseline | discovery + alignment `weights` | No: a `random` discovery (random frozen directions) |

- **Backbones:** CLIP-RN50 (CIFAR), ResNet-18 trained on CUB (CUB), ImageNet ResNet-50
  (Places365, ImageNet). `clip` and `resnet` exist. The CUB ResNet-18 needs a
  `checkpoint` kwarg on `resnet`, because its `weights` takes only torchvision names.
- **Datasets:** CUB and Waterbirds exist. CIFAR-10/100, Places365 and ImageNet are missing.
- **Metrics:** NEC is already logged as `nonzero_per_class` by the linear heads. ANEC-5 and
  ANEC-avg (accuracy at NEC = 5, and averaged over NEC ∈ {5, …, 30}) would come from a sweep over
  the NEC target plus a summary in `analysis.py`, not from an evaluator.

## SALF-CBM (Show and Tell)

*Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck
Models.* Benou, Riklin Raviv, CVPR 2025. [arXiv 2502.20134](https://arxiv.org/abs/2502.20134).

Label-free CBM made spatial. CLIP scores each concept at every cell of a 7×7 grid: a red circle is
drawn at the cell, and the circled image is compared with the concept's text. A 1×1 conv on the
backbone's feature map learns these maps. The maps are pooled into image-level concepts, which feed
a sparse linear layer.

| Paper step | Stage | Exists? |
|---|---|---|
| GPT concepts (features / surroundings / superclasses), LF-CBM filters | discovery `llm`, filtering `rules` → `clip` | Yes; check that the prompts match |
| Feature map resampled to the 7×7 grid | `instances: patches` | Yes for ResNet at 224 px (grid 7); other grids need resampling |
| Spatial targets: CLIP ViT-B/16 on images with a red circle (r = 32) at each grid point | alignment | No: a per-instance alignment like `segment_clip`, cached on disk |
| 1×1 conv CBL = the concept layer applied per patch | generation `scores` on patch bags | Yes |
| Cubic-cosine CBL loss (LF-CBM) | training concept loss | No: a cos³ concept loss |
| Global pooling of the concept maps, then a GLM-SAGA sparse layer | predictor | No: a bag-aware sparse head that pools over M (`mil` uses attention) |
| Sequential training | training `sequential` | Yes |

- **Backbones:** ResNet-18 trained on CUB, ImageNet ResNet-50, ViT-B/16. All except the CUB
  checkpoint exist (shared with VLG-CBM).
- **Datasets:** CUB exists. Places365, ImageNet and ImageNet-Segmentation (4,276 masks, for
  evaluation) are missing.
- **Metrics:** accuracy exists. Zero-shot segmentation from the concept heatmaps (mAP, mIoU,
  pixel accuracy) needs a new evaluator that reads foreground masks; its GradCAM / LRP baselines
  are out of scope. "Explain anything" ROI queries are interactive, so out of scope. Local
  interventions (editing a concept map inside a mask) could become an evaluator later.

## DCBM

*DCBM: Data-Efficient Visual Concept Bottleneck Models.* Prasse, Knab, Marton, Bartelt, Keuper,
ICML 2025. [arXiv 2412.11576](https://arxiv.org/abs/2412.11576),
[code](https://github.com/KathPra/DCBM).

Concepts are image regions, not text. SAM2 or Grounding DINO propose regions on 50 training images
per class. The crops are embedded with CLIP and clustered with k-means, and each centroid is one
concept. An image's concept activation is its projection onto a centroid. Text names come
afterwards, from the nearest word in a vocabulary.

| Paper step | Stage | Exists? |
|---|---|---|
| Region proposals on 50 images per class (SAM2 grid, or Grounding DINO with part prompts) | instance sources `sam_grid`, `grounded_boxes` | Partly: the segmentation exists |
| Crops embedded with the CLIP image encoder, k-means (k = 2048, median centroids) | discovery | Design: discovery needs segment features *without* turning the run's inputs into bags (`instances:` does both today) |
| Name each centroid by its nearest word in a 10k-word CLIP text vocabulary | discovery (`meta`) | No |
| Activation `⟨f(x), c_j⟩ / ‖c_j‖²` | alignment `weights` (frozen), generation `logits` | Partly: `ConceptLayer` standardizes features and scores, so check that this is equivalent up to an affine map |
| Linear layer, CE + λ‖w‖₁ (λ = 1e-4, Adam, 200 epochs) | predictor `sparse` (`alpha: 1`) | Yes, though the paper's optimizer is Adam and `sparse` uses proximal SGD |

- **Backbones:** CLIP RN50, ViT-B/16, ViT-L/14 (`clip`).
- **Datasets:** CUB exists. ImageNet, Places365, CIFAR-10/100 and ImageNet-R (as an
  `eval_datasets:` shift domain) are missing. MiT-States, ClimateTV, AWA2 and CelebA are
  secondary.
- **Metrics:** accuracy and OOD error change exist (`shift`). The grid pointing game (Grad-CAM
  on 2×2 image mosaics; Gini, relative %, absolute score) needs gradients through the backbone,
  but features are cached. A patch-map version is possible, but it would be a deviation from the
  paper and must be recorded as one.

## SL-CBM

*SL-CBM: Enhancing Concept Bottleneck Models with Semantic Locality for Better Interpretability.*
Zhang, Cheng, Wen, Zhang, Zhang, Hermanns, AAAI 2026.
[arXiv 2601.12804](https://arxiv.org/abs/2601.12804), [code](https://github.com/Uzukidd/sl-cbm).

A 1×1 conv on the spatial features gives concept saliency maps. The image summary (average pooling
or the CLS token) is projected onto the concept subspace, and cross-attention aligns it with the
maps. The model is trained jointly with a spatial entropy regularizer, so each location commits to
few concepts. Concepts are the dataset's annotations, so concept quality is held fixed.

| Paper step | Stage | Exists? |
|---|---|---|
| Predefined concept set (CUB attributes, RIVAL-10 attributes) | discovery `dataset`, alignment `human` | Yes for CUB |
| 1×1 conv saliency maps + summary projection + cross-attention | generation | Design: needs patch features *and* the image summary in one layer; with `instances: patches` the layer sees only the patches |
| Concept loss: L1 between softmax-normalized predictions and indicators (λ = 1e4) | training concept loss | No |
| Spatial entropy of the saliency maps (λ = 5), InfoNCE on concepts (λ = 0 by default) | training | No: needs a concept-layer penalty hook, like `PredictorHead.penalty()` |
| Joint training, Adam, lr 3e-4 | training `joint` | Yes |
| Linear head | predictor `dense` / `sparse` | Yes |

- **Backbones:** ResNet-50, ViT-B/16 (spatial tokens + CLS). Both exist.
- **Datasets:** CUB exists, but the paper reports a 4,796 / 5,794 split, so check that against
  the standard 5,994 train images. RIVAL-10 is missing.
- **Metrics:** concept and class accuracy exist (`concepts`, `shift`). For locality against masks,
  IoU exists (`part_iou`); Dice and compact IoU are missing. Average Drop / Increase / Gain need a
  new evaluator. NEC-5 / ANEC are shared with VLG-CBM. Intervention efficacy exists (`leakage`),
  but with a random order only; the paper's orderings (UCP, LCP, CCTP) are missing.
- **Baselines:** PCBM is close to `weights` alignment + `sparse`, and PCBM-h is `residual`.

## Shared gaps

Several models need the same piece, and building these first unlocks the most:

| Gap | Needed by | Kind |
|---|---|---|
| CIFAR-10/100, Places365, ImageNet (+ ImageNet-R, ImageNet-Segmentation), RIVAL-10 | all four | Datasets |
| `checkpoint` kwarg on `resnet` (CUB-trained ResNet-18) | VLG-CBM, SALF-CBM | Backbone kwarg |
| NEC-targeted `sparse` predictor + ANEC summary in `analysis.py` | VLG-CBM, SL-CBM, SALF-CBM | Predictor kwarg + analysis |
| Pooled sparse head over patch bags | SALF-CBM, SL-CBM | Predictor |
| Heatmap evaluators: foreground-mask segmentation, Dice / C-IoU, AD / AI / AG | SALF-CBM, SL-CBM, DCBM | Evaluators |
| Segment features available to discovery without bag inputs | DCBM | Design |
| Generation that sees patches and the global feature together | SL-CBM (and SALF-CBM with ViTs) | Design |
| Per-box training samples (crop augmentation) | VLG-CBM | Design |
