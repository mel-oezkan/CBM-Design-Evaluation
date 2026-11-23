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
- **CIFAR-10 / CIFAR-100** (`cifar10`, `cifar100`): the extracted python version from
  [the CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html). `root` is the extracted folder itself
  (`cifar-10-batches-py/` or `cifar-100-python/`); CIFAR-100 uses its 100 fine labels. Validation is a
  seeded holdout of the official train split (`val_fraction: 0.1`, `split_seed: 0`). There are no
  concept annotations, so use label-free discovery (`llm`, `kb`) and alignment (`clip`, `dino`); `human`
  alignment, `dataset` discovery and the `concepts`, `localization`, `part_iou` and `keypoint_distance`
  evaluators refuse the dataset. Images stay in memory, so stages that read image files (`vlm`
  discovery, the `grounding_dino` scorer, segment instance sources) refuse them too.
- **MetaShift**: a folder with `metadata.csv` (`filename, y, a, split`), as produced by the
  [SubpopBench](https://github.com/YyzHarry/SubpopBench) preprocessing scripts.
- **Synthetic**: generated in memory; no `root` needed. Pair it with the `toy` backbone for fully
  offline runs.

Backbone features are cached under `paths.cache/features/` and keyed by dataset and backbone. All
runs sharing a dataset and backbone reuse them, so training a CBM runs only on cached tensors.
Patch-feature caches also store each concept's part mask, so their key also hashes every sample's
keypoints: a changed concept-to-part mapping gets a new cache instead of reading stale masks.
LLM, VLM, ConceptNet and Grounding DINO outputs are cached under the same directory.

Two feature caches are large: augmented views (`augment: {views: V}`, about 1.6 GB for CUB train +
val at 64 views of Inception) and patch features (about 4–5 GB per backbone for CUB). Only some configs
read them, so they can be deleted when unused. Each time a run reads or writes one, a marker file
next to it (`<file>.last_used`) records the time. `cbm-eval cache prune` deletes every marked file
that no run used for 5 days (`--unused-days`; `--dry-run` only lists them). Clean features and the
LLM/VLM caches have no marker and are never deleted. Deleting a cache does not change results: the
next run that needs it extracts it again.

| Cache | Size (CUB) | Pruned when unused |
|---|---|---|
| Clean features, one vector per image | ~35–100 MB per backbone | no |
| Augmented views | ~1.6 GB per feature set | yes |
| Patch features | ~4–5 GB per backbone | yes |
| LLM, VLM, ConceptNet, Grounding DINO | small; costly to rebuild | no |

On Modal, `scripts/cache_prune_modal.py` prunes the volume's cache once a day while it is deployed
(`uv run modal deploy scripts/cache_prune_modal.py`), and `scripts/koh2020_modal.py` prunes once
before each launch.

`llm` and `vlm` discovery call Claude (`claude-opus-5-5` by default; override with `model:`). Server-side
refusal fallback is enabled. Credentials come from `ANTHROPIC_API_KEY` or an `ant auth login` profile.
A response cut off at `max_tokens` (default 4000) raises instead of being cached; raise it with
`{name: llm, max_tokens: 8000}`.
