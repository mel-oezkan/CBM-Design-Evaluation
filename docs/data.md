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
