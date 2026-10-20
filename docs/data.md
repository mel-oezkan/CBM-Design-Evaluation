# Data

`scripts/prepare_data.py` downloads each dataset, verifies its checksum, extracts it and checks
the result. One command per dataset gives a folder that the loader accepts:

```bash
uv run python scripts/prepare_data.py cub cub-parts waterbirds --root data
uv run python scripts/prepare_data.py --check all --root data     # validate only, download nothing
```

| Target | Folder under `--root` | Source | Prepared by |
|---|---|---|---|
| `cub` | `CUB_200_2011/` (+ `attributes.txt` next to it) | [Caltech data record](https://data.caltech.edu/records/65de6-vp158) | download |
| `cub-parts` | `CUB_200_2011/part_segmentations/AnnotationMasksPerclass/` | [CUB70 masks](https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset) (67 classes) | download |
| `cub-koh-split` | `CUB_processed/class_attr_data_10/` | Koh et al.'s Codalab bundle `0x5b9d528d2101418b87212db92fea6683` | manual, then check |
| `waterbirds` | `waterbirds/` | group DRO release `waterbird_complete95_forest2water2` | download |
| `metashift` | `metashift/` | [SubpopBench](https://github.com/YyzHarry/SubpopBench) cat/dog subset | manual, then convert |

The script prints each dataset's terms of use before it starts. Archives stay in `--root`, and a
later run reuses an archive while its SHA-256 matches. The script skips a folder that passes the
check. It never overwrites a folder that exists but fails the check; it prints the problems instead.

`--check` compares each folder with the published dataset: expected files, row counts (images,
attributes, part locations) and split sizes. Then it builds the registered loader on the folder.
Each problem comes as one line that says what is wrong, for example:

```text
cub: data/CUB_200_2011
  FAIL data/CUB_200_2011/images.txt not found; found data/CUB_200_2011/CUB_200_2011/images.txt: the folder is nested one level too deep, move its content up
  to fix: uv run python scripts/prepare_data.py cub --root data
```

## Folder layouts

Point `dataset.root` at:

- **Waterbirds**: the standard folder with `metadata.csv` (`img_filename, y, split, place`), as
  released with group DRO (`waterbird_complete95_forest2water2`). The script renames it to
  `waterbirds/`, the root the anchors use.
- **CUB**: the official `CUB_200_2011` directory
  ([Caltech data record](https://data.caltech.edu/records/65de6-vp158)). Attributes are class-level majority-voted by
  default (`majority_vote: mean`: >= 50% of the class's official-train images, counting "not
  visible" as absent; 93 concepts). `majority_vote: koh` follows Koh et al.'s code (train split only,
  "not visible" negatives ignored, ties -> present; 112 concepts), and `split_dir:` takes their
  `CUB_processed/class_attr_data_10` folder (Codalab worksheet 0x362911581fcd4e048ddfd84f47203fd2)
  to use their train/val split. Each attribute is mapped to its body part's keypoint for localization.
  Optional part segmentations for `part_iou` (CUB70, 67 classes) go to
  `<root>/part_segmentations/AnnotationMasksPerclass/` (target `cub-parts`), or set
  `dataset.part_segmentations`.
- **MetaShift**: a folder with `metadata.csv` (`filename, y, a, split`; `filename` relative to the
  folder). SubpopBench's preprocessing writes `metashift/metadata_metashift.csv` instead, with paths
  that start at its `--data_path`. `prepare_data.py metashift` prints the SubpopBench steps and
  converts that file to `metadata.csv`.
- **Synthetic**: generated in memory; no `root` needed. Pair it with the `toy` backbone for fully
  offline runs.

## Caches

Backbone features are cached under `paths.cache/features/` and keyed by dataset and backbone. All
runs sharing a dataset and backbone reuse them, so training a CBM runs only on cached tensors.
Patch-feature caches also store each concept's part mask, so their key also hashes every sample's
keypoints: a changed concept-to-part mapping gets a new cache instead of reading stale masks.
LLM, VLM, ConceptNet and Grounding DINO outputs are cached under the same directory.

`llm` and `vlm` discovery call Claude (`claude-opus-5-5` by default; override with `model:`). Server-side
refusal fallback is enabled. Credentials come from `ANTHROPIC_API_KEY` or an `ant auth login` profile.
A response cut off at `max_tokens` (default 4000) raises instead of being cached; raise it with
`{name: llm, max_tokens: 8000}`.
