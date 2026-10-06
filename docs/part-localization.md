# Part localization (CUB)

`keypoint_distance` and `part_iou` re-implement the ProtoCBM localization metrics
([pascal0012/ProtoCBM](https://github.com/pascal0012/ProtoCBM), `localization/`). Concept maps are
the concept layer on patch features, a patch bag's per-patch scores, or, for segment bags, the
best covering segment per patch. For each image and part group, `select: argmax` (the default)
scores the group's concept with the highest image-level score. `select: present` averages over
every concept annotated present. Annotations go through the backbone's resize and center crop and
are cached under `paths.cache/parts/`.

| Evaluator | Metrics | Options |
|---|---|---|
| `keypoint_distance` | `dist` (map peak → nearest visible keypoint, in image sides; mean over the 12 groups), `dist.<group>`, `pck` (share within `threshold`), `dist_center` / `pck_center` (a map that points at the center) | `select`, `threshold: 0.1` |
| `part_iou` | `miou` (over all image–group pairs), `iou.<group>` for the 8 mask groups, `miou_center` (center prior) | `select`, `hard: true`, `keep_ratio: 0.5`, `seg_size: 56` |

Differences from ProtoCBM: the peak is the center of the top patch rather than the argmax of a
bilinear upsample (the same up to sub-patch rounding, but without the corner bias of clamped
borders). Distances are normalized by the image side rather than measured in pixels. IoU is
computed at `seg_size` rather than at full image resolution.
