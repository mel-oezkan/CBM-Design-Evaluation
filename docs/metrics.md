# Metrics

All metrics are prefixed by their evaluator name, e.g. `shift.test.wga`.

- `shift`: per-split accuracy, per-group accuracy, worst-group accuracy (`wga`) and mean group
  accuracy. Effective robustness is reported when a `baseline: {slope, intercept}` is given.
  Otherwise, fit one over baseline runs with `analysis.fit_robustness_baseline`.
- `concepts`: `<split>.concept_error` (1 - accuracy of `c_hat >= 0.5` vs. annotations, as in
  Koh et al.'s Table 2), `<split>.concept_f1` and the annotations' `positive_rate`. Binary
  concept layers whose concepts are all annotated only.
- `leakage`:
  - `intervention.acc@f` / `intervention.gain`: accuracy after replacing a random fraction `f` of
    concept predictions with ground truth. Ground truth is human annotations when every concept is
    annotated and the layer predicts binary concepts. Otherwise it is the alignment's own targets,
    and gains are then expected to be near zero for pseudo-label alignment.
  - `probe.soft_hard_gap`: accuracy of a probe on soft scores minus one on binarized scores.
  - `probe.spurious_bacc`: how well the spurious attribute can be decoded from the concepts.
- `localization`: pointing game and locality (map mass inside the annotated region, with
  `locality_chance` for reference). Only concepts that match an annotated dataset concept by name
  are scored.
- `keypoint_distance`, `part_iou`: CUB part localization against keypoints and part masks (see
  [Part localization](part-localization.md)).
- `faithfulness`: deletion / insertion AUC (`dauc`, `iauc`, with `*_random` controls),
  `top1_insertion` and `top5_deletion` over segments ranked by their contribution to the
  prediction. Bag-aware heads (`predictor: mil`) only.
