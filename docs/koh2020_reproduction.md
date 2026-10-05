# Reproducing Koh et al. 2020 (Concept Bottleneck Models) with `cbm_eval` — lab notebook

Goal: reproduce the CUB results of *Concept Bottleneck Models* (Koh et al., ICML 2020,
arXiv:2007.04612) for the **independent, sequential and joint** bottlenecks **through the modular
`cbm_eval` pipeline**, as a test of how reliable that code is. Runs execute on Modal.
Status, findings and costs are kept current at the top; the dated log is at the bottom.

## Status

| Step | State |
|---|---|
| Read paper + official code (github.com/yewsiang/ConceptBottleneck) | done |
| Fidelity audit of `cbm_eval` vs. the paper / official code | done (findings below) |
| Additions needed to run Koh's setup in `cbm_eval` (registered variants, opt-in kwargs) | done, tests green (86) |
| Smoke run on Modal (anchor, joint, seed 1) | running |
| Fidelity ladder R0–R3 × {independent, sequential, joint} × seeds 1–3 | pending |
| Report (Table 1/2 + interventions) → `results/cub_koh2020/README.md` and below | pending |

## Targets (paper, CUB, mean ± 2 SD over 3 seeds)

| Model | Task error (Table 1) | Concept error (Table 2) |
|---|---|---|
| Independent | 0.240 ± 0.012 | 0.034 ± 0.002 |
| Sequential | 0.243 ± 0.006 | 0.034 ± 0.002 |
| Joint (λ = 0.01) | 0.199 ± 0.006 | 0.031 ± 0.000 |

## How the reproduction maps onto `cbm_eval`

Anchor: `configs/anchors/cub_koh2020.yaml`. Driver: `scripts/koh2020_modal.py`.

| Koh et al. | `cbm_eval` |
|---|---|
| CUB, 112 class-level concepts, authors' 80/20 train/val split | `dataset: {name: cub, majority_vote: koh, split_dir: <CUB_processed/class_attr_data_10>}` |
| Inception-v3 (ImageNet), **fine-tuned end to end** | `backbone: {name: inception}` — **frozen** (the pipeline only trains on cached features) |
| Concepts = annotated attributes | `discovery: dataset`, `filtering: rules(max_words=10, remove_class_names=false)` (keeps all 112) |
| Concept supervision | `alignment: human` |
| Linear c → y | `predictor: dense` |
| Independent: f trained on true c, tested on σ(ĝ(x)) | `training: independent[_weighted]`, `generation: scores` |
| Sequential / joint: f on concept **logits** | `generation: logits` + `training: sequential[_weighted]` / `joint[_weighted]` |
| Concept BCE weighted by neg/pos ratio | `*_weighted` training variants |
| Joint λ = 0.01 on a sum over 112 concepts | `concept_weight: 1.12` (our concept loss averages over concepts) |
| Task error / concept error | `shift.test.acc` / new `concepts.test.concept_error` |
| Test-time intervention (28 groups, visibility-aware) | `leakage.intervention.acc@f` (random individual concepts; approximate) |

**Fidelity ladder** (to attribute any gap): R0 = pipeline defaults; R1 = + Koh data;
R2 = + logit-connected heads; R3 = + weighted BCE (= anchor). The remaining gap at R3 is expected
to come mostly from the frozen backbone (plus optimizer differences: Adam with early stopping on
cached features, versus Koh's SGD with augmentation for up to 1000 epochs).

## Reliability findings about `cbm_eval`

| # | Finding | Severity | Status |
|---|---|---|---|
| F1 | `data/cub.py` says concepts are "majority-voted per class (as in Koh et al., 2020)", but it counts "not visible" annotations as negatives, votes over train+val, and has no tie rule. **Result: 93 concepts instead of 112** (19 missing). On the shared concepts, 4.5% of test labels differ from Koh's. | High for any CUB-CBM comparison with the literature | Opt-in fix `majority_vote: koh` (+ `split_dir`). Default unchanged to keep run ids/caches valid. Verified: **112 concepts, 100% identical labels and splits** vs. the authors' processed data. Docstring of the default still overclaims — suggest rewording. |
| F2 | Sequential and joint heads always see **sigmoid probabilities** (`ScoreLayer` + `activate`). Koh et al. connect f to logits (their sigmoid variant is a separate, worse-for-accuracy model). There was no way to configure this. | Medium | New `generation: logits` variant. |
| F3 | No imbalance-weighted concept loss. Koh weights each concept's BCE by its neg/pos ratio (~9:1 on average). | Medium (affects concept F1/error trade-off) | New `independent_weighted` / `sequential_weighted` / `joint_weighted`. |
| F4 | No evaluator reports concept accuracy, so Table 2 cannot be computed from pipeline outputs. | Medium | New `evaluation: concepts` (error, F1, positive rate). |
| F5 | No Inception-v3 backbone. | Low | New `backbone: inception`. |
| F6 | The pipeline cannot fine-tune the backbone (by design: frozen, cached features). An exact numerical reproduction of Koh et al.'s CUB numbers is therefore out of reach. | Design limitation | Documented. Measured via the ladder. |
| F7 | `leakage` interventions replace random individual concepts with ground truth. Koh intervene on concept *groups*, set invisible concepts to 0, and use 5th/95th-percentile logits for logit models. With `generation: logits`, an intervened 0/1 becomes ±logit(1−1e-6) ≈ ±13.8. | Low (evaluation-protocol difference) | Documented; curves are compared only qualitatively. |
| F9 | Koh's imbalance weighting (F3) is a near no-op for a frozen linear concept layer trained with Adam: a constant per-concept loss scale is cancelled by Adam's per-parameter normalization (each concept owns one weight row). A weighted and an unweighted probe gave **identical** results to 4 decimals. It can only matter through coupling (joint) or with SGD. | Insight (not a bug) | Documented; the ladder measures it. |
| F8 | `CUB(...)` reads every image header at construction (`_image_sizes`). On a network volume (Modal) this takes minutes, so the driver unpacks CUB to local disk first. | Low (performance) | Worked around. |

## Costs (Modal)

| When | What | Cost |
|---|---|---|
| 2026-10-05 | Data prep + first smoke attempt of the standalone port (stuck on slow volume reads, stopped) | ≈ $4.15 |
| 2026-10-05 | Modular pipeline smoke + ladder | pending |

All of it is covered by credits so far (billed $0). Check with `uv run modal billing report --for today`.

## Open problems / risks

- **Frozen vs. fine-tuned backbone** (F6) dominates any remaining gap. With frozen ImageNet
  features, CUB task error is expected to be well above the paper's 0.20–0.24. The interesting
  checks are therefore (a) whether concept error lands near 0.03–0.05, (b) the *ordering*
  (joint < independent ≈ sequential in task error), and (c) interventions improving accuracy.
- Optional, if wanted: the standalone faithful port (`src/cbm_eval/repro/koh2020`,
  `scripts/koh2020_reference_modal.py`) can produce reference numbers with fine-tuning. It costs about
  3 A100-hours per model (~$60–80 for x→c + joint, 3 seeds). It is **not** part of the reliability
  test and has not been run.

## Log

### 2026-10-05

- Read the paper (Tables 1–2, App. A.2, B.2, B.3) and the official code. Found the authors' Codalab
  artifacts still online (processed splits and training logs).
- First wrote a standalone port of the official training code and smoke-tested it on Modal. The
  smoke run stalled reading 12k JPEGs from the volume and was stopped (≈ $4).
- **Pivot (per request): the reproduction must use the `cbm_eval` modules.** The port stays in
  `src/cbm_eval/repro/koh2020` as an unrun reference, as `CLAUDE.md` allows.
- Audited `cbm_eval` against Koh et al. → findings F1–F8. Added the minimal registered variants and
  opt-in kwargs (defaults unchanged, per `CLAUDE.md`'s run-id rules), with tests. Suite: 86 passed.
- Verified F1's fix against the authors' processed pickles: 112/112 concepts in order and identical
  labels and split membership.
- Created the W&B project `deep_cv/cbm-koh2020-modular` (Modal secret `wandb`). Launched a smoke run
  of the anchor (R3 joint, seed 1) on an L4.
- Smoke result (R3 joint, seed 1): **task error 0.466, concept error 0.122**, 112 concepts.
  The pipeline ran end to end without errors.
- Ceilings from the frozen features, measured outside the pipeline with simple Adam probes on the
  same cached Inception features (authors' split):
  - linear concept probe: test concept error **0.103–0.114**, so the pipeline's 0.122 is close to it;
  - linear x→y probe, no bottleneck: test task error **0.374**.
  - **Conclusion:** with a frozen backbone, the paper's 0.031 concept error and 0.20–0.24 task errors are
    out of reach. The gap is the backbone, not the CBM stages.
- Launched the full ladder (4 levels × 3 schemes × seeds 1–3), detached.
