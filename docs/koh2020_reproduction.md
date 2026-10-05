# Reproducing Koh et al. 2020 (Concept Bottleneck Models) with `cbm_eval` — lab notebook

Goal: reproduce the CUB results of *Concept Bottleneck Models* (Koh et al., ICML 2020,
arXiv:2007.04612) for the **independent, sequential and joint** bottlenecks **through the modular
`cbm_eval` pipeline**, as a test of how reliable that code is. Runs execute on Modal.
Status, findings and costs are kept current at the top; the dated log is at the bottom.

## Status

| Step | State |
|---|---|
| Read paper + official code (github.com/yewsiang/ConceptBottleneck) | done |
| Fidelity audit of `cbm_eval` vs. the paper / official code | done (findings F1–F10) |
| Koh-specific options as registered variants / opt-in kwargs, per `CLAUDE.md` | done |
| **End-to-end backbone fine-tuning** in the architecture (`finetune:`; user decision) | done, 107 tests green |
| Anchors `configs/anchors/cub_koh2020{,_independent,_sequential}.yaml` (fine-tuned) | written |
| Ablations `configs/ablations/cub_koh2020*.yaml` (frozen arm + fidelity factors) | written |
| Fine-tuned anchors on Modal (3 schemes × seeds 1–3) | **waiting for user's go** |
| Frozen ablations on Modal (42 runs) | **waiting for user's go** |
| Report vs. Table 1 / 2 | pending |

## Targets (paper, CUB, mean ± 2 SD over 3 seeds)

| Model | Task error (Table 1) | Concept error (Table 2) |
|---|---|---|
| Independent | 0.240 ± 0.012 | 0.034 ± 0.002 |
| Sequential | 0.243 ± 0.006 | 0.034 ± 0.002 |
| Joint (λ = 0.01) | 0.199 ± 0.006 | 0.031 ± 0.000 |

## How the reproduction maps onto `cbm_eval`

Every run is a plain config executed by `cbm-eval run` / `cbm-eval sweep`.
`scripts/koh2020_modal.py` only unpacks the data on Modal, calls the CLI and mirrors rows to W&B.

| Koh et al. | `cbm_eval` |
|---|---|
| CUB, 112 class-level concepts, authors' 80/20 train/val split | `dataset: {name: cub, majority_vote: koh, split_dir: data/CUB_processed/class_attr_data_10}` |
| Inception-v3 (ImageNet), fine-tuned end to end, SGD, augmentation | `backbone: inception` + `training.finetune: {optimizer: sgd, lr, momentum: 0.9, weight_decay, batch_size: 64}` |
| x→c: lr 0.01, wd 4e-5 | independent / sequential anchors |
| joint: lr 0.001, wd 4e-4, loss ÷ (1 + λ·112) | joint anchor: lr 0.00047 (= 0.001 / 2.12, because our loss is not divided) |
| Concepts = annotated attributes | `discovery: dataset`, `filtering: rules(max_words=10, remove_class_names=false)` (keeps all 112) |
| Concept supervision | `alignment: human` |
| Linear c → y | `predictor: dense` |
| Independent: f on true c, tested on σ(ĝ(x)) | `training: independent_weighted`, `generation: scores` |
| Sequential / joint: f on concept **logits** | `generation: logits` + `sequential_weighted` / `joint_weighted` |
| Concept BCE weighted by neg/pos ratio | `*_weighted` training variants |
| Joint λ = 0.01 on a sum over 112 concepts | `concept_weight: 1.12` (our concept loss averages) |
| Task error / concept error | `shift.test.acc` / `concepts.test.concept_error` |
| Test-time intervention | `leakage.intervention.acc@f` (random single concepts; approximate) |

**Remaining gaps** (recorded in the anchor header): early stopping on val loss (ours) vs. up to
1000 epochs on train+val selected by training accuracy (theirs); Adam for concept layer and head
(ours) vs. SGD; no auxiliary Inception head; ImageNet normalization vs. their mean 0.5 / std 2;
resize vs. center crop at eval; interventions on single concepts vs. visibility-aware groups.

**Ablations** (`configs/ablations/cub_koh2020*.yaml`, frozen backbone via
`overrides: {stages.training.finetune: null}`): OFAT removal of each Koh option — `majority_vote:
mean`, no `split_dir`, `generation: scores` (sequential / joint), unweighted training.

## How fine-tuning fits the architecture (design)

- `Backbone.trainable` (class flag, default False) + `encoder()` (fresh trainable copy: images → (N, D))
  + `train_transform()` (augmentation). `inception` and `toy` are trainable. Frozen feature caching is unchanged.
- `training.finetune: {...}` (a frozen `Finetune` dataclass; unknown keys raise) on every training
  variant. Default `None` keeps existing behaviour and run ids.
- `PipelineBuilder.train` creates the encoder, passes it to `Training.fit(..., encoder=)`, then
  `ctx.use_encoder(encoder)`. From then on `ctx.split()` / `ctx.inputs()` return the fine-tuned features
  (in memory, run-specific), so `cache_scores` and every evaluator work unchanged. This is equivalent
  to evaluating the end-to-end model. The encoder is reset at the start of every run.
- Refused loudly: `finetune` with `instances:`, with patch-reading evaluators, or with a frozen-only backbone.
- `TrainedCBM.encoder` is saved as `encoder.pt`.
- Tests (`tests/test_finetune.py`): all schemes fine-tune and the encoder moves; evaluated inputs equal
  the fine-tuned encoder's features; reused `Context` resets; refusals; unknown keys rejected.

## Reliability findings about `cbm_eval`

| # | Finding | Severity | Status |
|---|---|---|---|
| F1 | `data/cub.py` says concepts are "majority-voted per class (as in Koh et al.)", but it counts "not visible" as negatives, votes over train+val and has no tie rule → **93 concepts instead of 112**; 4.5% of shared test labels differ. | High for comparisons with the literature | Opt-in `majority_vote: koh` (+ `split_dir`). Verified **112/112 concepts, 100% identical labels and splits** vs. the authors' data. Default unchanged. The README no longer claims Koh-style voting for the default. |
| F2 | Sequential and joint heads always saw sigmoid probabilities; Koh connect f to logits. | Medium | New `generation: logits`. |
| F3 | No imbalance-weighted concept loss. | Medium | New `*_weighted` training variants. |
| F4 | No evaluator for concept accuracy (Table 2). | Medium | New `evaluation: concepts`. |
| F5 | No Inception-v3 backbone. | Low | New `backbone: inception` (lazy weights, cache-key tested). |
| F6 | No backbone fine-tuning. Koh's numbers were unreachable: the frozen-feature ceiling is ≈ 0.37 task / ≈ 0.10 concept error. | Design limitation | **Added `finetune:`** (see design above). |
| F7 | `leakage` interventions are random single concepts, not Koh's visibility-aware groups; with `logits`, an intervened 0/1 becomes ±13.8 rather than the 5th/95th-percentile logits. | Low | Documented. |
| F8 | `CUB(...)` reads every image header at construction: minutes on a network volume. | Low (perf) | Worked around (local unpack). |
| F9 | Imbalance weighting is a near no-op for a frozen linear concept layer under Adam (per-row loss scale cancels). Weighted and unweighted probes were identical to 4 decimals. | Insight | Ablation measures it. |
| F10 | Feature caches are keyed by the full dataset key, so label-only options (`majority_vote`, `split_dir`) re-extract identical image features. | Low (perf) | Documented. |

## Results so far (frozen backbone, pre-restructure; indicative only)

| Config | Task error | Concept error |
|---|---|---|
| Joint, Koh data + logits + weighted (seed 1) | 0.466 | 0.122 |
| Independent, Koh options (seeds 1–3) | 0.522 / 0.525 / 0.526 | — |
| Independent, pipeline defaults (93 concepts; seeds 1–2) | 0.537 / 0.537 | 0.098 / 0.098 |
| Linear probe on frozen features (outside pipeline) | 0.374 (x→y) | 0.103–0.114 |

## Costs (Modal)

| When | What | Cost |
|---|---|---|
| 2026-10-05 | Standalone port (now removed): data prep + stalled smoke run | $4.21 |
| 2026-10-05 | Modular pipeline: smoke + partial frozen ladders | $0.37 |
| **Total** | | **$4.59** (covered by credits, billed $0) |

**Estimate for the next step (not started):** fine-tuned anchors = 9 A100 runs. Each runs up to 300
epochs with patience 30; at an estimated ~10–15 s per epoch, that is ~0.3–1 h, so **≈ $6–20** in
total. The frozen ablations (42 runs, one L4) cost **≈ $1**. A 2-epoch benchmark first (≈ $0.3) would
pin the epoch time down.

## Log

### 2026-10-05

- Read the paper and the official code. The authors' Codalab artifacts are still online (processed
  splits, training logs, checkpoints).
- Wrote a standalone port first. **Pivot (per request): the reproduction must use the `cbm_eval`
  modules.** Port removed per the updated `CLAUDE.md`. Copy kept outside the repo.
- Audited `cbm_eval` (F1–F10); added minimal registered variants / opt-in kwargs with tests.
- Frozen-backbone smoke (joint, seed 1): task error 0.466, concept error 0.122 — at the frozen ceiling.
- Seed check: independent seeds 1/2 gave identical task error. A synthetic test shows seeds do change
  the heads, so this is near-deterministic convergence on 200 fixed concept codes, not a seeding bug.
- Asked about fine-tuning (per `CLAUDE.md` "stop and ask"). **User chose: add true end-to-end
  fine-tuning.** Implemented as designed above; 107 tests pass.
- Restructured configs: fine-tuned anchors + frozen OFAT ablations. Modal glue split into parallel
  per-(anchor, seed) runs with separate result files, plus one sequential ablation container.
- **Waiting for the user's go before launching anything on Modal.**
