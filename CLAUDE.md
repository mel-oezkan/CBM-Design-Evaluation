# CLAUDE.md

Research code for a controlled design study of concept bottleneck models (CBMs). Read `README.md`
for an overview and `docs/` for the stage catalogue, config format, caching and metrics; this
file covers the rules for changing the code without breaking the architecture.

## Commands

```bash
uv sync                      # core + dev (pytest); `uv sync --all-extras` for CLIP / HF / Claude
uv run pytest -q             # fully offline, ~5 s; must stay green and offline
uv run --group docs properdocs build --strict   # docs site; must build without warnings
uv run cbm-eval list         # every registered component, per registry
uv run cbm-eval run configs/anchors/synthetic.yaml --seeds 0
```

## Architecture in one paragraph
<!-- --8<-- [start:architecture] -->

A run is an `ExperimentConfig` (`config.py`) → `PipelineBuilder` (`pipeline.py`) builds every
component by name from a `Registry` (`registry.py`) → stages run in a fixed order
(discovery → filtering chain → alignment → generation → predictor → training) against a shared
`Context` (`context.py`) → the `TrainedCBM` (`model.py`) is scored by evaluators → one row per
`(run_id, seed)` goes to the JSONL `ResultsStore` (`results.py`) → `analysis.py` aggregates.
Stages only talk through the typed containers in `structures.py` and `stages/base.py`
(`ConceptSet`, `AlignedConcepts`, `ConceptLayer`, `PredictorHead`, `FeatureSplit`, `Bag`, ...).
<!-- --8<-- [end:architecture] -->

<!-- --8<-- [start:rules] -->
## Core rules

1. **Everything pluggable goes through a registry.** A new variant is a class decorated with
   `@<REGISTRY>.register("name")` that subclasses the matching ABC. Never add `if name == ...`
   branches to `pipeline.py`, `cli.py` or another stage to select behaviour.
2. **One component = one interface.** Implement exactly the abstract method(s) of its base class
   and nothing the pipeline doesn't call. If a variant needs a new capability from the pipeline,
   express it as a class-level flag on the base class (like `Evaluator.needs_patches`,
   `Predictor.supports_bags`, `InstanceSource.needs_concepts`) with a safe default, and have
   `PipelineBuilder` read the flag — don't special-case the variant.
3. **Config kwargs = constructor kwargs.** `{name: x, **kw}` is passed straight to `__init__`.
   Every hyperparameter is an explicit keyword argument with a sensible default, and nothing reads
   the global config from inside a component. A subclass may forward `**kw` to its parent's
   explicit signature (`Joint`, `SAMGrid` do), since unknown keys still raise `TypeError`; never
   swallow or `.get()` from a `**kwargs` dict. Store kwargs on `self`. Stage components, scorers
   and instance sources are all built when `PipelineBuilder` is constructed, so keep their
   `__init__` cheap (no I/O, no model loading) and do heavy work lazily in the stage method.
   Datasets and backbones (built once per run in `context()`) and the LLM client (created lazily
   by discovery) may load files or weights in their `__init__`.
4. **Shared state comes from `Context`, never from globals or re-loading.** Use `ctx.split()`,
   `ctx.inputs()`, `ctx.teacher_split()`, `ctx.encode_text()`, `ctx.instances()`, `ctx.parts()`.
   A split name may be an `eval_datasets:` alias, so reach raw data through `ctx.samples(split)` /
   `ctx.source(split)` rather than `ctx.dataset.samples(split)`; `ctx.dataset` is the training dataset.
   If a stage needs something new and reusable, add a cached accessor to `Context` instead of
   computing it inside the stage.
5. **Stages are pure w.r.t. their inputs.** Return new objects (`concepts.subset(idx)`,
   a fresh `AlignedConcepts`) instead of mutating what you were given. Keep
   `ConceptSet.names`, `vectors` and `meta` row-aligned; record provenance in `meta`.
   `Context` is a cache, not a channel between stages: stages read it and fill it only through
   its cached accessors, and never assign its attributes. Only `PipelineBuilder` sets run state
   on it (e.g. `ctx.concepts` after filtering).
6. **Support instance bags or refuse them loudly.** Tensors may be `(N, D)` or `(N, M, D)` with a
   `Bag` mask. Use `instance_rows`, `Bag.rows/mean/max` rather than assuming 2-D. If a component
   genuinely can't handle bags, raise a clear `ValueError`/`RuntimeError` naming the config fix.
7. **Fail with actionable messages.** Errors should say which config key to change
   (see the bag/predictor check in `PipelineBuilder.__init__` and `Context.teacher_split`).

## Adding a component — checklist

| Kind | Base class | Registry | Package `__init__.py` to update |
|---|---|---|---|
| Dataset | `data.base.ImageDataset` | `DATASETS` | `data/__init__.py` |
| Backbone | `backbones.base.Backbone` | `BACKBONES` | `backbones/__init__.py` |
| Instance source | `instances.base.InstanceSource` | `INSTANCES` | `instances/__init__.py` |
| Discovery / Filter / Alignment / Generation / Predictor / Training | `stages.base.*` | `DISCOVERY` ... `TRAINING` | `stages/<stage>.py` (already imported by `stages/__init__.py`) |
| Concept scorer | duck-typed `score(concepts, ctx, split)` | `stages.scorers.SCORERS` | `stages/scorers.py` |
| Evaluator | `evaluation.base.Evaluator` | `EVALUATION` | `evaluation/__init__.py` |

Steps:

1. Put the class in the module for its kind (a new file only for a genuinely new family; add a
   module docstring saying what it is and, for paper methods, cite the paper).
2. Register it and **import the module from the package `__init__.py`** — registration is an
   import side effect and `registry.load_all()` only imports the packages.
3. Docstring the class with what it does and any deviation from the source paper.
4. Add a test on the synthetic dataset + `toy` backbone (extend the parametrized
   `test_every_stage_variant_runs` in `tests/test_pipeline.py` when it fits). Tests must not touch
   the network or download weights: inject stub clients/models via constructor args (see
   `StubLLM` in `tests/test_image_paths.py`, the stub detector/SAM in `tests/test_segmil.py`).
5. If the component caches anything on disk (dataset, backbone, scorer, instance source), add it
   to `CASES` in `tests/test_cache_keys.py`: every constructor arg must change the key or be listed
   as exempt with a reason.
6. Update the stage/variant tables in `docs/architecture.md` (and the summary in `README.md`),
   and add an ablation factor or anchor under `configs/` if the variant is part of the study.
   The class docstring and constructor signature are the variant's entry in the docs' component
   catalogue (`scripts/gen_component_docs.py`).

The docs pages under `docs/` include sections of `README.md` (intro, status) and `CLAUDE.md`
through snippet section markers (HTML comments starting with `--8<--`); keep them when editing
around them.

Kind-specific contracts:

- **Dataset**: implement `class_names` and `samples(split)` for `train`/`val`/`test`; set
  `n_attrs` and fill `Sample.attr` when there is a spurious attribute (groups depend on it).
  A shifted test set of another dataset sets `splits = ("test",)`, uses that dataset's class and
  concept names (matching is by name), and is used under `eval_datasets:`. Implement
  `class_concepts()` when concept labels are class-level, so unannotated eval domains inherit them.
  `cache_key()` must include every constructor arg that changes the data.
- **Backbone**: frozen; set `dim` (and `patch_grid`, `has_text` where applicable).
  `cache_key()` must identify the weights exactly. Patch and text embeddings must live in the
  same space as `encode_images`. Import optional heavy deps (`open_clip`, `transformers`) inside
  methods, not at module top level.
- **Generation**: subclass `ConceptLayer`, implement `represent` and set `rep_dim`; concept
  predictions stay in target space so interventions keep working. Set `represent_uses_x = False`
  if `represent` ignores `x`; change the scores in `logits_from_normalized`, not `concept_logits`. Don't override `forward`:
  joint training composes `logits_from_normalized`, `activate` and `represent` itself.
- **Predictor**: return a `PredictorHead`; use `penalty()`/`proximal_step()`/`phases()` hooks
  rather than editing the training stages. Set `supports_bags = True` only if it pools over M, and
  `uses_features = False` on the head only if `forward` ignores `x_norm`.
- **Training**: must respect `layer.frozen`, `aligned.has_targets`, `head.optimizer/lr` and
  `head.phases()`; return a flat `dict[str, float]` log. Losses are hooks on the training base
  class: `_task_loss(logits, batch)` (cross-entropy) and `_concept_loss(layer, batch)` (selected by
  `concept_loss:`, reading concept logits as `batch.concept_logits(layer)`). A new loss is a subclass of `Sequential`/`Joint`/`Independent` that overrides
  one hook and is registered under its own name; don't add a `loss:` string switch.
- **Concept scorer**: if it caches scores, keep its output-relevant args in `self.kw` and build the
  key from `cache_key()` (see `GroundingDINOScorer`).
- **Evaluator**: return a flat `dict[str, float]` with **unprefixed** keys — the pipeline prefixes
  them with the evaluator name. Read predictions through `TrainedCBM` (`predict`, cached scores),
  not by re-training. Set `needs_patches = True` if it reads patch features.
- **Instance source**: implement `build(ctx, split) -> InstanceSplit`; set `name`/`kw` so
  `cache_key` is correct, and `needs_concepts`/`needs_patches` honestly.

## Reproducibility invariants (do not break)

- `run_id` is a hash of `ExperimentConfig.to_dict()` minus `name/seed/paths/device/tags`.
  Adding a top-level config key or changing `to_dict()` changes every run id — only do it the way
  `instances` was added (omit the key when unset). Renaming a registered name or a constructor
  kwarg also invalidates existing results; prefer adding a new kwarg with the old behaviour as default.
- **The run_id hashes the config, not the code.** Constructor defaults and implementation details
  are invisible to it, and `sweep` skips every `(run_id, seed)` already in the results file. So
  never change a default or a variant's behaviour in place: add a kwarg whose default keeps the
  old behaviour, or register a new variant. If a behaviour change is unavoidable (a bug fix),
  say so in the commit message; results must be re-run with `--force` or into a new results file.
  Rows record `git_commit`, but the skip logic does not look at it.
- Every expensive or external result is cached on disk and keyed by `stable_hash` of everything
  that affects it (features: dataset+backbone; LLM/VLM: `JsonCache`; instances: dataset, encoders,
  source, concepts). New caches go under `ctx.cache_dir/<kind>/` with a complete key. Derive the key
  from a `self.kw` dict of the output-relevant constructor args (as `SyntheticDataset`, the
  instance sources and `GroundingDINOScorer` do) rather than listing fields by hand;
  `tests/test_cache_keys.py` enforces this. Args that only batch work stay out of the key.
  Never cache a partial or failed result: `ClaudeClient` raises on refused or cut-off responses
  instead of storing them, which is also why `max_tokens` can stay out of its key.
- Randomness flows from `cfg.seed` (`seed_everything`); don't create unseeded generators.
- Training defaults to CPU; devices come from `utils.resolve_device` and the config, not hard-coded `.cuda()`.

## Where things do NOT go

- **Paper reproductions are not separate pipelines.** A reproduction is an anchor under
  `configs/anchors/` (plus ablations), built from registered components. Where the paper differs
  from what exists, add a registered variant or an opt-in kwarg whose default keeps current
  behaviour (as `cub_koh2020.yaml` does with `majority_vote: koh`, `generation: logits`), and
  record remaining gaps in the anchor's header comment. Never re-implement data loading, models,
  training loops or metrics that a registry already provides. If the paper needs something the
  architecture can't express, stop and ask (open an issue to discuss the design) instead of
  building around it; backbone fine-tuning was added this way, as `Backbone.trainable` +
  `training.finetune:`.
- Analysis-only logic belongs in `analysis.py`, not in evaluators; evaluators produce per-run
  numbers only.
- Experiment settings belong in YAML under `configs/`, not as changed defaults in code (which
  would also break the run_id invariant above).

## Problems you find along the way

Every problem you find that the current task doesn't fix becomes a GitHub issue in this repository
(`gh issue create`): a bug, a crashing variant combination, a reproducibility gap, a cache-key
hole, or a doc that contradicts the code. Mentioning it in a reply or a commit message is not
enough. First check `gh issue list --search` for an existing issue and comment there instead of
duplicating it. An issue states:

- what goes wrong, with a minimal reproduction (a config or a few lines on the synthetic dataset and `toy` backbone);
- where it happens (`file:line`) and, if known, the cause;
- whether a fix would change existing results. If it would, existing runs need `--force` or a new
  results file (see the reproducibility invariants above), so say which runs are affected.

Then link the issue in your reply to the user, and in the PR or commit where the problem came up.

## Style

Match the surrounding code: `from __future__ import annotations`, type hints, module docstrings,
short docstrings that state shapes (`(N, M, D)`) and intent, compact multi-assignment in
`__init__`, comments only where the *why* isn't obvious. Shapes in comments use `N` images,
`M` instances, `K` concepts, `D` feature dim, `P` patches.
<!-- --8<-- [end:rules] -->
