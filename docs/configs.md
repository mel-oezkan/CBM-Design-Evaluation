# Config reference

This page defines the config format. For worked examples, see
[Designing an experiment](experiments.md).

An anchor fully specifies one run. Each stage is `{name: <variant>, **kwargs}`, and the kwargs go
straight to the variant's constructor. An ablation names an anchor and a set of factors:

```yaml
anchor: ../anchors/waterbirds_lfcbm.yaml
mode: ofat            # one-factor-at-a-time around the anchor (default), or grid
seeds: [0, 1, 2]
overrides: {teacher: {name: clip}}     # optional, applied to the anchor first
factors:
  stages.predictor: [{name: dense}, {name: residual}]
  stages.predictor.lam: [0.0001, 0.001]   # dotted keys reach into specs; list indices work too
```

A factor value **replaces** whatever sits at that key. For example, `stages.training: [{name: joint}]`
drops any `epochs` the anchor set, so constructor defaults apply. To vary one parameter, use a
dotted key instead.

In `grid` mode, a dotted key is applied to every combination. With both `stages.predictor: [{name:
sparse}, {name: dense}]` and `stages.predictor.lam: [...]`, `lam` is also written into the `dense`
spec, which rejects it. Put the parameter inside the spec (`{name: sparse, lam: 0.001}`) or split
the sweep into two ablations.

The `run_id` is a hash of everything that defines the model and its evaluation. Seed, name, paths
and tags are excluded. Sweeps skip `(run_id, seed)` pairs that already succeeded, so an interrupted
sweep can simply be restarted. The hash covers the config, not the code: after changing how a
variant behaves, re-run with `--force` or write to a new `paths.results` file, or the sweep will
skip those runs and keep the old numbers. Each row records `git_commit` to tell versions apart.

### Test-only domains

`eval_datasets:` adds shifted test sets, such as CUB as paintings, to a run. The CBM is trained on
`dataset` alone. Each alias names the eval dataset's `test` split, and an evaluator scores it only
when the alias is in its `splits`:

```yaml
dataset: {name: cub, root: data/CUB_200_2011}
eval_datasets:
  paintings: {name: cub_paintings, root: data/cub_paintings}
evaluation:
  - {name: shift, splits: [val, test, paintings]}
  - {name: concepts, splits: [test, paintings]}
```

Classes and concepts are matched to the training dataset's by name, and every eval class must exist
in the training dataset. An eval dataset without its own concept annotations inherits the training
dataset's class-level ones (`class_concepts()`, e.g. CUB's majority-voted attributes). A test-only
dataset cannot be used as `dataset:`.

`eval_datasets` is part of the `run_id`. To score runs that are already trained on a new domain, add
it to the anchor and re-evaluate the saved runs instead of retraining:

```bash
uv run cbm-eval evaluate runs/<run_id>-s0 runs/<run_id>-s1 --config configs/anchors/cub_human.yaml
```

`evaluate` takes `evaluation:` and `eval_datasets:` from `--config` (plus any `--set` overrides) and
refuses any change that would alter the model. It appends the row that `run` would write for the new
config, marked with `evaluated_from`, so a later `sweep` of that config skips it.

See [`ExperimentConfig`][cbm_eval.config.ExperimentConfig] and
[`Ablation`][cbm_eval.config.Ablation] for the parsed form, and the
[component catalogue](components.md) for the keys each `{name: ...}` spec accepts.
