# Python API

Everything the CLI does is a Python call. This page trains a CBM from a notebook or script,
predicts with it, intervenes on its concepts, reads its head, compares two configs and runs one
stage on its own. All code on this page comes from two scripts in `examples/`. The test suite runs
them offline on the synthetic dataset, so the code here works with the current version.

| You want to | Call | You get |
|---|---|---|
| describe a run | [`ExperimentConfig.from_dict(d)`][cbm_eval.config.ExperimentConfig] | the config; `cfg.run_id` is its hash |
| train and evaluate it | [`pipeline.run(cfg, store)`][cbm_eval.pipeline.run] | a [`RunResult`][cbm_eval.pipeline.RunResult]: `cbm`, `metrics`, `ctx`, `run_dir`, `concept_trace` |
| read a split | [`ctx.inputs(split)`][cbm_eval.context.Context.inputs], [`ctx.split(split)`][cbm_eval.context.Context.split] | the CBM's input features; labels and annotations |
| predict, intervene | [`cbm.predict(x, intervene, bag)`][cbm_eval.model.TrainedCBM.predict] | `(class logits, concept predictions)` |
| analyze many runs | [`ResultsStore(path).load()`][cbm_eval.results.ResultsStore.load], [`analysis`][cbm_eval.analysis] | a flat DataFrame; tables of means and effects |
| run one stage | `FILTERING.build({...})`, or the class itself | the stage component; call its one method |

Run both scripts from the repository root. Each takes a few seconds on a laptop CPU:

```bash
uv run python examples/quickstart.py
uv run python examples/one_stage.py
```

## Describe the run

A config dict has the same keys as an anchor YAML ([config reference](configs.md)). This one is a
classic CBM: the dataset's annotated concepts, trained on those annotations, with a sparse linear
head on top.

```python title="examples/quickstart.py"
--8<-- "examples/quickstart.py:config"
```

`from_dict` checks that every stage is present. Unknown kwargs fail later, when
`PipelineBuilder` calls each component's constructor. To change one key of an existing config,
use `cfg.with_overrides({"stages.predictor.lam": 0.01})`; it returns a new config with a new
`run_id`.

## Train and evaluate

```python
--8<-- "examples/quickstart.py:run"
```

`run` does what `cbm-eval run` does for one seed ([how a run works](how-it-works.md)): it builds
every component, encodes the images once, discovers and filters concepts, trains, evaluates, saves
the run folder and appends one row to `store`. Pass `store=None` to skip the row and `save=False`
to skip the folder.

The [`RunResult`][cbm_eval.pipeline.RunResult] holds:

| Field | What it is |
|---|---|
| `cbm` | the [`TrainedCBM`][cbm_eval.model.TrainedCBM]: concept layer, head, the concepts it uses (`cbm.concepts`), and its targets (`cbm.aligned`) |
| `metrics` | the flat metric dict, keys prefixed by evaluator (`shift.test.acc`) |
| `ctx` | the run's [`Context`][cbm_eval.context.Context]: dataset, backbone and cached features |
| `concept_trace` | `(step, n_concepts)` after discovery, each filter and alignment |
| `run_dir` | the saved run folder, or `None` with `save=False` |

## Predict

The CBM reads backbone features, not images. `ctx.inputs(split)` returns them for a split name
(`train`, `val`, `test` or an `eval_datasets:` alias):

```python
--8<-- "examples/quickstart.py:predict"
```

With `instances:` in the config, `x` is an `(N, M, D)` bag of instances and `bag` marks the real
ones; pass both to `predict` unchanged.

## Intervene on concepts

An intervention replaces some predicted concepts by given values before the head reads them.
`intervene=(mask, values)` takes two `(N, K)` tensors: `mask` says which concepts to replace in
each image, and `values` holds the replacements. This loop replaces the `k` concepts the model is
least sure of by their true labels:

```python
--8<-- "examples/quickstart.py:intervene"
```

```text
intervene on 0 concepts: accuracy 0.793
intervene on 2 concepts: accuracy 0.837
intervene on 4 concepts: accuracy 0.866
intervene on 8 concepts: accuracy 0.923
```

`cbm.aligned.targets(split)` holds the values the concept layer was trained to predict. With
`alignment: human` they are the dataset's annotations. With other alignments they are pseudo-labels
(for example CLIP similarities), not ground truth. The [`leakage`](components.md#evaluation)
evaluator runs random interventions of this kind for every run.

## Read the head

The head maps concepts to classes. For a linear head on `generation: scores`, its weight is
`(C, K)`: one row per class, one column per concept.

```python
--8<-- "examples/quickstart.py:weights"
```

```text
class_0: concept_18 (+0.91), concept_13 (+0.90), concept_02 (+0.83)
class_1: concept_21 (+0.99), concept_20 (+0.92), concept_08 (+0.77)
...
```

!!! warning "Only for linear heads on concept scores"
    `head.linear` exists on the `sparse` and `dense` heads only. `residual` and `mil` name the
    weight differently, and `generation: embeddings` gives several columns per concept.
    [#63](https://github.com/mel-oezkan/CBM-Design-Evaluation/issues/63) proposes one accessor
    for every head.

## Compare configs

Every run appends one row to the results file. Load the rows as a DataFrame and use
[`cbm_eval.analysis`][cbm_eval.analysis], as `cbm-eval analyze` does:

```python
--8<-- "examples/quickstart.py:analyze"
```

```text
 factor.predictor  metric.shift.test.acc.mean  metric.leakage.intervention.gain.mean
            dense                      0.7025                                0.10500
sparse(lam=0.001)                      0.7925                                0.15625
```

For many configs, build them with [`load_ablation(path).expand()`][cbm_eval.config.Ablation.expand]
and pass the list to [`pipeline.sweep`][cbm_eval.pipeline.sweep]. A sweep skips rows already in the
store and shares one `Context` between configs with the same data and backbone.
[`analysis.effects`][cbm_eval.analysis.effects] fits the effects model that
[Getting started](getting-started.md#analyze-the-results) shows.

## Run one stage on its own

Each stage is a class with one method ([stage interfaces](api/stages.md)). You can build it from
its class or from its registered name, and call that method yourself. Stages read data through a
`Context`, so build one first:

```python title="examples/one_stage.py"
--8<-- "examples/one_stage.py:stage"
```

```text
rules:  ['concept_03', 'concept_07', 'distractor_01', 'concept_11']
select: ['concept_07', 'concept_11']
```

The rule filter removes the duplicate `Concept 03`, the class name `class_2` and the name with too
many words. Stages return new objects and never change their inputs.

!!! note "A Context needs a full config"
    `PipelineBuilder(cfg).context()` needs a config with all six stages, although the `Context`
    only reads the dataset, backbone, device and cache path.
    [#64](https://github.com/mel-oezkan/CBM-Design-Evaluation/issues/64) proposes a way to build
    one from those alone.

`PipelineBuilder` also exposes the steps of `run` one by one. Use them to inspect the concepts
before training, or to train several heads on one concept set:

| Method | Does |
|---|---|
| [`context()`][cbm_eval.pipeline.PipelineBuilder.context] | builds the dataset, backbone and `Context` |
| [`concepts(ctx)`][cbm_eval.pipeline.PipelineBuilder.concepts] | runs discovery and the filter chain; returns a `ConceptSet` |
| [`train(ctx, concepts)`][cbm_eval.pipeline.PipelineBuilder.train] | fits alignment, concept layer and head; returns a `TrainedCBM` |
| [`evaluate(cbm, ctx)`][cbm_eval.pipeline.PipelineBuilder.evaluate] | runs the evaluators; returns the metric dict |
| [`load(ctx, run_dir)`][cbm_eval.pipeline.PipelineBuilder.load] | rebuilds a saved run's `TrainedCBM` without training |

Call `seed_everything(cfg.seed)` (from `cbm_eval.utils`) first if the result must equal that of
`run`.

## API reference

The reference pages list every public class and function with its signature and docstring:

- [Core](api/core.md): config, registries, pipeline, `Context`, data containers, `TrainedCBM`, results store.
- [Stage interfaces](api/stages.md): the base class of each stage.
- [Data, backbones, instances](api/inputs.md): the base classes of the inputs.
- [Evaluation and analysis](api/evaluation.md): the evaluator base class and `analysis`.
- [Component catalogue](components.md): every registered variant and its kwargs.
