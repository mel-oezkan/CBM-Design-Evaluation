# Designing an experiment

An experiment starts with a question about one or more design choices, such as "does joint training
leak more than sequential training?". You answer it with a sweep. The workflow is the same every
time:

1. **Anchor**: one YAML file that fully specifies a reference CBM.
2. **Ablation**: a second file that names the anchor and the factors to vary.
3. **Sweep**: `cbm-eval sweep` trains one model per variant and seed and appends one row each to
   the results file.
4. **Analysis**: `cbm-eval analyze` estimates the effect of each factor, with confidence intervals.

The examples below are complete files you can copy into `configs/`. Each one only combines
registered components. When your question needs a component that doesn't exist yet, see
[Contributing](contributing.md) and the [SEG-MIL-CBM example](instance-bags.md), which walks
through adding one. The full config format is in the [config reference](configs.md).

## Write an anchor

Start from the existing anchor closest to your question and change as little as possible. That
way, the comparison with the existing anchors stays controlled. For example, to ask whether a
label-free CBM matches the annotated one on CUB, copy `cub_human.yaml` and swap the concept stages
for the label-free ones from `waterbirds_lfcbm.yaml`:

```yaml title="configs/anchors/cub_lfcbm.yaml"
# Does a label-free CBM reach the accuracy of the annotated one on CUB?
# Same backbone and evaluation as cub_human; concepts come from Claude and are aligned by CLIP.
name: cub_lfcbm
dataset: {name: cub, root: data/CUB_200_2011}
backbone: {name: clip, model: ViT-B-16, pretrained: openai}
stages:
  discovery: {name: llm, per_class: 10}
  filtering:
    - {name: rules}
    - {name: clip, class_sim: 0.85, dedupe_sim: 0.9}
    - {name: select, k: 400, method: submodular}
  generation: {name: scores}
  alignment: {name: clip}
  predictor: {name: sparse, lam: 0.0007}
  training: {name: sequential}
evaluation:
  - {name: shift}
  - {name: leakage}
paths: {cache: .cache, runs: runs/cub, results: results/cub.jsonl}   # shared with cub_human
```

Some conventions:

- **Write the question in a header comment.** For a paper reproduction, also list where the
  anchor differs from the paper (see `configs/anchors/cub_koh2020.yaml`).
- **Share `paths.results` with the anchors you compare against.** Then one `analyze` call sees
  both designs, and every run keeps its own `run_id`.
- **Find the keys a variant accepts** in the [component catalogue](components.md). A misspelled key
  fails immediately with a `TypeError`, before any data is loaded.
- **Try changes with `--set` before writing them down:**
  `uv run cbm-eval run configs/anchors/cub_lfcbm.yaml --seeds 0 --set stages.filtering.2.k=200`.

## Vary one stage at a time

The default `ofat` mode (one factor at a time) runs the anchor plus one variant per factor value,
with everything else held at the anchor. It is the cheapest design and answers "what does this
choice do, all else equal?":

```yaml title="configs/ablations/cub_training.yaml"
# Does the training scheme change concept leakage on CUB?
anchor: ../anchors/cub_human.yaml
seeds: [0, 1, 2]
factors:
  stages.training:
    - {name: independent}
    - {name: joint, concept_weight: 0.01}
    - {name: joint, concept_weight: 1.0}
```

That is 4 configurations × 3 seeds = 12 runs. Always check the count before launching:

```bash
uv run cbm-eval sweep configs/ablations/cub_training.yaml --dry-run
uv run cbm-eval sweep configs/ablations/cub_training.yaml
uv run cbm-eval analyze --results results/cub.jsonl --metric leakage.intervention.gain \
    --query '`tags.ablation` == "cub_training"'
```

A factor value replaces the whole spec at its key. `{name: joint, concept_weight: 0.01}` doesn't
inherit the anchor's training kwargs, so anything else you need (e.g. `epochs`) goes into the
value. Every row is tagged with the ablation's file name, so `--query` can pick out one sweep from
a shared results file.

## Sweep a hyperparameter

To vary a single parameter instead of a whole stage, use a dotted key. List indices work too, which
is how you reach into the filtering chain:

```yaml title="configs/ablations/waterbirds_budget.yaml"
# How many concepts does the bottleneck need, and how sparse should the head be?
anchor: ../anchors/waterbirds_lfcbm.yaml
seeds: [0, 1, 2]
factors:
  stages.filtering.2.k: [10, 25, 100]          # the third filter in the chain (select)
  stages.predictor.lam: [0.0001, 0.002]
```

The concept budget only changes filtering and later stages, so the LLM concepts and the backbone
features are read from the cache. Only the training is repeated.

## Look for interactions

OFAT can't tell you whether the best choice for one stage depends on another. Use `mode: grid` to
run every combination. This example runs offline in a few seconds:

```yaml title="configs/ablations/synthetic_interactions.yaml"
# Does the best training scheme depend on what the predictor sees?
anchor: ../anchors/synthetic.yaml
mode: grid
seeds: [0, 1, 2]
factors:
  stages.generation: [{name: scores}, {name: embeddings}, {name: boc}]
  stages.training: [{name: sequential, epochs: 100}, {name: joint, epochs: 100}]
```

```bash
uv run cbm-eval sweep configs/ablations/synthetic_interactions.yaml     # 18 runs
uv run cbm-eval analyze --results results/synthetic.jsonl --metric shift.test.wga \
    --query '`tags.ablation` == "synthetic_interactions"'
```

```text
factor.generation        factor.training  metric.shift.test.wga.mean  ...std  ...count
              boc      joint(epochs=100)                    0.403913  0.031940       3
              boc sequential(epochs=100)                    0.302469  0.010692       3
       embeddings      joint(epochs=100)                    0.512346  0.021383       3
       embeddings sequential(epochs=100)                    0.493827  0.014259       3
           scores      joint(epochs=100)                    0.518751  0.009436       3
           scores sequential(epochs=100)                    0.521780  0.005648       3

metric.shift.test.wga  (n=18, reference levels: {'generation': 'scores', 'training': 'sequential(epochs=100)'})
  intercept                                                    +0.5008  [+0.4727, +0.5289]
  generation=boc                                               -0.1671  [-0.2015, -0.1326]
  generation=embeddings                                        -0.0172  [-0.0516, +0.0173]
  training=joint(epochs=100)                                   +0.0390  [+0.0109, +0.0671]
```

The effects model is additive, so it reports an average gain of +0.04 from joint training. The
per-configuration table shows where that gain comes from: joint training rescues the hard
bottleneck (`boc`, +0.10) and does nothing for `scores`. Read both parts.

Grids grow multiplicatively, so keep them to the two or three factors in question. In a grid, a
dotted key is applied to every combination, including specs that don't accept it. Put such a
parameter inside the spec instead (see the [config reference](configs.md)).

## Compare backbones

The `clip` alignment and the `clip`/`select` filters need text embeddings. Non-CLIP backbones
therefore keep CLIP as a separate `teacher:`, which you set with `overrides:`. Overrides are applied
to the anchor before the factors, so the anchor run of this sweep also gets the teacher:

```yaml title="configs/ablations/waterbirds_backbones.yaml"
anchor: ../anchors/waterbirds_lfcbm.yaml
mode: ofat
seeds: [0, 1, 2]
overrides:
  teacher: {name: clip, model: ViT-B-16, pretrained: openai}
factors:
  backbone:
    - {name: dinov2, model: dinov2_vitb14}
    - {name: resnet, model: resnet50}
```

Each new backbone extracts features once. After that, every run on the same dataset and backbone
trains on cached tensors.

## Add a test domain to finished runs

To measure robustness on a shifted test set, add it under `eval_datasets:` and include its alias in
the evaluators' `splits`. The models you already trained don't need retraining. Put the new
`evaluation:` and `eval_datasets:` into the anchor and re-score the saved runs:

```yaml
eval_datasets:
  paintings: {name: cub_paintings, root: data/cub_paintings}   # a test-only dataset you register
evaluation:
  - {name: shift, splits: [val, test, paintings]}
  - {name: concepts, splits: [test, paintings]}
```

```bash
uv run cbm-eval evaluate runs/cub/<run_id>-s0 runs/cub/<run_id>-s1 runs/cub/<run_id>-s2 \
    --config configs/anchors/cub_human.yaml
```

No test-only dataset ships yet. A new one follows the dataset contract in
[Contributing](contributing.md): it sets `splits = ("test",)` and uses the training dataset's class
and concept names.

## Analyze across experiments

All rows in one results file can be analyzed together. `--factors` restricts the effects model to
the factors you name, `--frontier` adds a Pareto frontier over a second metric (both maximized),
and `--group` fits a random intercept per group, e.g. per dataset (needs the `analysis` extra):

```bash
uv run cbm-eval analyze --results results/waterbirds.jsonl \
    --metric shift.test.wga --factors discovery alignment --frontier leakage.intervention.gain
```

For anything else, load the flat frame and use [`cbm_eval.analysis`][cbm_eval.analysis]:

```python
from cbm_eval import analysis
from cbm_eval.results import ResultsStore

df = ResultsStore("results/waterbirds.jsonl").load()
df = df[df["tags.ablation"] == "waterbirds_budget"]
print(analysis.aggregate(df, ["metric.shift.test.wga"], by=["factor.filtering", "factor.predictor"]))
```

## Reproducing a paper

A reproduction is an anchor (plus ablations) built from registered components, never a separate
pipeline. The recipe:

1. Map every part of the method onto a stage and look for a registered variant (`cbm-eval list`).
2. Where the paper differs from an existing variant, add an opt-in kwarg whose default keeps the
   current behaviour, or register a new variant. Never change a default: run ids hash the config,
   not the code, so old results would silently stop matching.
3. Build the anchor on the synthetic dataset first, so it runs offline in seconds.
4. Write the real anchor and list the remaining differences from the paper in its header comment.
5. Add an ablation that removes one paper-specific choice at a time.

Two worked examples:

- [SEG-MIL-CBM](instance-bags.md) needed new components and one architecture extension (instance
  bags).
- [Koh et al. 2020](koh2020_reproduction.md) needed only opt-in kwargs and weighted training
  variants, plus backbone fine-tuning.
