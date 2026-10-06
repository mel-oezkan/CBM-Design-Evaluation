# cbm-eval

--8<-- "README.md:intro"

## The problem

A concept bottleneck model (CBM) classifies an image in two steps. It first predicts a set of
human-readable concepts ("webbed feet", "water background"), then predicts the label from those
concepts alone. Because the label has to pass through the concepts, you can read off why the model
decided what it did, and you can correct a wrong concept at test time and watch the prediction
change.

Many CBM variants have been published since the original (Koh et al., 2020), and each paper changes
several things at once. Label-free CBM, for example, swaps the annotated concepts for LLM-generated
ones, the human labels for CLIP similarities, and the dense head for a sparse one, all in the same
paper, with its own backbone and datasets. When it reports a gain, you can't tell which of those
changes caused it, or whether the gain survives on another dataset.

## What this codebase does

It splits a CBM into six stages and offers several interchangeable implementations ("variants") of
each one:

| Stage | Question it answers | Variants |
|---|---|---|
| **Discovery** | Where do candidate concepts come from? | an LLM, a VLM, ConceptNet, a sparse autoencoder, the dataset's annotations, a fixed list |
| **Filtering** | Which candidates are kept? | text rules, CLIP-space filters, a detector, top-k selection (chained) |
| **Alignment** | How does each concept neuron learn what its concept means? | CLIP pseudo-labels, detector pseudo-labels, human annotations, fixed text-embedding weights |
| **Generation** | What does the label predictor see for each concept? | a score, a logit, an embedding (CEM), a hard 0/1 bit |
| **Predictor** | How are concepts turned into a label? | sparse linear, dense linear, linear plus a residual, attention MIL |
| **Training** | In what order are the two halves fitted? | independent, sequential, joint |

A CBM is one choice per stage, written as a YAML config. Every combination trains on the same
cached backbone features with the same splits, seeds and evaluators, so two configs that differ in
one stage differ *only* in that stage. A sweep varies stages in a controlled way, and the analysis
reports how much each choice changes each metric, with confidence intervals.

Several published designs are configs here. Each one shares most of its stages with the others:

| Design | Discovery | Alignment | Generation | Predictor | Training |
|---|---|---|---|---|---|
| Koh et al. 2020 | `dataset` | `human` | `logits` | `dense` | all three |
| Label-free CBM | `llm` | `clip` | `scores` | `sparse` | `sequential` |
| SEG-MIL-CBM | `llm` | `segment_clip` | `scores` | `mil` | `joint` |
| LaBo-style | `llm` | `weights` | `scores` | `dense` | `sequential` |
| CEM | any | with targets | `embeddings` | `dense` | `joint` |
| PCBM-h | any | any | `scores` | `residual` | `sequential` |

The first three are anchors under `configs/anchors/` (`cub_koh2020*.yaml`, `waterbirds_lfcbm.yaml`,
`waterbirds_segmil.yaml`); the Koh et al. independent model uses `scores` instead of `logits`, as
the paper does. The other three have no anchor: they reuse only the stage that makes each method
distinctive (frozen text-embedding concept neurons for LaBo, concept embeddings for CEM, a residual
path for post-hoc CBMs), and `waterbirds_bottleneck.yaml` sweeps the last two.

## What you get out

Each run trains one CBM and scores it on:

- **Robustness under shift**: accuracy and worst-group accuracy on test sets where a spurious
  feature (e.g. the background in Waterbirds) no longer predicts the label.
- **Concept quality**: how well predicted concepts match human annotations.
- **Leakage**: whether the predictor really relies on concept meaning, measured by test-time
  interventions and probes for information that shouldn't be in the concepts.
- **Localization and faithfulness**: whether a concept fires on the right image region, and whether
  the segments the model ranks highest are the ones its prediction depends on.

Every run appends one row to a results file. The analysis then answers questions like "does LLM
discovery help worst-group accuracy compared to annotated concepts?", "how much more do joint models
leak than sequential ones?" or "which filtering step actually matters?":

```text
metric.shift.test.wga  (n=45, reference levels: {...})
  intercept                                       +0.5218  [+0.4988, +0.5447]
  discovery=sae(epochs=30,k=6,n_latents=64)       -0.3983  [-0.4308, -0.3658]
  generation=boc                                  -0.2193  [-0.2518, -0.1868]
  predictor=dense                                 -0.1259  [-0.1584, -0.0935]
  ...
```

Each line is the change in worst-group accuracy from switching one stage away from the reference
design, with a 95% confidence interval over seeds.

## What to use it for

- **Answering a design question** about CBMs with an effect size rather than one comparison.
- **Reproducing a paper as a config**, then removing its choices one at a time to see which ones
  matter (see the [Koh et al. 2020 reproduction](koh2020_reproduction.md)).
- **Testing a new method fairly.** Add it as a variant of one stage, and it is compared against
  every existing variant under identical conditions (see the
  [SEG-MIL-CBM example](instance-bags.md)).

<div class="grid cards" markdown>

-   **Getting started**

    ---

    [Install, train one CBM and run a small study](getting-started.md) on the offline synthetic
    dataset in a few minutes.

-   **Architecture**

    ---

    [How a run works](how-it-works.md) step by step, [how training works](training.md), and the
    [component catalogue](components.md) of every variant and its options.

-   **Experiments**

    ---

    [Turning a design question into a sweep](experiments.md), and worked examples of adding a
    paper: [SEG-MIL-CBM](instance-bags.md) and [Koh et al. 2020](koh2020_reproduction.md).

-   **Extending**

    ---

    [Adding a component](contributing.md) without breaking the architecture: one registered
    subclass per variant, config kwargs as constructor kwargs, and an offline test.

</div>

--8<-- "README.md:status"
