# Architecture

--8<-- "CLAUDE.md:architecture"

## The pipeline map

Each box links to its entry in the [component catalogue](components.md), which lists the interface
and every registered variant with its kwargs. Stage numbers follow execution order: alignment runs
before generation because the concept layer is built from the aligned concepts.

<div class="cbm-map" markdown>
<div class="cbm-row cbm-inputs" markdown>
[**Experiment configs**<span>Anchors and ablations</span>](configs.md){.cbm-node}
[**Datasets**<span>Waterbirds, CUB, MetaShift, synthetic</span>](components.md#datasets){.cbm-node}
[**Backbones**<span>CLIP, DINOv2, ResNet, Inception, toy</span>](components.md#backbones){.cbm-node}
[**Instances**<span>Optional: image → bag of segments</span>](components.md#instance-sources){.cbm-node .cbm-optional}
</div>
<div class="cbm-arrow"></div>
<div class="cbm-row" markdown>
[**Pipeline builder**<span>Looks up each `name` in its registry · builds the shared `Context`</span>](api/core.md){.cbm-node .cbm-center}
</div>
<div class="cbm-arrow"></div>
<div class="cbm-band" markdown>
<div class="cbm-band-head"><b>Stage modules</b><span>one abstract base class each · <code>stages/base.py</code></span></div>
<div class="cbm-row cbm-stages" markdown>
[**Discovery**<span>LLM, VLM, ConceptNet, SAE, dataset, static</span><span class="cbm-io">ctx → ConceptSet</span>](components.md#discovery){.cbm-node .cbm-stage}
[**Filtering**<span>rules, CLIP, DINO, select (chained)</span><span class="cbm-io">ConceptSet → ConceptSet</span>](components.md#filtering){.cbm-node .cbm-stage}
[**Alignment**<span>CLIP, weights, DINO, human, segment</span><span class="cbm-io">ConceptSet → AlignedConcepts</span>](components.md#alignment){.cbm-node .cbm-stage}
[**Generation**<span>scores, logits, embeddings (CEM), BoC</span><span class="cbm-io">AlignedConcepts → ConceptLayer</span>](components.md#generation){.cbm-node .cbm-stage}
[**Predictor**<span>sparse, dense, residual, MIL</span><span class="cbm-io">dims → PredictorHead</span>](components.md#predictor){.cbm-node .cbm-stage}
[**Training**<span>independent, sequential, joint (+ weighted)</span><span class="cbm-io">layer + head → log</span>](components.md#training){.cbm-node .cbm-stage}
</div>
</div>
<div class="cbm-arrow"></div>
<div class="cbm-row" markdown>
[**Trained CBM**<span>Weights plus cached concept scores · `runs/<run_id>-s<seed>/`</span>](api/core.md){.cbm-node .cbm-center}
</div>
<div class="cbm-arrow"></div>
<div class="cbm-band" markdown>
<div class="cbm-band-head"><b>Evaluation suite</b><span>metrics prefixed by evaluator name</span></div>
<div class="cbm-row cbm-evals" markdown>
[**Shift**<span>WGA, effective robustness</span>](components.md#evaluation){.cbm-node .cbm-eval}
[**Leakage**<span>Interventions, probes</span>](components.md#evaluation){.cbm-node .cbm-eval}
[**Concepts**<span>Concept error, F1</span>](components.md#evaluation){.cbm-node .cbm-eval}
[**Localization**<span>Pointing game, part IoU, keypoints</span>](components.md#evaluation){.cbm-node .cbm-eval}
[**Faithfulness**<span>Deletion, insertion</span>](components.md#evaluation){.cbm-node .cbm-eval}
</div>
</div>
<div class="cbm-arrow"></div>
<div class="cbm-row cbm-tail" markdown>
[**Results store**<span>One JSONL row per run and seed</span>](api/evaluation.md){.cbm-node}
<div class="cbm-h-arrow">→</div>
[**Analysis**<span>Effects models, Pareto frontiers</span>](api/evaluation.md){.cbm-node}
</div>
<p class="cbm-map-note" markdown>Dashed borders mark optional parts. Without `instances:`, each image is one feature vector of shape (N, D). With it, every stage runs per instance on (N, M, D) plus a validity mask; see [instance bags](instance-bags.md).</p>
</div>

## What each stage hands to the next

The map shows which parts exist. This view adds the typed objects passed between the stages (arrow labels),
the config key or module behind each step (left), and which cached `Context` accessor each stage
reads (dashed). The steps are `PipelineBuilder.concepts()`, `.train()` and `.evaluate()` in
`pipeline.py`. Discovery and filtering only handle concept names (plus optional vectors); nothing is
trainable until alignment decides how each concept neuron gets its meaning.

<div class="cbm-figure">
--8<-- "docs/assets/architecture.svg"
</div>

[Open the data-flow diagram full size](assets/architecture.svg)

## Inside the trained CBM

The stages split the model as well as the pipeline. `ConceptLayer` in `stages/base.py` holds the
shared part (normalize, linear, activate, intervene); each generation variant only overrides
`represent()`. The head also receives the normalized features, so `residual` can bypass the
bottleneck and `mil` can compute attention over instances. Concept predictions stay in target
space, which is what lets interventions and the localization evaluators reuse the same layer on
patch features.

<div class="cbm-figure">
--8<-- "docs/assets/forward-pass.svg"
</div>

[Open the forward-pass diagram full size](assets/forward-pass.svg)

--8<-- "README.md:layout"
