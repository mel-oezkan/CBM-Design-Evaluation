# Example: implementing SEG-MIL-CBM

This page shows how a paper that the pipeline couldn't express yet was added to it. The paper is
SEG-MIL-CBM ([arXiv 2510.04180](https://arxiv.org/abs/2510.04180)), which classifies an image from a
*bag* of concept-guided segments instead of one feature vector. Follow the same steps for your own
method. The result is a set of registered components plus three configs:

| File | Purpose |
|---|---|
| `configs/anchors/synthetic_segmil.yaml` | Offline version on the synthetic data, runs in about a second |
| `configs/anchors/waterbirds_segmil.yaml` | The paper's method on Waterbirds |
| `configs/anchors/cub_segmil.yaml` | The paper's method on CUB, for the segment-faithfulness numbers |
| `configs/ablations/waterbirds_segmil.yaml` | Removes one SEG-MIL choice at a time |

## 1. Map the method onto the stages

Start by writing down each step of the method next to the stage that would own it, and check
`uv run cbm-eval list` for a variant that already does it:

| Paper step | Stage | Existed? |
|---|---|---|
| LLM concepts, filtered by CLIP | discovery `llm`, filtering chain | Yes (Label-free CBM) |
| Top-K concepts per image → Grounding DINO boxes → SAM masks; one feature per segment | none: changes the shape of every input | **No** |
| Frozen CLIP targets per segment, `q_k(s_i)` = softmax of cosines | alignment | **No** |
| Concept layer applied to each segment | generation `scores` | Yes |
| `L = CE + 0.1 · (−cos)` | training `joint` with `concept_weight` | Partly: new concept loss |
| Class-conditioned attention MIL with area weights | predictor | **No** |
| Deletion / insertion faithfulness over segments | evaluator | **No** |

Most gaps are new variants of existing stages. One isn't: segments change what an *input* is for
every stage, from `(N, D)` to `(N, M, D)`.

## 2. Separate variants from architecture changes

A new alignment, predictor or evaluator is one registered class each and touches nothing else. A
change that every stage has to understand is an architecture change. The rule for those is to stop
and discuss the design before building around it. Bags were added once, in a general form, so that
other bag-based methods can reuse them:

- **Containers**: `Bag` (a validity mask plus segment areas) and `InstanceSplit` in
  `structures.py`. Stages use `Bag.rows/mean/max` instead of assuming 2-D tensors.
- **A new registry**, `INSTANCES`, with the `InstanceSource` base class in `instances/base.py`. It
  has capability flags rather than special cases: `needs_concepts` (segmentation depends on the
  filtered concepts, so it runs after filtering) and `needs_patches`.
- **A cached accessor**, `ctx.instances(split)`, so that stages never segment or encode anything
  themselves.
- **A capability flag on the predictor**: `Predictor.supports_bags` (default `False`).
  `PipelineBuilder` refuses a bag config with a head that can't pool and names the fix:

    ```text
    ValueError: `instances: patches` gives each image a bag of instances; use a bag-aware
    predictor such as {name: mil}, not 'dense'
    ```

- **A top-level `instances:` config key**. `ExperimentConfig.to_dict()` leaves it out when it is
  unset, so the run ids of every existing config stay the same.

With that in place, everything else was a regular component, and existing variants such as the
`clip`, `human` and `dino` alignments work on bags without changes: their image-level targets
supervise the max over instances.

## 3. Write the components

Each component implements exactly its base class's abstract method and takes its hyperparameters
as explicit constructor kwargs, which become its config keys.

**Instance sources** (`instances/segments.py`): `grounded_sam` ranks each image's concepts by
teacher similarity, asks Grounding DINO for boxes and SAM for masks, and encodes each masked
segment with the frozen backbone. It declares that it depends on the concepts, and it loads its
models only on first use, so building the pipeline stays cheap:

```python
@INSTANCES.register("grounded_sam")
class GroundedSAM(SegmentSource):
    name = "grounded_sam"
    needs_concepts = True        # runs after filtering; the concept set is part of the cache key

    def __init__(self, top_k: int | None = 10, boxes_per_concept: int = 2, detector: Any = None,
                 sam: Any = None, **kw):
        ...                      # model settings only; tests pass stub detector/SAM objects here

    def propose(self, images, index, ctx):
        ...                      # Grounding DINO + SAM are created here, on first use
```

The sources that serve as controls came almost for free: `grounded_boxes` (the same boxes without
SAM), `sam_grid` (concept-agnostic SAM masks) and `patches` (ViT patch tokens, no segmentation).

**Alignment** (`stages/alignment.py`): `segment_clip` reads the cached bags through `Context` and
returns per-instance targets. Padding instances get zero targets:

```python
@ALIGNMENT.register("segment_clip")
class SegmentCLIPAlignment(Alignment):
    def __init__(self, targets: str = "softmax", temperature: float = 0.01): ...

    def fit(self, concepts, ctx):
        text = F.normalize(ctx.encode_text(concepts.names).float(), dim=1)

        def targets(split):                              # (N, M, K)
            inst = ctx.instances(split)
            sim = F.normalize(inst.teacher_features.float(), dim=-1) @ text.T
            return (sim / self.temperature).softmax(-1) * inst.mask[..., None]

        return AlignedConcepts(concepts, target_type="continuous", targets_fn=targets)
```

(Abridged: the real class also supports `standardized` and `cosine` targets, and it raises
actionable errors when `instances:` is missing.)

**Predictor** (`stages/predictor.py`): `mil` sets the capability flag and returns a `MILHead`,
which pools per-segment class evidence with attention and area weights. The head also exposes
`contributions()`, each segment's share of the logit, which the faithfulness evaluator reads:

```python
@PREDICTOR.register("mil")
class MIL(Predictor):
    supports_bags = True

    def __init__(self, pooling: str = "attention", attn_dim: int = 128, tau: float = 1.0,
                 area_gamma: float = 0.0, area_min: float = 0.01): ...

    def build(self, rep_dim, num_classes, feat_dim):
        return MILHead(rep_dim, num_classes, feat_dim, self.pooling, self.attn_dim, self.tau,
                       self.area_gamma, self.area_min)
```

**Training**: the paper's loss is the existing `joint` training with a new concept loss,
`concept_loss: cosine` (1 − cosine between each segment's concept vector and its target). No new
training loop was needed.

**Evaluator** (`evaluation/faithfulness.py`): `faithfulness` ranks segments by their contribution,
deletes or inserts them, and returns unprefixed keys (`dauc`, `iauc`, ...); the pipeline adds the
`faithfulness.` prefix. It returns nothing for heads without `contributions()`.

Every component has an offline test in `tests/test_segmil.py`. Grounding DINO and SAM are replaced
by stub objects passed through the `detector=` and `sam=` constructor args, so the tests never
download weights. The segment caches are listed in `tests/test_cache_keys.py`.

## 4. Run it offline first

Before touching real data, build the same design on the synthetic dataset. Patch tokens stand in
for segments, so no segmentation models are needed:

```yaml title="configs/anchors/synthetic_segmil.yaml"
name: synthetic_segmil
dataset: {name: synthetic}
backbone: {name: toy}
instances: {name: patches}
stages:
  discovery: {name: dataset}
  filtering: [{name: rules}]
  generation: {name: scores}
  alignment: {name: segment_clip}
  predictor: {name: mil, area_gamma: 0.25}
  training: {name: joint, concept_weight: 0.1, concept_loss: cosine, epochs: 100}
evaluation:
  - {name: shift}
  - {name: leakage}
  - {name: localization}
  - {name: faithfulness}
paths: {cache: .cache, runs: runs/synthetic, results: results/synthetic.jsonl}
```

```bash
uv run cbm-eval run configs/anchors/synthetic_segmil.yaml --seeds 0
```

```text
  "instances.per_image": 16.0,
  "shift.test.acc": 0.8487,
  "faithfulness.dauc": 0.2314,           # vs. 0.5887 for a random segment ranking
  "faithfulness.iauc": 0.8507,           # vs. 0.6095
  ...
```

Deleting the segments the model ranks highest hurts far more than deleting random ones, which is
the sanity check that the head, its contributions and the evaluator fit together.

## 5. Write the real anchor

The Waterbirds anchor reuses the label-free concept stages of `waterbirds_lfcbm.yaml`, so the two
differ only in the SEG-MIL parts. It also shares that anchor's results file, so a single `analyze`
call compares them:

```yaml title="configs/anchors/waterbirds_segmil.yaml"
# SEG-MIL-CBM (Eisenberg et al., arXiv 2510.04180) on Waterbirds: LF-CBM concepts, Grounding DINO
# boxes for each image's top-10 concepts, SAM masks, per-segment CLIP targets, attention-MIL head.
# Shares its results file with waterbirds_lfcbm so both land in one analysis.
name: waterbirds_segmil
dataset: {name: waterbirds, root: data/waterbirds}
backbone: {name: clip, model: ViT-B-16, pretrained: openai}
instances:
  name: grounded_sam
  top_k: 10
  boxes_per_concept: 2
  max_segments: 16
  min_pixels: 256
  max_frac: 0.9
  merge_iou: 0.8
  detector: {model: IDEA-Research/grounding-dino-tiny, box_threshold: 0.25, text_threshold: 0.25}
  sam: {model: facebook/sam-vit-base}
stages:
  discovery: {name: llm, per_class: 10}
  filtering:
    - {name: rules}
    - {name: clip, class_sim: 0.85, dedupe_sim: 0.9}
    - {name: select, k: 50, method: submodular}
  generation: {name: scores}
  alignment: {name: segment_clip, targets: softmax, temperature: 0.01}
  predictor: {name: mil, pooling: attention, attn_dim: 128, tau: 1.0, area_gamma: 0.25}
  training: {name: joint, concept_weight: 0.1, concept_loss: cosine, device: auto}
evaluation:
  - {name: shift}
  - {name: leakage}
  - {name: faithfulness, n_images: 3000}
paths: {cache: .cache, runs: runs/waterbirds, results: results/waterbirds.jsonl}
```

It needs the `clip`, `hf` and `llm` extras and Claude credentials. A GPU helps: segmentation and
encoding run on the top-level `device`, and `training.device: auto` trains on a GPU when there is
one.

How the anchor maps onto the paper:

| Paper component | Config |
|---|---|
| Top-K concepts per image → Grounding DINO boxes → SAM masks | `instances: {name: grounded_sam, top_k: 10}` |
| Segment features `h_i` (masked crop through the frozen backbone) | `instances.encode: crop` (or `pool`: patch tokens averaged under the mask) |
| Frozen CLIP targets `q_k(s_i)` = softmax of cosines | `alignment: {name: segment_clip, targets: softmax, temperature: 0.01}` |
| `L = CE + 0.1 · (−cos)` | `training: {name: joint, concept_weight: 0.1, concept_loss: cosine}` |
| Class-conditioned attention MIL with area weights | `predictor: {name: mil, pooling: attention, area_gamma: 0.25}` |
| Deletion / insertion faithfulness | `evaluation: [{name: faithfulness}]` |

## 6. Ablate the method

The ablation asks which part of SEG-MIL carries the worst-group gain, removing one choice at a time:

```yaml title="configs/ablations/waterbirds_segmil.yaml"
anchor: ../anchors/waterbirds_segmil.yaml
mode: ofat
seeds: [0, 1, 2]
factors:
  instances:                                   # does concept-guided segmentation matter?
    - {name: patches}                          # ViT patch tokens, no segmentation
    - {name: grounded_boxes, top_k: 10, boxes_per_concept: 2, max_segments: 16}   # no SAM
    - {name: sam_grid, points_per_side: 8, max_segments: 16}                       # concept-agnostic
  instances.encode: [pool]                     # pooled patch tokens instead of one forward per segment
  stages.predictor.pooling: [mean, max]        # does attention matter?
  stages.predictor.area_gamma: [0.0]
  stages.training.concept_weight: [0.0, 1.0]   # how much does concept supervision constrain the head?
  stages.alignment.targets: [standardized]
```

```bash
uv run cbm-eval sweep configs/ablations/waterbirds_segmil.yaml --dry-run    # 33 runs
```

Segmentation is the expensive step, so it is cached in two layers. Masks go to
`.cache/segments/` and depend on the dataset, the segmenter settings and, for grounded sources, the
filtered concept set, but not on the backbone. Segment features go to `.cache/instances/`, one
entry per backbone. Varying the head, loss or alignment therefore reuses both caches. Any change to
discovery or filtering re-segments the dataset, except for `sam_grid`, which doesn't depend on the
concepts.

## 7. Record the gaps

Whatever the anchor can't match goes into its header comment and onto this page:

- The backbone stays frozen (no warm-up phase).
- Faithfulness zeroes cached segment embeddings rather than re-encoding the edited images.
- Many hyperparameters are only in the paper's appendix, so `top_k`, `boxes_per_concept`, the
  detector thresholds and `tau` are our defaults.
- With the scale-free cosine loss, intervention metrics that write targets into the concept layer
  are not meaningful.
