# CBM-Design-Evaluation
<!-- --8<-- [start:intro] -->

Research code for a controlled design study of concept bottleneck models (CBMs). Published CBM
variants each change several design choices at once, so it is unclear which choice causes a reported
gain. Here a CBM is assembled from six interchangeable stages (concept discovery, filtering,
alignment, generation, predictor, training), and configs that differ in one stage differ *only* in
that stage: same features, splits, seeds and metrics. Sweeps over the stages measure the effect of
each choice on robustness, concept quality, leakage and localization, with confidence intervals.
<!-- --8<-- [end:intro] -->

![Data flow of one run: config, pipeline builder, stages, trained CBM, evaluators, results](docs/assets/architecture.svg)

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mel-oezkan/CBM-Design-Evaluation && cd CBM-Design-Evaluation
uv sync                  # core + tests
uv sync --all-extras     # + CLIP, Grounding DINO / SAM, Claude, statsmodels, W&B, Modal
```

## Quickstart

The synthetic anchor needs no data, weights or network access:

```bash
uv run cbm-eval run configs/anchors/synthetic.yaml --seeds 0 1 2      # train + evaluate one config
uv run cbm-eval sweep configs/ablations/synthetic_stages.yaml          # 45-run ablation, < 1 min on CPU
uv run cbm-eval analyze --results results/synthetic.jsonl --metric shift.test.wga
```

Results are appended to `results/synthetic.jsonl` (one row per run and seed); `analyze` reports
per-config means and the effect of each design choice with confidence intervals. Run
`uv run cbm-eval list` to see every registered component.

## What's inside

| Stage | Variants |
|---|---|
| Discovery | `llm`, `vlm`, `kb`, `sae`, `dataset`, `static` |
| Filtering | `rules`, `clip`, `dino`, `select` (chainable) |
| Alignment | `clip`, `weights`, `dino`, `human`, `segment_clip` |
| Generation | `scores`, `logits`, `embeddings` (CEM), `boc` |
| Predictor | `sparse`, `dense`, `residual`, `mil` |
| Training | `independent`, `sequential`, `joint` (+ `*_weighted`) |

- **Datasets:** Waterbirds, CUB, MetaShift, synthetic
- **Backbones:** OpenCLIP, DINOv2, ResNet, Inception-v3, toy
- **Evaluation:** distribution shift (worst-group accuracy), concept accuracy, leakage,
  localization, part localization, faithfulness

Ready-made configs live in [`configs/anchors/`](configs/anchors) (including the
[Koh et al. 2020 reproduction](docs/koh2020_reproduction.md) and SEG-MIL-CBM) and
[`configs/ablations/`](configs/ablations).

## Documentation

The full documentation is a docs site built from [`docs/`](docs):

```bash
uv run --group docs properdocs serve     # http://127.0.0.1:8000
```

- [Getting started](docs/getting-started.md): a first run, a sweep and its analysis, then real data
- [How a run works](docs/how-it-works.md) and [Training](docs/training.md): what each stage does, step by step
- [Pipeline map](docs/architecture.md): the code layout and what the stages pass to each other
- [Designing an experiment](docs/experiments.md): example anchors and ablations; worked examples
  of adding a paper ([SEG-MIL-CBM](docs/instance-bags.md), [Koh et al. 2020](docs/koh2020_reproduction.md))
- [Config reference](docs/configs.md): the config format, `run_id`, test-only domains
- [Data](docs/data.md), [Metrics](docs/metrics.md) and [Part localization](docs/part-localization.md)

## Contributing

Every variant is a registered class, and config kwargs are its constructor kwargs. See
[Contributing](docs/contributing.md) for how to add a component; the tests run fully offline:

```bash
uv run pytest -q
```

<!-- --8<-- [start:status] -->
## Status

This is research code under active development. The whole pipeline and every stage variant are
tested end-to-end on the synthetic dataset and toy backbone. The tests also cover the image-file
path: a fake Waterbirds folder, an untrained ResNet-18, and stub LLM/VLM clients.

The CUB loader and the Inception-v3 backbone have been run on the real dataset (frozen-backbone
runs of the Koh et al. 2020 reproduction). **Not yet run against real data or pretrained
weights:** OpenCLIP and DINOv2 feature extraction (including ViT patch tokens), the Waterbirds and
MetaShift loaders, live Claude/ConceptNet calls, Grounding DINO scoring, and end-to-end backbone
fine-tuning. Expect small fixes on first contact, and please open an issue when you hit one.
<!-- --8<-- [end:status] -->
