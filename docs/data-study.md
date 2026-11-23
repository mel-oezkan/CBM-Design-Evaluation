# Planned: data-quality study

!!! warning "Not implemented yet"
    This page records a plan made on 2026-10-20. None of the modules, kwargs or tools it names
    exist yet. The [to-do list](#to-do) at the end tracks the work, as does the GitHub tracker
    issue linked there.

The question behind the plan: **how much does the training data shape what a CBM learns?** That
covers how concept labels were annotated, how noisy they are, which concept vocabulary is used and
how well each concept is grounded in the image. This repo can answer the question in a controlled
way, because every data choice can become one config key while everything else stays fixed.

The full plan, with charts of the paper's results, a pipeline map and a mockup of the annotation
screen, is a standalone page: [open the plan with figures](assets/data-study-plan.html).

## Why: the OA-CBM paper

The plan started from *Rounded or Streamlined Head? Bridging Concept Bottleneck Models and
Attribute-Described Object Parts* (Liu et al., CVPR 2026). OA-CBM redefines concepts as
part–attribute pairs ("head: tubular nostrils atop the bill"), annotates two part-level grounding
datasets (PartAttrCUB on the same CUB70 masks this repo reads, and PartAttrImageNet), and adds two
learned modules:

- **Cost aggregation (CA)** reduces semantic inconsistency, where a concept is scored on the wrong
  part of the object.
- **Hierarchical clustering (HC)** produces an object mask, which reduces object inconsistency,
  where a concept is scored on the wrong object.

The paper credits its grounding gains to the finer-grained annotations. However, between the
baselines and OA-CBM it changes five things at once (concept format, grounding training data, text
encoder, CA and HC) and ablates only CA and HC. The data-related changes are never isolated. The
cleanest isolated result is the object mask: under the Pred-All protocol on PartAttrCUB, adding it
lifts h-IoU from 6.6 to 20.8.

## The study

The recommended framing is **measure first, then fix**:

1. Measure which data levers move accuracy, intervention gains, leakage and grounding.
2. Fix only the most damaging lever (for example by relabelling the worst concepts) and show the
   gain with the same measurements.

| # | Research question |
|---|---|
| RQ1 | How do concept-label noise, granularity (per image vs per class) and annotator certainty affect accuracy, concept accuracy and intervention gains? |
| RQ2 | At a fixed model, does the concept format matter: human attributes, object-level LLM concepts, part–attribute LLM concepts, SAE concepts? |
| RQ3 | What do keypoints, part masks or object masks add to grounding, and does better grounding transfer to accuracy or robustness? |
| RQ4 | How many concept-labelled images are needed before the results stop improving? |
| RQ5 | Do the backbone and training scheme change how sensitive a CBM is to data quality? |

## Data levers

"Yes" means a variant or kwarg already exists. "No" means a new kwarg or registered variant is
needed. "Design" means it doesn't fit the current interfaces and needs a design issue first, as
on [Planned models](models.md).

| Lever | Stage | Exists? |
|---|---|---|
| Label granularity (per image, per class, Koh vote) | dataset `cub`: `class_level_concepts`, `majority_vote` | Yes |
| Vocabulary source (human, `llm`, `vlm`, `sae`) | discovery | Yes |
| Vocabulary cleaning (dedupe, selection) | filtering `clip`, `select` | Yes |
| Injected concept-label noise (flip rate) | dataset kwarg | No |
| Annotator certainty (CUB levels 2–4) | dataset kwarg | No |
| Visibility mask on the concept loss | `Sample` field + training variant overriding `_concept_loss` | No |
| Annotation budget (share of images with concept labels) | dataset kwarg | No |
| Part–attribute vocabulary | discovery `part_llm` (two-step prompt, `meta["part"]`) | No |
| Object mask (none, oracle from CUB70, predicted by Grounding DINO + SAM) | instance source kwarg, masking `Bag.mask` | No |
| Relabelled CUB from the annotation platform | dataset kwarg `relabel:` | No |
| Grounding for any vocabulary, matched by part instead of name | part evaluators, kwarg `match:` (`name` or `part`) | No |
| Metrics split by data quality (part visible, annotator certain) | evaluator `slices` | No |
| Per-concept data audits (agreement, learnability, part signal, label noise) | new family + `cbm-eval audit` | Design |

Every new kwarg keeps the current behaviour as its default, so existing run ids and results stay
valid (see the reproducibility invariants in [Contributing](contributing.md)).

### What the CUB loader currently discards

These are the concrete starting points in [`data/cub.py`](https://github.com/mel-oezkan/CBM-Design-Evaluation/blob/main/src/cbm_eval/data/cub.py):

- **Annotator certainty.** Only "1 = not visible" is used, and only during majority voting. A
  guess counts the same as a confident label.
- **Per-image labels.** With `class_level_concepts: true`, every image gets its class's vector.
- **Part visibility.** Positives whose part is hidden still train the concept.
- **Coarse part mapping.** `malar`, `eyebrow`, `eyeline` and `masked` map to crown/forehead (see
  [issue #57](https://github.com/mel-oezkan/CBM-Design-Evaluation/issues/57)).

## Annotation platform

Relabelling by hand only works if it is targeted and cheap on disk. A full re-annotation of CUB
would be 3.7M labels (11,788 images × 312 attributes). Verifying the roughly 5,000 items the audits
flag would take a few hours.

The design keeps pixels remote and labels local:

1. **Task export (repo).** Audits run on Modal and pick the top-N items by priority, written to
   `tasks.jsonl` (image, part, attribute, current label, reason).
2. **Crop service (Modal).** Reads images and keypoints from the `cbm-koh2020` volume and cuts a
   224 px crop around the part, about 15 KB each. It runs either as a batch job that returns a zip,
   or as a web endpoint with `requires_proxy_auth=True`, since Modal web functions are public by
   default and CUB images must not be served openly.
3. **Annotation app (local).** A small FastAPI app with a grid verification screen: one attribute,
   many crops, keys 1–4 for present / absent / not visible / unsure. Labels go to SQLite; the crop
   cache has a size cap and is cleared after each batch. One batch of 2,000 crops is about 30 MB.
4. **Label snapshot (repo).** Exported as `labels/cub-relabel-<sha8>.jsonl` and never edited
   afterwards. One row per judgment, in CUB's own certainty terms:

    ```json
    {"image_id": 1234, "part": "head", "attribute": "has_head_pattern::malar",
     "label": "absent", "certainty": 4, "annotator": "a1", "task": "malar-probe-disagree",
     "gold": false, "ts": "2026-10-20T14:03:11Z", "tool": "0.1.0"}
    ```

5. **Back into experiments.** `dataset: {name: cub, relabel: labels/cub-relabel-<sha8>.jsonl}`,
   with the file's hash in `CUB.cache_key()` and a case in `tests/test_cache_keys.py`. The run id
   only hashes the config, so a labels file edited in place would silently reuse old results. This
   is why snapshots are immutable and named by hash.

Hidden gold items and a second annotator on a 10% subset give the agreement numbers (self-agreement,
Cohen's κ) that a data paper has to report.

For mask or box correction, an existing tool (CVAT, with SAM-assisted masks) fits better than a
custom app. It is only needed if part masks themselves get corrected for the grounding
evaluation.

## Open design questions

These should each be a GitHub issue (labelled `needs-decision`) before any code is written:

- **A home for data audits.** A new registry and `cbm-eval audit` command, or analysis functions
  in `analysis.py`? Audits read only the `Context` and train nothing, so they fit neither the
  evaluators nor the stages.
- **Per-entry concept masks.** A visibility mask needs `Sample` and `Batch` to carry a `(N, K)`
  mask next to the concept labels. This is a change to the shared containers.
- **The task export command** (`cbm-eval annotate export`) and where the annotation app lives:
  in this repo under `tools/`, or as a separate repo.
- **Out of scope unless decided otherwise:** OA-CBM's learned cost-aggregation module, the Llama3
  text encoder and its patch-guided score. None of these fit the current stage interfaces.

## To do

Ordered so that each step makes the next measurable. Tracked in the GitHub tracker issue
[#50](https://github.com/mel-oezkan/CBM-Design-Evaluation/issues/50).

| # | Task | Depends on | Status |
|---|---|---|---|
| 1 | Part-aware grounding: discovery records `meta["part"]`; part evaluators get `match:` (`name` or `part`) | — | Not started |
| 2 | `part_llm` discovery (parts, then attributes per part; fixed CUB parts via `parts:`) | 1 | Not started |
| 3 | CUB data levers: `label_noise`, `label_source` (certainty), `concept_budget`, with tests | — | Not started |
| 4 | Visibility-masked concept loss (needs the concept-mask design decision) | design issue | Not started |
| 5 | Object-mask instance source: `object:` (`none`, `oracle` or `predicted`) | — | Not started |
| 6 | `slices` evaluator | 3 | Not started |
| 7 | Data audits (needs the audit design decision) | design issue | Not started |
| 8 | Pilot on the synthetic dataset, where the true effect of noise is known | 3, 6 | Not started |
| 9 | Ablation configs under `configs/ablations/` and sweeps on Modal | 1–6 | Not started |
| 10 | Annotation pilot: 500 tasks, 3 attributes, crops as a zip, grid page + SQLite | 7 | Not started |
| 11 | `relabel:` kwarg with hash in the cache key; original vs relabelled ablation | 10 | Not started |
| 12 | Annotation quality: gold items, repeats, a second annotator on 10% | 10 | Not started |
