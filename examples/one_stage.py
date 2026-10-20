"""Use one stage on its own: run filters on a hand-written ConceptSet, outside a full run.

Runs offline in seconds on the synthetic dataset and the ``toy`` backbone::

    uv run python examples/one_stage.py

``docs/python-api.md`` includes the marked section.
"""

from __future__ import annotations

from pathlib import Path

from cbm_eval.config import load_config
from cbm_eval.pipeline import PipelineBuilder
from cbm_eval.registry import FILTERING
from cbm_eval.stages.filtering import RuleFilter
from cbm_eval.structures import ConceptSet

REPO = Path(__file__).resolve().parents[1]

# --8<-- [start:stage]
# A stage reads data through a Context. PipelineBuilder builds one from a config; the synthetic
# anchor is enough here, and only its dataset, backbone, device and paths matter.
cfg = load_config(REPO / "configs/anchors/synthetic.yaml", {"device": "cpu"})
ctx = PipelineBuilder(cfg).context()

concepts = ConceptSet(["concept_03", "Concept 03", "class_2", "concept_07", "distractor_01",
                       "concept_11", "a concept name with far too many words"])

# Build a stage from its class and constructor kwargs ...
rules = RuleFilter(max_words=4)
kept = rules.filter(concepts, ctx)  # a new ConceptSet; `concepts` is unchanged
print("rules: ", kept.names)

# ... or from its registered name and kwargs, as in a config.
select = FILTERING.build({"name": "select", "k": 2, "method": "discriminative"})
print("select:", select.filter(kept, ctx).names)
# --8<-- [end:stage]
