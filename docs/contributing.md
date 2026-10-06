# Contributing

## Development setup

```bash
uv sync                                          # core + pytest
uv run pytest -q                                 # fully offline, ~5 s; must stay green and offline
uv run --group docs properdocs serve             # live docs preview at http://127.0.0.1:8000
uv run --group docs properdocs build --strict    # must build without warnings
```

The tests never touch the network or download weights, so they run anywhere. Please keep it that
way: inject stub clients and models through constructor arguments instead (examples below).

## Adding a component

```python
from cbm_eval.registry import FILTERING
from cbm_eval.stages.base import Filter

@FILTERING.register("my_filter")
class MyFilter(Filter):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def filter(self, concepts, ctx):
        ...
        return concepts.subset(keep)
```

Import the module from its package `__init__.py`. It then becomes available as `{name: my_filter, threshold: 0.3}`.

A new loss is a training subclass that overrides one hook, `_task_loss(logits, batch)` or
`_concept_loss(layer, batch)`, registered under its own name:

```python
import torch.nn.functional as F

from cbm_eval.registry import TRAINING
from cbm_eval.stages.training import Joint

@TRAINING.register("joint_smooth")
class JointSmooth(Joint):
    def __init__(self, smoothing: float = 0.1, **kw):
        super().__init__(**kw)
        self.smoothing = smoothing

    def _task_loss(self, logits, b):
        return F.cross_entropy(logits, b.y, label_smoothing=self.smoothing)
```

The rules below are the full contract. They are also kept in `CLAUDE.md` at the repository root,
which AI coding assistants read, so both stay in sync.

--8<-- "CLAUDE.md:rules"

## Architecture review

Every pull request gets an automated architecture review as a PR comment
(`.github/workflows/architecture-review.yml`). The reviewer checks each commit against the rules
above and asks whether a one-stage ablation still measures only that stage and whether existing
results still match their `run_id`. It is advisory and does not block merging.

To run the same review locally before pushing, ask Claude Code to use the
`architecture-reviewer` agent (`.claude/agents/architecture-reviewer.md`), or headless:

```bash
claude -p "Use the architecture-reviewer agent to review origin/main..HEAD"
```
