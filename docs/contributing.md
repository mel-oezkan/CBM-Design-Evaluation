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

--8<-- "README.md:adding"

The rules below are the full contract. They are also kept in `CLAUDE.md` at the repository root,
which AI coding assistants read, so both stay in sync.

--8<-- "CLAUDE.md:rules"
