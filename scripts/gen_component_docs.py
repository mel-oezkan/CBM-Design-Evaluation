"""Generate ``components.md`` for the docs site from the registries (run by ``mkdocs-gen-files``).

Config kwargs are constructor kwargs, so each registered class's signature is its config reference.
Building the page from the registries at docs-build time keeps it in sync with the code.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import mkdocs_gen_files

from cbm_eval import registry
from cbm_eval.stages.scorers import SCORERS

REPO = "https://github.com/mel-oezkan/CBM-Design-Evaluation/blob/main"
ROOT = Path(__file__).resolve().parents[1]

# Section order follows the pipeline; ``registry.ALL`` keys are the registry kinds.
SECTIONS = [
    ("dataset", "Datasets", "`dataset:`"),
    ("backbone", "Backbones", "`backbone:` / `teacher:`"),
    ("instances", "Instance sources", "`instances:`"),
    ("discovery", "Discovery", "`stages.discovery:`"),
    ("filtering", "Filtering", "`stages.filtering:` (a list, applied in order)"),
    ("alignment", "Alignment", "`stages.alignment:`"),
    ("generation", "Generation", "`stages.generation:`"),
    ("predictor", "Predictor", "`stages.predictor:`"),
    ("training", "Training", "`stages.training:`"),
    ("evaluation", "Evaluation", "`evaluation:` (a list; metrics are prefixed with the name)"),
    ("concept scorer", "Concept scorers", "`scorer:` inside the `dino` alignment / filter"),
]


def _kwargs(obj) -> list[inspect.Parameter]:
    """Constructor kwargs, following ``**kw`` forwarding up the MRO (``Joint``, ``SAMGrid``)."""
    if not inspect.isclass(obj):
        return [p for p in inspect.signature(obj).parameters.values() if p.kind is not p.VAR_KEYWORD]
    seen: dict[str, inspect.Parameter] = {}
    for cls in obj.__mro__:
        init = cls.__dict__.get("__init__")
        if init is None:
            continue
        params = list(inspect.signature(init).parameters.values())[1:]
        for p in params:
            if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                seen.setdefault(p.name, p)
        if not any(p.kind is p.VAR_KEYWORD for p in params):
            break
    return list(seen.values())


def _interface(items: list) -> str | None:
    """The abstract base every item of a registry shares, rendered as its abstract method(s)."""
    classes = [o for o in items if inspect.isclass(o)]
    if not classes:
        return None
    shared = [b for b in classes[0].__mro__[1:] if inspect.isabstract(b) and all(issubclass(c, b) for c in classes)]
    if not shared:
        return None
    base = shared[0]
    methods = []
    for name in sorted(base.__abstractmethods__):
        attr = inspect.getattr_static(base, name)
        if isinstance(attr, property):
            methods.append(f"`{name}` (property)")
            continue
        sig = inspect.signature(attr)
        args = ", ".join(p for p in list(sig.parameters)[1:])
        ret = f" -> {sig.return_annotation}" if sig.return_annotation is not sig.empty else ""
        methods.append(f"`{name}({args}){ret}`")
    return f"Base class [`{base.__qualname__}`][{base.__module__}.{base.__qualname__}]: " + ", ".join(methods)


def _source(obj) -> str:
    path = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT)
    line = inspect.getsourcelines(obj)[1]
    return f"[`{obj.__module__}.{obj.__qualname__}`]({REPO}/{path.as_posix()}#L{line})"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _component(name: str, obj) -> list[str]:
    out = [f"### `{name}`", "", _source(obj), ""]
    out += [inspect.cleandoc(obj.__doc__), ""] if obj.__doc__ else ["*No docstring.*", ""]
    params = _kwargs(obj)
    if params:
        out += ["| kwarg | default | type |", "|---|---|---|"]
        for p in params:
            default = "*required*" if p.default is p.empty else f"`{_cell(repr(p.default))}`"
            ann = "" if p.annotation is p.empty else f"`{_cell(str(p.annotation))}`"
            out.append(f"| `{p.name}` | {default} | {ann} |")
        out.append("")
    else:
        out += ["No kwargs.", ""]
    return out


def render() -> str:
    registry.load_all()
    regs = {**registry.ALL, SCORERS.kind: SCORERS}
    missing = set(regs) - {kind for kind, *_ in SECTIONS}
    if missing:
        raise KeyError(f"Add {sorted(missing)} to SECTIONS in {Path(__file__).name}")
    lines = [
        "# Component catalogue",
        "",
        "Every registered component, generated from the registries at build time (the same data as",
        "`cbm-eval list`). In a config, write `{name: <name>, <kwarg>: <value>, ...}`; the kwargs go",
        "straight to the constructor, so the tables below are the full set of accepted keys.",
        "",
    ]
    for kind, title, key in SECTIONS:
        reg = regs[kind]
        items = [reg.get(n) for n in reg.names()]
        lines += [f"## {title}", "", f"Config key: {key}", ""]
        if (iface := _interface(items)) is not None:
            lines += [iface, ""]
        for name in reg.names():
            lines += _component(name, reg.get(name))
    return "\n".join(lines)


with mkdocs_gen_files.open("components.md", "w") as f:
    f.write(render())
mkdocs_gen_files.set_edit_path("components.md", "scripts/gen_component_docs.py")
