"""Download, verify and check the real datasets so that their loaders accept the folders.

    uv run python scripts/prepare_data.py cub cub-parts waterbirds --root data
    uv run python scripts/prepare_data.py metashift --root data          # SubpopBench steps, then converts
    uv run python scripts/prepare_data.py --check all --root data        # validate existing folders only

Downloads are verified against pinned SHA-256 checksums and kept in ``--root`` for reuse. Targets
whose data cannot be downloaded here (``metashift``, ``cub-koh-split``) print the manual steps.
The recipes and checks live in ``cbm_eval.data.prepare``. Exit code 1 when any target has problems.
"""

from __future__ import annotations

import argparse

from cbm_eval.data.prepare import RECIPES, SCRIPT, check, prepare


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", choices=[*RECIPES, "all"], metavar="DATASET",
                   help=f"one or more of: {', '.join(RECIPES)}, all")
    p.add_argument("--root", default="data", help="folder that holds the datasets (default: data)")
    p.add_argument("--check", action="store_true", help="only validate existing folders; download nothing")
    args = p.parse_args()
    names = list(RECIPES) if "all" in args.datasets else list(dict.fromkeys(args.datasets))

    failed = 0
    for name in names:
        if args.check:
            print(f"{name}: {args.root}/{RECIPES[name].folder}")
            problems = check(name, args.root)
        else:
            problems = prepare(name, args.root)
        for msg in problems:
            print(f"  FAIL {msg}")
        if problems:
            hint = RECIPES[name].manual or f"{SCRIPT} {name} --root {args.root}"
            print(f"  to fix: {hint.format(root=args.root).splitlines()[0]}")
        else:
            print("  ok: the loader accepts this folder")
        failed += bool(problems)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
