"""Budget-aware Modal launcher and spend watcher for this repository (the ``modal-operator`` agent's tool).

Every launch prices its **worst case** before anything starts:

    worst = (max_hours + OVERHEAD_HOURS) x hourly rate of the profile

The hourly rate comes from the live ``modal billing rates`` (GPU + CPU cores + memory GiB). The worst case
is a real bound: the container's ``timeout`` is ``max_hours``, its CPU and memory are hard limits, it gets
no retries and at most one container, so Modal stops it even if the Mac that launched it is off.
``launch`` refuses unless ``--approve-usd`` covers the worst case, and refuses to go beyond the monthly
free credit unless ``--allow-billing`` is also given. Both flags stand for a decision of the user.

    python scripts/modal_ops.py budget                        # this month: spend, credit left, running apps
    python scripts/modal_ops.py rates                         # $/h of every profile
    python scripts/modal_ops.py estimate --profile l4 --max-hours 1 [--hours 0.3]
    python scripts/modal_ops.py launch --profile cpu --max-hours 0.25 --approve-usd 0.06 \\
        --purpose "repro #61" -- cbm-eval run configs/anchors/synthetic.yaml --seeds 0
    python scripts/modal_ops.py watch <launch_id>             # state, cost so far, % of cap [--every 10]
    python scripts/modal_ops.py fetch <launch_id>             # ops/<launch_id>/ -> results/modal/<launch_id>/
    python scripts/modal_ops.py ledger                        # every launch: estimate vs. actual

Run it with the project environment (``uv run python scripts/modal_ops.py ...``); only ``launch``,
``watch`` and ``fetch`` import ``modal``. The ledger (one JSON line per launch) lives outside the repo, at
``$CBM_MODAL_LEDGER`` or ``~/.local/state/cbm-eval/modal-ledger.jsonl``, so every worktree shares it.
The monthly free credit is ``$CBM_MODAL_CREDIT`` (default 30, the Starter plan's).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _app_constants(path: Path = REPO / "scripts" / "cbm_modal.py") -> dict[str, object]:
    """Module-level literal constants of ``cbm_modal.py``, read from its source: importing it needs ``modal``."""
    tree = ast.parse(path.read_text())
    return {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
            for t in n.targets if isinstance(t, ast.Name)}


# Billed time outside the timeout (container start + idle scale-down of `cbm_modal.sh`), added to every worst
# case. Derived from the values the container really gets, so changing them there cannot break the cap.
_APP = _app_constants()
OVERHEAD_HOURS = (_APP["STARTUP_TIMEOUT_S"] + _APP["SCALEDOWN_WINDOW_S"]) / 3600
# Modal reports only the credit used, not the plan's grant, so the grant is configured: $CBM_MODAL_CREDIT
# overrides this Starter-plan default. If it is too high, `billed` in `budget` shows real charges at once.
DEFAULT_CREDIT = 30.0


@dataclass(frozen=True)
class Profile:
    """Resources of one container. CPU and memory are hard limits, so the bill cannot exceed them."""

    gpu: str | None
    cpu: float
    memory_gib: int
    note: str = ""

    def hourly(self, rates: dict[str, str]) -> float:
        """$/h of one container from ``modal billing rates --json``."""
        gpu = 0.0
        if self.gpu is not None:
            key = "gpu_hour_cost_" + self.gpu.lower().replace("-", "_")
            if key not in rates:
                raise KeyError(f"No rate '{key}' in `modal billing rates` for GPU {self.gpu}")
            gpu = float(rates[key])
        return gpu + self.cpu * float(rates["cpu_hour_cost"]) + self.memory_gib * float(rates["mem_gib_hour_cost"])


PROFILES = {
    "cpu": Profile(None, 2, 8, "training on cached features, tests, small configs"),
    "cpu-large": Profile(None, 8, 32, "many seeds / data preparation (unpacking, CSV work)"),
    "t4": Profile("T4", 4, 16, "small feature extraction (toy-sized sets, CLIP ViT-B)"),
    "l4": Profile("L4", 8, 32, "feature extraction on CUB / Waterbirds sized sets"),
    "a10g": Profile("A10G", 8, 32, "like l4, more memory bandwidth"),
    "a100": Profile("A100-40GB", 16, 32, "fine-tuning (koh2020_modal.py's setting); ask first"),
}


# ---------------------------------------------------------------------------------------------- pure logic


def worst_case(profile: Profile, rates: dict[str, str], max_hours: float) -> float:
    return (max_hours + OVERHEAD_HOURS) * profile.hourly(rates)


def month_status(summary: dict, credit: float) -> dict[str, float]:
    """This month's spend from ``modal billing summary --json``. ``counted`` is the usage the free credit
    pays for (metered minus the free storage and egress allowances); ``billed`` is what goes on the card."""
    adj = summary.get("adjustments", {})
    allowances = -sum(float(adj.get(k, 0)) for k in ("free_storage", "network_egress_allowance"))
    counted = max(0.0, float(summary["metered_cost"]) - allowances)
    return {"metered": float(summary["metered_cost"]), "counted": counted, "credit": credit,
            "credit_left": max(0.0, credit - counted), "billed": float(summary["billed_cost"])}


def check_launch(worst: float, approve_usd: float | None, credit_left: float, allow_billing: bool) -> list[str]:
    """Reasons to refuse a launch (empty: go)."""
    problems = []
    if approve_usd is None:
        problems.append(f"no --approve-usd: the worst case is ${worst:.2f}; ask the user to approve at least that")
    elif approve_usd < worst:
        problems.append(f"worst case ${worst:.2f} exceeds the approved ${approve_usd:.2f}: lower --max-hours, pick a "
                        f"cheaper profile, or ask the user to approve ${worst:.2f}")
    if worst > credit_left and not allow_billing:
        problems.append(f"worst case ${worst:.2f} exceeds the free credit left this month (${credit_left:.2f}); the "
                        f"rest would be billed. Ask the user, then pass --allow-billing")
    return problems


def app_cost(report: list[dict], app_id: str) -> float:
    """Billed cost of one app from ``modal billing report --json`` rows (one row per interval and resource)."""
    return sum(float(r["cost"]) for r in report if r.get("object_id") == app_id)


def _is_running(app: dict) -> bool:
    """An app from ``modal app list --json`` that has containers up (any state but stopped / stopping)."""
    return bool(app) and not app.get("state", "").startswith("stop") and int(app.get("tasks") or 0) > 0


def launch_state(app: dict, result: dict | None, error: str | None) -> str:
    """State of a launch from its app (``modal app list --json`` row, or ``{}``) and its call's outcome.
    ``pending``: the app is up but no container runs yet (image build, container start)."""
    if result is not None:
        return "finished" if result["exit_code"] == 0 else f"failed (exit {result['exit_code']})"
    if error is not None:
        return f"error: {error}"
    if _is_running(app):
        return "running"
    state = app.get("state", "unknown") if app else "unknown"
    return state if state.startswith(("stop", "disabled", "unknown")) else "pending"


def cost_so_far(billed_hours_cost: float, last_full_hour: datetime, now: datetime, hourly: float,
                running: bool) -> float:
    """Billed full hours plus, while the app runs, the current partial hour at the profile's rate."""
    partial = max(0.0, (now - last_full_hour).total_seconds() / 3600) if running else 0.0
    return billed_hours_cost + partial * hourly


# ------------------------------------------------------------------------------------------------ Modal I/O


def _modal_exe() -> str:
    """The ``modal`` CLI of this Python environment, else the one on the PATH."""
    exe = Path(sys.executable).with_name("modal")
    return str(exe) if exe.exists() else "modal"


def _modal(*args: str) -> object:
    out = subprocess.run([_modal_exe(), *args, "--json"], capture_output=True, text=True,
                         check=False)
    if out.returncode != 0:
        raise RuntimeError(f"`modal {' '.join(args)}` failed: {out.stderr.strip()[-500:]}")
    return json.loads(out.stdout or "null")


def _rates() -> dict[str, str]:
    return _modal("billing", "rates")


def _status(credit: float) -> dict[str, float]:
    return month_status(_modal("billing", "summary"), credit)


def _apps() -> dict[str, dict]:
    return {a["app_id"]: a for a in _modal("app", "list")}


def _report_since(start: datetime) -> list[dict]:
    return _modal("billing", "report", "--start", start.strftime("%Y-%m-%dT%H:00:00"), "--resolution", "h")


def _ledger_path() -> Path:
    return Path(os.environ.get("CBM_MODAL_LEDGER", Path.home() / ".local/state/cbm-eval/modal-ledger.jsonl"))


def _ledger() -> list[dict]:
    p = _ledger_path()
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def _append_ledger(entry: dict) -> None:
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _find(ref: str) -> dict:
    hits = [e for e in _ledger() if ref in (e["launch_id"], e["app_id"])]
    if not hits:
        raise SystemExit(f"No launch '{ref}' in the ledger ({_ledger_path()})")
    return hits[-1]


def _git() -> dict:
    def g(*a):
        r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain")), "checkout": REPO.name}


def _credit() -> float:
    return float(os.environ.get("CBM_MODAL_CREDIT", DEFAULT_CREDIT))


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------------------------------------- commands


def cmd_rates(a) -> None:
    rates = _rates()
    print(f"{'profile':10} {'gpu':10} {'cpu':>4} {'GiB':>4} {'$/h':>7}  use")
    for name, p in PROFILES.items():
        print(f"{name:10} {p.gpu or '-':10} {p.cpu:>4g} {p.memory_gib:>4} {p.hourly(rates):>7.3f}  {p.note}")


def cmd_budget(a) -> None:
    s = _status(_credit())
    print(f"this month: metered ${s['metered']:.2f}, counted against credit ${s['counted']:.2f}, "
          f"credit left ${s['credit_left']:.2f} of ${s['credit']:.0f}, billed ${s['billed']:.2f}")
    ours = {e["app_id"]: e for e in _ledger()}
    running = [x for x in _apps().values() if _is_running(x)]
    if not running:
        print("running apps: none")
    for x in running:
        e = ours.get(x["app_id"])
        tag = f"launch {e['launch_id']} ({e['profile']}, cap ${e['worst_usd']:.2f}): {e['purpose']}" if e else \
            "NOT launched through modal_ops (no worst-case bound known)"
        print(f"running: {x['app_id']} {x['description']} tasks={x['tasks']} since {x['created_at']} -- {tag}")


def cmd_estimate(a) -> None:
    rates, p = _rates(), PROFILES[a.profile]
    worst = worst_case(p, rates, a.max_hours)
    s = _status(_credit())
    print(f"profile {a.profile}: ${p.hourly(rates):.3f}/h ({p.gpu or 'no GPU'}, {p.cpu:g} CPU, {p.memory_gib} GiB)")
    if a.hours is not None:
        print(f"expected: {a.hours:g} h -> ${a.hours * p.hourly(rates):.2f}")
    print(f"worst case: timeout {a.max_hours:g} h + {OVERHEAD_HOURS:g} h overhead -> ${worst:.2f}")
    beyond = max(0.0, worst - s["credit_left"])
    print(f"free credit left: ${s['credit_left']:.2f}; billed beyond it in the worst case: ${beyond:.2f}")


def cmd_launch(a) -> None:
    if not a.command:
        raise SystemExit("Give the command after `--`")
    command = " ".join(a.command)
    rates, p = _rates(), PROFILES[a.profile]
    worst = worst_case(p, rates, a.max_hours)
    status = _status(_credit())
    if problems := check_launch(worst, a.approve_usd, status["credit_left"], a.allow_billing):
        raise SystemExit("Refused:\n- " + "\n- ".join(problems))
    launch_id = _now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    entry = {"launch_id": launch_id, "started": _now().isoformat(timespec="seconds"), "purpose": a.purpose,
             "command": command, "profile": a.profile, "resources": asdict(p), "hourly_usd": round(p.hourly(rates), 4),
             "max_hours": a.max_hours, "worst_usd": round(worst, 4), "approve_usd": a.approve_usd,
             "expected_hours": a.hours, "credit_left_before": round(status["credit_left"], 4),
             "allow_billing": a.allow_billing, "secrets": a.secret, "git": _git()}

    import modal

    sys.path.insert(0, str(REPO / "scripts"))
    import cbm_modal

    fn = cbm_modal.sh.with_options(
        gpu=p.gpu, cpu=(p.cpu, p.cpu), memory=(p.memory_gib * 1024, p.memory_gib * 1024),
        timeout=int(a.max_hours * 3600), retries=0, max_containers=1,
        secrets=[modal.Secret.from_name(s) for s in a.secret])
    with modal.enable_output(), cbm_modal.app.run(detach=True):
        call = fn.spawn(command, launch_id, entry)
        entry |= {"app_id": cbm_modal.app.app_id, "call_id": call.object_id}
    _append_ledger(entry)
    print(json.dumps({k: entry[k] for k in ("launch_id", "app_id", "call_id", "profile", "worst_usd", "max_hours")}))


def _watch_once(e: dict, rates_hourly: float) -> dict:
    import modal

    started = datetime.fromisoformat(e["started"])
    app = _apps().get(e["app_id"], {})
    running, result, error = _is_running(app), None, None
    try:
        result = modal.FunctionCall.from_id(e["call_id"]).get(timeout=0)
    except TimeoutError:
        pass
    except Exception as err:  # noqa: BLE001 -- re-raises the remote error, of any type (timeout, crash, stop)
        error = f"{type(err).__name__}: {str(err)[:200]}"
    state = launch_state(app, result, error)
    now = _now()
    last_full = now.replace(minute=0, second=0, microsecond=0)
    billed = app_cost(_report_since(started), e["app_id"])
    spent = cost_so_far(billed, last_full, now, rates_hourly, running)
    return {"launch_id": e["launch_id"], "state": state, "elapsed_h": round((now - started).total_seconds() / 3600, 3),
            "max_hours": e["max_hours"], "cost_so_far_usd": round(spent, 3), "billed_full_hours_usd": round(billed, 3),
            "worst_usd": e["worst_usd"], "cap_used": f"{100 * spent / e['worst_usd']:.0f}%",
            "result": result and {k: result[k] for k in ("exit_code", "seconds")},
            "log_tail": result and result["log_tail"][-15:]}


def cmd_watch(a) -> None:
    e = _find(a.ref)
    while True:
        w = _watch_once(e, e["hourly_usd"])
        print(json.dumps(w, indent=None if a.every else 2), flush=True)
        if a.stop_at_usd is not None and w["state"] == "running" and w["cost_so_far_usd"] >= a.stop_at_usd:
            subprocess.run([_modal_exe(), "app", "stop", e["app_id"]], check=False)
            print(f"stopped {e['app_id']}: cost ${w['cost_so_far_usd']:.2f} reached --stop-at-usd {a.stop_at_usd}")
            return
        if not a.every or w["state"] not in ("running", "pending"):
            return
        time.sleep(a.every * 60)


def cmd_fetch(a) -> None:
    """Copy ``ops/<launch_id>/`` to ``dest`` (files already there are overwritten, others kept).

    ``modal volume get VOL ops/<id> X`` writes ``X/<id>/...`` only if ``X`` is an existing directory; if
    ``X`` does not exist, every file is written to the path ``X`` itself. So it downloads into a fresh
    temporary directory and moves ``<id>/`` into place."""
    e = _find(a.ref)
    launch_id = e["launch_id"]
    dest = Path(a.dest or REPO / "results" / "modal" / launch_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "scripts"))
    from cbm_modal import VOLUME
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        subprocess.run([_modal_exe(), "volume", "get", VOLUME, f"ops/{launch_id}", tmp], check=True)
        shutil.copytree(Path(tmp, launch_id), dest, dirs_exist_ok=True)
    print(dest)


def cmd_ledger(a) -> None:
    entries = _ledger()
    if not entries:
        print(f"no launches yet ({_ledger_path()})")
        return
    since = min(datetime.fromisoformat(e["started"]) for e in entries)
    report = _report_since(max(since, _now() - timedelta(days=62)))
    print(f"{'launch':22} {'profile':9} {'worst $':>8} {'actual $':>9}  purpose")
    for e in entries:
        print(f"{e['launch_id']:22} {e['profile']:9} {e['worst_usd']:>8.2f} {app_cost(report, e['app_id']):>9.3f}  "
              f"{e['purpose']}")
    print("actual = billed full hours only; the current hour shows up once it ends")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="modal_ops", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget", help="this month's spend, credit left, running apps").set_defaults(fn=cmd_budget)
    sub.add_parser("rates", help="$/h of every profile").set_defaults(fn=cmd_rates)
    e = sub.add_parser("estimate", help="expected and worst-case cost of one launch")
    e.add_argument("--profile", choices=PROFILES, required=True)
    e.add_argument("--max-hours", type=float, required=True, help="the container timeout (the hard cap)")
    e.add_argument("--hours", type=float, help="the expected run time")
    e.set_defaults(fn=cmd_estimate)
    la = sub.add_parser("launch", help="run a command on Modal within an approved dollar cap")
    la.add_argument("--profile", choices=PROFILES, required=True)
    la.add_argument("--max-hours", type=float, required=True, help="the container timeout (the hard cap)")
    la.add_argument("--hours", type=float, help="the expected run time (recorded in the ledger)")
    la.add_argument("--approve-usd", type=float, help="the dollar cap the user approved (>= the worst case)")
    la.add_argument("--allow-billing", action="store_true", help="the user agreed to spend beyond the free credit")
    la.add_argument("--purpose", required=True, help="why (an issue number, the question it answers)")
    la.add_argument("--secret", action="append", default=[], help="a Modal Secret to attach (e.g. wandb)")
    la.add_argument("command", nargs=argparse.REMAINDER, help="-- the shell command to run in the repo")
    la.set_defaults(fn=cmd_launch)
    w = sub.add_parser("watch", help="state and cost so far of a launch")
    w.add_argument("ref", help="launch_id or app_id")
    w.add_argument("--every", type=float, help="repeat every N minutes until the launch ends")
    w.add_argument("--stop-at-usd", type=float, help="stop the app at this cost (only if the user said so)")
    w.set_defaults(fn=cmd_watch)
    f = sub.add_parser("fetch", help="download ops/<launch_id>/ (log, results, runs) from the Volume")
    f.add_argument("ref")
    f.add_argument("--dest")
    f.set_defaults(fn=cmd_fetch)
    sub.add_parser("ledger", help="every launch: worst case vs. billed").set_defaults(fn=cmd_ledger)
    a = p.parse_args(argv)
    if getattr(a, "command", None) and a.command[0] == "--":
        a.command = a.command[1:]
    a.fn(a)


if __name__ == "__main__":
    main()
