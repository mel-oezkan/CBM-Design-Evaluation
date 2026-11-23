"""Cost logic of ``scripts/modal_ops.py`` (offline: no ``modal`` import, no network)."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("modal_ops", Path(__file__).parents[1] / "scripts" / "modal_ops.py")
ops = importlib.util.module_from_spec(_spec)
sys.modules["modal_ops"] = ops  # dataclasses look their module up while the class is built
_spec.loader.exec_module(ops)

# `modal billing rates --json` (excerpt, 2026-11-23)
RATES = {"cpu_hour_cost": "0.04730", "mem_gib_hour_cost": "0.00800", "gpu_hour_cost_a100_40gb": "2.10000",
         "gpu_hour_cost_l4": "0.80000", "gpu_hour_cost_t4": "0.59000", "gpu_hour_cost_a10g": "1.10000"}
# `modal billing summary --for 2026-10 --json`: $30 credit used up, $10.30 billed
OCTOBER = {"metered_cost": "40.50115868", "billed_cost": "10.30000000",
           "adjustments": {"network_egress_allowance": "-0.00206205", "free_storage": "-0.19909663",
                           "reservation_adjustment": "-0E-8", "credits": "-30.00000000", "plan_cost": "0E-8"}}
NOVEMBER = {"metered_cost": "0.28974228", "billed_cost": "0E-8",
            "adjustments": {"free_storage": "-0.28974228", "credits": "-0E-8"}}


def test_a100_profile_matches_the_october_bill():
    # 2.10 GPU + 16 x 0.0473 CPU + 32 x 0.008 GiB: the ~$3.10/h per container seen in October
    assert ops.PROFILES["a100"].hourly(RATES) == pytest.approx(3.1128)


def test_every_profile_has_a_rate():
    for p in ops.PROFILES.values():
        assert p.hourly(RATES) > 0


def test_unknown_gpu_rate_raises():
    with pytest.raises(KeyError, match="gpu_hour_cost_h100"):
        ops.Profile("H100", 1, 1).hourly(RATES)


def test_month_status_counts_usage_against_the_credit():
    oct_, nov = ops.month_status(OCTOBER, 30), ops.month_status(NOVEMBER, 30)
    assert oct_["counted"] == pytest.approx(40.30, abs=0.01) and oct_["credit_left"] == 0
    assert oct_["counted"] - 30 == pytest.approx(oct_["billed"], abs=0.01)  # what went beyond the credit
    assert nov["counted"] == 0 and nov["credit_left"] == 30  # storage only: covered by the allowance


def test_overhead_follows_the_container_settings(tmp_path):
    # the live app: 300 s start + 60 s scale-down
    assert ops.OVERHEAD_HOURS == pytest.approx(0.1)
    # a change in cbm_modal.py reaches the worst case without a second edit
    app = tmp_path / "cbm_modal.py"
    app.write_text("import modal\nSTARTUP_TIMEOUT_S = 600\nSCALEDOWN_WINDOW_S = 120\n")
    c = ops._app_constants(app)
    assert (c["STARTUP_TIMEOUT_S"] + c["SCALEDOWN_WINDOW_S"]) / 3600 == pytest.approx(0.2)
    # and `sh` really uses the constants, not literals that could drift from them
    src = ast.parse((Path(ops.__file__).parent / "cbm_modal.py").read_text())
    sh = next(f for f in src.body if isinstance(f, ast.FunctionDef) and f.name == "sh")
    kw = {k.arg: k.value for k in sh.decorator_list[0].keywords}
    assert ast.unparse(kw["startup_timeout"]) == "STARTUP_TIMEOUT_S"
    assert ast.unparse(kw["scaledown_window"]) == "SCALEDOWN_WINDOW_S"


def test_worst_case_includes_overhead():
    p = ops.PROFILES["cpu"]
    assert ops.worst_case(p, RATES, 1.0) == pytest.approx((1 + ops.OVERHEAD_HOURS) * p.hourly(RATES))


@pytest.mark.parametrize("approve, credit_left, allow, n_problems", [
    (None, 30, False, 1),   # no approval at all
    (0.5, 30, False, 1),    # approval below the worst case
    (2.0, 30, False, 0),    # approved, within the credit
    (2.0, 0.5, False, 1),   # approved, but it would be billed
    (2.0, 0.5, True, 0),    # approved, billing agreed
])
def test_check_launch(approve, credit_left, allow, n_problems):
    assert len(ops.check_launch(1.0, approve, credit_left, allow)) == n_problems


def test_app_cost_sums_only_that_app():
    report = [{"object_id": "ap-a", "cost": "1.5"}, {"object_id": "ap-b", "cost": "9"},
              {"object_id": "ap-a", "cost": "0.25"}]
    assert ops.app_cost(report, "ap-a") == pytest.approx(1.75)


def test_cost_so_far_adds_the_partial_hour_while_running():
    hour = datetime(2026, 11, 23, 10, tzinfo=timezone.utc)
    now = hour + timedelta(minutes=30)
    assert ops.cost_so_far(1.0, hour, now, 2.0, running=True) == pytest.approx(2.0)
    assert ops.cost_so_far(1.0, hour, now, 2.0, running=False) == pytest.approx(1.0)


@pytest.mark.parametrize("app, result, error, state", [
    ({"state": "ephemeral (detached)", "tasks": "0"}, None, None, "pending"),  # image build, container start
    ({"state": "ephemeral (detached)", "tasks": "1"}, None, None, "running"),
    ({"state": "stopped", "tasks": "0"}, None, None, "stopped"),
    ({}, None, None, "unknown"),
    ({"state": "stopped", "tasks": "0"}, {"exit_code": 0}, None, "finished"),
    ({"state": "ephemeral (detached)", "tasks": "1"}, {"exit_code": 2}, None, "failed (exit 2)"),
    ({"state": "stopped", "tasks": "0"}, None, "FunctionTimeoutError: x", "error: FunctionTimeoutError: x"),
])
def test_launch_state(app, result, error, state):
    # `watch --every` keeps polling while the state is pending or running
    assert ops.launch_state(app, result, error) == state


def test_launch_refuses_without_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(ops, "_rates", lambda: RATES)
    monkeypatch.setattr(ops, "_status", lambda credit: ops.month_status(NOVEMBER, credit))
    monkeypatch.setenv("CBM_MODAL_LEDGER", str(tmp_path / "ledger.jsonl"))
    with pytest.raises(SystemExit, match="no --approve-usd"):
        ops.main(["launch", "--profile", "cpu", "--max-hours", "1", "--purpose", "t", "--", "echo", "hi"])
    assert not (tmp_path / "ledger.jsonl").exists()


def test_fetch_puts_the_launch_dir_at_dest(monkeypatch, tmp_path):
    # `modal volume get VOL ops/<id> X` writes X/<id>/... into an existing directory X
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"launch_id": "L1", "app_id": "ap-1"}) + "\n")
    monkeypatch.setenv("CBM_MODAL_LEDGER", str(ledger))
    monkeypatch.setitem(sys.modules, "cbm_modal", types.SimpleNamespace(VOLUME="vol"))

    def fake_get(cmd, check):
        assert cmd[-2] == "ops/L1" and Path(cmd[-1]).is_dir()
        (Path(cmd[-1]) / "L1" / "results").mkdir(parents=True)
        (Path(cmd[-1]) / "L1" / "log.txt").write_text("new")
        (Path(cmd[-1]) / "L1" / "results" / "r.jsonl").write_text("{}")

    monkeypatch.setattr(ops.subprocess, "run", fake_get)
    dest = tmp_path / "out" / "L1"
    dest.mkdir(parents=True)
    (dest / "log.txt").write_text("old")
    (dest / "notes.txt").write_text("kept")
    ops.main(["fetch", "L1", "--dest", str(dest)])
    assert (dest / "log.txt").read_text() == "new" and (dest / "results" / "r.jsonl").exists()
    assert (dest / "notes.txt").exists() and sorted(p.name for p in dest.parent.iterdir()) == ["L1"]
