from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


_ROOT = Path(__file__).resolve().parents[1]
_PROBE = _ROOT / "scripts" / "probe_streaming_fresh_process.py"
_SEED = 73021


def _invoke(request: Mapping[str, Any], *, expect_success: bool = True):
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [sys.executable, str(_PROBE)],
        input=json.dumps(request, allow_nan=False, sort_keys=True),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=_ROOT,
        env=environment,
        check=False,
        timeout=90,
    )
    assert completed.stdout, completed.stderr
    response = json.loads(completed.stdout)
    if expect_success:
        assert completed.returncode == 0, response
        assert response["ok"] is True
    else:
        assert completed.returncode != 0
        assert response["ok"] is False
    return completed, response


def test_checkpoint_continuation_is_exact_across_three_fresh_processes() -> None:
    split_intervals = 31
    tail_intervals = 44

    _baseline_process, baseline = _invoke(
        {
            "operation": "run",
            "seed": _SEED,
            "interval_count": split_intervals + tail_intervals,
        }
    )
    _prefix_process, prefix = _invoke(
        {
            "operation": "run",
            "seed": _SEED,
            "interval_count": split_intervals,
        }
    )
    _resume_process, resumed = _invoke(
        {
            "operation": "resume",
            "seed": _SEED,
            "interval_count": tail_intervals,
            "bridge_checkpoint": prefix["bridge_checkpoint"],
            "mechanics_checkpoint": prefix["mechanics_checkpoint"],
            "expected_runtime_receipt": prefix["runtime_receipt"],
        }
    )

    assert prefix["final_bridge_tick"] == split_intervals
    assert prefix["final_mechanics_tick"] == 5 * split_intervals
    assert resumed["final_bridge_tick"] == split_intervals + tail_intervals
    assert resumed["final_mechanics_tick"] == 5 * (
        split_intervals + tail_intervals
    )
    assert resumed["runtime_receipt"] == baseline["runtime_receipt"]
    assert resumed["scientific_status"] == "software_only_manufactured_physics"
    assert resumed["tail"] == baseline["tail"][split_intervals:]
    assert resumed["bridge_checkpoint"] == baseline["bridge_checkpoint"]
    assert resumed["mechanics_checkpoint"] == baseline["mechanics_checkpoint"]
    assert (
        resumed["bridge_checkpoint_sha256"]
        == baseline["bridge_checkpoint_sha256"]
    )
    assert (
        resumed["mechanics_checkpoint_sha256"]
        == baseline["mechanics_checkpoint_sha256"]
    )


def test_resume_rejects_a_corrupt_checkpoint_in_a_fresh_process() -> None:
    _prefix_process, prefix = _invoke(
        {"operation": "run", "seed": _SEED, "interval_count": 23}
    )
    corrupted = copy.deepcopy(prefix["bridge_checkpoint"])
    corrupted["state"]["tick_index"] += 1

    _process, response = _invoke(
        {
            "operation": "resume",
            "seed": _SEED,
            "interval_count": 2,
            "bridge_checkpoint": corrupted,
            "mechanics_checkpoint": prefix["mechanics_checkpoint"],
            "expected_runtime_receipt": prefix["runtime_receipt"],
        },
        expect_success=False,
    )
    assert response["error_type"] == "ValueError"
    assert "SHA-256 mismatch" in response["error"]


def test_resume_rejects_a_runtime_or_source_receipt_mismatch() -> None:
    _prefix_process, prefix = _invoke(
        {"operation": "run", "seed": _SEED, "interval_count": 19}
    )
    wrong_receipt = copy.deepcopy(prefix["runtime_receipt"])
    wrong_receipt["module_sha256"]["flight/streaming_bridge.py"] = "0" * 64

    _process, response = _invoke(
        {
            "operation": "resume",
            "seed": _SEED,
            "interval_count": 1,
            "bridge_checkpoint": prefix["bridge_checkpoint"],
            "mechanics_checkpoint": prefix["mechanics_checkpoint"],
            "expected_runtime_receipt": wrong_receipt,
        },
        expect_success=False,
    )
    assert response["error_type"] == "ValueError"
    assert response["error"] == "runtime receipt mismatch"
