import copy
import json
from pathlib import Path

import pytest

from scripts.register_flybody_smoke_fixture import register


DERIVED_FIELDS = {
    "final_body_quaternion_wxyz",
    "first_control_actuator_torque_n_m",
    "first_control_hold_max_delta_n_m",
    "first_control_hold_sample_count",
    "integrated_root_fluid_impulse_n_s",
    "maximum_measured_wing_excursion_rad",
}


def _registered_fixture():
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "reference"
        / "flybody_analytic_wingbeat_smoke.v3.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_v3_registration_recomputes_trace_derived_summary_deterministically():
    expected = _registered_fixture()
    raw_summary = {
        key: value
        for key, value in expected["summary"].items()
        if key not in DERIVED_FIELDS
    }
    observed = register({"summary": raw_summary, "trace": expected["trace"]})

    assert observed == expected
    assert observed["summary"]["first_control_hold_sample_count"] == 4
    assert observed["summary"]["first_control_hold_max_delta_n_m"] == 0.0


def test_registration_rejects_any_released_policy_equivalence_claim():
    expected = _registered_fixture()
    summary = copy.deepcopy(expected["summary"])
    summary["released_policy_topology_equivalent"] = True

    with pytest.raises(ValueError, match="released-policy equivalence"):
        register({"summary": summary, "trace": expected["trace"]})
