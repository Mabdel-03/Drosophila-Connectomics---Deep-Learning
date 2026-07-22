from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import fly_sensor2behavior.evaluators as evaluator_module
from fly_sensor2behavior.evaluators import (
    _evaluate_registered_fly_fgs_to_flybody_with_adapter,
    evaluate_fly_fgs_checkpoint_reentry,
    evaluate_fly_fgs_fixed_step_integrity,
    evaluate_fly_fgs_incremental_runtime_parity,
    evaluate_registered_fly_fgs_to_flybody_vertical_slice,
)
from fly_sensor2behavior.flight import AerodynamicWrench, RigidBodyState
from fly_sensor2behavior.validation import load_benchmark_registry


SOURCE_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PATH = (
    SOURCE_ROOT / "data" / "reference" / "fly_fgs" / "fixed_step_capture.v1.json.gz"
)
MANIFEST_PATH = (
    SOURCE_ROOT / "data" / "reference" / "fly_fgs" / "source_manifest.v1.json"
)
CAPTURE_SCRIPT = SOURCE_ROOT / "scripts" / "capture_fly_fgs_fixed_step.mjs"


class _DeterministicTelemetryPhysics:
    """Protocol-compatible manufactured adapter; never worker evidence."""

    backend_name = "flybody"
    aerodynamic_owner = "flybody"
    body_state_reference = "manufactured root frame"
    wing_joint_order = tuple("wing-axis-%d" % index for index in range(6))

    def __init__(self):
        self.state = self.default_initial_state()
        self.angles = np.zeros(6, dtype=float)
        self.velocities = np.zeros(6, dtype=float)
        self.last_actuator_torque_n_m = np.zeros(6, dtype=float)
        self._wrench = self._make_wrench(self.last_actuator_torque_n_m)

    @staticmethod
    def default_initial_state():
        state = RigidBodyState()
        state.position_world_m[2] = 0.1
        return state

    @staticmethod
    def _make_wrench(axis_torque):
        lateral = float(axis_torque[0] - axis_torque[3]) * 100.0
        return AerodynamicWrench(
            force_body_n=np.array((lateral, 0.0, 1.0e-6)),
            torque_body_n_m=np.array((0.0, 0.0, lateral * 1.0e-3)),
            left_force_body_n=np.zeros(3),
            right_force_body_n=np.zeros(3),
            mechanical_power_w=float(np.sum(np.abs(axis_torque))),
        )

    def provenance_metadata(self):
        return {
            "engine": "manufactured deterministic telemetry adapter",
            "status": "unit-test-only; not FlyBody evidence",
        }

    def reset(self, initial_state):
        self.state = initial_state.copy()
        self.angles = np.zeros(6, dtype=float)
        self.velocities = np.zeros(6, dtype=float)
        self.last_actuator_torque_n_m = np.zeros(6, dtype=float)
        self._wrench = self._make_wrench(self.last_actuator_torque_n_m)

    def step(self, wings, force_body_n, torque_body_n_m, dt_s):
        np.testing.assert_array_equal(force_body_n, np.zeros(3))
        np.testing.assert_array_equal(torque_body_n_m, np.zeros(3))
        axis_torque = np.asarray(wings.wing_axis_torque_n_m, dtype=float)
        self.last_actuator_torque_n_m = axis_torque.copy()
        self.velocities += axis_torque * 1.0e10 * dt_s
        self.angles += self.velocities * dt_s
        asymmetry = axis_torque[:3] - axis_torque[3:]
        self.state.angular_velocity_body_rad_s += asymmetry * 1.0e10 * dt_s
        self.state.velocity_world_m_s[0] += (
            float(np.sum(asymmetry)) * 1.0e8 * dt_s
        )
        self.state.position_world_m += self.state.velocity_world_m_s * dt_s
        self._wrench = self._make_wrench(axis_torque)
        return self.state.copy()

    def aerodynamic_wrench(self):
        return self._wrench

    def wing_joint_state(self):
        return self.angles.copy(), self.velocities.copy()

    def whole_fly_com_position_m(self):
        return self.state.position_world_m + np.array((1.0e-4, 0.0, 2.0e-4))

    def ground_contact_count(self):
        return 0


def test_registered_fixed_step_integrity_gate_recomputes_every_metric():
    case = load_benchmark_registry().case("fly_fgs.fixed_step_integrity")

    result = evaluate_fly_fgs_fixed_step_integrity(case)
    values = {metric.metric_id: metric.value for metric in result.values}

    assert result.blocked_reason is None
    assert set(values) == {
        "registered_source_inventory_match",
        "fixed_step_contract_match",
        "registered_trace_digest_match",
        "full_state_readout_match",
        "t4_activity_peak_to_peak",
        "nod1_voltage_peak_to_peak_v",
        "forbidden_downstream_field_count",
    }
    assert values["registered_source_inventory_match"] == 1.0
    assert values["fixed_step_contract_match"] == 1.0
    assert values["registered_trace_digest_match"] == 1.0
    assert values["full_state_readout_match"] == 1.0
    assert values["t4_activity_peak_to_peak"] > 0.0
    assert values["nod1_voltage_peak_to_peak_v"] > 0.0
    assert values["forbidden_downstream_field_count"] == 0.0


def test_registered_trace_digest_check_is_not_a_constant_pass(monkeypatch):
    case = load_benchmark_registry().case("fly_fgs.fixed_step_integrity")
    monkeypatch.setattr(
        evaluator_module,
        "_FLY_FGS_REGISTERED_TRACE_SHA256",
        "0" * 64,
    )

    result = evaluate_fly_fgs_fixed_step_integrity(case)
    values = {metric.metric_id: metric.value for metric in result.values}

    assert values["registered_trace_digest_match"] == 0.0


def test_node_fixed_step_runner_is_exactly_repeatable_when_node_is_available():
    """A real repeated execution is distinct from the frozen integrity gate."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed on this test host")
    command = (
        node,
        str(CAPTURE_SCRIPT),
        "--manifest",
        str(MANIFEST_PATH),
        "--include-full-cell-state",
    )
    first = subprocess.run(
        command,
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        timeout=60,
    )
    second = subprocess.run(
        command,
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        timeout=60,
    )

    assert first.stderr == b""
    assert second.stderr == b""
    assert first.stdout == second.stdout
    assert first.stdout == gzip.decompress(CAPTURE_PATH.read_bytes())
    payload = json.loads(first.stdout)
    assert max(
        max(values) - min(values)
        for values in payload["pooled_readout_traces"]["t4_activity"].values()
    ) > 0.0
    assert max(
        max(values) - min(values)
        for values in payload["nod1_voltage_v"].values()
    ) > 0.0


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is unavailable")
def test_incremental_runtime_parity_gate_passes_every_registered_metric():
    case = load_benchmark_registry().case("fly_fgs.incremental_runtime_parity")
    result = evaluate_fly_fgs_incremental_runtime_parity(case)
    observed = {metric.metric_id: metric.value for metric in result.values}

    assert result.blocked_reason is None
    assert set(observed) == {metric.metric_id for metric in case.metrics}
    assert all(
        metric.observe(observed[metric.metric_id]).passed
        for metric in case.metrics
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is unavailable")
def test_checkpoint_reentry_gate_passes_and_rejects_tampering():
    case = load_benchmark_registry().case("fly_fgs.checkpoint_reentry")
    result = evaluate_fly_fgs_checkpoint_reentry(case)
    observed = {metric.metric_id: metric.value for metric in result.values}

    assert result.blocked_reason is None
    assert set(observed) == {metric.metric_id for metric in case.metrics}
    assert observed["corrupted_checkpoint_rejection_count"] == 1.0
    assert observed["resumed_final_state_digest_match"] == 1.0
    assert all(
        metric.observe(observed[metric.metric_id]).passed
        for metric in case.metrics
    )
def test_registered_fly_fgs_worker_gate_blocks_without_native_worker(
    monkeypatch,
):
    case = load_benchmark_registry().case(
        "pipeline.registered_fly_fgs_to_flybody_vertical_slice"
    )
    monkeypatch.setattr(evaluator_module, "_flybody_worker_available", lambda: False)

    result = evaluate_registered_fly_fgs_to_flybody_vertical_slice(case)

    assert result.values == ()
    assert "flygym==2.1.0" in result.blocked_reason
    assert "MuJoCo 3.9.x" in result.blocked_reason


def test_registered_worker_invariants_execute_with_manufactured_adapter():
    """Exercise evaluator logic without promoting the fake as FlyBody evidence."""

    case = load_benchmark_registry().case(
        "pipeline.registered_fly_fgs_to_flybody_vertical_slice"
    )
    result = _evaluate_registered_fly_fgs_to_flybody_with_adapter(
        _DeterministicTelemetryPhysics(),
        case=case,
        fixture_path=CAPTURE_PATH,
    )
    observed = {metric.metric_id: metric.value for metric in result.values}

    assert set(observed) == {metric.metric_id for metric in case.metrics}
    assert all(
        metric.observe(observed[metric.metric_id]).passed
        for metric in case.metrics
    )
    assert observed["registered_fly_fgs_source_scope_match"] == 1.0
    assert observed["circuit_driven_motor_event_count"] > 0.0
    assert observed["circuit_driven_active_muscle_count"] > 0.0
