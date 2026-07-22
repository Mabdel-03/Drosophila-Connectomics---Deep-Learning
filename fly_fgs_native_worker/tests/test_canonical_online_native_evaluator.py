from __future__ import annotations

import copy
import dataclasses
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import fly_sensor2behavior.evaluators as evaluator_module
from fly_sensor2behavior.evaluators import (
    _CANONICAL_ONLINE_PINNED_NODE_VERSION,
    _load_canonical_online_native_pair_contract,
    _new_canonical_online_circuit_runtime,
    _online_native_compiled_receipt_matches,
    _online_native_event_to_next_wing_command_violation_count,
    _online_native_pre_onset_transition_applied_mismatch_count,
    _online_native_source_receipt_matches,
    default_evaluators,
    evaluate_canonical_online_fly_fgs_to_flybody,
)
from fly_sensor2behavior.flight.streaming_bridge import (
    RawAppSide,
    StreamingMotorEvent,
)
from fly_sensor2behavior.flight.streaming_mechanics import (
    MechanicsInterventionMode,
    StreamingMechanicsConfig,
    StreamingMuscleIntervention,
    StreamingMuscleWingStepper,
)
from fly_sensor2behavior.schema import REVIEWED_FLYBODY_WING_AXIS_ORDER
from fly_sensor2behavior.schema import AnatomicalSide
from fly_sensor2behavior.validation import (
    PromotionGate,
    RuntimeTier,
    load_benchmark_registry,
)


CASE_ID = "pipeline.canonical_online_fly_fgs_to_flybody"


@dataclass(frozen=True)
class _AppliedPhysicsTransition:
    interval_start_s: float
    command_n_m: tuple[float, ...]
    body_state_after: tuple[float, ...]


def _assert_registered_metrics_pass(case, result) -> None:
    assert result.blocked_reason is None
    observed = {value.metric_id: value.value for value in result.values}
    assert set(observed) == {metric.metric_id for metric in case.metrics}
    assert all(
        metric.observe(observed[metric.metric_id]).passed
        for metric in case.metrics
    )


def test_registry_declares_scheduled_native_gate_without_scientific_promotion():
    registry = load_benchmark_registry()
    case = registry.case(CASE_ID)

    assert registry.version == "1.18.0"
    assert len(registry.cases) == 29
    assert case.runtime_tier is RuntimeTier.SCHEDULED
    assert case.hard_gate is True
    assert case.promotion_gate is PromotionGate.SOFTWARE_CORRECT
    assert (
        case.fixture_uri
        == "data/benchmarks/scenarios/canonical-online-native-pair.v1.json"
    )
    assert case.input_sha256 == (
        "b133bce4878e18db4eb860a3dc37436cd9825f10d1b5d3eef25c90954ece0361"
    )
    assert {
        "effector.raw_lane_physical_hypotheses",
        "feedback.canonical_fly_fgs_closed_loop",
        "flybody.checkpoint_reentry",
        "muscle.force_stage_intervention_contract",
        "pipeline.registered_fly_fgs_to_flybody_vertical_slice",
    } == set(case.prerequisite_case_ids)
    assert "live content-addressed fly-FGS Node circuit" in case.claim
    assert "anatomically unknown" in case.claim
    assert "signed yaw/roll conclusions are prohibited" in case.claim
    assert "stable flight" in case.claim
    assert "behavior" in case.claim
    metric_ids = {metric.metric_id for metric in case.metrics}
    assert {
        "online_native_feedback_mismatch_count",
        "online_native_source_runtime_receipt_match",
        "online_native_compiled_model_receipt_match",
        "online_native_generated_event_count",
        "online_native_applied_event_count",
        "online_native_targeted_natural_muscle_response_count",
        "online_native_targeted_effective_muscle_response_count",
        "online_native_command_actuator_max_abs_delta_n_m",
        "online_native_intervention_pre_onset_numeric_mismatch_count",
        "online_native_intervention_pre_onset_generated_event_mismatch_count",
        "online_native_intervention_target_suppressed_count",
        "online_native_intervention_event_locality_mismatch_count",
        "online_native_intervention_post_onset_measured_wing_delta_rad",
        "online_native_intervention_final_root_position_delta_m",
        "online_native_ground_contact_count",
        "online_native_composite_checkpoint_tail_digest_match",
        "online_native_composite_checkpoint_final_match",
        "online_native_scientific_limitations_preserved",
    } <= metric_ids
    assert CASE_ID in default_evaluators(registry)


def test_paired_scenario_contract_loads_exact_config_and_intervention():
    scenario_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "benchmarks"
        / "scenarios"
        / "canonical-online-native-pair.v1.json"
    )

    scenario, intervention = _load_canonical_online_native_pair_contract(
        scenario_path
    )

    common = scenario["common_configuration"]
    assert common["duration_s"] == 0.1
    assert common["checkpoint_time_s"] == 0.05
    assert common["figure_velocity_rad_s"] == pytest.approx(math.pi / 6)
    assert common["seed"] == 73
    assert intervention == {
        "end_s": 0.1,
        "intervention_id": "raw-l-iv2-silence-60-100ms",
        "mode": "silence",
        "muscle": "iv2",
        "output_scale": 0.0,
        "raw_app_side": "L",
        "start_s": 0.06,
    }


def test_public_native_gate_blocks_when_node_is_unavailable(monkeypatch):
    case = load_benchmark_registry().case(CASE_ID)
    monkeypatch.setattr(evaluator_module.shutil, "which", lambda _name: None)

    result = evaluate_canonical_online_fly_fgs_to_flybody(case)

    assert result.values == ()
    assert "requires Node.js" in result.blocked_reason
    assert "live circuit sidecar" in result.blocked_reason


def test_public_native_gate_blocks_without_pinned_flybody_worker(monkeypatch):
    case = load_benchmark_registry().case(CASE_ID)
    monkeypatch.setattr(
        evaluator_module.shutil, "which", lambda _name: "/usr/bin/node"
    )
    monkeypatch.setattr(evaluator_module, "_flybody_worker_available", lambda: False)

    result = evaluate_canonical_online_fly_fgs_to_flybody(case)

    assert result.values == ()
    assert "flygym==2.1.0" in result.blocked_reason
    assert "MuJoCo 3.9.x" in result.blocked_reason
    assert "no reduced or manufactured physics fallback" in result.blocked_reason


def test_scheduled_circuit_runtime_construction_pins_exact_node_release(monkeypatch):
    import fly_sensor2behavior.fly_fgs_runtime as runtime_module

    observed = {}
    sentinel = object()

    def fake_runtime(**kwargs):
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(runtime_module, "NodeFlyFGSCircuitRuntime", fake_runtime)

    assert _new_canonical_online_circuit_runtime() is sentinel
    assert _CANONICAL_ONLINE_PINNED_NODE_VERSION == "v22.22.1"
    assert observed == {
        "request_timeout_s": 180.0,
        "expected_node_version": "v22.22.1",
    }


def test_live_source_receipt_checker_rejects_source_or_policy_drift():
    from fly_sensor2behavior.fly_fgs import (
        FLY_FGS_SOURCE_MANIFEST_SHA256,
        FLY_FGS_SNAPSHOT_ID,
    )
    from fly_sensor2behavior.fly_fgs_runtime import FLY_FGS_RUNTIME_PROTOCOL_VERSION

    receipt = {
        "event": "ready",
        "protocol_version": FLY_FGS_RUNTIME_PROTOCOL_VERSION,
        "node_version": "v22.22.1",
        "snapshot_id": FLY_FGS_SNAPSHOT_ID,
        "source_manifest_sha256": FLY_FGS_SOURCE_MANIFEST_SHA256,
        "circuit_engine_sha256": "1" * 64,
        "circuit_bundle_sha256": "2" * 64,
        "dt_s": 0.005,
        "sample_count": 100,
        "pre_roll_steps": 48,
        "motor_input_policy": "four individual NOD1 voltage channels only",
        "full_cell_state_eligible_motor_input": False,
        "executable_asset_ids": ["circuit_engine", "circuit_bundle"],
    }

    assert _online_native_source_receipt_matches(receipt)
    changed = dict(receipt)
    changed["motor_input_policy"] = "full circuit state"
    assert not _online_native_source_receipt_matches(changed)
    changed = dict(receipt)
    changed["source_manifest_sha256"] = "0" * 64
    assert not _online_native_source_receipt_matches(changed)
    for wrong_node_version in ("v22.22.0", "v24.0.0", "22.22.1", ""):
        changed = dict(receipt)
        changed["node_version"] = wrong_node_version
        assert not _online_native_source_receipt_matches(changed)


def test_compiled_receipt_checker_binds_model_versions_and_axis_order(monkeypatch):
    import fly_sensor2behavior.flybody_adapter as adapter_module

    versions = {"flygym": "2.1.0", "mujoco": "3.9.0"}
    records = {
        "flygym": "sha256:" + "3" * 64,
        "mujoco": "sha256:" + "4" * 64,
    }
    lock_digest = "sha256:" + "5" * 64
    model_digest = "sha256:" + "6" * 64
    worker_image_digest = "sha256:" + "a" * 64
    monkeypatch.setattr(adapter_module, "dependency_versions", lambda: versions)
    monkeypatch.setattr(
        adapter_module, "dependency_record_fingerprints", lambda: records
    )
    monkeypatch.setattr(
        adapter_module, "worker_dependency_lock_sha256", lambda: lock_digest
    )
    monkeypatch.setattr(
        adapter_module,
        "declared_worker_image_digest",
        lambda: worker_image_digest,
    )
    receipt = {
        "schema_version": "1.0.0",
        "engine": "FlyGym/FlyBody with native MuJoCo",
        "worker_versions": versions,
        "worker_image_digest": worker_image_digest,
        "worker_image_digest_status": "declared_oci_digest",
        "worker_dependency_lock": {
            "name": "requirements.lock",
            "sha256": lock_digest,
        },
        "dependency_record_sha256": records,
        "compiled_model_fingerprint": {"sha256": model_digest},
        "worker_config": {
            "timestep_s": 0.0001,
            "spawn_height_m": 0.1,
            "max_abs_wing_torque_n_m": 3.0e-6,
            "add_aerodynamic_geoms": True,
        },
        "wing_dof_order": list(REVIEWED_FLYBODY_WING_AXIS_ORDER),
        "fluid_geoms_contact_disabled": True,
        "ground_contact_topology": "legs_only",
        "released_policy_topology_equivalent": False,
        "tendon_count": 8,
        "source": {
            "flygym_tag": "v2.1.0",
            "flygym_commit": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f",
        },
        "checkpoint_contract": {
            "schema_version": "1.0.0",
            "state_spec": "mjSTATE_INTEGRATION",
            "float_encoding": "float64_le_base64",
            "compatibility_bound_to_compiled_model": True,
            "fresh_adapter_exact_reentry_required": True,
        },
        "public_units": "SI",
        "root_fluid_wrench_scope": (
            "root-total; no reviewed per-wing decomposition"
        ),
    }
    checkpoint = {
        "components": {
            "physics": {
                "compiled_model_sha256": model_digest,
                "worker_versions": versions,
                "physics_timestep_s": 0.0001,
                "wing_dof_order": list(REVIEWED_FLYBODY_WING_AXIS_ORDER),
            }
        }
    }

    assert _online_native_compiled_receipt_matches(receipt, checkpoint)
    changed = copy.deepcopy(checkpoint)
    changed["components"]["physics"]["compiled_model_sha256"] = (
        "sha256:" + "7" * 64
    )
    assert not _online_native_compiled_receipt_matches(receipt, changed)

    for invalid_digest, invalid_status in (
        (None, "not_declared"),
        ("", "declared_oci_digest"),
        ("sha256:test-only", "declared_oci_digest"),
        ("sha256:" + "A" * 64, "declared_oci_digest"),
        (worker_image_digest, "not_declared"),
        (worker_image_digest, ""),
        ("sha256:" + "b" * 64, "declared_oci_digest"),
    ):
        changed_receipt = copy.deepcopy(receipt)
        changed_receipt["worker_image_digest"] = invalid_digest
        changed_receipt["worker_image_digest_status"] = invalid_status
        assert not _online_native_compiled_receipt_matches(
            changed_receipt, checkpoint
        )

    monkeypatch.setattr(
        adapter_module, "declared_worker_image_digest", lambda: None
    )
    assert not _online_native_compiled_receipt_matches(receipt, checkpoint)


def test_pre_onset_comparison_uses_transition_applied_state_at_exact_grid_onset():
    """An onset endpoint may gate next state without changing prior physics."""

    silence = StreamingMuscleIntervention(
        intervention_id="on-grid-iv2-cut",
        muscle="iv2",
        start_s=0.0001,
        end_s=0.0003,
        mode=MechanicsInterventionMode.SILENCE,
        raw_app_side=RawAppSide.L,
        output_scale=0.0,
    )
    baseline = StreamingMuscleWingStepper()
    intervention = StreamingMuscleWingStepper(
        StreamingMechanicsConfig(muscle_interventions=(silence,))
    )
    preloaded_event = StreamingMotorEvent(
        event_id="preloaded-before-onset",
        motor_neuron="MN-iv2",
        muscle="iv2",
        raw_app_side=RawAppSide.L,
        anatomical_side=AnatomicalSide.UNKNOWN,
        event_time_s=0.0,
        availability_time_s=0.0,
        wingbeat_phase_rad=0.20 * 2.0 * math.pi,
        rate_hz=200.0,
        emission_probability=1.0,
        source_measurement_time_s=0.0,
        generator_seed=73,
        phase_crossing_index=0,
    )
    baseline.push_delivered_events((preloaded_event,))
    intervention.push_delivered_events((preloaded_event,))

    baseline_frame = baseline.step()
    intervention_frame = intervention.step()
    baseline_endpoint = baseline_frame.muscle_snapshot.individual["left:iv2"]
    intervention_endpoint = intervention_frame.muscle_snapshot.individual[
        "left:iv2"
    ]

    assert baseline_frame.applied_event_ids == ("preloaded-before-onset",)
    assert intervention_frame.applied_event_ids == ("preloaded-before-onset",)
    assert baseline_endpoint.force_n > 0.0
    # The endpoint is exactly 0.1 ms, so it is already the effective state for
    # the intervention-active next interval.
    assert intervention_endpoint.force_n == 0.0
    np.testing.assert_array_equal(
        baseline_frame.actuation_wing_kinematics.wing_axis_torque_n_m,
        intervention_frame.actuation_wing_kinematics.wing_axis_torque_n_m,
    )

    applied_transition = _AppliedPhysicsTransition(
        interval_start_s=0.0,
        command_n_m=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        body_state_after=(0.1, 0.2, 0.3),
    )
    assert (
        _online_native_pre_onset_transition_applied_mismatch_count(
            baseline_frame,
            intervention_frame,
            applied_transition,
            applied_transition,
            onset_s=0.0001,
        )
        == 0
    )

    changed_pre_onset_transition = dataclasses.replace(
        applied_transition,
        command_n_m=(9.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    assert (
        _online_native_pre_onset_transition_applied_mismatch_count(
            baseline_frame,
            intervention_frame,
            applied_transition,
            changed_pre_onset_transition,
            onset_s=0.0001,
        )
        == 1
    )

    # A transition whose left boundary is exactly the onset is correctly not
    # constrained by pre-onset identity under half-open interval semantics.
    baseline_onset = dataclasses.replace(
        applied_transition, interval_start_s=0.0001
    )
    intervention_onset = dataclasses.replace(
        changed_pre_onset_transition, interval_start_s=0.0001
    )
    assert (
        _online_native_pre_onset_transition_applied_mismatch_count(
            baseline_frame,
            intervention_frame,
            baseline_onset,
            intervention_onset,
            onset_s=0.0001,
        )
        == 0
    )


def test_event_counterfactual_requires_next_wing_command_propagation():
    assert (
        _online_native_event_to_next_wing_command_violation_count(
            StreamingMuscleWingStepper
        )
        == 0
    )

    class _TelemetryOnlyEventMutant(StreamingMuscleWingStepper):
        """Update true muscle state but substitute an event-free hinge output."""

        def __init__(self):
            super().__init__()
            self._event_free_hinge_path = StreamingMuscleWingStepper()

        def step(self, *, measured_phase_end_unwrapped_rad=None):
            actual = super().step(
                measured_phase_end_unwrapped_rad=measured_phase_end_unwrapped_rad
            )
            event_free = self._event_free_hinge_path.step(
                measured_phase_end_unwrapped_rad=measured_phase_end_unwrapped_rad
            )
            return dataclasses.replace(
                actual,
                actuation_wing_kinematics=(
                    event_free.actuation_wing_kinematics
                ),
                wing_kinematics=event_free.wing_kinematics,
            )

    assert (
        _online_native_event_to_next_wing_command_violation_count(
            _TelemetryOnlyEventMutant
        )
        >= 1
    )


@pytest.mark.skipif(
    shutil.which("node") is None or not evaluator_module._flybody_worker_available(),
    reason="pinned FlyBody worker and Node.js are unavailable",
)
def test_native_worker_executes_every_registered_online_metric():
    case = load_benchmark_registry().case(CASE_ID)

    result = evaluate_canonical_online_fly_fgs_to_flybody(case)

    _assert_registered_metrics_pass(case, result)
    observed = {value.metric_id: value.value for value in result.values}
    assert observed["online_native_circuit_sample_count"] == 20.0
    assert observed["online_native_physics_transition_count"] == 1000.0
    assert observed["online_native_intervention_physics_transition_count"] == 1000.0
    assert observed["online_native_generated_event_count"] >= 1.0
    assert observed["online_native_applied_event_count"] >= 1.0
    assert observed["online_native_intervention_target_suppressed_count"] >= 1.0
    assert (
        observed[
            "online_native_intervention_pre_onset_generated_event_mismatch_count"
        ]
        == 0.0
    )
    assert observed["online_native_intervention_pre_onset_numeric_mismatch_count"] == 0.0
    assert observed["online_native_intervention_event_locality_mismatch_count"] == 0.0
    assert observed["online_native_intervention_max_target_actuation_force_n"] == 0.0
    assert observed["online_native_intervention_post_onset_measured_wing_delta_rad"] > 0.0
    assert observed["online_native_intervention_final_root_position_delta_m"] > 0.0
    assert observed["online_native_ground_contact_count"] == 0.0
    assert observed["online_native_composite_checkpoint_final_match"] == 1.0
