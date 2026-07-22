import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fly_sensor2behavior.cli import main
from fly_sensor2behavior.evaluators import (
    _COMPARISON_SHAPE_MISMATCH,
    FixtureIntegrityError,
    _evaluate_registered_nod1_to_flybody_with_adapter,
    _evaluate_retinal_to_flybody_vertical_slice_with_adapter,
    _integrated_interval_fluid_impulse,
    _intervention_nontarget_muscle_state_mismatch_count,
    _motor_channel_max_abs_rate_hz,
    _nonsteering_activation_max_abs_delta,
    _open_loop_flybody_torque_program,
    _paired_initial_state_standardized_delta,
    _paired_timebase_max_abs_delta_s,
    _visual_pipeline_repeat_max_standardized_delta,
    _visual_repeat_max_standardized_delta,
    default_evaluators,
    evaluate_cross_atlas_integrity,
    evaluate_flybody_timestep_convergence,
    evaluate_no_future_reads,
)
from fly_sensor2behavior.flight import AerodynamicWrench, RigidBodyState
from fly_sensor2behavior.validation import (
    BenchmarkRegistry,
    GateStatus,
    ValidationReport,
    ValidationRunner,
    default_validation_source_digests,
    load_benchmark_registry,
)


EXECUTABLE_CASES = (
    "contracts.json_roundtrip",
    "timing.no_future_reads",
    "vision.causal_golden_suite",
    "pipeline.retinal_to_body_vertical_slice",
    "feedback.reduced_closed_loop_yaw",
    "nod1.hines_dense_manufactured",
    "physics.reduced_timestep_convergence",
    "nod1.browser_python_parity",
    "evidence.cross_atlas_integrity",
    "evidence.banc_fanc_wing_pathway",
    "fly_fgs.fixed_step_integrity",
    "fly_fgs.incremental_runtime_parity",
    "fly_fgs.checkpoint_reentry",
    "streaming.causal_neuromuscular_runtime",
    "muscle.force_stage_intervention_contract",
    "streaming.fresh_process_checkpoint_reentry",
    "effector.raw_lane_physical_hypotheses",
    "feedback.canonical_fly_fgs_closed_loop",
)

BLOCKED_CASES = (
    "physics.timestep_convergence",
    "hinge.heldout_wing_prediction",
    "power_muscle.heldout_calcium_flight_state",
    "dng02.heldout_wingbeat_amplitude",
)

WORKER_CAPABILITY_CASES = (
    "flybody.adapter_analytic_smoke",
    "physics.flybody_open_loop_convergence",
    "pipeline.retinal_to_flybody_vertical_slice",
    "pipeline.registered_nod1_to_flybody_vertical_slice",
    "pipeline.canonical_online_fly_fgs_to_flybody",
)


def test_registered_timing_evaluator_exercises_schema_to_bridge_availability():
    registry = BenchmarkRegistry.from_json(
        Path("data/benchmarks/registry.v1.json").read_text()
    )
    result = evaluate_no_future_reads(registry.case("timing.no_future_reads"))
    assert {value.metric_id: value.value for value in result.values} == {
        "future_read_count": 0.0,
        "bridge_future_read_count": 0.0,
    }


def test_released_policy_gate_reports_structured_intake_blockers(monkeypatch):
    import fly_sensor2behavior.evaluators as evaluator_module

    monkeypatch.setattr(evaluator_module, "_flybody_worker_available", lambda: True)
    monkeypatch.delenv("FLY_S2B_FLYBODY_RELEASE_ROOT", raising=False)
    monkeypatch.delenv("FLY_S2B_FLYBODY_RELEASE_AUDIT", raising=False)
    case = load_benchmark_registry().case("physics.timestep_convergence")

    result = evaluate_flybody_timestep_convergence(case)

    assert result.blocked_reason is not None
    assert "archive_receipt_missing" in result.blocked_reason
    assert "checkpoint_descriptor_missing" in result.blocked_reason
    assert "saved_model_signature_audit_missing" in result.blocked_reason
    assert "51196886 is excluded" in result.blocked_reason


def test_registered_nod1_worker_evaluator_covers_full_fixture_and_typed_cut():
    registry = load_benchmark_registry()
    case = registry.case("pipeline.registered_nod1_to_flybody_vertical_slice")
    result = _evaluate_registered_nod1_to_flybody_with_adapter(
        _CausalTelemetryPhysics(),
        case=case,
        fixture_path=Path(case.fixture_uri),
    )
    observed = {value.metric_id: value.value for value in result.values}
    assert set(observed) == {metric.metric_id for metric in case.metrics}
    failed = {
        metric.metric_id: observed[metric.metric_id]
        for metric in case.metrics
        if not metric.observe(observed[metric.metric_id]).passed
    }
    assert failed == {}


class _CausalTelemetryPhysics:
    """Deterministic manufactured adapter for the worker evaluator contract."""

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
        axis_torque = np.asarray(axis_torque, dtype=float)
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
            "engine": "manufactured causal telemetry adapter",
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
        self.state.velocity_world_m_s[0] += float(np.sum(asymmetry)) * 1.0e8 * dt_s
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


def _paired_flight_output_fixture():
    rows = 3
    return SimpleNamespace(
        time_s=np.arange(rows, dtype=float) * 1.0e-3,
        position_world_m=np.zeros((rows, 3), dtype=float),
        velocity_world_m_s=np.zeros((rows, 3), dtype=float),
        quaternion_body_to_world=np.tile(
            np.array((1.0, 0.0, 0.0, 0.0)), (rows, 1)
        ),
        angular_velocity_body_rad_s=np.zeros((rows, 3), dtype=float),
        wing_stroke_rad=np.zeros((rows, 2), dtype=float),
        wing_angle_of_attack_rad=np.zeros((rows, 2), dtype=float),
        aerodynamic_force_body_n=np.zeros((rows, 3), dtype=float),
        aerodynamic_torque_body_n_m=np.zeros((rows, 3), dtype=float),
        measured_wing_joint_angle_rad=np.zeros((rows, 6), dtype=float),
        measured_wing_joint_velocity_rad_s=np.zeros((rows, 6), dtype=float),
        measured_wing_joint_order=tuple("wing-%d" % index for index in range(6)),
        whole_fly_com_position_world_m=np.zeros((rows, 3), dtype=float),
        ground_contact_count=np.zeros(rows, dtype=np.int64),
        external_actuator_torque_n_m=np.zeros((rows, 6), dtype=float),
        physics_time_s=np.arange(rows, dtype=float) * 1.0e-3,
        ground_contact_transition_point_count=np.zeros(rows, dtype=np.int64),
        external_actuator_torque_physics_n_m=np.zeros((rows, 6), dtype=float),
        motor_event_times_s={"left:b1": np.array((0.001,))},
        motor_event_phases_rad={"left:b1": np.array((0.2,))},
        muscle_activation={
            "left:b1": np.array((0.0, 0.2, 0.1)),
            "left:DLM": np.array((0.4, 0.4, 0.4)),
        },
        muscle_force_n={
            "left:b1": np.array((0.0, 1.0e-6, 0.0)),
            "left:DLM": np.array((2.0e-5, 2.0e-5, 2.0e-5)),
        },
        muscle_phase_effect={
            "left:b1": np.array((0.0, 0.1, 0.0)),
            "left:DLM": np.ones(rows, dtype=float),
        },
        muscle_work_j={
            "left:b1": np.array((0.0, 1.0e-10, 0.0)),
            "left:DLM": np.array((0.0, 2.0e-9, 4.0e-9)),
        },
    )


def _paired_run_fixture(flight):
    frame = SimpleNamespace(
        exposure_start_s=0.0,
        exposure_end_s=1.0e-3,
        measurement_time_s=5.0e-4,
        availability_time_s=1.5e-3,
    )
    return SimpleNamespace(
        flight=flight,
        circuit=SimpleNamespace(sample_times_s=(0.0, 1.0e-3, 2.0e-3)),
        retinal_frames=(frame,),
    )


def test_paired_flybody_helpers_match_initial_state_and_all_clocks():
    first = _paired_flight_output_fixture()
    second = deepcopy(first)
    expected_root = SimpleNamespace(
        position_world_m=np.zeros(3),
        velocity_world_m_s=np.zeros(3),
        quaternion_body_to_world=np.array((1.0, 0.0, 0.0, 0.0)),
        angular_velocity_body_rad_s=np.zeros(3),
    )

    assert _paired_initial_state_standardized_delta(
        (first, second),
        expected_root,
        np.zeros(6),
        np.zeros(6),
        first.measured_wing_joint_order,
        np.zeros(3),
    ) == 0.0
    assert _paired_timebase_max_abs_delta_s(
        _paired_run_fixture(first), (_paired_run_fixture(second),)
    ) == 0.0

    second.measured_wing_joint_angle_rad[0, 2] = 0.25
    assert _paired_initial_state_standardized_delta(
        (first, second),
        expected_root,
        np.zeros(6),
        np.zeros(6),
        first.measured_wing_joint_order,
        np.zeros(3),
    ) == pytest.approx(0.25)
    shifted = _paired_run_fixture(deepcopy(first))
    shifted.retinal_frames[0].availability_time_s = 1.6e-3
    assert _paired_timebase_max_abs_delta_s(
        _paired_run_fixture(first), (shifted,)
    ) == pytest.approx(1.0e-4)


def test_visual_repeat_metric_covers_trajectory_force_and_muscle_state():
    reference = _paired_flight_output_fixture()
    repeated = deepcopy(reference)

    assert _visual_repeat_max_standardized_delta(reference, repeated) == 0.0

    repeated.aerodynamic_force_body_n[1, 0] = 2.0e-7
    assert _visual_repeat_max_standardized_delta(reference, repeated) == pytest.approx(
        0.2
    )
    repeated = deepcopy(reference)
    repeated.muscle_activation["left:DLM"][2] += 0.1
    assert _visual_repeat_max_standardized_delta(reference, repeated) == pytest.approx(
        0.1
    )
    repeated = deepcopy(reference)
    repeated.ground_contact_count[1] = 1
    assert _visual_repeat_max_standardized_delta(reference, repeated) == pytest.approx(
        1.0
    )
    repeated = deepcopy(reference)
    repeated.external_actuator_torque_n_m[2, 4] = 2.0e-10
    assert _visual_repeat_max_standardized_delta(reference, repeated) == pytest.approx(
        0.2
    )
    repeated = deepcopy(reference)
    repeated.external_actuator_torque_physics_n_m[1, 3] = 3.0e-10
    assert _visual_repeat_max_standardized_delta(reference, repeated) == pytest.approx(
        0.3
    )
    reference.diagnostics = SimpleNamespace(
        integrated_aerodynamic_impulse_n_s=(0.0, 0.0, 0.0),
        metrics={"ground_contact_transition_count": 0.0},
        warnings=("contact-free",),
        physics_provenance={"engine": "manufactured"},
    )
    repeated = deepcopy(reference)
    assert _visual_repeat_max_standardized_delta(reference, repeated) == 0.0
    repeated.diagnostics.metrics["ground_contact_transition_count"] = 1.0
    assert (
        _visual_repeat_max_standardized_delta(reference, repeated)
        == _COMPARISON_SHAPE_MISMATCH
    )


def test_visual_pipeline_repeat_metric_covers_upstream_records():
    flight = _paired_flight_output_fixture()
    first = SimpleNamespace(
        retinal_frames=("retina",),
        circuit=("circuit",),
        bridge=("bridge",),
        flight_config=("config",),
        source_metadata={"source": "fixture"},
        flight=flight,
    )
    repeated = deepcopy(first)
    assert _visual_pipeline_repeat_max_standardized_delta(first, repeated) == 0.0
    repeated.bridge = ("changed",)
    assert _visual_pipeline_repeat_max_standardized_delta(
        first, repeated
    ) == 1.0e300


def test_visual_ablation_invariance_is_scoped_to_nonsteering_muscles():
    neutral = _paired_flight_output_fixture()
    moving = deepcopy(neutral)
    moving.muscle_activation["left:b1"][1] += 0.5

    assert _nonsteering_activation_max_abs_delta(
        moving, neutral, ("left:b1",)
    ) == 0.0

    moving.muscle_activation["left:DLM"][1] += 0.03
    assert _nonsteering_activation_max_abs_delta(
        moving, neutral, ("left:b1",)
    ) == pytest.approx(0.03)


def test_mn_intervention_cut_contract_passes_with_causal_fake_adapter():
    raw = _evaluate_retinal_to_flybody_vertical_slice_with_adapter(
        _CausalTelemetryPhysics()
    )
    benchmark = load_benchmark_registry().case(
        "pipeline.retinal_to_flybody_vertical_slice"
    )
    values = {item.metric_id: item.value for item in raw.values}

    assert set(values) == {metric.metric_id for metric in benchmark.metrics}
    assert all(metric.observe(values[metric.metric_id]).passed for metric in benchmark.metrics)
    assert values["mn_silence_baseline_target_event_count"] >= 1.0
    assert values["mn_silence_target_event_count"] == 0.0
    assert values["mn_silence_target_rate_max_abs_hz"] == 0.0
    assert values["mn_silence_upstream_mismatch_count"] == 0.0
    assert values["mn_silence_nontarget_event_mismatch_count"] == 0.0
    assert values["mn_silence_nontarget_muscle_state_mismatch_count"] == 0.0


def test_target_rate_invariant_detects_a_negative_nonzero_sample():
    channel = SimpleNamespace(
        rate_samples=(
            SimpleNamespace(rate_hz=0.0),
            SimpleNamespace(rate_hz=-1.0e-12),
        )
    )

    assert _motor_channel_max_abs_rate_hz(channel) == pytest.approx(1.0e-12)


def test_nontarget_muscle_locality_excludes_downstream_work():
    reference = _paired_flight_output_fixture()
    intervention = deepcopy(reference)
    intervention.muscle_activation["left:b1"][:] = 0.0
    intervention.muscle_force_n["left:b1"][:] = 0.0
    intervention.muscle_phase_effect["left:b1"][:] = 0.0
    # Work is a downstream hinge-velocity outcome and may change even for an
    # intrinsically unchanged non-target muscle.
    intervention.muscle_work_j["left:DLM"][-1] += 1.0e-9
    assert _intervention_nontarget_muscle_state_mismatch_count(
        reference,
        intervention,
        target_muscle_key="left:b1",
    ) == 0.0


@pytest.mark.parametrize("attribute", ("muscle_force_n", "muscle_phase_effect"))
def test_nontarget_muscle_locality_detects_tiny_force_or_phase_mutation(attribute):
    reference = _paired_flight_output_fixture()
    intervention = deepcopy(reference)
    values = getattr(intervention, attribute)["left:DLM"]
    values[-1] = np.nextafter(values[-1], np.inf)

    assert _intervention_nontarget_muscle_state_mismatch_count(
        reference,
        intervention,
        target_muscle_key="left:b1",
    ) == 1.0


def test_open_loop_flybody_stress_program_preserves_control_ticks():
    coarse = _open_loop_flybody_torque_program(0.001, 1.0e-4)
    fine = _open_loop_flybody_torque_program(0.001, 5.0e-5)
    ultrafine = _open_loop_flybody_torque_program(0.001, 2.5e-5)

    assert coarse.shape == (10, 6)
    assert fine.shape == (20, 6)
    assert ultrafine.shape == (40, 6)
    assert coarse[::2] == pytest.approx(fine[::4])
    assert coarse[::2] == pytest.approx(ultrafine[::8])
    assert coarse.reshape(5, 2, 6) == pytest.approx(
        coarse[::2, None, :].repeat(2, axis=1)
    )


def test_flybody_impulse_integrates_applied_interval_samples_not_endpoints():
    class Trace:
        time_s = np.array([0.0, 0.1, 0.2])
        # Index zero is an initial-state diagnostic.  Samples one and two are
        # the forces actually applied over [0,.1] and [.1,.2], respectively.
        root_fluid_force_n = np.array(
            [[100.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]
        )

    assert _integrated_interval_fluid_impulse(Trace()) == pytest.approx(
        [0.4, 0.6, 0.0]
    )


def test_builtin_executable_cases_pass_registered_metrics():
    registry = load_benchmark_registry()
    report = ValidationRunner(registry, default_evaluators(registry)).run(
        evaluation_id="built-in-local-cases",
        case_ids=EXECUTABLE_CASES,
        include_dependents=False,
    )
    results = {result.case_id: result for result in report.results}

    assert set(results) == set(EXECUTABLE_CASES)
    assert all(result.status is GateStatus.PASS for result in results.values())
    assert all(result.observations for result in results.values())


def test_worker_cross_runtime_and_empirical_claims_are_explicitly_blocked():
    registry = load_benchmark_registry()
    report = ValidationRunner(registry, default_evaluators(registry)).run(
        evaluation_id="unsupported-cases",
        case_ids=BLOCKED_CASES,
        include_dependents=False,
        include_prerequisites=False,
    )

    assert {result.case_id for result in report.results} == set(BLOCKED_CASES)
    assert all(result.status is GateStatus.BLOCKED for result in report.results)
    assert all(result.reason and len(result.reason) > 40 for result in report.results)
    assert not any(result.observations for result in report.results)


def test_worker_capability_cases_pass_in_pinned_worker_or_block_elsewhere():
    from fly_sensor2behavior.flybody_adapter import (
        PINNED_FLYGYM_VERSION,
        PINNED_MUJOCO_SERIES,
        dependency_versions,
    )

    registry = load_benchmark_registry()
    report = ValidationRunner(registry, default_evaluators(registry)).run(
        evaluation_id="worker-capability-cases",
        case_ids=WORKER_CAPABILITY_CASES,
        include_dependents=False,
        include_prerequisites=False,
    )
    results = {result.case_id: result for result in report.results}

    assert set(results) == set(WORKER_CAPABILITY_CASES)
    versions = dependency_versions()
    native_worker = (
        versions["flygym"] == PINNED_FLYGYM_VERSION
        and isinstance(versions["mujoco"], str)
        and versions["mujoco"].startswith(PINNED_MUJOCO_SERIES + ".")
    )
    for case_id, result in results.items():
        requires_node = case_id == "pipeline.canonical_online_fly_fgs_to_flybody"
        expected_pass = native_worker and (
            not requires_node or shutil.which("node") is not None
        )
        if expected_pass:
            assert result.status is GateStatus.PASS
            assert result.observations
        else:
            assert result.status is GateStatus.BLOCKED
            assert result.reason


def test_evidence_evaluator_rejects_fixture_digest_drift(tmp_path, monkeypatch):
    registry = load_benchmark_registry()
    benchmark = registry.case("evidence.cross_atlas_integrity")
    changed_fixture = tmp_path / "seed_graph.v1.json"
    changed_fixture.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "fly_sensor2behavior.evaluators.default_seed_graph_path",
        lambda: changed_fixture,
    )

    with pytest.raises(FixtureIntegrityError, match="fixture digest mismatch"):
        evaluate_cross_atlas_integrity(benchmark)


def test_validate_cli_writes_exclusive_canonical_report_and_dependency_selection(
    tmp_path, capsys
):
    output = tmp_path / "validation.json"
    return_code = main(
        (
            "validate",
            "--output",
            str(output),
            "--evaluation-id",
            "dependency-validation",
            "--dependency",
            "anatomy.crosswalk",
            "--no-dependents",
        )
    )
    summary = capsys.readouterr().out
    raw = output.read_text(encoding="utf-8")
    report = ValidationReport.from_json(raw)

    assert return_code == 0
    assert '"status": "blocked"' not in summary
    assert report.selected_case_ids == (
        "contracts.json_roundtrip",
        "evidence.cross_atlas_integrity",
        "evidence.banc_fanc_wing_pathway",
    )
    assert raw == report.to_json()
    assert hashlib.sha256(raw.encode("utf-8")).hexdigest() == report.content_sha256
    with pytest.raises(SystemExit) as exc_info:
        main(("validate", "--output", str(output), "--case", EXECUTABLE_CASES[0]))
    assert exc_info.value.code == 2


def test_validate_cli_records_declared_worker_image_digest(
    tmp_path, capsys, monkeypatch
):
    digest = "ab" * 32
    monkeypatch.setenv("FLY_S2B_WORKER_IMAGE_DIGEST", "sha256:" + digest)
    output = tmp_path / "worker-bound-validation.json"

    return_code = main(
        (
            "validate",
            "--output",
            str(output),
            "--evaluation-id",
            "worker-bound-validation",
            "--case",
            "contracts.json_roundtrip",
            "--no-dependents",
            "--no-prerequisites",
        )
    )
    capsys.readouterr()
    report = ValidationReport.from_json(output.read_text(encoding="utf-8"))

    assert return_code == 0
    image_receipts = [
        item
        for item in report.source_digests
        if item.kind == "image" and item.name == "fly-s2b-worker-image"
    ]
    assert len(image_receipts) == 1
    assert image_receipts[0].sha256 == digest
    lock_receipts = [
        item
        for item in report.source_digests
        if item.kind == "dependency_lock" and item.name == "requirements.lock"
    ]
    assert len(lock_receipts) == 1
    expected_lock = hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "requirements.lock").read_bytes()
    ).hexdigest()
    assert lock_receipts[0].sha256 == expected_lock
    protocol_receipts = [
        item for item in report.source_digests if item.kind == "protocol"
    ]
    assert len(protocol_receipts) == 4
    assert all(item.name.startswith("data/benchmarks/protocols/") for item in protocol_receipts)


def test_validate_cli_binds_generated_report_before_writing(
    tmp_path, monkeypatch
):
    registry = load_benchmark_registry()
    valid = ValidationRunner(registry, default_evaluators(registry)).run(
        evaluation_id="forged-runner-output",
        case_ids=("contracts.json_roundtrip",),
        include_dependents=False,
        include_prerequisites=False,
    )
    forged = replace(valid, registry_sha256="0" * 64)
    monkeypatch.setattr(
        "fly_sensor2behavior.cli.ValidationRunner.run",
        lambda self, **kwargs: forged,
    )
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(SystemExit) as exc_info:
        main(("validate", "--output", str(output)))

    assert exc_info.value.code == 2
    assert not output.exists()


def test_export_web_cli_requires_exact_registry_for_attached_report(
    tmp_path, capsys
):
    canonical = load_benchmark_registry()
    contract = canonical.case("contracts.json_roundtrip")
    custom = BenchmarkRegistry(
        registry_id="custom-web-export-registry",
        version="1.0.0",
        description="Exact report-binding CLI fixture",
        cases=(contract,),
    )
    registry_path = tmp_path / "custom-registry.json"
    registry_path.write_text(custom.to_json(), encoding="utf-8")
    report = ValidationRunner(custom, default_evaluators(custom)).run(
        evaluation_id="custom-web-export",
        source_digests=default_validation_source_digests(custom, registry_path),
    )
    report_path = tmp_path / "custom-report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")

    rejected_output = tmp_path / "rejected-web"
    with pytest.raises(SystemExit) as exc_info:
        main(
            (
                "export-web",
                "--output",
                str(rejected_output),
                "--validation-report",
                str(report_path),
            )
        )
    assert exc_info.value.code == 2
    assert not rejected_output.exists()

    accepted_output = tmp_path / "accepted-web"
    assert (
        main(
            (
                "export-web",
                "--output",
                str(accepted_output),
                "--scenarios",
                "baseline",
                "--duration-s",
                "0.012",
                "--sample-rate-hz",
                "100",
                "--validation-report",
                str(report_path),
                "--validation-registry",
                str(registry_path),
            )
        )
        == 0
    )
    capsys.readouterr()
    attached_manifest = json.loads(
        (accepted_output / "manifest.json").read_text(encoding="utf-8")
    )
    attached_report_path = accepted_output / attached_manifest[
        "validation_report_url"
    ].removeprefix("data/")
    attached = ValidationReport.from_json(
        attached_report_path.read_text(encoding="utf-8"),
        registry=custom,
    )
    assert attached == report
    attached_registry_path = accepted_output / attached_manifest[
        "validation_registry_url"
    ].removeprefix("data/")
    attached_registry = load_benchmark_registry(
        attached_registry_path
    )
    assert attached_registry == custom
    assert attached_manifest["validation_registry_canonical_sha256"] == (
        custom.content_sha256
    )


def test_validate_cli_returns_zero_for_blocked_and_one_only_for_failed(
    tmp_path, capsys
):
    blocked_output = tmp_path / "blocked.json"
    blocked_code = main(
        (
            "validate",
            "--output",
            str(blocked_output),
            "--case",
            "physics.timestep_convergence",
            "--no-dependents",
            "--no-prerequisites",
        )
    )
    capsys.readouterr()
    blocked = ValidationReport.from_json(blocked_output.read_text(encoding="utf-8"))
    assert blocked_code == 0
    assert blocked.overall_status is GateStatus.BLOCKED

    default = load_benchmark_registry()
    contract = default.case("contracts.json_roundtrip")
    failing_metric = replace(contract.metrics[0], target_value=0.0)
    failing_case = replace(contract, metrics=(failing_metric,))
    custom = BenchmarkRegistry(
        registry_id="deliberately-failing-registry",
        version="1.0.0",
        description="CLI exit-code fixture",
        cases=(failing_case,),
    )
    registry_path = tmp_path / "failing-registry.json"
    registry_path.write_text(custom.to_json(), encoding="utf-8")
    failed_output = tmp_path / "failed.json"
    failed_code = main(
        (
            "validate",
            "--output",
            str(failed_output),
            "--registry",
            str(registry_path),
        )
    )
    capsys.readouterr()
    failed = ValidationReport.from_json(failed_output.read_text(encoding="utf-8"))

    assert failed_code == 1
    assert failed.overall_status is GateStatus.FAIL
