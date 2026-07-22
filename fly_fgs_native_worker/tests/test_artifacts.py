import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import fly_sensor2behavior.artifacts as artifacts_module

from fly_sensor2behavior.artifacts import (
    AUTHORITY_NOTICE,
    GROUND_CONTACT_SAMPLE_SEMANTICS,
    GROUND_CONTACT_TELEMETRY_KIND,
    SCENARIO_NAMES,
    audit_evidence,
    build_scenario_config,
    episode_to_web_replay,
    export_web_replay,
    read_chunked_array,
    run_scenario,
    sha256_file,
    web_replay_projection_sha256,
    write_episode_artifact,
)
from fly_sensor2behavior.flight import (
    AerodynamicWrench,
    FlightEpisodeRunner,
    FlightSimulationConfig,
    MotorCommand,
    MotorSignalKind,
    MuscleClass,
    RigidBodyState,
    Side,
)
from fly_sensor2behavior.evaluators import default_evaluators
from fly_sensor2behavior.validation import (
    SourceDigest,
    ValidationRunner,
    default_benchmark_registry_path,
    default_validation_source_digests,
    load_benchmark_registry,
)


def _release_source_digests(registry):
    return default_validation_source_digests(
        registry, default_benchmark_registry_path()
    )


class _ArtifactTelemetryPhysics:
    backend_name = "flybody"
    aerodynamic_owner = "flybody"
    body_state_reference = "mock FlyBody root/thorax frame; not whole-fly COM"
    wing_joint_order = (
        "c_thorax-l_wing-yaw",
        "c_thorax-l_wing-roll",
        "c_thorax-l_wing-pitch",
        "c_thorax-r_wing-yaw",
        "c_thorax-r_wing-roll",
        "c_thorax-r_wing-pitch",
    )

    def __init__(self):
        self.calls = 0
        self.state = RigidBodyState()
        self.last_actuator_torque_n_m = np.zeros(6)

    def default_initial_state(self):
        state = RigidBodyState()
        state.position_world_m[2] = 0.1
        return state

    def provenance_metadata(self):
        return {
            "engine": "mock pinned FlyBody",
            "compiled_model_fingerprint": {"sha256": "sha256:" + "1" * 64},
            "dependency_record_sha256": {"flygym": "sha256:" + "2" * 64},
            "licenses": {"flygym": "Apache-2.0", "flybody_model_assets": "unresolved"},
        }

    def reset(self, initial_state):
        self.state = initial_state.copy()

    def step(self, wings, force_body_n, torque_body_n_m, dt_s):
        del wings, force_body_n, torque_body_n_m
        self.calls += 1
        self.state.position_world_m[0] += dt_s
        return self.state.copy()

    def aerodynamic_wrench(self):
        return AerodynamicWrench(
            force_body_n=np.array([0.0, 0.0, 1.0e-6]),
            torque_body_n_m=np.array([0.0, 0.0, 1.0e-9]),
            left_force_body_n=np.zeros(3),
            right_force_body_n=np.zeros(3),
            mechanical_power_w=2.0e-6,
        )

    def wing_joint_state(self):
        base = 0.002 * self.calls
        angles = np.array([base, 0.1, 0.2, -base, -0.1, -0.2])
        velocities = np.array([2.0, 3.0, 4.0, -2.0, -3.0, -4.0])
        return angles, velocities

    def whole_fly_com_position_m(self):
        return self.state.position_world_m + np.array([1.0e-4, 2.0e-4, 3.0e-4])

    def ground_contact_count(self):
        # A single 0.1 ms contact is deliberately absent from the 1 ms replay
        # frames. The full-step summary must still disclose it.
        return int(self.calls == 3)


def test_evidence_audit_exposes_only_provisional_full_pathway_coverage():
    audit = audit_evidence()
    records = audit["provisional_full_pathway_coverage"]
    assert audit["coverage_status"] == "provisional"
    assert {item["branch"] for item in records} == {
        "LLPC1_full_pathway_provisional",
        "NOD1_full_pathway_provisional",
    }
    assert records[0]["resolved_structural_synapses"] < records[0]["total_structural_synapses"]
    assert len(audit["evidence_sha256"]) == 64
    assert any("not physiological weights" in warning for warning in audit["warnings"])


def test_all_named_scenarios_are_deterministic_and_exploratory():
    assert SCENARIO_NAMES == (
        "baseline",
        "vch_dch_ablation",
        "dng02_activation",
        "dna04_activation",
        "dnp26_activation",
        "dng32_activation",
        "gust",
    )
    for name in SCENARIO_NAMES:
        first_config, first = run_scenario(name, duration_s=0.012, seed=23)
        second_config, second = run_scenario(name, duration_s=0.012, seed=23)
        assert first_config == second_config
        np.testing.assert_array_equal(first.position_world_m, second.position_world_m)
        assert first.diagnostics.status.value == "exploratory"
        assert any("FlyBody/MuJoCo was not executed" in warning for warning in first.diagnostics.warnings)


def test_artifact_chunks_are_hashed_reconstructable_and_keep_signal_provenance(tmp_path):
    config, result = run_scenario("dna04_activation", duration_s=0.025, seed=4)
    run_dir = tmp_path / "run"
    manifest = write_episode_artifact(
        run_dir,
        "dna04_activation",
        config,
        result,
        chunk_samples=7,
        created_at_utc="2026-07-17T00:00:00Z",
    )

    assert manifest["validation_status"] == "exploratory"
    assert manifest["calibration"] == {
        "status": "none",
        "calibrated_by": [],
        "datasets": [],
    }
    assert manifest["scientific_label"] == "exploratory; calibrated-by-none"
    assert manifest["run_id"].startswith("dna04_activation-")
    assert manifest["source_kind"] == "exploratory_reduced_order_numpy"
    assert manifest["schema_version"] == "2.0.0"
    assert "not FlyBody/MuJoCo" in manifest["authority_notice"]
    assert set(manifest["model_hashes"]) == {
        "reduced_order_flight_source",
        "visual_neural_pipeline_source",
        "flybody_adapter_source",
        "artifact_contract_source",
        "flight_model_registry",
    }
    assert len(manifest["evidence"]["sha256"]) == 64
    assert manifest["configuration"]["motor_commands"][0]["signal_kind"] == "inferred_rate"
    assert manifest["runtime"]["numpy"]
    assert manifest["licenses"]["numpy"] == "BSD-3-Clause"
    assert {item["branch"] for item in manifest["evidence"]["coverage"]} == {
        "LLPC1_full_pathway_provisional",
        "NOD1_full_pathway_provisional",
    }

    time_descriptor = manifest["arrays"]["time_s"]
    assert time_descriptor["unit"] == "s"
    assert len(time_descriptor["chunks"]) > 1
    np.testing.assert_array_equal(read_chunked_array(run_dir, time_descriptor), result.time_s)
    for chunk in time_descriptor["chunks"]:
        assert sha256_file(run_dir / chunk["path"]) == chunk["sha256"]

    for provenance in manifest["signal_provenance"].values():
        assert provenance["signal_kind"] == "inferred_rate"
        assert provenance["event_timing"] == "inferred_from_rate_by_physiology_scaffold"
    inferred_events = manifest["arrays"]["motor_event_times_s/MN-b1-left"]
    assert "inferred" in inferred_events["provenance"]
    assert manifest["diagnostics"]["physics_backend"] == "reduced_order"
    assert manifest["diagnostics"]["aerodynamic_owner"] == "reduced_order_quasi_steady"
    for suffix, unit in (
        ("muscle_force_n/left:iv2", "N"),
        ("muscle_phase_effect/left:iv2", "1"),
        ("muscle_work_j/left:iv2", "J"),
    ):
        assert manifest["arrays"][suffix]["unit"] == unit
    assert manifest["arrays"]["muscle_work_j/left:iv2"]["provenance"] == (
        "virtual_hinge_work_estimate_model_inference"
    )

    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    sidecar_digest = (run_dir / "manifest.sha256").read_text().split()[0]
    assert sidecar_digest == hashlib.sha256(manifest_bytes).hexdigest()


def test_chunk_reader_detects_tampering(tmp_path):
    config, result = run_scenario("baseline", duration_s=0.010)
    run_dir = tmp_path / "run"
    manifest = write_episode_artifact(run_dir, "baseline", config, result, chunk_samples=8)
    descriptor = manifest["arrays"]["time_s"]
    chunk_path = run_dir / descriptor["chunks"][0]["path"]
    chunk_path.write_bytes(chunk_path.read_bytes() + b"tamper")
    try:
        read_chunked_array(run_dir, descriptor)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("tampered artifact unexpectedly verified")


def test_artifact_identity_locks_configuration_and_nonempty_targets_are_refused(tmp_path):
    config_a, result_a = run_scenario("baseline", duration_s=0.010, seed=1)
    config_b, result_b = run_scenario("baseline", duration_s=0.010, seed=2)
    manifest_a = write_episode_artifact(tmp_path / "a", "baseline", config_a, result_a)
    manifest_b = write_episode_artifact(tmp_path / "b", "baseline", config_b, result_b)
    assert manifest_a["run_id"] != manifest_b["run_id"]

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "stale.txt").write_text("partial")
    try:
        write_episode_artifact(partial, "baseline", config_a, result_a)
    except FileExistsError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("nonempty artifact target was mutated")


def test_artifact_preflight_rejects_config_result_duration_mismatch_without_output(
    tmp_path,
):
    short_config, _ = run_scenario("baseline", duration_s=0.010)
    _, long_result = run_scenario("baseline", duration_s=0.020)
    output = tmp_path / "mismatched-run"

    with pytest.raises(ValueError, match="shape|duration|clock"):
        write_episode_artifact(output, "baseline", short_config, long_result)

    assert not output.exists()
    assert not list(tmp_path.glob(".mismatched-run.staging-*"))


@pytest.mark.parametrize(
    "nonfinite",
    (float("nan"), float("inf"), float("-inf")),
    ids=("nan", "positive-infinity", "negative-infinity"),
)
def test_artifact_preflight_rejects_nonfinite_core_arrays_without_output(
    tmp_path, nonfinite
):
    config, result = run_scenario("baseline", duration_s=0.010)
    position = result.position_world_m.copy()
    position[1, 0] = nonfinite
    invalid = replace(result, position_world_m=position)
    output = tmp_path / ("nonfinite-%s" % repr(nonfinite))

    with pytest.raises(ValueError, match="finite"):
        write_episode_artifact(output, "baseline", config, invalid)

    assert not output.exists()


def test_artifact_preflight_rejects_shape_quaternion_and_diagnostic_clock_errors(
    tmp_path,
):
    config, result = run_scenario("baseline", duration_s=0.010)
    nonunit_quaternion = result.quaternion_body_to_world.copy()
    nonunit_quaternion[2] *= 2.0
    cases = (
        (
            "bad-shape",
            replace(result, position_world_m=result.position_world_m[:, :2]),
            "shape",
        ),
        (
            "bad-quaternion",
            replace(result, quaternion_body_to_world=nonunit_quaternion),
            "unit quaternion",
        ),
        (
            "bad-physics-clock",
            replace(
                result,
                diagnostics=replace(
                    result.diagnostics,
                    physics_steps=result.diagnostics.physics_steps + 1,
                ),
            ),
            "physics_steps",
        ),
    )

    for name, invalid, error_match in cases:
        output = tmp_path / name
        with pytest.raises(ValueError, match=error_match):
            write_episode_artifact(output, "baseline", config, invalid)
        assert not output.exists()


def test_artifact_preflight_rejects_out_of_contract_muscle_state(tmp_path):
    config, result = run_scenario("baseline", duration_s=0.010)
    activation = dict(result.muscle_activation)
    muscle_id = next(iter(activation))
    invalid_values = activation[muscle_id].copy()
    invalid_values[0] = 1.5001
    activation[muscle_id] = invalid_values
    invalid = replace(result, muscle_activation=activation)

    with pytest.raises(ValueError, match=r"\[0, 1\.5\]"):
        write_episode_artifact(
            tmp_path / "invalid-muscle-state", "baseline", config, invalid
        )
    assert not (tmp_path / "invalid-muscle-state").exists()


def test_artifact_preflight_rejects_unknown_motor_keys_and_bad_supplement_alignment(
    tmp_path,
):
    config, result = run_scenario("baseline", duration_s=0.010)
    event_times = dict(result.motor_event_times_s)
    event_times["unknown-MN"] = np.array([], dtype=float)
    unknown_motor = replace(result, motor_event_times_s=event_times)

    with pytest.raises(ValueError, match="configured neuron IDs"):
        write_episode_artifact(
            tmp_path / "unknown-motor", "baseline", config, unknown_motor
        )
    assert not (tmp_path / "unknown-motor").exists()

    misaligned_supplement = (
        ("descending_rate_hz/DNp26/left", np.zeros(2), "Hz", "fixture"),
        (
            "descending_measurement_time_s/DNp26/left",
            np.zeros(1),
            "s",
            "fixture",
        ),
        (
            "descending_availability_time_s/DNp26/left",
            np.zeros(2),
            "s",
            "fixture",
        ),
    )
    with pytest.raises(ValueError, match="equal lengths"):
        write_episode_artifact(
            tmp_path / "misaligned-supplement",
            "baseline",
            config,
            result,
            supplemental_arrays=misaligned_supplement,
        )
    assert not (tmp_path / "misaligned-supplement").exists()

    with pytest.raises(ValueError, match="finite"):
        write_episode_artifact(
            tmp_path / "nonfinite-supplement",
            "baseline",
            config,
            result,
            supplemental_arrays=(
                ("custom_metric", np.array([np.inf]), "1", "fixture"),
            ),
        )
    assert not (tmp_path / "nonfinite-supplement").exists()

    causal_supplement = (
        ("descending_rate_hz/DNp26/left", np.ones(2), "Hz", "fixture"),
        (
            "descending_measurement_time_s/DNp26/left",
            np.array((0.001, 0.002)),
            "s",
            "fixture",
        ),
        (
            "descending_availability_time_s/DNp26/left",
            np.array((0.0015, 0.0019)),
            "s",
            "fixture",
        ),
    )
    with pytest.raises(ValueError, match="causality"):
        write_episode_artifact(
            tmp_path / "acausal-supplement",
            "baseline",
            config,
            result,
            supplemental_arrays=causal_supplement,
        )
    assert not (tmp_path / "acausal-supplement").exists()


def test_artifact_write_failure_does_not_publish_a_partial_target(
    tmp_path, monkeypatch
):
    config, result = run_scenario("baseline", duration_s=0.010)
    output = tmp_path / "transactional-run"
    original_write_npy = artifacts_module._write_npy
    calls = 0

    def fail_after_first_chunk(path, array):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected artifact write failure")
        return original_write_npy(path, array)

    monkeypatch.setattr(artifacts_module, "_write_npy", fail_after_first_chunk)
    with pytest.raises(OSError, match="injected"):
        write_episode_artifact(output, "baseline", config, result)

    assert not output.exists()
    assert not list(tmp_path.glob(".transactional-run.staging-*"))


def test_exact_spike_provenance_is_not_relabelled_as_inferred(tmp_path):
    command = MotorCommand(
        neuron_id="MN-DLM-left-exact",
        muscle="DLM",
        side=Side.LEFT,
        muscle_class=MuscleClass.ASYNCHRONOUS_POWER,
        signal_kind=MotorSignalKind.EXACT_SPIKES,
        spike_times_s=(0.003, 0.007),
        measurement_spike_times_s=(0.002, 0.006),
        provenance="measured test fixture",
    )
    config = FlightSimulationConfig(
        duration_s=0.010,
        physics_dt_s=0.0001,
        neural_dt_s=0.005,
        logging_dt_s=0.001,
        motor_commands=(command,),
    )
    result = FlightEpisodeRunner().run(config)
    manifest = write_episode_artifact(tmp_path, "baseline", config, result)
    provenance = manifest["signal_provenance"][command.neuron_id]
    assert provenance["signal_kind"] == "exact_spikes"
    assert provenance["event_timing"] == (
        "exact_input_spike_runtime_availability_times"
    )
    assert (
        manifest["arrays"]["motor_event_times_s/%s" % command.neuron_id]["provenance"]
        == "exact_input_spike_runtime_availability_times"
    )
    measurement_descriptor = manifest["arrays"][
        "motor_input_measurement_spike_times_s/%s" % command.neuron_id
    ]
    assert measurement_descriptor["provenance"] == (
        "exact_input_spike_measurement_times"
    )
    assert manifest["configuration"]["motor_commands"][0][
        "measurement_spike_times_s"
    ] == list(command.measurement_spike_times_s)
    np.testing.assert_allclose(
        read_chunked_array(
            tmp_path,
            manifest["arrays"]["motor_event_times_s/%s" % command.neuron_id],
        ),
        command.spike_times_s,
    )
    np.testing.assert_allclose(
        read_chunked_array(tmp_path, measurement_descriptor),
        command.measurement_spike_times_s,
    )


def test_web_episode_matches_typescript_contract_and_is_decimated():
    config, result = run_scenario("dng02_activation", duration_s=0.025)
    replay = episode_to_web_replay(
        "dng02_activation", config, result, target_sample_rate_hz=100.0
    )
    assert replay["status"] == "exploratory"
    assert replay["source_kind"] == "exploratory_reduced_order_numpy"
    assert replay["authority_notice"] == AUTHORITY_NOTICE
    assert replay["frames"][0]["t"] == 0.0
    assert replay["frames"][-1]["t"] == config.duration_s
    assert len(replay["frames"]) < len(result.time_s)
    frame = replay["frames"][0]
    assert len(frame["position_m"]) == 3
    assert len(frame["orientation_rad"]) == 3
    assert len(frame["velocity_m_s"]) == 3
    assert len(frame["wing_envelope_deg"]) == 4
    assert len(frame["force_world_n"]) == 3
    assert len(frame["moment_world_n_m"]) == 3
    assert len(frame["circuit"]) == 7
    assert len(frame["muscles"]) == 5
    assert 0.0 <= frame["evidence_incompleteness"] <= 1.0
    assert "no upstream neural trace" in replay["circuit_activity_status"]
    assert "not a trajectory posterior" in replay["evidence_incompleteness_status"]
    assert replay["signal_semantics"] == {
        "neural_origin": "unavailable",
        "motor_timing": "inferred_rate",
        "mechanics_origin": "simulated",
        "confidence": "low",
        "provenance": (
            "No upstream circuit trace was ingested; motor, muscle, and mechanics "
            "signals are uncalibrated reduced-order model outputs."
        ),
    }
    assert replay["individual_muscle_channels"]
    muscle_id = replay["individual_muscle_channels"][0]["id"]
    assert set(replay["frames"][0]["individual_muscles"][muscle_id]) == {
        "activation",
        "force_n",
        "phase_effect",
        "work_j",
    }


def test_flybody_measured_telemetry_keeps_artifact_and_replay_provenance(tmp_path):
    config = FlightSimulationConfig(
        duration_s=0.005,
        physics_dt_s=0.0001,
        neural_dt_s=0.005,
        logging_dt_s=0.001,
    )
    result = FlightEpisodeRunner(
        physics_adapter=_ArtifactTelemetryPhysics()
    ).run(config)

    run_dir = tmp_path / "flybody-run"
    manifest = write_episode_artifact(
        run_dir,
        "baseline",
        config,
        result,
        created_at_utc="2026-07-18T00:00:00Z",
    )
    assert manifest["source_kind"] == "exploratory_flybody_mujoco_worker"
    assert "pinned FlyGym/FlyBody MuJoCo worker" in manifest["authority_notice"]
    assert manifest["runtime"]["physics_backend"] == "flybody"
    assert manifest["runtime"]["physics_provenance"]["engine"] == (
        "mock pinned FlyBody"
    )
    assert manifest["licenses"]["external_physics"]["flygym"] == "Apache-2.0"
    assert manifest["arrays"]["wing_stroke_rad"]["provenance"] == (
        "virtual_hinge_desired_kinematics_not_measured_flybody_state"
    )
    assert manifest["arrays"]["measured_wing_joint_angle_rad"]["provenance"] == (
        "external_physics_measured_output"
    )
    assert manifest["arrays"]["measured_wing_joint_velocity_rad_s"][
        "provenance"
    ] == "external_physics_measured_output"
    assert manifest["arrays"]["whole_fly_com_position_world_m"]["provenance"] == (
        "flybody_mujoco_articulated_subtree_com"
    )
    assert manifest["arrays"]["ground_contact_count"]["provenance"] == (
        "decimated_projection_of_mujoco_transition_contact_points"
    )
    assert manifest["arrays"]["ground_contact_transition_point_count"][
        "provenance"
    ] == GROUND_CONTACT_TELEMETRY_KIND
    assert manifest["arrays"]["ground_contact_transition_point_count"][
        "sample_semantics"
    ] == GROUND_CONTACT_SAMPLE_SEMANTICS
    assert manifest["arrays"]["external_actuator_torque_physics_n_m"][
        "axis_labels"
    ] == list(_ArtifactTelemetryPhysics.wing_joint_order)
    np.testing.assert_array_equal(
        read_chunked_array(
            run_dir, manifest["arrays"]["measured_wing_joint_angle_rad"]
        ),
        result.measured_wing_joint_angle_rad,
    )
    np.testing.assert_array_equal(
        read_chunked_array(
            run_dir, manifest["arrays"]["whole_fly_com_position_world_m"]
        ),
        result.whole_fly_com_position_world_m,
    )
    np.testing.assert_array_equal(
        read_chunked_array(
            run_dir,
            manifest["arrays"]["ground_contact_transition_point_count"],
        ),
        result.ground_contact_transition_point_count,
    )
    assert len(result.physics_time_s) == result.diagnostics.physics_steps + 1
    assert np.count_nonzero(result.ground_contact_transition_point_count) == 1
    assert np.max(result.ground_contact_count) == 0

    replay = episode_to_web_replay(
        "baseline", config, result, target_sample_rate_hz=1000.0
    )
    assert replay["source_kind"] == "exploratory_flybody_mujoco_worker"
    assert replay["physics_backend"] == "flybody"
    assert replay["body_state_reference"] == (
        "whole-fly articulated subtree COM for display; root/thorax retained separately"
    )
    assert replay["wing_kinematics_source"] == "measured_flybody_yaw_axes"
    assert replay["measured_wing_joint_order"] == list(
        _ArtifactTelemetryPhysics.wing_joint_order
    )
    assert replay["contact_telemetry_status"] == GROUND_CONTACT_TELEMETRY_KIND
    assert replay["ground_contact_summary"] == {
        "telemetry": GROUND_CONTACT_TELEMETRY_KIND,
        "sample_semantics": GROUND_CONTACT_SAMPLE_SEMANTICS,
        "initial_count": 0,
        "occurred": True,
        "first_transition_start_s": pytest.approx(0.0002),
        "first_transition_end_s": pytest.approx(0.0003),
        "transition_count": 1,
        "maximum_count": 1,
    }
    inspection = replay["wingbeat_inspection"]
    assert inspection["sample_rate_hz"] == pytest.approx(10_000.0)
    assert inspection["start_s"] == 0.0
    assert inspection["end_s"] == pytest.approx(config.duration_s)
    assert len(inspection["time_s"]) == result.diagnostics.physics_steps + 1
    assert inspection["wing_joint_order"] == list(
        _ArtifactTelemetryPhysics.wing_joint_order
    )
    assert all(frame["ground_contact_count"] == 0 for frame in replay["frames"])

    inconsistent_metrics = dict(result.diagnostics.metrics)
    inconsistent_metrics["ground_contact_transition_count"] = 0.0
    inconsistent_result = replace(
        result,
        diagnostics=replace(result.diagnostics, metrics=inconsistent_metrics),
    )
    with pytest.raises(ValueError, match="full-step diagnostics are inconsistent"):
        write_episode_artifact(
            tmp_path / "inconsistent-contact-run",
            "baseline",
            config,
            inconsistent_result,
        )
    zero_time_metrics = dict(result.diagnostics.metrics)
    zero_time_metrics[
        "first_ground_contact_transition_start_s_or_duration_s"
    ] = 0.0
    zero_time_result = replace(
        result,
        diagnostics=replace(result.diagnostics, metrics=zero_time_metrics),
    )
    with pytest.raises(ValueError, match="full-step diagnostics are inconsistent"):
        write_episode_artifact(
            tmp_path / "zero-time-post-reset-contact-run",
            "baseline",
            config,
            zero_time_result,
        )
    frame = replay["frames"][0]
    np.testing.assert_allclose(frame["position_m"], [1.0e-4, 2.0e-4, 0.1003])
    np.testing.assert_allclose(frame["root_position_m"], [0.0, 0.0, 0.1])
    np.testing.assert_array_equal(
        frame["measured_wing_joint_angle_rad"], [0.0, 0.1, 0.2, -0.0, -0.1, -0.2]
    )
    np.testing.assert_array_equal(
        frame["measured_wing_joint_velocity_rad_s"], [2.0, 3.0, 4.0, -2.0, -3.0, -4.0]
    )
    assert len(frame["desired_wing_stroke_rad"]) == 2

    root_only_replay = episode_to_web_replay(
        "baseline",
        config,
        replace(result, whole_fly_com_position_world_m=None),
        target_sample_rate_hz=1000.0,
    )
    assert root_only_replay["body_state_reference"] == (
        "FlyBody root/thorax frame; whole-fly COM telemetry unavailable"
    )
    np.testing.assert_array_equal(
        root_only_replay["frames"][0]["position_m"],
        result.position_world_m[0],
    )

    with pytest.raises(ValueError, match="reviewed FlyGym 2.1.0.*axis order"):
        episode_to_web_replay(
            "baseline",
            config,
            replace(
                result,
                measured_wing_joint_order=tuple(
                    "unreviewed-axis-%d" % index for index in range(6)
                ),
            ),
            target_sample_rate_hz=1000.0,
        )


def test_web_export_contains_every_scenario_and_provisional_coverage(tmp_path):
    manifest = export_web_replay(
        tmp_path,
        duration_s=0.012,
        target_sample_rate_hz=100.0,
    )
    assert [item["id"] for item in manifest["episodes"]] == list(SCENARIO_NAMES)
    assert all(item["status"] == "exploratory" for item in manifest["episodes"])
    assert len(manifest["coverage"]) == 2
    assert all("Provisional" in item["note"] for item in manifest["coverage"])
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
    for summary in manifest["episodes"]:
        assert summary["data_url"] == "data/episodes/%s.json" % summary["id"]
        assert summary["artifact_manifest_url"].startswith("data/runs/")
        episode = json.loads((tmp_path / "episodes" / (summary["id"] + ".json")).read_text())
        assert episode["id"] == summary["id"]
        artifact_path = tmp_path / summary["artifact_manifest_url"].removeprefix("data/")
        artifact = json.loads(artifact_path.read_text())
        assert artifact["run_id"] in summary["artifact_manifest_url"]
        assert episode["source_run_id"] == artifact["run_id"]
        assert episode["source_artifact_manifest_sha256"] == sha256_file(
            artifact_path
        )
        assert episode["source_artifact_schema_version"] == artifact["schema_version"]
        assert artifact["web_replay_projection"]["sha256"] == (
            web_replay_projection_sha256(episode)
        )
        replay_path = tmp_path / summary["data_url"].removeprefix("data/")
        assert summary["replay_sha256"] == sha256_file(replay_path)
        assert summary["artifact_manifest_sha256"] == sha256_file(artifact_path)
        projection_path = tmp_path / summary["artifact_projection_url"].removeprefix(
            "data/"
        )
        projection_bytes = projection_path.read_bytes()
        assert hashlib.sha256(projection_bytes).hexdigest() == artifact[
            "web_replay_projection"
        ]["sha256"]
        assert json.loads(projection_bytes) == {
            key: value
            for key, value in episode.items()
            if key
            not in (
                "source_artifact_manifest_sha256",
                "source_artifact_schema_version",
            )
        }


def test_vch_ablation_and_gust_configuration_are_explicit():
    ablation = build_scenario_config("vch_dch_ablation", duration_s=0.020)
    assert ablation.vch_dch_factor == 0.0
    gust = build_scenario_config("gust", duration_s=0.020)
    assert len(gust.perturbations) == 1
    assert gust.perturbations[0].mode.value == "gust"


def test_web_export_attaches_a_validated_native_report(tmp_path):
    registry = load_benchmark_registry()
    report = ValidationRunner(registry, default_evaluators(registry)).run(
        evaluation_id="web-test",
        source_digests=_release_source_digests(registry),
    )
    report_path = tmp_path / "source-report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")
    output = tmp_path / "web"
    manifest = export_web_replay(
        output,
        scenarios=("baseline",),
        duration_s=0.012,
        target_sample_rate_hz=100.0,
        validation_report_path=report_path,
    )
    report_relative = Path(manifest["validation_report_url"].removeprefix("data/"))
    registry_relative = Path(
        manifest["validation_registry_url"].removeprefix("data/")
    )
    assert report_relative.name == "report.%s.json" % report.content_sha256
    assert registry_relative.name.startswith("registry.v1.")
    assert manifest["validation_report_sha256"] == sha256_file(
        output / report_relative
    )
    assert manifest["validation_registry_file_sha256"] == sha256_file(
        output / registry_relative
    )
    assert manifest["validation_registry_canonical_sha256"] == (
        registry.content_sha256
    )
    attached = json.loads((output / report_relative).read_text())
    assert attached == report.to_dict()
    by_label = {item["label"]: item for item in manifest["provenance"]}
    for label in (
        "Evidence graph",
        "Model registry",
        "BANC/FANC evidence",
        "Validation report",
        "Evaluation registry",
    ):
        assert len(by_label[label]["sha256"]) == 64
        assert by_label[label]["url"].startswith("data/")
    assert by_label["Evaluation registry"]["value"].count("sha256:") == 1
    assert manifest["validation_registry_canonical_sha256"] in (
        by_label["Evaluation registry"]["value"]
    )
    protocol_entries = {
        label: item
        for label, item in by_label.items()
        if label.startswith("Evaluation protocol · ")
    }
    assert len(protocol_entries) == 4
    for item in protocol_entries.values():
        assert item["url"].startswith("data/benchmarks/protocols/")
        assert sha256_file(output / item["url"].removeprefix("data/")) == item["sha256"]


def test_web_export_rejects_complete_report_without_release_source_receipts(tmp_path):
    registry = load_benchmark_registry()
    report = ValidationRunner(registry, default_evaluators(registry)).run(
        evaluation_id="web-missing-source-receipts"
    )
    report_path = tmp_path / "source-report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="source receipt"):
        export_web_replay(
            tmp_path / "web",
            scenarios=("baseline",),
            duration_s=0.012,
            validation_report_path=report_path,
        )


def test_web_export_binds_worker_image_source_digest_into_provenance(tmp_path):
    registry = load_benchmark_registry()
    report = ValidationRunner(registry, default_evaluators(registry)).run(
        evaluation_id="web-worker-receipt-test",
        source_digests=_release_source_digests(registry),
    )
    image_sha = hashlib.sha256(b"manufactured worker image").hexdigest()
    report = replace(
        report,
        source_digests=report.source_digests
        + (
            SourceDigest(
                kind="image",
                name="fly-s2b-worker-image",
                sha256=image_sha,
            ),
        ),
    )
    report_path = tmp_path / "source-report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")

    manifest = export_web_replay(
        tmp_path / "web",
        scenarios=("baseline",),
        duration_s=0.012,
        validation_report_path=report_path,
    )

    worker_entries = [
        item for item in manifest["provenance"] if item["label"] == "Worker image"
    ]
    assert worker_entries == [
        {
            "label": "Worker image",
            "sha256": image_sha,
            "value": "fly-s2b-worker-image · sha256:%s" % image_sha,
        }
    ]


def test_web_export_can_attach_trace_backed_reduced_retinal_episode(tmp_path):
    manifest = export_web_replay(
        tmp_path,
        scenarios=("baseline",),
        duration_s=0.020,
        neural_dt_s=0.005,
        target_sample_rate_hz=200.0,
        include_reduced_retinal_pipeline=True,
    )

    assert [item["id"] for item in manifest["episodes"]] == [
        "baseline",
        "retinal_reduced_nod1",
    ]
    assert manifest["source_kind"] == "mixed_exploratory_replay_with_pipeline_traces"
    summary = manifest["episodes"][1]
    replay = json.loads(
        (tmp_path / summary["data_url"].removeprefix("data/")).read_text()
    )
    assert replay["neural_model_scope"]["kind"] == "reduced_nod1_surrogate"
    assert replay["neural_trace_channels"]
    assert replay["pathway_channels"][0]["status"] == "attached"
    artifact_path = tmp_path / summary["artifact_manifest_url"].removeprefix("data/")
    artifact = json.loads(artifact_path.read_text())
    assert artifact["pipeline"]["mode"] == "open_loop_causal_retinal_frames"
    assert artifact["web_replay_projection"]["sha256"] == (
        web_replay_projection_sha256(replay)
    )
    assert summary["replay_sha256"] == sha256_file(
        tmp_path / summary["data_url"].removeprefix("data/")
    )
    assert summary["artifact_manifest_sha256"] == sha256_file(artifact_path)
    assert hashlib.sha256(
        (tmp_path / summary["artifact_projection_url"].removeprefix("data/")).read_bytes()
    ).hexdigest() == artifact["web_replay_projection"]["sha256"]
    assert any(item["label"] == "Attached neural replay" for item in manifest["provenance"])


def test_web_export_places_authoritative_pipeline_after_baseline(tmp_path):
    runner = FlightEpisodeRunner(physics_adapter=_ArtifactTelemetryPhysics())
    manifest = export_web_replay(
        tmp_path,
        scenarios=("baseline", "vch_dch_ablation"),
        duration_s=0.020,
        neural_dt_s=0.005,
        target_sample_rate_hz=200.0,
        include_reduced_retinal_pipeline=True,
        pipeline_runner=runner,
        retinal_receptor_count=16,
        retinal_angular_velocity_rad_s=-5.0,
        retinal_sensor_latency_s=0.0005,
    )

    assert [item["id"] for item in manifest["episodes"]] == [
        "baseline",
        "retinal_reduced_nod1",
        "vch_dch_ablation",
    ]
    assert manifest["source_kind"] == (
        "mixed_exploratory_flybody_and_reduced_replay"
    )
    assert "explicitly labelled FlyBody" in manifest["authority_notice"]
    by_label = {item["label"]: item for item in manifest["provenance"]}
    assert "Pinned FlyGym/FlyBody MuJoCo" in by_label["Physics"]["value"]
    assert by_label["Reduced-order executable model"]["value"].endswith(
        by_label["Reduced-order executable model"]["sha256"]
    )
    assert by_label["FlyBody adapter executable model"]["value"].endswith(
        by_label["FlyBody adapter executable model"]["sha256"]
    )
    flybody_summary = manifest["episodes"][1]
    flybody_episode = json.loads(
        (tmp_path / flybody_summary["data_url"].removeprefix("data/")).read_text()
    )
    assert flybody_episode["physics_backend"] == "flybody"
