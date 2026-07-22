import json
from dataclasses import replace

import numpy as np

from fly_sensor2behavior.artifacts import web_replay_projection_sha256
from fly_sensor2behavior.pipeline import (
    NOD1FlightPipelineConfig,
    _pipeline_arrays,
    _pipeline_manifest,
    load_legacy_nod1_result,
    nod1_run_to_web_replay,
    run_nod1_flight_pipeline,
    run_retinal_flight_pipeline,
    write_nod1_flight_artifact,
    write_nod1_web_replay,
)
from fly_sensor2behavior.schema import CircuitSignalKind
from fly_sensor2behavior.vision import AnalyticGratingScene, PanoramicRetina


NOD1_LEFT = ("720575940628438427", "720575940625528556")
NOD1_RIGHT = ("720575940623997949", "720575940629456860")


def _legacy_result(left_mv=-50.0, right_mv=-60.0):
    times = [index * 0.005 for index in range(21)]
    voltage = {}
    for root_id in NOD1_LEFT:
        voltage[root_id] = [left_mv] * len(times)
    for root_id in NOD1_RIGHT:
        voltage[root_id] = [right_mv] * len(times)
    return {
        "time": times,
        "voltage": voltage,
        "steering": [999.0] * len(times),
        "meta": {
            "logicVersion": "legacy-nod1-v0.5.0",
            "stimulus": {"kind": "figure_ground", "seed": 7},
        },
    }


def test_pipeline_runs_circuit_bridge_muscles_and_mechanics():
    run = run_nod1_flight_pipeline(
        _legacy_result(),
        config=NOD1FlightPipelineConfig(seed=41),
    )

    assert run.bridge.descending.channel("DNp26", "right").samples[-1].rate_hz > 0.0
    assert run.bridge.descending.channel("DNp26", "left").samples[-1].rate_hz == 0.0
    assert np.max(run.flight.muscle_force_n["right:iv2"]) > 0.0
    assert np.max(run.flight.muscle_force_n["left:iv2"]) == 0.0
    assert np.all(np.isfinite(run.flight.position_world_m))
    assert "steering" in run.circuit.provenance.filters["excluded_motor_fields"]


def test_pipeline_artifact_contains_circuit_dn_motor_and_physics(tmp_path):
    run = run_nod1_flight_pipeline(
        _legacy_result(),
        config=NOD1FlightPipelineConfig(seed=41),
    )
    target = tmp_path / "run"
    replay = nod1_run_to_web_replay(run, target_sample_rate_hz=100.0)
    manifest = write_nod1_flight_artifact(
        target,
        run,
        created_at_utc="2026-07-17T00:00:00Z",
        chunk_samples=64,
        web_replay=replay,
    )

    arrays = manifest["arrays"]
    assert "circuit_voltage/720575940628438427" in arrays
    assert "descending_rate_hz/DNp26/right" in arrays
    assert "wing_motor_rate_hz/MN-iv2/right" in arrays
    assert "wing_motor_rate_measurement_time_s/MN-iv2/right" in arrays
    assert "wing_motor_rate_availability_time_s/MN-iv2/right" in arrays
    assert "wing_motor_event_availability_time_s/MN-iv2/right" in arrays
    assert "muscle_force_n/right:iv2" in arrays
    assert "position_world_m" in arrays
    assert manifest["pipeline"]["pipeline_id"] == "legacy-nod1-to-dnp26-to-flight-v1"
    assert manifest["pipeline"]["stages"][0] == "saved_legacy_nod1_voltage"
    assert manifest["pipeline"]["visual_boundary"]["status"] == "unavailable_in_legacy_result"
    assert "not source-of-record retinal frames" in manifest["pipeline"]["visual_boundary"]["notice"]
    assert arrays["circuit_voltage/720575940628438427"]["provenance"] == "legacy_nod1_simulated_trace"
    assert manifest["pipeline"]["scientific_status"] == "exploratory_uncalibrated"
    circuit_contract = manifest["pipeline"]["circuit_output_contract"]
    assert circuit_contract["dataset"]["materialization"] == 783
    assert circuit_contract["dataset"]["coordinate_units"] == "nm"
    assert circuit_contract["dataset_identity_space"] == "flywire_fafb:FAFB@783"
    assert circuit_contract["sample_time_array"] == "circuit_sample_time_s"
    assert {
        signal["neuron"]["entity_id"] for signal in circuit_contract["signals"]
    } == set(NOD1_LEFT + NOD1_RIGHT)
    assert manifest["pipeline"]["identity_boundary"]["join_policy"].endswith(
        "never directly joined"
    )
    assert (target / "manifest.sha256").is_file()
    assert manifest["web_replay_projection"]["sha256"] == (
        web_replay_projection_sha256(replay)
    )


def test_pipeline_preserves_exact_circuit_spike_availability_arrays_and_manifest():
    run = run_nod1_flight_pipeline(
        _legacy_result(),
        config=NOD1FlightPipelineConfig(seed=41),
    )
    source_signal = run.circuit.signals[0]
    spike_times_s = (0.001, 0.006)
    availability_times_s = (0.003, 0.009)
    spike_signal = replace(
        source_signal,
        signal_kind=CircuitSignalKind.SPIKE_EVENTS,
        unit="1",
        values=(),
        spike_times_s=spike_times_s,
        uncertainty=(),
        availability_times_s=availability_times_s,
    )
    spike_run = replace(
        run,
        circuit=replace(run.circuit, signals=(spike_signal,)),
    )

    arrays = {
        name: {"values": values, "unit": unit, "provenance": provenance}
        for name, values, unit, provenance in _pipeline_arrays(spike_run)
    }
    spike_path = "circuit_spike_times_s/%s" % source_signal.neuron.entity_id
    availability_path = (
        "circuit_spike_availability_time_s/%s" % source_signal.neuron.entity_id
    )
    np.testing.assert_array_equal(
        arrays[spike_path]["values"], np.asarray(spike_times_s)
    )
    np.testing.assert_array_equal(
        arrays[availability_path]["values"], np.asarray(availability_times_s)
    )
    assert arrays[spike_path]["unit"] == "s"
    assert arrays[availability_path]["unit"] == "s"

    circuit_contract = _pipeline_manifest(spike_run)["circuit_output_contract"]
    assert circuit_contract["sample_time_array"] == "circuit_sample_time_s"
    assert circuit_contract["signals"] == [
        {
            "neuron": spike_signal.neuron.to_dict(),
            "signal_kind": CircuitSignalKind.SPIKE_EVENTS.value,
            "unit": "1",
            "origin": spike_signal.origin.value,
            "confidence": spike_signal.confidence.to_dict(),
            "provenance": spike_signal.provenance.to_dict(),
            "value_array": spike_path,
            "availability_time_array": availability_path,
        }
    ]


def test_pipeline_input_trace_changes_content_identity(tmp_path):
    first = run_nod1_flight_pipeline(_legacy_result(-50.0), config=NOD1FlightPipelineConfig(seed=2))
    second = run_nod1_flight_pipeline(_legacy_result(-49.0), config=NOD1FlightPipelineConfig(seed=2))
    first_manifest = write_nod1_flight_artifact(
        tmp_path / "first", first, created_at_utc="2026-07-17T00:00:00Z"
    )
    second_manifest = write_nod1_flight_artifact(
        tmp_path / "second", second, created_at_utc="2026-07-17T00:00:00Z"
    )
    assert first.circuit_trace_sha256 != second.circuit_trace_sha256
    assert first_manifest["run_id"] != second_manifest["run_id"]


def test_legacy_loader_rejects_non_object(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    try:
        load_legacy_nod1_result(path)
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("non-object JSON must be rejected")


def test_causal_retinal_frames_run_to_body_and_are_preserved(tmp_path):
    retina = PanoramicRetina(
        receptor_count=32,
        sensor_latency_s=0.0005,
        integration_subsamples=3,
    )
    scene = AnalyticGratingScene(angular_velocity_rad_s=1.0)
    frames = tuple(
        retina.sample(
            scene,
            exposure_start_s=index * 0.001,
            exposure_end_s=(index + 1) * 0.001,
        )
        for index in range(101)
    )
    run = run_retinal_flight_pipeline(
        frames,
        config=NOD1FlightPipelineConfig(seed=17, neural_dt_s=0.001),
    )

    assert len(run.retinal_frames) == 101
    assert max(run.circuit.signals[0].values) > -0.060
    assert run.bridge.descending.channel("DNp26", "left").samples[-1].rate_hz > 0.0
    assert run.bridge.descending.channel("DNp26", "right").samples[-1].rate_hz > 0.0
    assert np.all(np.isfinite(run.flight.position_world_m))

    manifest = write_nod1_flight_artifact(
        tmp_path / "retinal-run",
        run,
        created_at_utc="2026-07-17T00:00:00Z",
    )
    assert "retinal_normalized_luminance" in manifest["arrays"]
    assert "wing_motor_rate_measurement_time_s/MN-iv2/right" in manifest["arrays"]
    assert "wing_motor_rate_availability_time_s/MN-iv2/right" in manifest["arrays"]
    assert manifest["pipeline"]["visual_boundary"]["status"] == "provided"
    assert manifest["pipeline"]["mode"] == "open_loop_causal_retinal_frames"
    assert manifest["pipeline"]["pipeline_id"] == (
        "causal-retinal-reduced-nod1-to-dnp26-to-flight-v1"
    )
    assert manifest["pipeline"]["stages"][:2] == [
        "causal_finite_exposure_retinal_frames",
        "reduced_nod1_motion_surrogate",
    ]
    assert "preserved in this artifact" in manifest["pipeline"]["visual_boundary"]["notice"]
    assert manifest["arrays"]["circuit_voltage/720575940628438427"]["provenance"] == (
        "reduced_nod1_simulated_trace"
    )
    circuit_contract = manifest["pipeline"]["circuit_output_contract"]
    assert circuit_contract["dataset"]["materialization"] == 783
    assert circuit_contract["dataset"]["coordinate_units"] == "nm"
    assert circuit_contract["exact_timebase"] is True
    assert all(
        isinstance(signal["neuron"]["entity_id"], str)
        for signal in circuit_contract["signals"]
    )


def test_saved_nod1_web_replay_exposes_raw_causal_pipeline_channels(tmp_path):
    run = run_nod1_flight_pipeline(
        _legacy_result(),
        config=NOD1FlightPipelineConfig(seed=41),
    )
    replay = nod1_run_to_web_replay(run, target_sample_rate_hz=100.0)

    assert replay["id"] == "saved_nod1_circuit"
    assert replay["neural_model_scope"] == {
        "kind": "full_legacy_circuit_cable_export",
        "label": "Frozen 1,208-cell browser circuit · cable-neuron export",
        "full_circuit_executed": True,
        "circuit_cell_count": 1208,
        "exported_circuit_channel_count": 4,
        "notice": (
            "The saved result was produced by the frozen 1,208-cell browser circuit; "
            "this replay carries only its exported vCH/DCH/NOD1 cable-neuron voltages."
        ),
    }
    channels = {channel["id"]: channel for channel in replay["neural_trace_channels"]}
    nod1_id = "circuit:720575940628438427"
    assert channels[nod1_id]["signal_kind"] == "voltage"
    assert channels[nod1_id]["unit"] == "V"
    assert channels["dn:DNp26:right"]["origin"] == "inferred"
    assert channels["muscle:right:iv2"]["causal_role"] == "circuit_driven"
    assert channels["muscle:left:DLM"]["causal_role"] == "airborne_baseline_drive"
    assert replay["pathway_channels"][0]["status"] == "unavailable"
    assert replay["pathway_channels"][2]["status"] == "unavailable"
    assert replay["pathway_channels"][3]["status"] == "attached"
    assert replay["frames"][0]["neural_signals"][nod1_id] == -0.05
    assert replay["frames"][0]["pathway_values"][3] == -0.055
    # The DN encoder declares delay, so its value is not exposed at t=0.
    assert "dn:DNp26:right" not in replay["frames"][0]["neural_signals"]
    assert "dn:DNp26:right" in replay["frames"][1]["neural_signals"]
    assert "pipeline-derived raw" in replay["circuit_activity_status"]

    path = tmp_path / "saved-nod1-replay.json"
    written = write_nod1_web_replay(path, run, target_sample_rate_hz=100.0)
    assert json.loads(path.read_text()) == written


def test_retinal_web_replay_is_explicitly_reduced_and_preserves_availability():
    retina = PanoramicRetina(
        receptor_count=16,
        sensor_latency_s=0.0005,
        integration_subsamples=3,
    )
    scene = AnalyticGratingScene(angular_velocity_rad_s=1.0)
    frames = tuple(
        retina.sample(
            scene,
            exposure_start_s=index * 0.001,
            exposure_end_s=(index + 1) * 0.001,
        )
        for index in range(21)
    )
    run = run_retinal_flight_pipeline(
        frames,
        config=NOD1FlightPipelineConfig(seed=17, neural_dt_s=0.001),
    )
    replay = nod1_run_to_web_replay(run, target_sample_rate_hz=200.0)

    scope = replay["neural_model_scope"]
    assert scope["kind"] == "reduced_nod1_surrogate"
    assert scope["full_circuit_executed"] is False
    assert scope["circuit_cell_count"] is None
    assert "not the frozen 1,208-cell" in scope["notice"]
    retinal_id = "retina:binocular:mean"
    assert any(channel["id"] == retinal_id for channel in replay["neural_trace_channels"])
    assert replay["pathway_channels"][0]["status"] == "attached"
    # First retinal/circuit sample becomes available after t=0 and stays absent before then.
    assert retinal_id not in replay["frames"][0]["neural_signals"]
    assert replay["frames"][0]["pathway_values"][0] is None
    assert retinal_id in replay["frames"][1]["neural_signals"]

    # Physics provenance is independent of neural-model scope. A retinal
    # surrogate executed through FlyBody must never retain the reduced-physics
    # source label in its public replay.
    sample_count = len(run.flight.time_s)
    flybody_flight = replace(
        run.flight,
        diagnostics=replace(run.flight.diagnostics, physics_backend="flybody"),
        measured_wing_joint_angle_rad=np.zeros((sample_count, 6)),
        measured_wing_joint_velocity_rad_s=np.zeros((sample_count, 6)),
        measured_wing_joint_order=(
            "c_thorax-l_wing-yaw",
            "c_thorax-l_wing-roll",
            "c_thorax-l_wing-pitch",
            "c_thorax-r_wing-yaw",
            "c_thorax-r_wing-roll",
            "c_thorax-r_wing-pitch",
        ),
        whole_fly_com_position_world_m=run.flight.position_world_m.copy(),
    )
    flybody_replay = nod1_run_to_web_replay(
        replace(run, flight=flybody_flight), target_sample_rate_hz=200.0
    )
    assert flybody_replay["source_kind"] == (
        "exploratory_retinal_nod1_flybody_pipeline"
    )
    assert flybody_replay["label"].endswith("→ FlyBody")
    assert flybody_replay["physics_backend"] == "flybody"
