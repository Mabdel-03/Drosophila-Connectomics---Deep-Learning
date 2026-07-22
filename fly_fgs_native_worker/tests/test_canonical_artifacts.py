from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fly_sensor2behavior.artifacts import read_chunked_array, sha256_file
from fly_sensor2behavior.canonical_artifacts import (
    CANONICAL_ARTIFACT_SCHEMA_VERSION,
    bind_canonical_web_replay_to_artifact,
    canonical_array_bundle,
    canonical_closed_loop_to_web_replay,
    canonical_run_id,
    canonical_web_projection_sha256,
    write_canonical_closed_loop_artifact,
)
from fly_sensor2behavior.flight.canonical_closed_loop import CanonicalClosedLoopConfig
from fly_sensor2behavior.flight.canonical_closed_loop import (
    CanonicalClosedLoopCheckpoint,
    CanonicalClosedLoopSimulator,
)
from fly_sensor2behavior.flight.streaming_mechanics import (
    MechanicsInterventionMode,
    StreamingMechanicsConfig,
    StreamingMuscleIntervention,
    StreamingMuscleWingStepper,
)
from fly_sensor2behavior.flight.types import AerodynamicWrench
from fly_sensor2behavior.fly_fgs import (
    FLY_FGS_NOD1_ROOT_IDS,
    FLY_FGS_SOURCE_MANIFEST_SHA256,
    load_registered_fly_fgs_fixture,
)
from fly_sensor2behavior.fly_fgs_runtime import (
    NodeFlyFGSCircuitRuntime,
    default_fly_fgs_runtime_script_path,
)

from test_canonical_closed_loop import (
    _DEFAULT_EFFECTOR_HYPOTHESIS,
    _high_gain_bridge,
    _initial_body,
    _runner,
    FakeCircuitRuntime,
    FakeFlightPhysics,
)


NODE_AVAILABLE = shutil.which("node") is not None and Path(
    default_fly_fgs_runtime_script_path()
).is_file()


def _canonical_run(duration_s: float = 0.020):
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=duration_s,
        include_retinal_input=True,
        include_full_cell_state=True,
    )
    simulator, _, _ = _runner(config, bridge=_high_gain_bridge(seed=31))
    result = simulator.run()
    checkpoint = simulator.checkpoint()
    return simulator, result, checkpoint


def _digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _with_registered_display_axes(result):
    """Replace only fake display lanes with exact registered axis-width rows."""

    fixture = load_registered_fly_fgs_fixture()
    cell_ids = list(fixture.full_cell_state["cell_ids"])
    nod_indices = {
        root_id: cell_ids.index("r%s" % root_id)
        for root_id in FLY_FGS_NOD1_ROOT_IDS
    }
    intervals = []
    for interval in result.intervals:
        observation = interval.circuit_observation
        if observation is None:
            intervals.append(interval)
            continue
        sample = observation.sample
        source_index = sample.sample_index
        voltage = list(fixture.full_cell_state["voltage_v"][source_index])
        for root_id, index in nod_indices.items():
            voltage[index] = sample.nod1_voltage_v[root_id]
        registered_sample = replace(
            sample,
            retinal_input_luminance=tuple(
                fixture.stimulus["retinal_input"]["luminance"][source_index]
            ),
            full_cell_voltage_v=tuple(voltage),
            full_cell_activity=tuple(
                fixture.full_cell_state["activity"][source_index]
            ),
        )
        intervals.append(
            replace(
                interval,
                circuit_observation=replace(observation, sample=registered_sample),
            )
        )
    return replace(result, intervals=tuple(intervals)), fixture


def test_array_bundle_preserves_all_clocks_and_does_not_backfill_natural_t0():
    simulator, result, _ = _canonical_run(0.006)
    try:
        bundle = canonical_array_bundle(result)
        transition_count = 60
        physics_time = bundle.arrays["physics_time_s"][0]
        np.testing.assert_allclose(
            physics_time,
            np.arange(transition_count + 1, dtype=float) * 0.0001,
            rtol=0.0,
            atol=1.0e-15,
        )
        assert bundle.arrays["bridge_interval_start_s"][0].shape == (12,)
        assert bundle.arrays["bridge_phase_path_unwrapped_rad"][0].shape == (12, 6)
        assert bundle.arrays["circuit_measurement_time_s"][0].tolist() == [
            0.0,
            0.005,
        ]
        assert bundle.arrays["muscle_effective_activation"][0].shape[0] == 61
        assert bundle.arrays["muscle_natural_activation"][0].shape[0] == 60
        np.testing.assert_array_equal(
            bundle.arrays["muscle_natural_endpoint_time_s"][0],
            physics_time[1:],
        )
        assert bundle.axis_metadata["nod1_voltage_v"]["side_semantics"].endswith(
            "anatomy unknown"
        )
        expected_mn_axis = [
            "raw_app_%s:%s" % (lane, motor)
            for lane in ("L", "R")
            for motor in ("MN-iv2", "MN-i1", "MN-iv1", "MN-b3")
        ]
        assert bundle.axis_metadata["generated_mn_rate_hz"][
            "axis_labels"
        ] == expected_mn_axis
        first_rates = {
            "raw_app_%s:%s" % (sample.raw_app_side.value, sample.target_name): sample.rate_hz
            for sample in result.intervals[0].bridge.generated_motor_rates
        }
        np.testing.assert_array_equal(
            bundle.arrays["generated_mn_rate_hz"][0][0],
            [first_rates[channel] for channel in expected_mn_axis],
        )
        assert np.all(bundle.arrays["held_dn_rate_present"][0])
        assert np.all(bundle.arrays["held_mn_rate_present"][0])
        assert {record["record_kind"] for record in bundle.rate_records} == {
            "generated",
            "held_at_interval_start",
        }
        assert len(bundle.intervention_records) == transition_count + 12
        np.testing.assert_array_equal(
            bundle.arrays["aerodynamic_force_body_n"][0][0], np.zeros(3)
        )
        assert result.initial_articulated_physics_telemetry is not None
        np.testing.assert_array_equal(
            result.initial_articulated_physics_telemetry.measured_wing_position_rad,
            np.zeros(6),
        )
    finally:
        simulator.close()


def test_registered_display_axes_and_online_circuit_replay_are_exact_and_motor_ineligible():
    simulator, result, checkpoint = _canonical_run(0.006)
    try:
        registered_result, fixture = _with_registered_display_axes(result)
        bundle = canonical_array_bundle(registered_result)
        full_axis = bundle.axis_metadata["full_circuit_voltage_v"]
        retinal_axis = bundle.axis_metadata["retinal_input_luminance"]
        expected_cell_ids = list(fixture.full_cell_state["cell_ids"])
        expected_retinal = fixture.circuit_topology["retinotopic_t4a"]

        assert full_axis["axis_labels"] == expected_cell_ids
        assert full_axis["axis_registration"] == (
            "registered_fly_fgs_circuit_bundle_order"
        )
        assert full_axis["eligible_motor_input"] is False
        assert len(full_axis["axis_sha256"]) == 64
        assert retinal_axis["axis_labels"] == expected_retinal["cell_ids"]
        assert retinal_axis["parent_cell_indices"] == expected_retinal[
            "cell_indices"
        ]
        assert retinal_axis["azimuth_deg"] == expected_retinal["azimuth_deg"]
        assert retinal_axis["elevation_deg"] == expected_retinal["elevation_deg"]
        assert retinal_axis["eligible_motor_input"] is False

        replay = canonical_closed_loop_to_web_replay(
            registered_result,
            checkpoint=checkpoint,
        )
        assert "circuit_replay" not in replay
        attachment = replay["online_circuit_replay"]
        assert attachment["source_kind"] == (
            "registered_fly_fgs_online_circuit_replay"
        )
        assert attachment["display_only"] is True
        assert attachment["eligible_motor_input"] is False
        assert attachment["source_receipts"]["source_manifest_sha256"] == (
            FLY_FGS_SOURCE_MANIFEST_SHA256
        )
        assert attachment["circuit_topology"] == fixture.circuit_topology
        assert attachment["axis_registration"]["full_state_axis_labels"] == (
            expected_cell_ids
        )
        assert attachment["motor_boundary"]["eligible_root_ids"] == list(
            FLY_FGS_NOD1_ROOT_IDS
        )
        assert attachment["motor_boundary"]["full_cell_state_eligible"] is False
        np.testing.assert_array_equal(
            attachment["full_cell_state"]["voltage_v"],
            bundle.arrays["full_circuit_voltage_v"][0],
        )
        np.testing.assert_array_equal(
            attachment["retinal_input"]["luminance"],
            bundle.arrays["retinal_input_luminance"][0],
        )
        assert [
            row["measurement_time_s"] for row in attachment["causal_samples"]
        ] == bundle.arrays["circuit_measurement_time_s"][0].tolist()
        assert attachment["causal_samples"][1]["applied_circuit_control"] == (
            attachment["causal_samples"][1]["requested_control"]
        )
        unsigned = {
            key: value
            for key, value in attachment.items()
            if key != "attachment_sha256"
        }
        assert attachment["attachment_sha256"] == _digest(unsigned)

        tampered = json.loads(json.dumps(replay))
        tampered["online_circuit_replay"]["full_cell_state"]["voltage_v"][0][
            0
        ] += 1.0e-12
        assert canonical_web_projection_sha256(tampered) != (
            canonical_web_projection_sha256(replay)
        )
    finally:
        simulator.close()


def test_registered_width_requires_exact_retinal_and_nod1_axis_alignment():
    simulator, result, _ = _canonical_run(0.001)
    try:
        registered_result, fixture = _with_registered_display_axes(result)
        first = registered_result.intervals[0]
        assert first.circuit_observation is not None
        sample = first.circuit_observation.sample
        cell_ids = list(fixture.full_cell_state["cell_ids"])
        nod_index = cell_ids.index("r%s" % FLY_FGS_NOD1_ROOT_IDS[0])
        bad_voltage = list(sample.full_cell_voltage_v)
        bad_voltage[nod_index] += 1.0e-12
        bad_sample = replace(sample, full_cell_voltage_v=tuple(bad_voltage))
        with pytest.raises(ValueError, match="registered NOD1 cell axis"):
            canonical_array_bundle(
                replace(
                    registered_result,
                    intervals=(
                        replace(
                            first,
                            circuit_observation=replace(
                                first.circuit_observation, sample=bad_sample
                            ),
                        ),
                        *registered_result.intervals[1:],
                    ),
                )
            )

        wrong_retina = replace(
            sample,
            retinal_input_luminance=(0.0, 1.0),
        )
        with pytest.raises(ValueError, match="unregistered retinal axis"):
            canonical_array_bundle(
                replace(
                    registered_result,
                    intervals=(
                        replace(
                            first,
                            circuit_observation=replace(
                                first.circuit_observation, sample=wrong_retina
                            ),
                        ),
                        *registered_result.intervals[1:],
                    ),
                )
            )
    finally:
        simulator.close()


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node fly-FGS runtime is unavailable")
def test_real_online_runtime_publishes_registered_rich_display_attachment():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.005,
        include_retinal_input=True,
        include_full_cell_state=True,
    )
    body = _initial_body(0.0, 0.0)
    simulator = CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=NodeFlyFGSCircuitRuntime(request_timeout_s=180.0),
        bridge=_high_gain_bridge(seed=47),
        mechanics=StreamingMuscleWingStepper(),
        physics_adapter=FakeFlightPhysics(body),
        initial_body_state=body,
    )
    try:
        result = simulator.run()
        checkpoint = simulator.checkpoint()
        bundle = canonical_array_bundle(result)
        assert bundle.arrays["retinal_input_luminance"][0].shape == (1, 1441)
        assert bundle.arrays["full_circuit_voltage_v"][0].shape == (1, 1684)
        assert bundle.axis_metadata["full_circuit_voltage_v"][
            "axis_registration"
        ] == "registered_fly_fgs_circuit_bundle_order"
        replay = canonical_closed_loop_to_web_replay(
            result,
            checkpoint=checkpoint,
        )
        attachment = replay["online_circuit_replay"]
        assert attachment["clocks"]["sample_count"] == 1
        assert len(attachment["retinal_input"]["luminance"][0]) == 1441
        assert len(attachment["full_cell_state"]["voltage_v"][0]) == 1684
        assert attachment["motor_boundary"]["eligible_signal"] == "nod1_voltage_v"

        tampered = checkpoint.to_dict()
        tampered["components"]["circuit"]["dynamic_state"][
            "sample_index"
        ] += 1
        tampered["payload_sha256"] = _digest(
            {
                key: value
                for key, value in tampered.items()
                if key != "payload_sha256"
            }
        )
        with pytest.raises(ValueError, match="component payload SHA-256"):
            canonical_closed_loop_to_web_replay(
                result,
                checkpoint=CanonicalClosedLoopCheckpoint(tampered),
            )
    finally:
        simulator.close()


def test_online_web_projection_uses_exported_phase_and_discrete_event_receipts():
    simulator, result, checkpoint = _canonical_run()
    try:
        replay = canonical_closed_loop_to_web_replay(
            result,
            checkpoint=checkpoint,
        )
        assert replay["wing_phase_source"] == "exported_model_owned_oscillator_endpoints"
        assert replay["neural_model_scope"]["kind"] == "online_registered_fly_fgs_closed_loop"
        assert replay["online_closed_loop"]["anatomical_laterality"] == "unknown"
        assert "prohibited" in replay["online_closed_loop"]["signed_behavior_claim_policy"]
        assert replay["online_closed_loop"]["motor_events"]
        assert len(replay["frames"]) == 201
        exported_phase = [frame["wing_phase_unwrapped_rad"] for frame in replay["frames"]]
        np.testing.assert_array_equal(
            exported_phase,
            canonical_array_bundle(result).arrays["wing_phase_unwrapped_rad"][0],
        )
        event_ids = {
            event["event_id"]
            for event in replay["online_closed_loop"]["motor_events"]
        }
        received_ids = {
            event_id
            for frame in replay["frames"]
            for event_id in (
                frame["applied_motor_event_ids"]
                + frame["suppressed_motor_event_ids"]
            )
        }
        assert received_ids
        assert received_ids.issubset(event_ids)
        assert all(
            event["availability_time_s"] >= event["event_time_s"]
            for event in replay["online_closed_loop"]["motor_events"]
        )
        events = replay["online_closed_loop"]["motor_events"]
        assert {event["mechanics_disposition"] for event in events}.issubset(
            {"pending_at_episode_end", "applied", "suppressed"}
        )
        assert all(event["generated_by_bridge"] is True for event in events)
        receipt_counts = Counter(
            event_id
            for frame in replay["frames"]
            for event_id in (
                frame["applied_motor_event_ids"]
                + frame["suppressed_motor_event_ids"]
            )
        )
        assert set(receipt_counts) == received_ids
        assert set(receipt_counts.values()) == {1}
        assert replay["frames"][0]["applied_motor_event_ids"] == []
        assert replay["frames"][0]["suppressed_motor_event_ids"] == []
        assert all(
            channel["side"] == "unknown"
            for channel in replay["individual_muscle_channels"]
        )
        # Strict JSON preflight is part of the publication function.
        json.dumps(replay, allow_nan=False, sort_keys=True)
    finally:
        simulator.close()


def test_online_web_projection_updates_held_rates_at_exact_bridge_boundaries():
    simulator, result, checkpoint = _canonical_run(0.006)
    try:
        replay = canonical_closed_loop_to_web_replay(
            result,
            checkpoint=checkpoint,
        )
        frames = replay["frames"]
        physics_per_bridge = int(round(0.0005 / 0.0001))
        assert physics_per_bridge == 5

        # Bridge-held DN/MN rates belong to the outgoing half-open interval,
        # including its exact left boundary.  Select a changing channel so the
        # regression cannot pass if the prior interval is repeated at t=5 ms.
        changing_channel = None
        changing_tick = None
        for tick in range(1, len(result.intervals)):
            previous = {
                "dn:raw_app_%s:%s"
                % (sample.raw_app_side.value, sample.target_name): sample.rate_hz
                for sample in result.intervals[tick - 1].bridge.held_dn_rates
            }
            current = {
                "dn:raw_app_%s:%s"
                % (sample.raw_app_side.value, sample.target_name): sample.rate_hz
                for sample in result.intervals[tick].bridge.held_dn_rates
            }
            for channel, value in current.items():
                if channel in previous and value != previous[channel]:
                    changing_channel = channel
                    changing_tick = tick
                    break
            if changing_channel is not None:
                break
        assert changing_channel is not None
        assert changing_tick is not None

        boundary_frame = frames[changing_tick * physics_per_bridge]
        current_rate = {
            "dn:raw_app_%s:%s"
            % (sample.raw_app_side.value, sample.target_name): sample.rate_hz
            for sample in result.intervals[changing_tick].bridge.held_dn_rates
        }[changing_channel]
        previous_rate = {
            "dn:raw_app_%s:%s"
            % (sample.raw_app_side.value, sample.target_name): sample.rate_hz
            for sample in result.intervals[changing_tick - 1].bridge.held_dn_rates
        }[changing_channel]
        assert boundary_frame["t"] == pytest.approx(
            result.intervals[changing_tick].interval_start_s,
            abs=1.0e-15,
        )
        assert boundary_frame["neural_signals"][changing_channel] == current_rate
        assert boundary_frame["neural_signals"][changing_channel] != previous_rate

        # The terminal endpoint has no outgoing interval; it intentionally
        # retains the final interval's held rates.
        final_interval = result.intervals[-1]
        final_dn = {
            "dn:raw_app_%s:%s"
            % (sample.raw_app_side.value, sample.target_name): sample.rate_hz
            for sample in final_interval.bridge.held_dn_rates
        }
        for channel, rate_hz in final_dn.items():
            assert frames[-1]["neural_signals"][channel] == rate_hz
    finally:
        simulator.close()


def test_immutable_canonical_artifact_binds_arrays_tables_and_checkpoint(tmp_path):
    simulator, result, checkpoint = _canonical_run(0.006)
    try:
        run_dir = tmp_path / "run"
        expected_run_id = canonical_run_id(
            result, scenario_id="online-test", checkpoint=checkpoint
        )
        replay = canonical_closed_loop_to_web_replay(
            result,
            checkpoint=checkpoint,
            scenario_id="online-test",
            source_run_id=expected_run_id,
        )
        manifest = write_canonical_closed_loop_artifact(
            run_dir,
            result,
            checkpoint=checkpoint,
            scenario_id="online-test",
            created_at_utc="2026-07-18T00:00:00Z",
            chunk_samples=17,
            runtime_receipts={"worker_image_digest": "sha256:test-only"},
            web_replay=replay,
        )
        assert manifest["run_id"] == expected_run_id
        assert manifest["schema_version"] == CANONICAL_ARTIFACT_SCHEMA_VERSION
        assert len(manifest["scientific_content_sha256"]) == 64
        assert manifest["checkpoint"]["payload_sha256"] == checkpoint.to_dict()[
            "payload_sha256"
        ]
        assert manifest["checkpoint"]["file_sha256"] == sha256_file(
            run_dir / "checkpoint.json"
        )
        assert manifest["tables"]["motor_events"]["timing_semantics"].endswith(
            "never interpolate"
        )
        assert manifest["tables"]["intervention_activity"]["count"] == 72
        assert manifest["tables"]["intervention_activity"]["definitions"] == manifest[
            "checkpoint"
        ]["intervention_contracts"]
        np.testing.assert_array_equal(
            read_chunked_array(run_dir, manifest["arrays"]["physics_time_s"]),
            np.arange(61, dtype=float) * 0.0001,
        )
        sidecar_digest = (run_dir / "manifest.sha256").read_text(
            encoding="ascii"
        ).split()[0]
        assert sidecar_digest == sha256_file(run_dir / "manifest.json")
        assert manifest["web_replay_projection"]["sha256"] == canonical_web_projection_sha256(replay)
        bound = bind_canonical_web_replay_to_artifact(
            replay, manifest, sidecar_digest
        )
        assert bound["source_artifact_manifest_sha256"] == sidecar_digest
        assert canonical_web_projection_sha256(bound) == manifest[
            "web_replay_projection"
        ]["sha256"]
        with pytest.raises(ValueError, match="does not bind"):
            bind_canonical_web_replay_to_artifact(replay, manifest, "0" * 64)
        with pytest.raises(FileExistsError, match="empty directory"):
            write_canonical_closed_loop_artifact(
                run_dir,
                result,
                checkpoint=checkpoint,
            )
    finally:
        simulator.close()


def test_web_artifact_reference_requires_run_identity_and_sha256():
    simulator, result, _ = _canonical_run(0.001)
    try:
        with pytest.raises(ValueError, match="requires source_run_id"):
            canonical_closed_loop_to_web_replay(
                result,
                source_artifact_manifest_sha256="0" * 64,
            )
        replay = canonical_closed_loop_to_web_replay(
            result,
            source_run_id="run-1",
            source_artifact_manifest_sha256="a" * 64,
        )
        assert replay["source_run_id"] == "run-1"
        assert replay["source_artifact_schema_version"] == CANONICAL_ARTIFACT_SCHEMA_VERSION
    finally:
        simulator.close()


def test_held_rate_arrays_use_presence_masks_and_only_latest_causal_samples():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.003,
    )
    simulator, _, _ = _runner(config)
    try:
        result = simulator.run()
        bundle = canonical_array_bundle(result)
        for stage in ("dn", "mn"):
            present = bundle.arrays["held_%s_rate_present" % stage][0]
            values = bundle.arrays["held_%s_rate_hz" % stage][0]
            measurement = bundle.arrays[
                "held_%s_rate_measurement_time_s" % stage
            ][0]
            availability = bundle.arrays[
                "held_%s_rate_availability_time_s" % stage
            ][0]
            assert not np.any(present[0])
            assert np.all(values[~present] == 0.0)
            assert np.all(measurement[~present] == 0.0)
            assert np.all(availability[~present] == 0.0)
            bridge_time = bundle.arrays["bridge_interval_start_s"][0]
            assert np.all(
                availability[present]
                <= np.broadcast_to(bridge_time[:, None], availability.shape)[present]
                + 1.0e-12
            )
        held_records = [
            record
            for record in bundle.rate_records
            if record["record_kind"] == "held_at_interval_start"
        ]
        assert held_records
        assert all(
            record["availability_time_s"]
            <= result.intervals[record["bridge_tick_index"]].interval_start_s
            + 1.0e-12
            for record in held_records
        )
    finally:
        simulator.close()


def test_publication_rejects_forged_reset_clock_circuit_and_rate_receipts():
    simulator, result, _ = _canonical_run(0.006)
    try:
        first = result.intervals[0]
        bad_clock = replace(first, interval_start_s=0.0001)
        with pytest.raises(ValueError, match="bridge interval clock"):
            canonical_array_bundle(
                replace(result, intervals=(bad_clock, *result.intervals[1:]))
            )

        assert first.circuit_observation is not None
        bad_observation = replace(
            first.circuit_observation,
            initialization_mode="body_scene_control_applied",
        )
        with pytest.raises(ValueError, match="initialization/reset mode"):
            canonical_array_bundle(
                replace(
                    result,
                    intervals=(
                        replace(first, circuit_observation=bad_observation),
                        *result.intervals[1:],
                    ),
                )
            )

        missing_retina_sample = replace(
            first.circuit_observation.sample,
            retinal_input_luminance=None,
        )
        missing_retina = replace(
            first.circuit_observation,
            sample=missing_retina_sample,
        )
        with pytest.raises(ValueError, match="retinal trace presence"):
            canonical_array_bundle(
                replace(
                    result,
                    intervals=(
                        replace(first, circuit_observation=missing_retina),
                        *result.intervals[1:],
                    ),
                )
            )

        generated = first.bridge.generated_motor_rates
        duplicate_rates = (generated[0], generated[0], *generated[2:])
        bad_bridge = replace(first.bridge, generated_motor_rates=duplicate_rates)
        bad_bridge_start = replace(
            first.bridge_start, generated_motor_rates=duplicate_rates
        )
        with pytest.raises(ValueError, match="duplicate rate channel"):
            canonical_array_bundle(
                replace(
                    result,
                    intervals=(
                        replace(
                            first,
                            bridge=bad_bridge,
                            bridge_start=bad_bridge_start,
                        ),
                        *result.intervals[1:],
                    ),
                )
            )
    finally:
        simulator.close()


def test_checkpoint_endpoint_components_and_event_ledgers_are_revalidated():
    simulator, result, checkpoint = _canonical_run(0.006)
    try:
        bad_component = checkpoint.to_dict()
        bad_component["components"]["mechanics"]["state"][
            "applied_event_ids"
        ].append("invented-event")
        mechanics = bad_component["components"]["mechanics"]
        mechanics["payload_sha256"] = _digest(
            {key: value for key, value in mechanics.items() if key != "payload_sha256"}
        )
        bad_component["payload_sha256"] = _digest(
            {
                key: value
                for key, value in bad_component.items()
                if key != "payload_sha256"
            }
        )
        forged = CanonicalClosedLoopCheckpoint(bad_component)
        with pytest.raises(ValueError, match="event/component state"):
            canonical_run_id(result, checkpoint=forged)

        bad_body = checkpoint.to_dict()
        bad_body["state"]["body_state"]["position_world_m"][0] += 1.0e-6
        bad_body["payload_sha256"] = _digest(
            {key: value for key, value in bad_body.items() if key != "payload_sha256"}
        )
        with pytest.raises(ValueError, match="body state"):
            canonical_run_id(
                result,
                checkpoint=CanonicalClosedLoopCheckpoint(bad_body),
            )

        broken_child_digest = checkpoint.to_dict()
        broken_child_digest["components"]["bridge"]["payload_sha256"] = "0" * 64
        broken_child_digest["payload_sha256"] = _digest(
            {
                key: value
                for key, value in broken_child_digest.items()
                if key != "payload_sha256"
            }
        )
        with pytest.raises(ValueError, match="component payload SHA-256"):
            canonical_run_id(
                result,
                checkpoint=CanonicalClosedLoopCheckpoint(broken_child_digest),
            )
    finally:
        simulator.close()


def test_checkpoint_receipt_accepts_verified_flybody_style_physics_digest_prefix(
    tmp_path,
):
    simulator, result, checkpoint = _canonical_run(0.006)
    try:
        prefixed = checkpoint.to_dict()
        physics = prefixed["components"]["physics"]
        bare_digest = physics["payload_sha256"]
        physics["payload_sha256"] = "sha256:" + bare_digest
        prefixed["payload_sha256"] = _digest(
            {
                key: value
                for key, value in prefixed.items()
                if key != "payload_sha256"
            }
        )
        wrapped = CanonicalClosedLoopCheckpoint(prefixed)
        run_id = canonical_run_id(result, checkpoint=wrapped)
        assert run_id.startswith("canonical_closed_loop-")

        run_dir = tmp_path / "prefixed-physics-checkpoint"
        manifest = write_canonical_closed_loop_artifact(
            run_dir,
            result,
            checkpoint=wrapped,
        )
        assert manifest["checkpoint"]["component_payload_sha256"]["physics"] == (
            bare_digest
        )
        assert json.loads((run_dir / "checkpoint.json").read_text())["components"][
            "physics"
        ]["payload_sha256"] == "sha256:" + bare_digest

        corrupt = prefixed.copy()
        corrupt = json.loads(json.dumps(corrupt))
        corrupt["components"]["physics"]["payload_sha256"] = (
            "sha256:" + "0" * 64
        )
        corrupt["payload_sha256"] = _digest(
            {
                key: value
                for key, value in corrupt.items()
                if key != "payload_sha256"
            }
        )
        with pytest.raises(ValueError, match="component payload SHA-256"):
            canonical_run_id(
                result,
                checkpoint=CanonicalClosedLoopCheckpoint(corrupt),
            )
    finally:
        simulator.close()


def test_run_identity_binds_authoritative_trajectory_content():
    simulator, result, _ = _canonical_run(0.001)
    try:
        original_id = canonical_run_id(result, scenario_id="content-bound")
        first_interval = result.intervals[0]
        first_transition = first_interval.physics[0]
        wrench = first_transition.aerodynamic_wrench_after
        altered_wrench = AerodynamicWrench(
            force_body_n=wrench.force_body_n + np.array([1.0e-12, 0.0, 0.0]),
            torque_body_n_m=wrench.torque_body_n_m,
            left_force_body_n=wrench.left_force_body_n,
            right_force_body_n=wrench.right_force_body_n,
            mechanical_power_w=wrench.mechanical_power_w,
        )
        altered_transition = replace(
            first_transition, aerodynamic_wrench_after=altered_wrench
        )
        altered_interval = replace(
            first_interval,
            physics=(altered_transition, *first_interval.physics[1:]),
        )
        altered_result = replace(
            result,
            intervals=(altered_interval, *result.intervals[1:]),
        )
        altered_id = canonical_run_id(altered_result, scenario_id="content-bound")
        assert altered_id != original_id
    finally:
        simulator.close()


def test_force_stage_intervention_contracts_and_suppressed_events_are_sealed():
    duration_s = 0.020
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=duration_s,
        include_retinal_input=True,
        include_full_cell_state=True,
    )
    interventions = tuple(
        StreamingMuscleIntervention(
            intervention_id="silence-%s" % muscle,
            muscle=muscle,
            start_s=0.0,
            end_s=duration_s,
            mode=MechanicsInterventionMode.SILENCE,
            raw_app_side=None,
        )
        for muscle in ("iv2", "i1", "iv1", "b3")
    )
    body = _initial_body(0.0, 2.0)
    physics = FakeFlightPhysics(body)
    circuit = FakeCircuitRuntime(lambda: physics.step_count)
    simulator = CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=circuit,
        bridge=_high_gain_bridge(seed=31),
        mechanics=StreamingMuscleWingStepper(
            StreamingMechanicsConfig(muscle_interventions=interventions)
        ),
        physics_adapter=physics,
        initial_body_state=body,
    )
    try:
        result = simulator.run()
        checkpoint = simulator.checkpoint()
        bundle = canonical_array_bundle(result)
        delivered = [
            event for event in bundle.event_records if event["delivered_to_mechanics"]
        ]
        assert delivered
        assert all(
            event["mechanics_disposition"] == "suppressed" for event in delivered
        )
        assert all(
            event["suppressed_by_mechanics_intervention"] is True
            and event["applied_to_muscle_state"] is False
            for event in delivered
        )
        replay = canonical_closed_loop_to_web_replay(result, checkpoint=checkpoint)
        suppressed_frame_ids = {
            event_id
            for frame in replay["frames"]
            for event_id in frame["suppressed_motor_event_ids"]
        }
        assert suppressed_frame_ids == {event["event_id"] for event in delivered}
        run_id = canonical_run_id(
            result,
            scenario_id="force-stage-silence",
            checkpoint=checkpoint,
        )
        assert run_id.startswith("force-stage-silence-")
        checkpoint_contracts = checkpoint.to_dict()["components"]["mechanics"][
            "config"
        ]["muscle_interventions"]
        assert [item["intervention_id"] for item in checkpoint_contracts] == [
            "silence-b3",
            "silence-i1",
            "silence-iv1",
            "silence-iv2",
        ]
    finally:
        simulator.close()
