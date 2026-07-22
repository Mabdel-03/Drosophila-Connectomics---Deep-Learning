from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import fly_sensor2behavior.pipeline as pipeline_module
from fly_sensor2behavior.flight import (
    AerodynamicWrench,
    FlightEpisodeRunner,
    RigidBodyState,
)
from fly_sensor2behavior.nod1_parity import load_registered_nod1_browser_fixture
from fly_sensor2behavior.pipeline import (
    NOD1FlightPipelineConfig,
    nod1_run_identity_context,
    nod1_run_to_web_replay,
    run_registered_fly_fgs_flight_pipeline,
    write_nod1_flight_artifact,
)
from fly_sensor2behavior.vision import AnalyticGratingScene, PanoramicRetina


class _FakePhysics:
    backend_name = "fake_fly_fgs_physics"
    aerodynamic_owner = "fake_fly_fgs_physics"
    body_state_reference = "manufactured pipeline test frame"

    def __init__(self) -> None:
        self.calls = 0
        self.state = RigidBodyState()
        self.last_actuator_torque_n_m = np.zeros(6, dtype=float)

    def default_initial_state(self) -> RigidBodyState:
        state = RigidBodyState()
        state.position_world_m[2] = 0.1
        return state

    def reset(self, initial_state: RigidBodyState) -> None:
        self.calls = 0
        self.state = initial_state.copy()

    def step(self, wings, force_body_n, torque_body_n_m, dt_s):
        del wings, force_body_n, torque_body_n_m
        self.calls += 1
        self.state.position_world_m[0] += dt_s
        return self.state.copy()

    def aerodynamic_wrench(self) -> AerodynamicWrench:
        return AerodynamicWrench(
            force_body_n=np.array([1.0e-6, 0.0, 0.0]),
            torque_body_n_m=np.array([0.0, 2.0e-9, 0.0]),
            left_force_body_n=np.zeros(3),
            right_force_body_n=np.zeros(3),
            mechanical_power_w=3.0e-6,
        )

    def provenance_metadata(self):
        return {
            "engine": "manufactured deterministic fake",
            "authority": "software-interface-test-only",
        }


class _FakeFlyFGSFixture:
    def __init__(self, *, circuit_replay=None) -> None:
        self.circuit_trace = load_registered_nod1_browser_fixture().circuit_trace
        self.source_manifest_sha256 = "1" * 64
        self.capture_sha256 = "2" * 64
        self.source_manifest_uri = "urn:test:fly-fgs-manifest"
        self.capture_uri = "urn:test:fly-fgs-capture"
        self.snapshot_id = "fly-fgs-manufactured-pipeline-test"
        self.fixed_step = {
            "dt_s": 0.005,
            "duration_s": 0.5,
            "sample_count": 100,
            "sample_interval": "half-open [0, 0.5 s)",
        }
        self.stimulus = {
            "configuration": {"figure_enabled": True},
            "schedule": {"figure_world_azimuth_deg": [0.0, 1.0]},
            "retinal_display": {
                "unit": "normalized luminance (1)",
                "azimuth_deg": [-1.0, 1.0],
                "luminance": [[0.0, 1.0], [1.0, 0.0]],
            },
        }
        self.circuit_inventory = {
            "cell_count": 1684,
            "event_count": 61789,
            "cell_type_counts": {
                "T4a": 1457,
                "LLPC1": 219,
                "Nod1": 4,
                "VCH": 2,
                "DCH": 2,
            },
        }
        self.rejected_downstream_fields = (
            "SCALE_M",
            "state.muscleAct",
            "state.ampL",
            "state.ampR",
            "state.yaw",
        )
        self.retinal_frames = (
            PanoramicRetina(receptor_count=8, sensor_latency_s=0.0).sample(
                AnalyticGratingScene(angular_velocity_rad_s=1.0),
                exposure_start_s=0.0,
                exposure_end_s=0.005,
            ),
        )
        self._circuit_replay = (
            {
                "schema_version": "1.0.0",
                "source_kind": "registered_fly_fgs_fixed_step_circuit_replay",
                "fixed_step": {"time_s": [0.0, 0.005]},
                "pooled_readout_traces": {
                    "nod1_activity": {
                        "raw_app_L": [0.1, 0.2],
                        "raw_app_R": [0.2, 0.1],
                    }
                },
                "full_cell_state_scope": {
                    "included": False,
                    "eligible_motor_input": False,
                },
                "rejected_downstream_fields": list(
                    self.rejected_downstream_fields
                ),
            }
            if circuit_replay is None
            else circuit_replay
        )

    def circuit_replay_attachment(self):
        return self._circuit_replay

    @property
    def dt_s(self):
        return float(self.fixed_step["dt_s"])

    @property
    def duration_s(self):
        return float(self.fixed_step["duration_s"])

    def pipeline_source_metadata(self):
        return {
            "input_mode": "registered_fly_fgs_fixed_step_circuit",
            "visual_source": "manufactured_fly_fgs_test_scene",
            "retinal_frames_present": True,
            "retinal_frame_count": len(self.retinal_frames),
            "source_manifest_uri": self.source_manifest_uri,
            "source_manifest_sha256": self.source_manifest_sha256,
            "capture_uri": self.capture_uri,
            "capture_sha256": self.capture_sha256,
            "source_duration_s": self.duration_s,
            "source_sample_dt_s": self.dt_s,
            "source_sample_count": 100,
            "source_sample_interval": self.fixed_step["sample_interval"],
            "source_sample_interval_end_s": self.duration_s,
            "circuit_model_scope": "manufactured_fly_fgs_pipeline_fixture",
            "circuit_cell_count": 1684,
            "stimulus": {"figure_enabled": True},
            "rejected_downstream_fields": list(
                self.rejected_downstream_fields
            ),
        }


def _run_with_fixture(monkeypatch, fixture, *, use_fake_physics=True):
    observed = {}

    def fake_loader(**kwargs):
        observed.update(kwargs)
        return fixture

    monkeypatch.setattr(
        pipeline_module,
        "load_registered_fly_fgs_fixture",
        fake_loader,
    )
    physics = _FakePhysics() if use_fake_physics else None
    run = run_registered_fly_fgs_flight_pipeline(
        manifest_path=Path("manifest.json"),
        capture_path=Path("capture.json.gz"),
        config=NOD1FlightPipelineConfig(
            physics_dt_s=0.005,
            neural_dt_s=0.005,
            logging_dt_s=0.005,
            seed=23,
        ),
        runner=(
            FlightEpisodeRunner(physics_adapter=physics)
            if physics is not None
            else None
        ),
    )
    return run, physics, observed


def test_registered_fly_fgs_uses_only_circuit_trace_and_project_mechanics(
    monkeypatch,
):
    fixture = _FakeFlyFGSFixture()
    run, physics, observed = _run_with_fixture(monkeypatch, fixture)

    assert observed == {
        "manifest_path": Path("manifest.json"),
        "capture_path": Path("capture.json.gz"),
    }
    assert run.circuit is fixture.circuit_trace
    assert len(run.circuit.signals) == 4
    assert run.retinal_frames == fixture.retinal_frames
    assert run.flight_config.duration_s == pytest.approx(0.5)
    assert run.flight.diagnostics.physics_backend == "fake_fly_fgs_physics"
    assert physics.calls == 100
    assert run.source_metadata["input_mode"] == (
        "registered_fly_fgs_fixed_step_circuit"
    )
    assert run.source_metadata["motor_input_scope"] == (
        "four_registered_nod1_voltage_channels_only"
    )
    assert run.source_metadata["circuit_cell_count"] == 1684
    assert run.source_metadata["retinal_frames_present"] is True
    assert run.source_metadata["retinal_frame_count"] == 1
    assert run.source_metadata["fly_fgs_downstream_mechanics_imported"] is False
    assert run.source_metadata["full_cell_state_eligible_motor_input"] is False
    assert run.circuit_replay == fixture.circuit_replay_attachment()
    assert run.circuit_replay is not fixture.circuit_replay_attachment()
    json.dumps(run.circuit_replay, allow_nan=False, sort_keys=True)


def test_fly_fgs_artifact_and_replay_are_canonical_and_json_safe(
    monkeypatch,
    tmp_path,
):
    fixture = _FakeFlyFGSFixture()
    run, _, _ = _run_with_fixture(
        monkeypatch,
        fixture,
        use_fake_physics=False,
    )

    manifest = write_nod1_flight_artifact(
        tmp_path / "run",
        run,
        chunk_samples=32,
        created_at_utc="2026-07-18T00:00:00Z",
    )
    pipeline = manifest["pipeline"]
    assert pipeline["pipeline_id"] == "fly-fgs-retina-circuit-to-flight-v1"
    assert pipeline["mode"] == (
        "open_loop_registered_fly_fgs_fixed_step_circuit_output"
    )
    assert pipeline["visual_boundary"]["status"] == (
        "registered_retinal_frames_and_circuit_replay"
    )
    assert "fly-FGS muscle" in pipeline["visual_boundary"]["notice"]
    assert pipeline["source_metadata"]["capture_sha256"] == "2" * 64

    replay = nod1_run_to_web_replay(run, target_sample_rate_hz=100.0)
    assert replay["id"] == "fly_fgs_canonical"
    assert replay["source_kind"] == (
        "exploratory_fly_fgs_circuit_reduced_pipeline"
    )
    assert replay["neural_model_scope"]["kind"] == (
        "registered_fly_fgs_fixed_step_circuit"
    )
    assert replay["neural_model_scope"]["circuit_cell_count"] == 1684
    assert replay["circuit_replay"] == run.circuit_replay
    assert replay["circuit_replay"]["full_cell_state_scope"][
        "eligible_motor_input"
    ] is False
    json.dumps(replay, allow_nan=False, sort_keys=True)

    identity = nod1_run_identity_context(run)
    assert identity["fly_fgs_source_manifest_sha256"] == "1" * 64
    assert identity["fly_fgs_registered_capture_sha256"] == "2" * 64
    assert len(identity["fly_fgs_circuit_replay_sha256"]) == 64


def test_non_json_circuit_replay_fails_before_physics(monkeypatch):
    fixture = _FakeFlyFGSFixture(circuit_replay={"bad": float("nan")})
    physics = _FakePhysics()
    monkeypatch.setattr(
        pipeline_module,
        "load_registered_fly_fgs_fixture",
        lambda **_kwargs: fixture,
    )

    with pytest.raises(ValueError, match="circuit_replay must contain only finite JSON"):
        run_registered_fly_fgs_flight_pipeline(
            config=NOD1FlightPipelineConfig(
                physics_dt_s=0.005,
                neural_dt_s=0.005,
                logging_dt_s=0.005,
                seed=23,
            ),
            runner=FlightEpisodeRunner(physics_adapter=physics),
        )
    assert physics.calls == 0
