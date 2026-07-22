from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_sensor2behavior.cli import build_parser, main
from fly_sensor2behavior.artifacts import web_replay_projection_sha256
from fly_sensor2behavior.flight import (
    AerodynamicWrench,
    FlightEpisodeRunner,
    RigidBodyState,
)
from fly_sensor2behavior.nod1_parity import (
    NOD1_BROWSER_PARITY_APP_SIDES,
    NOD1_BROWSER_PARITY_ROOT_SIDES,
    load_registered_nod1_browser_fixture,
    parse_registered_nod1_browser_fixture_bytes,
)
from fly_sensor2behavior.pipeline import (
    NOD1FlightPipelineConfig,
    nod1_run_to_web_replay,
    run_registered_nod1_browser_flight_pipeline,
    write_nod1_flight_artifact,
)
from fly_sensor2behavior.schema import (
    CircuitSignalKind,
    ConfidenceLevel,
    EvidenceTier,
    SideMappingMethod,
)


SOURCE_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = SOURCE_ROOT / "data" / "reference" / "nod1_browser_python_parity.v1.json.gz"
EXPECTED_SHA256 = "a76794e630533822468971cbdbe2164a3d1ddce8a8a38811c9b9c1ad5b97a8a4"


class _FakeFixturePhysics:
    backend_name = "fake_fixture_physics"
    aerodynamic_owner = "fake_fixture_physics"
    body_state_reference = "manufactured rigid-body test frame"

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
        del wings
        np.testing.assert_array_equal(force_body_n, np.zeros(3))
        np.testing.assert_array_equal(torque_body_n_m, np.zeros(3))
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


@pytest.fixture(scope="module")
def compressed_fixture_bytes() -> bytes:
    return FIXTURE_PATH.read_bytes()


@pytest.fixture(scope="module")
def raw_fixture(compressed_fixture_bytes):
    return json.loads(gzip.decompress(compressed_fixture_bytes))


@pytest.fixture(scope="module")
def validated_registered_fixture():
    return load_registered_nod1_browser_fixture()


@pytest.fixture(scope="module")
def registered_pipeline_runs(validated_registered_fixture):
    import fly_sensor2behavior.pipeline as pipeline_module

    physics = _FakeFixturePhysics()
    config = NOD1FlightPipelineConfig(
        physics_dt_s=0.005,
        neural_dt_s=0.005,
        logging_dt_s=0.005,
        seed=17,
    )
    original_loader = pipeline_module.load_registered_nod1_browser_fixture
    pipeline_module.load_registered_nod1_browser_fixture = (
        lambda **_kwargs: validated_registered_fixture
    )
    try:
        fake_run = run_registered_nod1_browser_flight_pipeline(
            config=config,
            runner=FlightEpisodeRunner(physics_adapter=physics),
        )
        reduced_run = run_registered_nod1_browser_flight_pipeline(config=config)
    finally:
        pipeline_module.load_registered_nod1_browser_fixture = original_loader
    return {"fake": fake_run, "physics": physics, "reduced": reduced_run}


def test_fixture_availability_is_explicitly_an_inferred_offline_schedule(
    validated_registered_fixture,
):
    trace = validated_registered_fixture.circuit_trace
    metadata = validated_registered_fixture.pipeline_source_metadata()

    assert all(
        signal.availability_times_s == trace.sample_times_s
        for signal in trace.signals
    )
    assert all(
        signal.provenance.filters["availability_schedule"]
        == "inferred_offline_playback_at_sample_time"
        and signal.provenance.filters["captured_streaming_availability"] is False
        and "did not capture streaming availability" in signal.provenance.notes
        for signal in trace.signals
    )
    assert metadata["source_availability_semantics"].startswith(
        "inferred_offline_playback_at_sample_time"
    )


def test_registered_loader_imports_only_real_chromium_browser_si_voltage(
    compressed_fixture_bytes,
    raw_fixture,
    validated_registered_fixture,
    registered_pipeline_runs,
):
    trace = validated_registered_fixture.circuit_trace
    run = registered_pipeline_runs["reduced"]

    assert hashlib.sha256(compressed_fixture_bytes).hexdigest() == EXPECTED_SHA256
    assert trace.dataset.materialization == 783
    assert trace.dataset.neuron_universe == "proofread_139255"
    assert trace.dataset.coordinate_units == "nm"
    assert trace.sample_times_s == pytest.approx(tuple(index * 0.005 for index in range(100)))
    assert trace.sample_times_s[-1] == pytest.approx(0.495)
    assert trace.sample_interval_end_s == pytest.approx(0.5)
    assert trace.exact_timebase is True
    assert len(trace.signals) == 4

    by_root = {signal.neuron.entity_id: signal for signal in trace.signals}
    assert set(by_root) == set(NOD1_BROWSER_PARITY_ROOT_SIDES)
    for root_id, signal in by_root.items():
        record = raw_fixture["readouts"][root_id]
        assert signal.signal_kind is CircuitSignalKind.VOLTAGE
        assert signal.unit == "V"
        assert signal.values == tuple(record["browser_voltage_v"])
        assert signal.values != tuple(record["python_voltage_v"])
        assert signal.availability_times_s == trace.sample_times_s
        assert signal.confidence.level is ConfidenceLevel.LOW
        assert signal.confidence.tier is EvidenceTier.MODEL_INFERENCE
        assert signal.neuron.anatomical_side.value == "unknown"
        assert signal.neuron.side_context.anatomical_side.value == "unknown"
        assert (
            signal.neuron.side_context.app_rendering_side
            is NOD1_BROWSER_PARITY_APP_SIDES[root_id]
        )
        assert signal.neuron.side_context.mapping_method is SideMappingMethod.SIMULATION_CONVENTION
        assert "no soma-x coordinate" in signal.neuron.side_context.notes
        assert signal.provenance.artifact_hash == "sha256:" + EXPECTED_SHA256
        assert signal.provenance.filters["included_voltage_field"] == "browser_voltage_v"
        assert "python_voltage_v" in signal.provenance.filters["excluded_comparison_fields"]

    assert run.retinal_frames == ()
    assert run.source_metadata["retinal_frames_present"] is False
    assert "no source retinal frames" in run.source_metadata["retinal_boundary_notice"]


def test_compressed_byte_tamper_is_rejected_before_gzip(
    compressed_fixture_bytes,
):
    tampered = bytearray(compressed_fixture_bytes)
    tampered[-11] ^= 0x01
    with pytest.raises(ValueError, match="compressed fixture SHA-256 mismatch"):
        parse_registered_nod1_browser_fixture_bytes(
            bytes(tampered),
            expected_sha256=EXPECTED_SHA256,
            fixture_uri="data/reference/nod1_browser_python_parity.v1.json.gz",
            benchmark_case_version="2.0.0",
        )


@pytest.mark.parametrize(
    ("raw_json", "message"),
    (
        (b'{"schema_version":"1.0.0","schema_version":"1.0.0"}', "duplicate JSON key"),
        (b'{"value":NaN}', "non-finite JSON constant"),
    ),
)
def test_manufactured_non_strict_json_is_rejected(raw_json, message):
    compressed = gzip.compress(raw_json, mtime=0)
    with pytest.raises(ValueError, match=message):
        parse_registered_nod1_browser_fixture_bytes(
            compressed,
            expected_sha256=hashlib.sha256(compressed).hexdigest(),
            fixture_uri="urn:test:manufactured",
            benchmark_case_version="2.0.0",
        )


def test_deprecated_motor_proxy_is_rejected_recursively(raw_fixture):
    stimulus = raw_fixture["configuration"]["stimulus"]
    stimulus["steering"] = [999.0]
    try:
        encoded = json.dumps(
            raw_fixture,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    finally:
        del stimulus["steering"]
    compressed = gzip.compress(encoded, mtime=0)
    with pytest.raises(ValueError, match="deprecated motor field 'steering'"):
        parse_registered_nod1_browser_fixture_bytes(
            compressed,
            expected_sha256=hashlib.sha256(compressed).hexdigest(),
            fixture_uri="urn:test:deprecated-motor-proxy",
            benchmark_case_version="2.0.0",
        )


def test_registered_fixture_runs_full_half_open_capture_through_fake_physics(
    registered_pipeline_runs,
):
    run = registered_pipeline_runs["fake"]
    physics = registered_pipeline_runs["physics"]

    assert run.source_metadata["source_sample_interval"] == "half-open [0, duration_s)"
    assert run.source_metadata["source_duration_s"] == pytest.approx(0.5)
    assert run.source_metadata["source_sample_interval_end_s"] == pytest.approx(0.5)
    assert "causal availability" in run.source_metadata["latency_tail_semantics"]
    assert run.bridge.descending.duration_s == pytest.approx(0.5)
    assert run.bridge.descending.update_count == 1000
    assert run.bridge.wing_motor.duration_s == pytest.approx(0.5)
    assert run.bridge.wing_motor.update_count == 1000
    assert run.flight_config.duration_s == pytest.approx(0.5)
    assert run.flight.time_s[-1] == pytest.approx(0.5)
    assert run.flight.diagnostics.physics_steps == 100
    assert run.flight.diagnostics.physics_backend == "fake_fixture_physics"
    assert physics.calls == 100
    assert len(run.bridge.descending.channels) == 2
    assert len(run.bridge.wing_motor.channels) == 8
    final_dn_sample = next(
        sample
        for channel in run.bridge.descending.channels
        for sample in channel.samples
        if abs(sample.measurement_time_s - 0.495) < 1.0e-12
        and sample.source_measurement_time_s == pytest.approx(0.495)
    )
    assert final_dn_sample.availability_time_s == pytest.approx(0.498)
    latency_tail = next(
        sample
        for channel in run.bridge.wing_motor.channels
        for sample in channel.rate_samples
        if abs(sample.measurement_time_s - 0.498) < 1.0e-12
        and sample.source_measurement_time_s == pytest.approx(0.495)
    )
    assert latency_tail.availability_time_s == pytest.approx(0.5)
    assert max(
        sample.rate_hz
        for channel in run.bridge.descending.channels
        for sample in channel.samples
    ) > 0.0
    assert max(
        sample.rate_hz
        for channel in run.bridge.wing_motor.channels
        for sample in channel.rate_samples
    ) > 0.0
    assert np.all(np.isfinite(run.flight.position_world_m))
    assert run.flight.position_world_m[-1, 0] == pytest.approx(0.5)


def test_registered_fixture_artifact_and_replay_preserve_scope_without_retina(
    tmp_path,
    registered_pipeline_runs,
):
    run = registered_pipeline_runs["reduced"]
    manifest = write_nod1_flight_artifact(
        tmp_path / "run",
        run,
        chunk_samples=32,
        created_at_utc="2026-07-18T00:00:00Z",
    )

    pipeline = manifest["pipeline"]
    assert pipeline["pipeline_id"] == "frozen-browser-nod1-parity-to-dnp26-to-flight-v1"
    assert pipeline["mode"] == "open_loop_frozen_browser_visual_circuit_output"
    assert pipeline["visual_boundary"]["status"] == "frozen_browser_output_no_retinal_frames"
    assert "does not establish a calibrated visual model" in pipeline["visual_boundary"]["notice"]
    assert manifest["configuration"]["duration_s"] == pytest.approx(0.5)
    assert pipeline["source_metadata"]["stimulus"]["preset"] == "FG"
    assert pipeline["source_metadata"]["configuration"]["duration_s"] == pytest.approx(0.5)
    assert pipeline["source_metadata"]["circuit_inventory"]["cells"] == 1208
    assert pipeline["source_metadata"]["fixture_sha256"] == EXPECTED_SHA256
    assert not any(name.startswith("retinal_") for name in manifest["arrays"])
    assert not any(name in ("steering", "forward_looking_phase_T") for name in manifest["arrays"])
    assert manifest["arrays"]["circuit_voltage/720575940628438427"]["provenance"] == (
        "registered_frozen_chromium_browser_voltage_v"
    )

    replay = nod1_run_to_web_replay(run, target_sample_rate_hz=100.0)
    assert replay["id"] == "frozen_browser_nod1"
    assert replay["neural_model_scope"]["full_circuit_executed"] is True
    assert replay["neural_model_scope"]["circuit_cell_count"] == 1208
    assert replay["neural_model_scope"]["exported_circuit_channel_count"] == 4
    assert replay["pathway_channels"][0]["status"] == "unavailable"
    assert not any(
        channel["id"].startswith("retina:")
        for channel in replay["neural_trace_channels"]
    )


def test_fixture_cli_writes_artifact_and_replay_without_duration_override(
    tmp_path,
    capsys,
    monkeypatch,
    registered_pipeline_runs,
):
    import fly_sensor2behavior.pipeline as pipeline_module

    run = registered_pipeline_runs["reduced"]
    observed = {}

    def fake_pipeline(**kwargs):
        observed.update(kwargs)
        return run

    monkeypatch.setattr(
        pipeline_module,
        "run_registered_nod1_browser_flight_pipeline",
        fake_pipeline,
    )
    output = tmp_path / "cli-run"
    replay = tmp_path / "cli-replay.json"
    assert main(
        (
            "simulate-nod1-fixture",
            "--registry",
            str(tmp_path / "registry.json"),
            "--fixture",
            str(tmp_path / "fixture.json.gz"),
            "--output",
            str(output),
            "--physics-dt-s",
            "0.005",
            "--logging-dt-s",
            "0.005",
            "--neural-dt-s",
            "0.005",
            "--seed",
            "17",
            "--web-replay-output",
            str(replay),
        )
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pipeline_mode"] == "open_loop_frozen_browser_visual_circuit_output"
    assert summary["visual_boundary"] == "frozen_browser_output_no_retinal_frames"
    assert replay.is_file()
    artifact = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    replay_payload = json.loads(replay.read_text(encoding="utf-8"))
    assert artifact["web_replay_projection"]["sha256"] == (
        web_replay_projection_sha256(replay_payload)
    )
    assert observed["registry_path"] == tmp_path / "registry.json"
    assert observed["fixture_path"] == tmp_path / "fixture.json.gz"
    assert observed["config"].physics_dt_s == pytest.approx(0.005)
    assert not hasattr(observed["config"], "duration_s")

    parser = build_parser()
    parsed = parser.parse_args(
        ("simulate-nod1-fixture", "--output", str(tmp_path / "unused"))
    )
    assert parsed.command == "simulate-nod1-fixture"
    assert not hasattr(parsed, "duration_s")
