from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from fly_sensor2behavior.fly_fgs import (
    FLY_FGS_NOD1_RAW_APP_SIDES,
    FLY_FGS_NOD1_ROOT_IDS,
    FLY_FGS_REGISTERED_CAPTURE_SHA256,
    FLY_FGS_SOURCE_MANIFEST_SHA256,
    default_fly_fgs_registered_capture_path,
    default_fly_fgs_source_manifest_path,
    load_registered_fly_fgs_fixture,
    parse_fly_fgs_fixed_step_capture_bytes,
)
from fly_sensor2behavior.schema import (
    AnatomicalSide,
    ConfidenceLevel,
    EyeSide,
    RetinalSignalKind,
    SideMappingMethod,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SCRIPT = REPO_ROOT / "scripts" / "capture_fly_fgs_fixed_step.mjs"


@pytest.fixture(scope="module")
def registered_fixture():
    return load_registered_fly_fgs_fixture()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mutated_capture(payload: dict) -> tuple[bytes, str]:
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    return compressed, hashlib.sha256(compressed).hexdigest()


def test_registered_source_and_capture_are_content_addressed(registered_fixture) -> None:
    manifest_path = default_fly_fgs_source_manifest_path()
    capture_path = default_fly_fgs_registered_capture_path()
    assert _sha256(manifest_path) == FLY_FGS_SOURCE_MANIFEST_SHA256
    assert _sha256(capture_path) == FLY_FGS_REGISTERED_CAPTURE_SHA256
    assert manifest_path.stat().st_size == registered_fixture.source_manifest_bytes == 8209
    assert capture_path.stat().st_size == registered_fixture.capture_bytes == 380776

    manifest = json.loads(manifest_path.read_text())
    receipts = {item["asset_id"]: item for item in manifest["assets"]}
    assert {
        name for name, receipt in receipts.items() if receipt["eligible_circuit_input"]
    } == {"circuit_engine", "circuit_bundle"}
    assert receipts["page_snapshot"]["role"] == "source_receipt_only"
    assert (
        receipts["wing_dns_excluded"]["role"]
        == "excluded_downstream_source_receipt_only"
    )
    for receipt in receipts.values():
        asset_path = manifest_path.parent / receipt["path"]
        assert asset_path.stat().st_size == receipt["bytes"]
        assert _sha256(asset_path) == receipt["sha256"]


def test_exact_nod1_trace_side_boundary_and_event_filter(registered_fixture) -> None:
    fixture = registered_fixture
    assert fixture.dt_s == 0.005
    assert fixture.duration_s == 0.5
    assert fixture.circuit_trace.sample_times_s == tuple(
        index * 0.005 for index in range(100)
    )
    assert tuple(
        signal.neuron.entity_id for signal in fixture.circuit_trace.signals
    ) == FLY_FGS_NOD1_ROOT_IDS
    assert all(signal.unit == "V" for signal in fixture.circuit_trace.signals)
    assert all(
        max(signal.values) - min(signal.values) > 1e-12
        for signal in fixture.circuit_trace.signals
    )
    for signal in fixture.circuit_trace.signals:
        context = signal.neuron.side_context
        assert context is not None
        assert context.raw_dataset_side == FLY_FGS_NOD1_RAW_APP_SIDES[
            signal.neuron.entity_id
        ]
        assert context.anatomical_side is AnatomicalSide.UNKNOWN
        assert context.visual_field_side is AnatomicalSide.UNKNOWN
        assert context.effector_side is AnatomicalSide.UNKNOWN
        assert context.mapping_method is SideMappingMethod.SIMULATION_CONVENTION
        assert context.confidence.level is ConfidenceLevel.LOW

    inventory = fixture.circuit_inventory
    assert inventory["event_count"] == inventory["raw_event_count"] == 61789
    assert inventory["removed_llpc1_to_llpc1_event_count"] == 1589
    assert inventory["effective_runtime_event_count"] == 60200
    assert inventory["cell_type_counts"] == {
        "DCH": 2,
        "LLPC1": 219,
        "Nod1": 4,
        "T4a": 1457,
        "VCH": 2,
    }
    assert inventory["t5_cell_count"] == 0
    assert inventory["retinotopic_t4a_input_count"] == 1441
    observations = fixture.fixed_step["invariant_observations"]
    assert observations["t4_activity_excursion"] == pytest.approx(
        0.007244982389009451, abs=1e-15
    )
    assert observations["nod1_voltage_excursion_v"] == pytest.approx(
        0.024822540388854418, abs=1e-15
    )


def test_exact_t4a_retinal_input_is_exposed_as_causal_frames(
    registered_fixture,
) -> None:
    fixture = registered_fixture
    frames = fixture.retinal_frames
    retinal_input = fixture.stimulus["retinal_input"]
    assert len(frames) == 99
    assert len(retinal_input["cell_ids"]) == 1441
    assert len(retinal_input["luminance"]) == 100
    assert retinal_input["cell_ids"] == fixture.circuit_topology["retinotopic_t4a"][
        "cell_ids"
    ]
    assert retinal_input["cell_indices"] == fixture.circuit_topology[
        "retinotopic_t4a"
    ]["cell_indices"]

    for source_index, frame in enumerate(frames, start=1):
        time_s = fixture.circuit_trace.sample_times_s[source_index]
        assert frame.measurement_time_s == time_s
        assert frame.availability_time_s == time_s
        assert frame.exposure_start_s == fixture.circuit_trace.sample_times_s[
            source_index - 1
        ]
        assert frame.exposure_start_s == pytest.approx(time_s - fixture.dt_s)
        assert frame.exposure_end_s == time_s
        assert frame.eye_side is EyeSide.BINOCULAR
        assert frame.signal_kind is RetinalSignalKind.NORMALIZED_LUMINANCE
        assert frame.unit == "1"
        assert frame.samples == tuple(retinal_input["luminance"][source_index])
    assert frames[0].exposure_start_s == 0.0
    assert frames[-1].measurement_time_s == 0.495
    assert "pre-roll" in frames[0].provenance.filters["excluded_sample_0_reason"]
    assert "do not constitute calibrated biological ommatidia" in frames[0].provenance.notes

    for index in (0, 720, 1440):
        azimuth = math.radians(retinal_input["azimuth_deg"][index])
        elevation = math.radians(retinal_input["elevation_deg"][index])
        expected = (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        )
        assert frames[0].ommatidial_directions_body[index] == pytest.approx(
            expected, abs=1e-15
        )
        assert math.sqrt(sum(value * value for value in expected)) == pytest.approx(
            1.0, abs=1e-15
        )


def test_replay_attachment_is_bounded_json_and_contains_no_mechanics(
    registered_fixture,
) -> None:
    attachment = registered_fixture.circuit_replay_attachment()
    json.dumps(attachment, sort_keys=True, allow_nan=False)
    assert len(attachment["full_cell_state"]["voltage_v"]) == 100
    assert len(attachment["full_cell_state"]["voltage_v"][0]) == 1684
    assert len(attachment["full_cell_state"]["activity"]) == 100
    assert len(attachment["circuit_topology"]["cell_axis"]) == 1684
    assert attachment["full_cell_state_scope"]["eligible_motor_input"] is False
    assert attachment["retinal_frame_scope"]["source_sample_indices"] == list(
        range(1, 100)
    )
    assert attachment["retinal_frame_scope"]["sample_0_excluded"] is True
    assert attachment["asset_receipts"]["source_manifest"]["bytes"] == 8209
    assert attachment["asset_receipts"]["registered_capture"]["bytes"] == 380776

    forbidden_keys = {
        "SCALE_M",
        "MUSCLE_META",
        "FG_DN",
        "computeControl",
        "yawToRate",
        "muscleAct",
        "ampL",
        "ampR",
        "aoaL",
        "aoaR",
        "yaw",
        "wingR",
        "wingL",
    }

    def keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield key
                if key != "rejected_downstream_fields":
                    yield from keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from keys(nested)

    assert forbidden_keys.isdisjoint(set(keys(attachment)))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_node_capture_is_byte_deterministic_and_engine_only(tmp_path: Path) -> None:
    rerun = tmp_path / "capture.json"
    subprocess.run(
        ["node", str(CAPTURE_SCRIPT), "--output", str(rerun)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    registered_raw = gzip.decompress(
        default_fly_fgs_registered_capture_path().read_bytes()
    )
    assert rerun.read_bytes() == registered_raw
    payload = json.loads(registered_raw)
    assert set(payload) == {
        "schema_version",
        "fixture_kind",
        "snapshot_id",
        "source_receipt",
        "execution",
        "circuit_inventory",
        "stimulus",
        "nod1_voltage_v",
        "pooled_readout_traces",
        "rejected_downstream_fields",
        "full_cell_state",
    }


def test_parser_rejects_hash_tamper_and_dead_or_inconsistent_state() -> None:
    registered = default_fly_fgs_registered_capture_path().read_bytes()
    changed = bytearray(registered)
    changed[-8] ^= 1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_fly_fgs_fixed_step_capture_bytes(bytes(changed))

    dead_payload = json.loads(gzip.decompress(registered))
    dead_payload["pooled_readout_traces"]["t4_activity"]["raw_app_L"] = [
        0.0
    ] * 100
    dead_payload["pooled_readout_traces"]["t4_activity"]["raw_app_R"] = [
        0.0
    ] * 100
    dead_payload["execution"]["invariant_observations"][
        "t4_activity_excursion"
    ] = 0.0
    compressed, digest = _mutated_capture(dead_payload)
    with pytest.raises(ValueError, match="T4 activity trace is dead"):
        parse_fly_fgs_fixed_step_capture_bytes(
            compressed, expected_capture_sha256=digest
        )

    inconsistent_payload = json.loads(gzip.decompress(registered))
    root_id = FLY_FGS_NOD1_ROOT_IDS[0]
    cell_index = inconsistent_payload["full_cell_state"]["cell_ids"].index(
        "r%s" % root_id
    )
    inconsistent_payload["full_cell_state"]["voltage_v"][0][cell_index] += 1e-6
    compressed, digest = _mutated_capture(inconsistent_payload)
    with pytest.raises(ValueError, match="differs from full-cell state"):
        parse_fly_fgs_fixed_step_capture_bytes(
            compressed, expected_capture_sha256=digest
        )
