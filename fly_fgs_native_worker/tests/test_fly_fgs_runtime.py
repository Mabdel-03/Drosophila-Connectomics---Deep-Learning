from __future__ import annotations

import copy
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pytest

from fly_sensor2behavior.fly_fgs import (
    FLY_FGS_SOURCE_MANIFEST_SHA256,
    load_registered_fly_fgs_fixture,
)
from fly_sensor2behavior.fly_fgs_runtime import (
    FlyFGSCircuitCheckpoint,
    FlyFGSCircuitSample,
    FlyFGSRuntimeError,
    FlyFGSSceneBodyInput,
    NodeFlyFGSCircuitRuntime,
)


NODE_AVAILABLE = shutil.which("node") is not None


def _source_control(fixture, sample_index: int) -> FlyFGSSceneBodyInput:
    schedule = fixture.stimulus["schedule"]
    return FlyFGSSceneBodyInput(
        heading_rad=math.radians(schedule["heading_deg"][sample_index]),
        heading_velocity_rad_s=math.radians(
            schedule["heading_velocity_deg_s"][sample_index]
        ),
        figure_world_azimuth_rad=math.radians(
            schedule["figure_world_azimuth_deg"][sample_index]
        ),
        figure_velocity_rad_s=math.radians(
            schedule["figure_velocity_deg_s"][sample_index]
        ),
        ground_velocity_rad_s=math.radians(
            schedule["grating_velocity_deg_s"][sample_index]
        ),
    )


def _assert_sample_exact(sample, fixture, sample_index: int) -> None:
    assert sample.sample_index == sample_index
    assert sample.measurement_time_s == sample_index * fixture.dt_s
    assert sample.availability_time_s == sample.measurement_time_s
    assert sample.full_cell_state_eligible_motor_input is False
    assert sample.retinal_input_luminance == tuple(
        fixture.stimulus["retinal_input"]["luminance"][sample_index]
    )
    assert sample.full_cell_voltage_v == tuple(
        fixture.full_cell_state["voltage_v"][sample_index]
    )
    assert sample.full_cell_activity == tuple(
        fixture.full_cell_state["activity"][sample_index]
    )
    for signal in fixture.circuit_trace.signals:
        assert sample.nod1_voltage_v[signal.neuron.entity_id] == signal.values[
            sample_index
        ]

    pooled = fixture.pooled_readout_traces
    readout = sample.pooled_readout
    for trace_name, runtime_left, runtime_right in (
        ("nod1_activity", "nod1L", "nod1R"),
        ("vch_activity", "vchL", "vchR"),
        ("dch_activity", "dchL", "dchR"),
        ("t4_activity", "t4L", "t4R"),
        ("llpc1_activity", "llpcL", "llpcR"),
    ):
        assert readout[runtime_left] == pooled[trace_name]["raw_app_L"][
            sample_index
        ]
        assert readout[runtime_right] == pooled[trace_name]["raw_app_R"][
            sample_index
        ]
    for trace_name, runtime_left, runtime_right in (
        ("nod1_voltage_v", "nod1L", "nod1R"),
        ("vch_voltage_v", "vchL", "vchR"),
        ("dch_voltage_v", "dchL", "dchR"),
    ):
        assert readout["mv"][runtime_left] / 1000 == pooled[trace_name][
            "raw_app_L"
        ][sample_index]
        assert readout["mv"][runtime_right] / 1000 == pooled[trace_name][
            "raw_app_R"
        ][sample_index]


@dataclass(frozen=True)
class RuntimeRecord:
    samples: Tuple[FlyFGSCircuitSample, ...]
    checkpoint: FlyFGSCircuitCheckpoint
    finalization: dict
    node_version: str


@pytest.fixture(scope="module")
def runtime_record() -> RuntimeRecord:
    if not NODE_AVAILABLE:
        pytest.skip("Node.js is unavailable")
    fixture = load_registered_fly_fgs_fixture()
    samples = []
    checkpoint = None
    with NodeFlyFGSCircuitRuntime(request_timeout_s=180.0) as runtime:
        samples.append(
            runtime.initialize(
                include_retinal_input=True,
                include_full_cell_state=True,
            )
        )
        for sample_index in range(1, 100):
            samples.append(
                runtime.advance(
                    _source_control(fixture, sample_index),
                    include_retinal_input=True,
                    include_full_cell_state=True,
                )
            )
            if sample_index == 50:
                checkpoint = runtime.checkpoint()
        finalization = dict(runtime.finalize())
        node_version = str(runtime.ready_receipt["node_version"])
    assert checkpoint is not None
    return RuntimeRecord(
        samples=tuple(samples),
        checkpoint=checkpoint,
        finalization=finalization,
        node_version=node_version,
    )


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js is unavailable")
def test_incremental_runtime_exactly_reproduces_registered_full_capture(
    runtime_record,
) -> None:
    fixture = load_registered_fly_fgs_fixture()
    assert runtime_record.node_version.startswith("v")
    assert len(runtime_record.samples) == 100
    for sample_index, sample in enumerate(runtime_record.samples):
        _assert_sample_exact(sample, fixture, sample_index)
    assert runtime_record.finalization["final_sample_index"] == 99
    assert runtime_record.finalization["final_measurement_time_s"] == 0.495
    assert len(runtime_record.finalization["state_sha256"]) == 64


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js is unavailable")
def test_checkpoint_restores_exact_state_in_fresh_process(runtime_record) -> None:
    fixture = load_registered_fly_fgs_fixture()
    with NodeFlyFGSCircuitRuntime(request_timeout_s=180.0) as runtime:
        # The Python wrapper deliberately treats the payload as opaque. The
        # authoritative sidecar detects tampering before modifying its state.
        corrupted = copy.deepcopy(runtime_record.checkpoint.to_dict())
        corrupted["dynamic_state"]["sample_index"] = 49
        with pytest.raises(FlyFGSRuntimeError, match="payload SHA-256 mismatch"):
            runtime.restore(FlyFGSCircuitCheckpoint(corrupted))

        restored = runtime.restore(
            runtime_record.checkpoint,
            include_retinal_input=True,
            include_full_cell_state=True,
        )
        assert restored == runtime_record.samples[50]
        _assert_sample_exact(restored, fixture, 50)
        for sample_index in range(51, 100):
            resumed = runtime.advance(
                _source_control(fixture, sample_index),
                include_retinal_input=True,
                include_full_cell_state=True,
            )
            assert resumed == runtime_record.samples[sample_index]
        assert runtime.finalize() == runtime_record.finalization
        with pytest.raises(FlyFGSRuntimeError, match="sample interval is exhausted"):
            runtime.advance(_source_control(fixture, 99))


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js is unavailable")
def test_runtime_never_opens_receipt_only_page_or_wing_source(tmp_path) -> None:
    source = Path("data/reference/fly_fgs")
    target = tmp_path / "fly_fgs"
    assets = target / "assets"
    assets.mkdir(parents=True)
    shutil.copy2(source / "source_manifest.v1.json", target)
    shutil.copy2(source / "assets/fgmodel.snapshot.js", assets)
    shutil.copy2(source / "assets/bundle_fg.snapshot.json", assets)
    # index.snapshot.html and wing_dns.excluded.snapshot.json are absent. A
    # ready receipt proves the runtime loaded only manifest-eligible inputs.
    with NodeFlyFGSCircuitRuntime(
        source_manifest=target / "source_manifest.v1.json",
        source_manifest_sha256=FLY_FGS_SOURCE_MANIFEST_SHA256,
    ) as runtime:
        receipt = runtime.ready_receipt
        assert receipt["executable_asset_ids"] == [
            "circuit_engine",
            "circuit_bundle",
        ]
        assert receipt["motor_input_policy"] == (
            "four individual NOD1 voltage channels only"
        )


def test_scene_body_input_and_checkpoint_contracts_reject_invalid_values() -> None:
    with pytest.raises(FlyFGSRuntimeError, match="must be finite"):
        FlyFGSSceneBodyInput(
            heading_rad=float("nan"),
            heading_velocity_rad_s=0.0,
            figure_world_azimuth_rad=0.0,
            figure_velocity_rad_s=0.0,
        )
    with pytest.raises(FlyFGSRuntimeError, match="schema version mismatch"):
        FlyFGSCircuitCheckpoint(
            {
                "schema_version": "2.0.0",
                "protocol_version": "1.0.0",
                "snapshot_id": "fly-fgs-live-2026-07-18",
                "source_manifest_sha256": "0" * 64,
                "circuit_engine_sha256": "0" * 64,
                "circuit_bundle_sha256": "0" * 64,
                "dt_s": 0.005,
                "float_encoding": "float64_le_base64",
                "dynamic_state": {},
                "payload_sha256": "0" * 64,
            }
        )


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node.js is unavailable")
def test_scene_body_boundary_is_exactly_common_angle_and_velocity_gauge_invariant():
    """Only world-to-body relative motion may reach the captured visual scene."""

    first_control = FlyFGSSceneBodyInput(
        heading_rad=0.1,
        heading_velocity_rad_s=0.2,
        figure_world_azimuth_rad=0.3,
        figure_velocity_rad_s=0.4,
        ground_velocity_rad_s=0.1,
    )
    angle_shift = 1.7
    velocity_shift = 0.7
    gauge_shifted = FlyFGSSceneBodyInput(
        heading_rad=first_control.heading_rad + angle_shift,
        heading_velocity_rad_s=(
            first_control.heading_velocity_rad_s + velocity_shift
        ),
        figure_world_azimuth_rad=(
            first_control.figure_world_azimuth_rad + angle_shift
        ),
        figure_velocity_rad_s=first_control.figure_velocity_rad_s + velocity_shift,
        ground_velocity_rad_s=first_control.ground_velocity_rad_s + velocity_shift,
    )
    with NodeFlyFGSCircuitRuntime() as runtime:
        runtime.initialize(
            include_retinal_input=True,
            include_full_cell_state=True,
        )
        initial = runtime.checkpoint()
        first = runtime.advance(
            first_control,
            include_retinal_input=True,
            include_full_cell_state=True,
        )
        runtime.restore(initial)
        shifted = runtime.advance(
            gauge_shifted,
            include_retinal_input=True,
            include_full_cell_state=True,
        )

    assert shifted.nod1_voltage_v == first.nod1_voltage_v
    assert shifted.pooled_readout == first.pooled_readout
    assert shifted.retinal_input_luminance == first.retinal_input_luminance
    assert shifted.full_cell_voltage_v == first.full_cell_voltage_v
    assert shifted.full_cell_activity == first.full_cell_activity
