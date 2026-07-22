#!/usr/bin/env python3
"""Fresh-process checkpoint probe for the streaming neural-to-wing stack.

The probe deliberately communicates only canonical JSON on stdin/stdout.  It
is used by tests and release evaluators to prove that continuation does not
depend on a Python object, process-local RNG, or hidden module state surviving
the checkpoint boundary.  The mechanics in this probe are the manufactured,
exploratory virtual-hinge model; this is a software re-entry check, not an
empirical flight-validation result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import platform
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Keep direct invocation from a source checkout hermetic; installation in the
# caller's site-packages is neither required nor relied upon.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from fly_sensor2behavior import fly_fgs_runtime
from fly_sensor2behavior.fly_fgs import (
    FLY_FGS_NOD1_RAW_APP_SIDES,
    FLY_FGS_NOD1_ROOT_IDS,
)
from fly_sensor2behavior.fly_fgs_runtime import (
    FlyFGSCircuitSample,
    FlyFGSSceneBodyInput,
)
from fly_sensor2behavior.flight import hinge, muscles, types
from fly_sensor2behavior.flight.streaming_bridge import (
    STREAMING_BRIDGE_RUNTIME_VERSION,
    STREAMING_BRIDGE_SCHEMA_VERSION,
    StreamingBridgeCheckpoint,
    StreamingBridgeConfig,
    StreamingNOD1MotorBridge,
)
from fly_sensor2behavior.flight.streaming_mechanics import (
    STREAMING_MECHANICS_RUNTIME_VERSION,
    STREAMING_MECHANICS_SCHEMA_VERSION,
    StreamingMechanicsCheckpoint,
    StreamingMuscleWingStepper,
)
from fly_sensor2behavior.flight import streaming_bridge, streaming_mechanics


PROBE_SCHEMA_VERSION = "1.0.0"
PROBE_KIND = "manufactured_streaming_software_reentry"
_BRIDGE_DT_S = 0.0005
_CIRCUIT_STRIDE = 10
_CIRCUIT_DT_S = 0.005
_MAX_INTERVALS = 1000


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_path(module: Any) -> Path:
    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError("cannot resolve module source for runtime receipt")
    return Path(source).resolve()


def _runtime_receipt() -> Mapping[str, Any]:
    probe_path = Path(__file__).resolve()
    modules = {
        "fly_fgs_runtime.py": _module_path(fly_fgs_runtime),
        "flight/hinge.py": _module_path(hinge),
        "flight/muscles.py": _module_path(muscles),
        "flight/streaming_bridge.py": _module_path(streaming_bridge),
        "flight/streaming_mechanics.py": _module_path(streaming_mechanics),
        "flight/types.py": _module_path(types),
        "scripts/probe_streaming_fresh_process.py": probe_path,
    }
    module_sha256 = {
        label: _sha256_file(path) for label, path in sorted(modules.items())
    }
    aggregate = hashlib.sha256(
        _canonical_json(module_sha256).encode("ascii")
    ).hexdigest()
    return {
        "probe_schema_version": PROBE_SCHEMA_VERSION,
        "probe_kind": PROBE_KIND,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_hexversion": sys.hexversion,
        "numpy_version": np.__version__,
        "bridge_schema_version": STREAMING_BRIDGE_SCHEMA_VERSION,
        "bridge_runtime_version": STREAMING_BRIDGE_RUNTIME_VERSION,
        "mechanics_schema_version": STREAMING_MECHANICS_SCHEMA_VERSION,
        "mechanics_runtime_version": STREAMING_MECHANICS_RUNTIME_VERSION,
        "module_sha256": module_sha256,
        "module_set_sha256": aggregate,
    }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(child) for child in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sample(sample_index: int) -> FlyFGSCircuitSample:
    # A deterministic bilateral waveform exercises source holds, encoder/VNC
    # delay queues, phase-coded seeded event generation, and the NMJ queue.
    phase = 0.61 * sample_index
    raw_l = -0.0560 + 0.0022 * math.sin(phase)
    raw_r = -0.0572 + 0.0018 * math.cos(phase + 0.37)
    voltages = {
        root_id: (
            raw_l
            if FLY_FGS_NOD1_RAW_APP_SIDES[root_id] == "L"
            else raw_r
        )
        for root_id in FLY_FGS_NOD1_ROOT_IDS
    }
    time_s = sample_index * _CIRCUIT_DT_S
    control = FlyFGSSceneBodyInput(
        heading_rad=0.0,
        heading_velocity_rad_s=0.0,
        figure_world_azimuth_rad=0.3,
        figure_velocity_rad_s=0.0,
        ground_velocity_rad_s=0.0,
    )
    return FlyFGSCircuitSample(
        sample_index=sample_index,
        measurement_time_s=time_s,
        availability_time_s=time_s,
        nod1_voltage_v=voltages,
        pooled_readout={},
        last_control=control,
        retinal_input_luminance=None,
        full_cell_voltage_v=None,
        full_cell_activity=None,
        full_cell_state_eligible_motor_input=False,
    )


def _parse_nonnegative_int(
    value: Any, label: str, *, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds the probe safety limit")
    return value


def _new_stack(seed: int) -> tuple[StreamingNOD1MotorBridge, StreamingMuscleWingStepper]:
    return (
        StreamingNOD1MotorBridge(config=StreamingBridgeConfig(), seed=seed),
        StreamingMuscleWingStepper(),
    )


def _restored_stack(
    seed: int,
    bridge_payload: Any,
    mechanics_payload: Any,
) -> tuple[StreamingNOD1MotorBridge, StreamingMuscleWingStepper]:
    bridge_checkpoint = StreamingBridgeCheckpoint(bridge_payload)
    mechanics_checkpoint = StreamingMechanicsCheckpoint(mechanics_payload)
    bridge = StreamingNOD1MotorBridge(config=StreamingBridgeConfig(), seed=seed)
    bridge.restore(bridge_checkpoint)
    mechanics_stepper = StreamingMuscleWingStepper.from_checkpoint(
        mechanics_checkpoint
    )
    if bridge.tick_index * 5 != mechanics_stepper.tick_index:
        raise ValueError("bridge/mechanics checkpoint clocks disagree")
    return bridge, mechanics_stepper


def _advance(
    bridge: StreamingNOD1MotorBridge,
    mechanics_stepper: StreamingMuscleWingStepper,
    interval_count: int,
) -> list[Mapping[str, Any]]:
    tail: list[Mapping[str, Any]] = []
    for _ in range(interval_count):
        bridge_tick = bridge.tick_index
        sample = None
        if bridge_tick % _CIRCUIT_STRIDE == 0:
            sample = _sample(bridge_tick // _CIRCUIT_STRIDE)
        interval = bridge.begin_interval(circuit_sample=sample)
        mechanics_frames = mechanics_stepper.advance_bridge_interval(interval)
        phase_path = (
            interval.wing_phase_start_unwrapped_rad,
            *(frame.phase_end_unwrapped_rad for frame in mechanics_frames),
        )
        bridge_frame = bridge.end_interval(phase_path)
        # A compact but information-rich transition receipt.  Its digest makes
        # exact continuation comparisons independent of object repr details.
        transition = {
            "bridge_tick": bridge_frame.tick_index,
            "start_s": bridge_frame.interval_start_s,
            "end_s": bridge_frame.interval_end_s,
            "phase_path_unwrapped_rad": list(
                bridge_frame.wing_phase_path_unwrapped_rad
            ),
            "generated_event_ids": [
                event.event_id for event in bridge_frame.generated_events
            ],
            "delivered_event_ids": [
                event.event_id for event in bridge_frame.delivered_events
            ],
            "mechanics": [
                {
                    "tick": frame.tick_index,
                    "applied_event_ids": list(frame.applied_event_ids),
                    "suppressed_event_ids": list(frame.suppressed_event_ids),
                    "phase_end_unwrapped_rad": frame.phase_end_unwrapped_rad,
                    "frequency_hz": frame.kinematic_frequency_hz,
                    "wing_kinematics": _jsonable(frame.wing_kinematics),
                }
                for frame in mechanics_frames
            ],
        }
        tail.append(
            {
                "bridge_tick": bridge_frame.tick_index,
                "transition_sha256": hashlib.sha256(
                    _canonical_json(transition).encode("ascii")
                ).hexdigest(),
                "transition": transition,
            }
        )
    return tail


def _handle(request: Any) -> Mapping[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("request must be a JSON object")
    allowed = {
        "operation",
        "seed",
        "interval_count",
        "bridge_checkpoint",
        "mechanics_checkpoint",
        "expected_runtime_receipt",
    }
    if set(request) - allowed:
        raise ValueError("request contains unknown fields")
    operation = request.get("operation")
    if operation not in ("run", "resume"):
        raise ValueError("operation must be run or resume")
    seed = _parse_nonnegative_int(request.get("seed", 0), "seed")
    interval_count = _parse_nonnegative_int(
        request.get("interval_count"), "interval_count", maximum=_MAX_INTERVALS
    )
    receipt = _runtime_receipt()
    expected_receipt = request.get("expected_runtime_receipt")
    if expected_receipt is not None and expected_receipt != receipt:
        raise ValueError("runtime receipt mismatch")

    if operation == "run":
        if "bridge_checkpoint" in request or "mechanics_checkpoint" in request:
            raise ValueError("run must not include checkpoints")
        bridge, mechanics_stepper = _new_stack(seed)
    else:
        if "bridge_checkpoint" not in request or "mechanics_checkpoint" not in request:
            raise ValueError("resume requires both checkpoints")
        bridge, mechanics_stepper = _restored_stack(
            seed,
            request["bridge_checkpoint"],
            request["mechanics_checkpoint"],
        )

    tail = _advance(bridge, mechanics_stepper, interval_count)
    bridge_checkpoint = bridge.checkpoint().to_dict()
    mechanics_checkpoint = mechanics_stepper.checkpoint().to_dict()
    return {
        "ok": True,
        "probe_schema_version": PROBE_SCHEMA_VERSION,
        "probe_kind": PROBE_KIND,
        "scientific_status": "software_only_manufactured_physics",
        "runtime_receipt": receipt,
        "start_operation": operation,
        "intervals_advanced": interval_count,
        "final_bridge_tick": bridge.tick_index,
        "final_mechanics_tick": mechanics_stepper.tick_index,
        "tail": tail,
        "tail_sha256": hashlib.sha256(
            _canonical_json(tail).encode("ascii")
        ).hexdigest(),
        "bridge_checkpoint": bridge_checkpoint,
        "mechanics_checkpoint": mechanics_checkpoint,
        "bridge_checkpoint_sha256": bridge_checkpoint["payload_sha256"],
        "mechanics_checkpoint_sha256": mechanics_checkpoint["payload_sha256"],
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
        response = _handle(request)
    except Exception as exc:  # The CLI boundary returns a machine-readable fail-stop.
        response = {
            "ok": False,
            "probe_schema_version": PROBE_SCHEMA_VERSION,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        sys.stdout.write(_canonical_json(response) + "\n")
        return 2
    sys.stdout.write(_canonical_json(response) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
