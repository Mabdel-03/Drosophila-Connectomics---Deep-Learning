"""Re-entrant process boundary for the captured fly-FGS circuit engine.

The captured JavaScript engine remains the numerical authority. This module
does not translate it into Python and does not load the source page's DN,
muscle, wing, yaw, or toy-body code. It launches the packaged JSON-RPC
sidecar, sends SI-valued scene/body observations, and validates every response
before exposing the four individual NOD1 voltages to downstream code.

The runtime is an exploratory circuit compatibility component. Its analytic
T4a input is normalized luminance, not calibrated compound-eye irradiance; its
raw application L/R labels are not anatomical laterality; and subprocess wall
time is never interpreted as physiological latency.
"""

from __future__ import annotations

import json
import math
import select
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .fly_fgs import (
    FLY_FGS_NOD1_ROOT_IDS,
    FLY_FGS_SOURCE_MANIFEST_SHA256,
    FLY_FGS_SNAPSHOT_ID,
    default_fly_fgs_source_manifest_path,
)


FLY_FGS_RUNTIME_PROTOCOL_VERSION = "1.0.0"
FLY_FGS_RUNTIME_CHECKPOINT_SCHEMA_VERSION = "1.0.0"
FLY_FGS_RUNTIME_SCRIPT_URI = "scripts/fly_fgs_runtime_rpc.mjs"
_EXPECTED_READY_FIELDS = {
    "event",
    "protocol_version",
    "node_version",
    "snapshot_id",
    "source_manifest_sha256",
    "circuit_engine_sha256",
    "circuit_bundle_sha256",
    "dt_s",
    "sample_count",
    "pre_roll_steps",
    "motor_input_policy",
    "full_cell_state_eligible_motor_input",
    "executable_asset_ids",
}
_EXPECTED_SAMPLE_FIELDS = {
    "sample_index",
    "measurement_time_s",
    "availability_time_s",
    "nod1_voltage_v",
    "pooled_readout",
    "last_control",
    "full_cell_state_eligible_motor_input",
}
_CONTROL_FIELDS = {
    "heading_rad",
    "heading_velocity_rad_s",
    "figure_world_azimuth_rad",
    "figure_velocity_rad_s",
    "ground_velocity_rad_s",
}


class FlyFGSRuntimeError(RuntimeError):
    """Raised when the circuit sidecar violates or rejects its contract."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FlyFGSRuntimeError("%s must be numeric" % label)
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise FlyFGSRuntimeError("%s must be numeric" % label) from exc
    if not math.isfinite(converted):
        raise FlyFGSRuntimeError("%s must be finite" % label)
    return converted


def _exact_fields(
    value: Any, expected: Sequence[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FlyFGSRuntimeError("%s must be an object" % label)
    if set(value) != set(expected):
        raise FlyFGSRuntimeError("%s fields do not match the protocol" % label)
    return value


def _finite_tree(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _finite_tree(child, "%s.%s" % (label, key))
            for key, child in value.items()
        }
    return _finite(value, label)


def default_fly_fgs_runtime_script_path() -> Path:
    """Locate the runtime sidecar in a checkout or an installed wheel."""

    source_tree = Path(__file__).resolve().parents[2] / FLY_FGS_RUNTIME_SCRIPT_URI
    if source_tree.is_file():
        return source_tree
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / "runtime"
        / "fly_fgs_runtime_rpc.mjs"
    )


@dataclass(frozen=True)
class FlyFGSSceneBodyInput:
    """One causal scene/body observation expressed in SI angular units."""

    heading_rad: float
    heading_velocity_rad_s: float
    figure_world_azimuth_rad: float
    figure_velocity_rad_s: float
    ground_velocity_rad_s: float = 0.0

    def __post_init__(self) -> None:
        for name in _CONTROL_FIELDS:
            object.__setattr__(self, name, _finite(getattr(self, name), name))

    def to_dict(self) -> Mapping[str, float]:
        return {
            "heading_rad": self.heading_rad,
            "heading_velocity_rad_s": self.heading_velocity_rad_s,
            "figure_world_azimuth_rad": self.figure_world_azimuth_rad,
            "figure_velocity_rad_s": self.figure_velocity_rad_s,
            "ground_velocity_rad_s": self.ground_velocity_rad_s,
        }


@dataclass(frozen=True)
class FlyFGSCircuitSample:
    """Validated state sample from the full circuit runtime.

    Only ``nod1_voltage_v`` is eligible to cross the motor boundary. The
    optional full-cell state and T4a luminance are display/audit state.
    """

    sample_index: int
    measurement_time_s: float
    availability_time_s: float
    nod1_voltage_v: Mapping[str, float]
    pooled_readout: Mapping[str, Any]
    last_control: FlyFGSSceneBodyInput
    retinal_input_luminance: Optional[Tuple[float, ...]] = None
    full_cell_voltage_v: Optional[Tuple[float, ...]] = None
    full_cell_activity: Optional[Tuple[float, ...]] = None
    full_cell_state_eligible_motor_input: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.sample_index, bool) or not isinstance(
            self.sample_index, int
        ):
            raise FlyFGSRuntimeError("sample_index must be an integer")
        if self.sample_index < 0:
            raise FlyFGSRuntimeError("sample_index must be non-negative")
        measurement = _finite(self.measurement_time_s, "measurement_time_s")
        availability = _finite(self.availability_time_s, "availability_time_s")
        if availability < measurement:
            raise FlyFGSRuntimeError(
                "circuit availability cannot precede measurement"
            )
        if tuple(self.nod1_voltage_v) != FLY_FGS_NOD1_ROOT_IDS:
            raise FlyFGSRuntimeError("NOD1 root inventory/order mismatch")
        for root_id, voltage in self.nod1_voltage_v.items():
            _finite(voltage, "NOD1 %s voltage" % root_id)
        if not isinstance(self.last_control, FlyFGSSceneBodyInput):
            raise FlyFGSRuntimeError("last_control must be FlyFGSSceneBodyInput")
        if self.full_cell_state_eligible_motor_input is not False:
            raise FlyFGSRuntimeError(
                "full circuit state must never be motor eligible"
            )
        for name in (
            "retinal_input_luminance",
            "full_cell_voltage_v",
            "full_cell_activity",
        ):
            values = getattr(self, name)
            if values is not None and any(
                not math.isfinite(float(item)) for item in values
            ):
                raise FlyFGSRuntimeError("%s contains a non-finite value" % name)


@dataclass(frozen=True)
class FlyFGSCircuitCheckpoint:
    """Opaque, hash-protected state suitable for fresh-process restore."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "schema_version",
            "protocol_version",
            "snapshot_id",
            "source_manifest_sha256",
            "circuit_engine_sha256",
            "circuit_bundle_sha256",
            "dt_s",
            "float_encoding",
            "dynamic_state",
            "payload_sha256",
        }
        _exact_fields(self.payload, expected, "checkpoint")
        if (
            self.payload["schema_version"]
            != FLY_FGS_RUNTIME_CHECKPOINT_SCHEMA_VERSION
        ):
            raise FlyFGSRuntimeError("checkpoint schema version mismatch")
        if self.payload["protocol_version"] != FLY_FGS_RUNTIME_PROTOCOL_VERSION:
            raise FlyFGSRuntimeError("checkpoint protocol version mismatch")
        if self.payload["snapshot_id"] != FLY_FGS_SNAPSHOT_ID:
            raise FlyFGSRuntimeError("checkpoint snapshot mismatch")
        digest = self.payload["payload_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise FlyFGSRuntimeError("checkpoint digest is invalid")

    def to_dict(self) -> Mapping[str, Any]:
        return json.loads(json.dumps(self.payload, allow_nan=False))


class NodeFlyFGSCircuitRuntime:
    """Manage one deterministic fly-FGS sidecar process."""

    def __init__(
        self,
        *,
        node_executable: Optional[str] = None,
        runtime_script: Optional[Path] = None,
        source_manifest: Optional[Path] = None,
        source_manifest_sha256: str = FLY_FGS_SOURCE_MANIFEST_SHA256,
        request_timeout_s: float = 120.0,
        expected_node_version: Optional[str] = None,
    ) -> None:
        executable = node_executable or shutil.which("node")
        if not executable:
            raise FlyFGSRuntimeError(
                "Node.js is required for the canonical fly-FGS runtime"
            )
        self.node_executable = str(executable)
        self.runtime_script = Path(
            runtime_script or default_fly_fgs_runtime_script_path()
        )
        self.source_manifest = Path(
            source_manifest or default_fly_fgs_source_manifest_path()
        )
        if not self.runtime_script.is_file():
            raise FlyFGSRuntimeError("fly-FGS runtime sidecar is unavailable")
        if not self.source_manifest.is_file():
            raise FlyFGSRuntimeError("fly-FGS source manifest is unavailable")
        if (
            not isinstance(source_manifest_sha256, str)
            or len(source_manifest_sha256) != 64
        ):
            raise FlyFGSRuntimeError(
                "source manifest SHA-256 must be a 64-character digest"
            )
        self.source_manifest_sha256 = source_manifest_sha256
        self.request_timeout_s = _finite(request_timeout_s, "request_timeout_s")
        if self.request_timeout_s <= 0.0:
            raise FlyFGSRuntimeError("request_timeout_s must be positive")
        self.expected_node_version = expected_node_version
        self._process: Optional[subprocess.Popen[str]] = None
        self._next_request_id = 1
        self._ready: Optional[Mapping[str, Any]] = None

    @property
    def ready_receipt(self) -> Mapping[str, Any]:
        if self._ready is None:
            raise FlyFGSRuntimeError("runtime process has not been started")
        return dict(self._ready)

    def _stderr_tail(self) -> str:
        process = self._process
        if process is None or process.stderr is None or process.poll() is None:
            return ""
        try:
            value = process.stderr.read()
        except OSError:
            return ""
        return value[-2000:].strip()

    def _read_line(self) -> str:
        process = self._process
        if process is None or process.stdout is None:
            raise FlyFGSRuntimeError("runtime process is unavailable")
        readable, _, _ = select.select(
            [process.stdout], [], [], self.request_timeout_s
        )
        if not readable:
            raise FlyFGSRuntimeError("fly-FGS runtime response timed out")
        line = process.stdout.readline()
        if line == "":
            detail = self._stderr_tail()
            raise FlyFGSRuntimeError(
                "fly-FGS runtime exited before responding%s"
                % ((": " + detail) if detail else "")
            )
        return line

    @staticmethod
    def _strict_json(line: str, label: str) -> Mapping[str, Any]:
        def reject_constant(value: str) -> None:
            raise ValueError("non-finite constant %s" % value)

        def reject_duplicates(
            pairs: Sequence[Tuple[str, Any]]
        ) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate key %s" % key)
                result[key] = value
            return result

        try:
            value = json.loads(
                line,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise FlyFGSRuntimeError("%s is not strict JSON" % label) from exc
        if not isinstance(value, Mapping):
            raise FlyFGSRuntimeError("%s must be a JSON object" % label)
        return value

    def start(self) -> "NodeFlyFGSCircuitRuntime":
        if self._process is not None:
            raise FlyFGSRuntimeError("runtime process is already started")
        command = [
            self.node_executable,
            str(self.runtime_script),
            "--manifest",
            str(self.source_manifest),
            "--manifest-sha256",
            self.source_manifest_sha256,
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
        )
        try:
            ready = self._strict_json(
                self._read_line(), "runtime ready receipt"
            )
            _exact_fields(ready, _EXPECTED_READY_FIELDS, "runtime ready receipt")
            if ready["event"] != "ready":
                raise FlyFGSRuntimeError("runtime did not emit a ready event")
            if ready["protocol_version"] != FLY_FGS_RUNTIME_PROTOCOL_VERSION:
                raise FlyFGSRuntimeError("runtime protocol version mismatch")
            if ready["snapshot_id"] != FLY_FGS_SNAPSHOT_ID:
                raise FlyFGSRuntimeError("runtime source snapshot mismatch")
            if ready["source_manifest_sha256"] != self.source_manifest_sha256:
                raise FlyFGSRuntimeError("runtime manifest receipt mismatch")
            if ready["dt_s"] != 0.005 or ready["sample_count"] != 100:
                raise FlyFGSRuntimeError("runtime registered clock mismatch")
            if ready["pre_roll_steps"] != 48:
                raise FlyFGSRuntimeError("runtime pre-roll mismatch")
            if ready["executable_asset_ids"] != [
                "circuit_engine",
                "circuit_bundle",
            ]:
                raise FlyFGSRuntimeError(
                    "runtime executable asset boundary mismatch"
                )
            if ready["full_cell_state_eligible_motor_input"] is not False:
                raise FlyFGSRuntimeError("runtime widened the motor-input boundary")
            if (
                self.expected_node_version is not None
                and ready["node_version"] != self.expected_node_version
            ):
                raise FlyFGSRuntimeError(
                    "Node.js version does not match the pinned worker"
                )
            self._ready = dict(ready)
            return self
        except Exception:
            self.close()
            raise

    def _request(
        self, method: str, params: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise FlyFGSRuntimeError("runtime process has not been started")
        request_id = self._next_request_id
        self._next_request_id += 1
        request = {"id": request_id, "method": method, "params": dict(params)}
        try:
            encoded = json.dumps(
                request,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            process.stdin.write(encoded + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise FlyFGSRuntimeError(
                "could not send request to fly-FGS runtime"
            ) from exc
        response = self._strict_json(self._read_line(), "runtime response")
        expected = (
            {"id", "ok", "result"}
            if response.get("ok") is True
            else {"id", "ok", "error"}
        )
        _exact_fields(response, expected, "runtime response")
        if response["id"] != request_id:
            raise FlyFGSRuntimeError("runtime response ID mismatch")
        if response["ok"] is not True:
            raise FlyFGSRuntimeError(
                "fly-FGS runtime rejected %s: %s"
                % (method, response.get("error"))
            )
        result = response["result"]
        if not isinstance(result, Mapping):
            raise FlyFGSRuntimeError("runtime result must be an object")
        return result

    @staticmethod
    def _sample_options(
        *, include_retinal_input: bool, include_full_cell_state: bool
    ) -> Mapping[str, bool]:
        if not isinstance(include_retinal_input, bool) or not isinstance(
            include_full_cell_state, bool
        ):
            raise FlyFGSRuntimeError("sample inclusion flags must be boolean")
        return {
            "include_retinal_input": include_retinal_input,
            "include_full_cell_state": include_full_cell_state,
        }

    @staticmethod
    def _parse_control(value: Any) -> FlyFGSSceneBodyInput:
        mapping = _exact_fields(value, _CONTROL_FIELDS, "last_control")
        return FlyFGSSceneBodyInput(**mapping)

    @classmethod
    def _parse_sample(
        cls,
        value: Mapping[str, Any],
        *,
        expected_retinal: bool,
        expected_full: bool,
    ) -> FlyFGSCircuitSample:
        expected = set(_EXPECTED_SAMPLE_FIELDS)
        if expected_retinal:
            expected.add("retinal_input_luminance")
        if expected_full:
            expected.update(("full_cell_voltage_v", "full_cell_activity"))
        mapping = _exact_fields(value, expected, "circuit sample")
        nod1_raw = mapping["nod1_voltage_v"]
        if not isinstance(nod1_raw, Mapping):
            raise FlyFGSRuntimeError("NOD1 voltage payload must be an object")
        nod1 = {
            str(key): _finite(item, "NOD1 voltage")
            for key, item in nod1_raw.items()
        }
        pooled = _finite_tree(mapping["pooled_readout"], "pooled_readout")
        retinal = None
        if expected_retinal:
            raw = mapping["retinal_input_luminance"]
            if not isinstance(raw, list) or len(raw) != 1441:
                raise FlyFGSRuntimeError(
                    "retinal T4a payload must have 1,441 values"
                )
            retinal = tuple(
                _finite(item, "retinal luminance") for item in raw
            )
        voltage = activity = None
        if expected_full:
            raw_voltage = mapping["full_cell_voltage_v"]
            raw_activity = mapping["full_cell_activity"]
            if (
                not isinstance(raw_voltage, list)
                or not isinstance(raw_activity, list)
                or len(raw_voltage) != 1684
                or len(raw_activity) != 1684
            ):
                raise FlyFGSRuntimeError(
                    "full-cell payload must have 1,684 values"
                )
            voltage = tuple(
                _finite(item, "full-cell voltage") for item in raw_voltage
            )
            activity = tuple(
                _finite(item, "full-cell activity") for item in raw_activity
            )
        return FlyFGSCircuitSample(
            sample_index=mapping["sample_index"],
            measurement_time_s=_finite(
                mapping["measurement_time_s"], "measurement_time_s"
            ),
            availability_time_s=_finite(
                mapping["availability_time_s"], "availability_time_s"
            ),
            nod1_voltage_v=nod1,
            pooled_readout=pooled,
            last_control=cls._parse_control(mapping["last_control"]),
            retinal_input_luminance=retinal,
            full_cell_voltage_v=voltage,
            full_cell_activity=activity,
            full_cell_state_eligible_motor_input=mapping[
                "full_cell_state_eligible_motor_input"
            ],
        )

    def initialize(
        self,
        *,
        include_retinal_input: bool = False,
        include_full_cell_state: bool = False,
    ) -> FlyFGSCircuitSample:
        options = self._sample_options(
            include_retinal_input=include_retinal_input,
            include_full_cell_state=include_full_cell_state,
        )
        result = self._request("initialize", {"sample_options": options})
        return self._parse_sample(
            result,
            expected_retinal=include_retinal_input,
            expected_full=include_full_cell_state,
        )

    def advance(
        self,
        control: FlyFGSSceneBodyInput,
        *,
        include_retinal_input: bool = False,
        include_full_cell_state: bool = False,
    ) -> FlyFGSCircuitSample:
        if not isinstance(control, FlyFGSSceneBodyInput):
            raise FlyFGSRuntimeError("control must be FlyFGSSceneBodyInput")
        options = self._sample_options(
            include_retinal_input=include_retinal_input,
            include_full_cell_state=include_full_cell_state,
        )
        result = self._request(
            "advance",
            {"control": control.to_dict(), "sample_options": options},
        )
        return self._parse_sample(
            result,
            expected_retinal=include_retinal_input,
            expected_full=include_full_cell_state,
        )

    def checkpoint(self) -> FlyFGSCircuitCheckpoint:
        return FlyFGSCircuitCheckpoint(self._request("checkpoint", {}))

    def restore(
        self,
        checkpoint: FlyFGSCircuitCheckpoint,
        *,
        include_retinal_input: bool = False,
        include_full_cell_state: bool = False,
    ) -> FlyFGSCircuitSample:
        if not isinstance(checkpoint, FlyFGSCircuitCheckpoint):
            raise FlyFGSRuntimeError(
                "checkpoint must be FlyFGSCircuitCheckpoint"
            )
        options = self._sample_options(
            include_retinal_input=include_retinal_input,
            include_full_cell_state=include_full_cell_state,
        )
        result = self._request(
            "restore",
            {"checkpoint": checkpoint.to_dict(), "sample_options": options},
        )
        return self._parse_sample(
            result,
            expected_retinal=include_retinal_input,
            expected_full=include_full_cell_state,
        )

    def finalize(self) -> Mapping[str, Any]:
        result = self._request("finalize", {})
        expected = {
            "final_sample_index",
            "final_measurement_time_s",
            "state_sha256",
        }
        _exact_fields(result, expected, "runtime finalization")
        if (
            isinstance(result["final_sample_index"], bool)
            or not isinstance(result["final_sample_index"], int)
            or result["final_sample_index"] < 0
        ):
            raise FlyFGSRuntimeError("final sample index is invalid")
        _finite(result["final_measurement_time_s"], "final_measurement_time_s")
        digest = result["state_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise FlyFGSRuntimeError("final state digest is invalid")
        return dict(result)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                self._request("shutdown", {})
            except FlyFGSRuntimeError:
                process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self._process = None
        self._ready = None

    def __enter__(self) -> "NodeFlyFGSCircuitRuntime":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "FLY_FGS_RUNTIME_CHECKPOINT_SCHEMA_VERSION",
    "FLY_FGS_RUNTIME_PROTOCOL_VERSION",
    "FlyFGSCircuitCheckpoint",
    "FlyFGSCircuitSample",
    "FlyFGSRuntimeError",
    "FlyFGSSceneBodyInput",
    "NodeFlyFGSCircuitRuntime",
    "default_fly_fgs_runtime_script_path",
]
