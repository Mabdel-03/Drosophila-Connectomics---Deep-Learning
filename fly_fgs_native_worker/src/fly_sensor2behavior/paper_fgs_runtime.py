"""Python process boundary for the 400 Hz paper-mode JavaScript circuit."""

from __future__ import annotations

import json
import math
import select
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .fly_fgs import FLY_FGS_NOD1_ROOT_IDS
from .fly_fgs_runtime import FlyFGSCircuitSample, FlyFGSSceneBodyInput


PAPER_FGS_RUNTIME_PROTOCOL_VERSION = "1.0.0"
PAPER_FGS_CIRCUIT_DT_S = 0.0025


class PaperFGSRuntimeError(RuntimeError):
    pass


def default_runtime_script_path() -> Path:
    checkout = Path(__file__).resolve().parents[2] / "scripts/paper_fgs_runtime_rpc.mjs"
    installed = (
        Path(sysconfig.get_path("data"))
        / "share/fly-sensor2behavior/runtime/paper_fgs_runtime_rpc.mjs"
    )
    for candidate in (checkout, installed):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("paper figure-ground runtime sidecar is unavailable")


def default_source_root() -> Path:
    checkout = Path(__file__).resolve().parents[3] / "fly_fgs_source"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "paper runtime needs the fly_fgs_source directory; pass source_root explicitly"
    )


def default_content_manifest_path() -> Path:
    checkout = (
        Path(__file__).resolve().parents[2]
        / "data/reference/paper_fgs/paper_runtime_manifest.v1.json"
    )
    installed = (
        Path(sysconfig.get_path("data"))
        / "share/fly-sensor2behavior/reference/paper_fgs/paper_runtime_manifest.v1.json"
    )
    for candidate in (checkout, installed):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("paper runtime content manifest is unavailable")


@dataclass(frozen=True)
class PaperFGSCircuitSample:
    sample_index: int
    measurement_time_s: float
    availability_time_s: float
    stimulus_time_s: float
    stimulus: Mapping[str, Any]
    nod1_voltage_v: Mapping[str, float]
    pooled_readout: Mapping[str, Any]
    retinal_input_luminance: Optional[Tuple[float, ...]] = None
    full_cell_voltage_v: Optional[Tuple[float, ...]] = None
    full_cell_activity: Optional[Tuple[float, ...]] = None
    full_cell_state_eligible_motor_input: bool = False
    circuit_dt_s: float = PAPER_FGS_CIRCUIT_DT_S

    def __post_init__(self) -> None:
        if isinstance(self.sample_index, bool) or self.sample_index < 0:
            raise PaperFGSRuntimeError("paper sample_index must be non-negative")
        if self.circuit_dt_s not in (0.0025, 0.00125):
            raise PaperFGSRuntimeError("paper circuit rate must be 400 or 800 Hz")
        expected = self.sample_index * self.circuit_dt_s
        if not math.isclose(self.measurement_time_s, expected, rel_tol=0.0, abs_tol=1e-12):
            raise PaperFGSRuntimeError("paper circuit measurement clock diverged")
        if self.availability_time_s != self.measurement_time_s:
            raise PaperFGSRuntimeError("paper circuit availability clock diverged")
        if tuple(self.nod1_voltage_v) != FLY_FGS_NOD1_ROOT_IDS:
            raise PaperFGSRuntimeError("paper NOD1 root inventory/order mismatch")
        if self.full_cell_state_eligible_motor_input is not False:
            raise PaperFGSRuntimeError("full circuit state cannot become motor input")

    def to_bridge_sample(self, bridge_sample_index: int) -> FlyFGSCircuitSample:
        """Downsample the 400/800 Hz state onto the 200 Hz motor boundary."""

        samples_per_boundary = int(round(0.005 / self.circuit_dt_s))
        expected_paper_index = bridge_sample_index * samples_per_boundary
        if self.sample_index != expected_paper_index:
            raise PaperFGSRuntimeError(
                "paper sample is not aligned to the 200 Hz motor boundary"
            )
        stimulus = self.stimulus
        radians = math.pi / 180.0
        return FlyFGSCircuitSample(
            sample_index=bridge_sample_index,
            measurement_time_s=bridge_sample_index * 0.005,
            availability_time_s=bridge_sample_index * 0.005,
            nod1_voltage_v=dict(self.nod1_voltage_v),
            pooled_readout=dict(self.pooled_readout),
            last_control=FlyFGSSceneBodyInput(
                heading_rad=0.0,
                heading_velocity_rad_s=0.0,
                figure_world_azimuth_rad=float(
                    stimulus["figure_angle_realized_deg"]
                )
                * radians,
                figure_velocity_rad_s=float(
                    stimulus["figure_velocity_command_deg_s"]
                )
                * radians,
                ground_velocity_rad_s=float(
                    stimulus["ground_velocity_command_deg_s"]
                )
                * radians,
            ),
            retinal_input_luminance=self.retinal_input_luminance,
            full_cell_voltage_v=self.full_cell_voltage_v,
            full_cell_activity=self.full_cell_activity,
            full_cell_state_eligible_motor_input=False,
        )


class NodePaperFGSCircuitRuntime:
    """Manage one deterministic paper-mode Node sidecar."""

    def __init__(
        self,
        *,
        source_root: Optional[Path] = None,
        runtime_script: Optional[Path] = None,
        node_executable: Optional[str] = None,
        request_timeout_s: float = 120.0,
        expected_node_version: Optional[str] = None,
        circuit_dt_s: float = PAPER_FGS_CIRCUIT_DT_S,
        content_manifest: Optional[Path] = None,
    ) -> None:
        executable = node_executable or shutil.which("node")
        if not executable:
            raise PaperFGSRuntimeError("Node.js is required for paper circuit execution")
        self.node_executable = str(executable)
        self.source_root = Path(source_root or default_source_root()).resolve()
        self.runtime_script = Path(runtime_script or default_runtime_script_path()).resolve()
        self.request_timeout_s = float(request_timeout_s)
        self.expected_node_version = expected_node_version
        self.circuit_dt_s = float(circuit_dt_s)
        if self.circuit_dt_s not in (0.0025, 0.00125):
            raise ValueError("circuit_dt_s must select 400 or 800 Hz")
        self.content_manifest = Path(
            content_manifest or default_content_manifest_path()
        ).resolve()
        self._process: Optional[subprocess.Popen[str]] = None
        self._ready: Optional[Mapping[str, Any]] = None
        self._next_request_id = 1
        self._last_sample: Optional[PaperFGSCircuitSample] = None

    @property
    def ready_receipt(self) -> Mapping[str, Any]:
        if self._ready is None:
            raise PaperFGSRuntimeError("paper runtime has not started")
        return json.loads(json.dumps(self._ready, allow_nan=False))

    def start(self) -> None:
        if self._process is not None:
            return
        if not self.source_root.is_dir() or not self.runtime_script.is_file():
            raise PaperFGSRuntimeError("paper runtime source assets are unavailable")
        self._process = subprocess.Popen(
            [
                self.node_executable,
                str(self.runtime_script),
                "--source-root",
                str(self.source_root),
                "--dt-s",
                str(self.circuit_dt_s),
                "--expected-manifest",
                str(self.content_manifest),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        line = self._readline()
        try:
            ready = json.loads(line)
        except json.JSONDecodeError as exc:
            self.close()
            raise PaperFGSRuntimeError("paper runtime emitted invalid readiness JSON") from exc
        if (
            ready.get("event") != "ready"
            or ready.get("protocol_version") != PAPER_FGS_RUNTIME_PROTOCOL_VERSION
            or ready.get("dt_s") != self.circuit_dt_s
            or tuple(ready.get("nod1_root_ids", ())) != FLY_FGS_NOD1_ROOT_IDS
            or not isinstance(ready.get("runtime_manifest"), Mapping)
        ):
            self.close()
            raise PaperFGSRuntimeError("paper runtime readiness contract mismatch")
        if self.expected_node_version and ready.get("node_version") != self.expected_node_version:
            self.close()
            raise PaperFGSRuntimeError("paper runtime Node version mismatch")
        self._ready = ready

    def _readline(self) -> str:
        if self._process is None or self._process.stdout is None:
            raise PaperFGSRuntimeError("paper runtime is not running")
        readable, _, _ = select.select(
            [self._process.stdout], [], [], self.request_timeout_s
        )
        if not readable:
            raise PaperFGSRuntimeError("paper runtime request timed out")
        line = self._process.stdout.readline()
        if not line:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            raise PaperFGSRuntimeError(
                "paper runtime exited unexpectedly: {}".format(stderr.strip())
            )
        return line

    def _request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        self.start()
        assert self._process is not None and self._process.stdin is not None
        request_id = self._next_request_id
        self._next_request_id += 1
        self._process.stdin.write(
            json.dumps(
                {"id": request_id, "method": method, "params": dict(params)},
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._process.stdin.flush()
        response = json.loads(self._readline())
        if response.get("id") != request_id:
            raise PaperFGSRuntimeError("paper runtime response ID mismatch")
        if "error" in response:
            raise PaperFGSRuntimeError(response["error"].get("message", "runtime error"))
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise PaperFGSRuntimeError("paper runtime result is invalid")
        return result

    def _parse_sample(self, result: Mapping[str, Any]) -> PaperFGSCircuitSample:
        def optional_tuple(name: str) -> Optional[Tuple[float, ...]]:
            value = result.get(name)
            return None if value is None else tuple(float(item) for item in value)

        return PaperFGSCircuitSample(
            sample_index=int(result["sample_index"]),
            measurement_time_s=float(result["measurement_time_s"]),
            availability_time_s=float(result["availability_time_s"]),
            stimulus_time_s=float(result["stimulus_time_s"]),
            stimulus=dict(result["stimulus"]),
            nod1_voltage_v={
                str(root): float(value)
                for root, value in result["nod1_voltage_v"].items()
            },
            pooled_readout=dict(result["pooled_readout"]),
            retinal_input_luminance=optional_tuple("retinal_input_luminance"),
            full_cell_voltage_v=optional_tuple("full_cell_voltage_v"),
            full_cell_activity=optional_tuple("full_cell_activity"),
            full_cell_state_eligible_motor_input=result[
                "full_cell_state_eligible_motor_input"
            ],
            circuit_dt_s=self.circuit_dt_s,
        )

    def initialize(
        self,
        protocol_id: str,
        *,
        texture_relationship_mode: str = "registered_copy",
        texture_seed: int = 123456,
        stimulus_time_s: float = -0.4,
        include_retinal_input: bool = False,
        include_full_cell_state: bool = False,
    ) -> PaperFGSCircuitSample:
        result = self._request(
            "initialize",
            {
                "protocol_id": protocol_id,
                "texture_relationship_mode": texture_relationship_mode,
                "texture_seed": int(texture_seed),
                "stimulus_time_s": float(stimulus_time_s),
                "sample_options": {
                    "include_retinal_input": bool(include_retinal_input),
                    "include_full_cell_state": bool(include_full_cell_state),
                },
            },
        )
        self._last_sample = self._parse_sample(result)
        return self._last_sample

    def advance(
        self,
        stimulus_time_s: float,
        *,
        include_retinal_input: bool = False,
        include_full_cell_state: bool = False,
    ) -> PaperFGSCircuitSample:
        if self._last_sample is None:
            raise PaperFGSRuntimeError("paper runtime must be initialized first")
        result = self._request(
            "advance",
            {
                "sample_index": self._last_sample.sample_index + 1,
                "stimulus_time_s": float(stimulus_time_s),
                "sample_options": {
                    "include_retinal_input": bool(include_retinal_input),
                    "include_full_cell_state": bool(include_full_cell_state),
                },
            },
        )
        self._last_sample = self._parse_sample(result)
        return self._last_sample

    def close(self) -> None:
        process = self._process
        self._process = None
        self._ready = None
        self._last_sample = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=2.0)
        except Exception:
            process.kill()
            process.wait(timeout=2.0)

    def __enter__(self) -> "NodePaperFGSCircuitRuntime":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "PAPER_FGS_CIRCUIT_DT_S",
    "PAPER_FGS_RUNTIME_PROTOCOL_VERSION",
    "NodePaperFGSCircuitRuntime",
    "PaperFGSCircuitSample",
    "PaperFGSRuntimeError",
    "default_content_manifest_path",
]
