"""Resumable, disk-backed execution for the paper figure-ground assay.

One trial is integrated at a time.  Every returned native channel is copied to
an immutable-shape NumPy memmap in 4,096-sample blocks before the trial is
marked complete.  A completed trial is trusted only after its receipt and all
array slices reproduce the checksums in ``release_state.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .flight.paper_open_loop import PaperOpenLoopConfig, PaperOpenLoopSimulator
from .flybody_adapter import FlyBodyWorkerConfig
from .paper_fgs import (
    PAPER_FGS_PROTOCOL_IDS,
    _sha256_file,
    analyze_phase_locked_trials,
    compare_convergence,
    compare_to_paper_figure3,
    load_protocol,
    write_paper_artifact,
)
from .paper_fgs_runtime import NodePaperFGSCircuitRuntime
from .paper_tether import PAPER_TORQUE_METER_MODES, make_paper_torque_meter


PAPER_STREAMING_STATE_SCHEMA_VERSION = "paper_fgs_release_state.v1"
PAPER_STREAMING_ARRAY_SCHEMA_VERSION = "paper_fgs_memmap_store.v1"
PAPER_STREAMING_BLOCK_SAMPLES = 4096
PAPER_STREAMING_MAX_RSS_BYTES = 16 * 1024**3


class PaperStreamingError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    payload = json.dumps(
        _jsonable(value), allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_array_slice(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(array.shape)))
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()
    for start in range(0, array.shape[0], PAPER_STREAMING_BLOCK_SAMPLES):
        stop = min(start + PAPER_STREAMING_BLOCK_SAMPLES, array.shape[0])
        digest.update(np.ascontiguousarray(array[start:stop]).tobytes())
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.  The locked worker is Linux, but
    # retaining this distinction makes local unit tests unambiguous.
    return value * 1024 if sys.platform.startswith("linux") else value


def _worker_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_hashes(source_root: Path) -> Mapping[str, str]:
    required = (
        "lib/fgmodel.js",
        "lib/paper_stimulus.js",
        "data/bundle_fg.json",
        "data/paper_protocols.json",
        "data/nod1_laterality.v1.json",
        "data/wing_dns.json",
    )
    output = {}
    for relative in required:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError("paper browser runtime input is missing: {}".format(path))
        output[relative] = _sha256_file(path)
    return output


def build_run_identity(
    config: PaperOpenLoopConfig,
    *,
    source_root: Path,
    worker_manifest: Optional[Path] = None,
) -> Mapping[str, Any]:
    worker = _worker_root()
    reference_root = worker / "reference"
    manifest_path = worker_manifest or worker / "VENDORED_SOURCE_MANIFEST.json"
    required_files = {
        "worker_manifest": manifest_path,
        "requirements_lock": worker / "requirements.lock",
        "paper_pdf": reference_root / "BF00595226.pdf",
        "paper_specification": (
            reference_root / "figure_ground_relative_motion_simulation_spec.md"
        ),
        "protocols": worker / "data/reference/paper_fgs/paper_protocols.v1.json",
        "release_matrix": (
            worker / "data/reference/paper_fgs/scientific_release_matrix.v1.json"
        ),
        "figure3_reference": (
            worker / "data/reference/paper_fgs/figure3_reference.v1.json"
        ),
        "runtime_manifest": (
            worker / "data/reference/paper_fgs/paper_runtime_manifest.v1.json"
        ),
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("locked run inputs are missing: {}".format(", ".join(missing)))
    identity = {
        "schema_version": "paper_fgs_run_identity.v1",
        "config": _jsonable(asdict(config)),
        "worker_files": {
            name: _sha256_file(path) for name, path in required_files.items()
        },
        "browser_runtime_files": dict(_source_hashes(source_root.resolve())),
        "expected_reference_hashes": {
            "paper_pdf": "9159ec24548e1ee0eaf4edca0d4790d7656890d2fb611dd0b7d27cc85a6f05d7",
            "paper_specification": "c43b9ec417a52d5fac5333c1df01ca0765961f5c88779a3769fc9b3a2968647c",
        },
    }
    for name, expected in identity["expected_reference_hashes"].items():
        if identity["worker_files"][name] != expected:
            raise PaperStreamingError("locked {} checksum mismatch".format(name))
    return identity


class StreamingPaperRunStore:
    """Atomic journal plus preallocated arrays for one matrix run."""

    def __init__(
        self,
        root: Path,
        *,
        repetitions: int,
        identity: Mapping[str, Any],
        resume: bool,
    ) -> None:
        self.root = Path(root).resolve()
        self.arrays_root = self.root / "arrays"
        self.receipts_root = self.root / "trial_receipts"
        self.state_path = self.root / "release_state.json"
        self.specs_path = self.root / "array_specs.json"
        self.repetitions = int(repetitions)
        self.identity = _jsonable(identity)
        self.identity_sha256 = _sha256_bytes(_canonical_json_bytes(self.identity))
        if self.root.exists() and not resume:
            raise FileExistsError(
                "streaming run already exists; pass --resume after verification: {}".format(
                    self.root
                )
            )
        if self.root.exists():
            if not self.state_path.is_file():
                raise PaperStreamingError("existing run lacks release_state.json")
            self.state = json.loads(self.state_path.read_text("utf-8"))
            self._validate_state_identity()
            self._verify_completed_trials()
        else:
            self.arrays_root.mkdir(parents=True)
            self.receipts_root.mkdir(parents=True)
            self.state = {
                "schema_version": PAPER_STREAMING_STATE_SCHEMA_VERSION,
                "identity_sha256": self.identity_sha256,
                "identity": self.identity,
                "repetitions": self.repetitions,
                "block_samples": PAPER_STREAMING_BLOCK_SAMPLES,
                "status": "running",
                "completed_trials": {},
                "in_progress_trial_id": None,
                "peak_rss_bytes": _peak_rss_bytes(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _atomic_write_json(self.state_path, self.state)

    def _validate_state_identity(self) -> None:
        if self.state.get("schema_version") != PAPER_STREAMING_STATE_SCHEMA_VERSION:
            raise PaperStreamingError("release state schema mismatch")
        if self.state.get("identity_sha256") != self.identity_sha256:
            raise PaperStreamingError(
                "resume identity mismatch; start a new immutable run directory"
            )
        if int(self.state.get("repetitions", -1)) != self.repetitions:
            raise PaperStreamingError("resume repetition count mismatch")

    def _load_specs(self) -> Mapping[str, Any]:
        if not self.specs_path.is_file():
            return {}
        payload = json.loads(self.specs_path.read_text("utf-8"))
        if payload.get("schema_version") != PAPER_STREAMING_ARRAY_SCHEMA_VERSION:
            raise PaperStreamingError("array store schema mismatch")
        if int(payload.get("repetitions", -1)) != self.repetitions:
            raise PaperStreamingError("array store repetition count mismatch")
        return payload.get("arrays", {})

    def _initialize_arrays(self, trial: Mapping[str, np.ndarray]) -> None:
        if self.specs_path.exists():
            return
        specs = {}
        for name in sorted(trial):
            value = np.asarray(trial[name])
            if value.dtype.kind not in "biufc":
                raise PaperStreamingError("unsupported trial dtype for {}".format(name))
            shape = (self.repetitions,) + value.shape
            filename = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16] + ".npy"
            path = self.arrays_root / filename
            memmap = np.lib.format.open_memmap(
                path, mode="w+", dtype=value.dtype, shape=shape
            )
            memmap.flush()
            del memmap
            specs[name] = {
                "file": filename,
                "dtype": str(value.dtype),
                "trial_shape": list(value.shape),
                "store_shape": list(shape),
            }
        _atomic_write_json(
            self.specs_path,
            {
                "schema_version": PAPER_STREAMING_ARRAY_SCHEMA_VERSION,
                "repetitions": self.repetitions,
                "arrays": specs,
            },
        )

    def _validate_trial_against_specs(
        self, trial: Mapping[str, np.ndarray], specs: Mapping[str, Any]
    ) -> None:
        if set(trial) != set(specs):
            raise PaperStreamingError("trial channel set changed during the run")
        for name, spec in specs.items():
            value = np.asarray(trial[name])
            if str(value.dtype) != spec["dtype"] or list(value.shape) != spec["trial_shape"]:
                raise PaperStreamingError("trial array contract changed for {}".format(name))

    def _write_array(
        self, trial_id: int, name: str, value: np.ndarray, spec: Mapping[str, Any]
    ) -> str:
        destination = np.load(self.arrays_root / spec["file"], mmap_mode="r+")
        source = np.asarray(value)
        if source.ndim == 0:
            destination[trial_id] = source
            destination.flush()
        else:
            for start in range(0, source.shape[0], PAPER_STREAMING_BLOCK_SAMPLES):
                stop = min(start + PAPER_STREAMING_BLOCK_SAMPLES, source.shape[0])
                destination[trial_id, start:stop] = source[start:stop]
                destination.flush()
        digest = _sha256_array_slice(destination[trial_id])
        del destination
        file_descriptor = os.open(self.arrays_root / spec["file"], os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        return digest

    def begin_trial(self, trial_id: int) -> None:
        if str(trial_id) in self.state["completed_trials"]:
            raise PaperStreamingError("completed trials are immutable")
        self.state["in_progress_trial_id"] = trial_id
        self.state["status"] = "running"
        _atomic_write_json(self.state_path, self.state)

    def commit_trial(
        self,
        trial_id: int,
        trial: Mapping[str, np.ndarray],
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.state.get("in_progress_trial_id") != trial_id:
            raise PaperStreamingError("trial journal is not in the expected state")
        self._initialize_arrays(trial)
        specs = self._load_specs()
        self._validate_trial_against_specs(trial, specs)
        array_sha256 = {
            name: self._write_array(trial_id, name, np.asarray(trial[name]), spec)
            for name, spec in sorted(specs.items())
        }
        receipt_path = self.receipts_root / "trial_{:03d}.json".format(trial_id)
        _atomic_write_json(receipt_path, receipt)
        receipt_sha256 = _sha256_file(receipt_path)
        trial_digest = _sha256_bytes(
            _canonical_json_bytes(
                {"arrays": array_sha256, "receipt_sha256": receipt_sha256}
            )
        )
        record = {
            "trial_id": trial_id,
            "trial_sha256": trial_digest,
            "receipt": receipt_path.name,
            "receipt_sha256": receipt_sha256,
            "arrays": array_sha256,
        }
        self.state["completed_trials"][str(trial_id)] = record
        self.state["in_progress_trial_id"] = None
        self.state["peak_rss_bytes"] = max(
            int(self.state.get("peak_rss_bytes", 0)), _peak_rss_bytes()
        )
        _atomic_write_json(self.state_path, self.state)
        return record

    def _verify_completed_trials(self) -> None:
        specs = self._load_specs()
        completed = self.state.get("completed_trials", {})
        if completed and not specs:
            raise PaperStreamingError("completed trials exist without array specifications")
        arrays = {
            name: np.load(self.arrays_root / spec["file"], mmap_mode="r")
            for name, spec in specs.items()
        }
        for trial_key, record in sorted(completed.items(), key=lambda item: int(item[0])):
            trial_id = int(trial_key)
            receipt_path = self.receipts_root / record["receipt"]
            if not receipt_path.is_file() or _sha256_file(receipt_path) != record["receipt_sha256"]:
                raise PaperStreamingError("completed trial receipt checksum mismatch")
            observed_arrays = {
                name: _sha256_array_slice(array[trial_id])
                for name, array in arrays.items()
            }
            if observed_arrays != record["arrays"]:
                raise PaperStreamingError("completed trial array checksum mismatch")
            observed_trial = _sha256_bytes(
                _canonical_json_bytes(
                    {
                        "arrays": observed_arrays,
                        "receipt_sha256": record["receipt_sha256"],
                    }
                )
            )
            if observed_trial != record["trial_sha256"]:
                raise PaperStreamingError("completed trial aggregate checksum mismatch")

    @property
    def completed_trial_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(int(item) for item in self.state["completed_trials"]))

    def trial_views(self) -> "MemmapTrialSequence":
        if len(self.completed_trial_ids) != self.repetitions:
            raise PaperStreamingError("cannot finalize an incomplete run")
        return MemmapTrialSequence(self.root, self.repetitions)

    def receipts(self) -> Sequence[Mapping[str, Any]]:
        output = []
        for trial_id in range(self.repetitions):
            record = self.state["completed_trials"][str(trial_id)]
            output.append(
                json.loads((self.receipts_root / record["receipt"]).read_text("utf-8"))
            )
        return output

    def mark_finalized(self, manifest_path: Path) -> None:
        self.state["status"] = "complete"
        self.state["artifact_manifest"] = str(Path(manifest_path).resolve())
        self.state["artifact_manifest_sha256"] = _sha256_file(manifest_path)
        self.state["completed_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        _atomic_write_json(self.state_path, self.state)


class MemmapTrialSequence:
    """Read-only trial mappings backed by one memmap per channel."""

    def __init__(self, root: Path, repetitions: int) -> None:
        payload = json.loads((Path(root) / "array_specs.json").read_text("utf-8"))
        self.root = Path(root)
        self.repetitions = int(repetitions)
        self.specs = payload["arrays"]
        self.arrays = {
            name: np.load(self.root / "arrays" / spec["file"], mmap_mode="r")
            for name, spec in self.specs.items()
        }

    def __len__(self) -> int:
        return self.repetitions

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self.repetitions))]
        if index < 0:
            index += self.repetitions
        if not 0 <= index < self.repetitions:
            raise IndexError(index)
        return {name: values[index] for name, values in self.arrays.items()}

    def stacked_array(self, name: str) -> np.ndarray:
        return self.arrays[name]


def _mechanical_analysis(
    protocol: Any,
    arrays: Mapping[str, np.ndarray],
    analysis: Dict[str, Any],
) -> None:
    sources = {
        "authoritative_total": "reported_yaw_torque_Nm",
        "left_wing": "wing_reported_yaw_torque_Nm",
        "right_wing": "wing_reported_yaw_torque_Nm",
        "wing_sum": "wing_sum_reported_yaw_torque_Nm",
        "nonwing_residual": "nonwing_reported_yaw_torque_residual_Nm",
        "left_wing_aerodynamic": "wing_aerodynamic_reported_yaw_torque_Nm",
        "right_wing_aerodynamic": "wing_aerodynamic_reported_yaw_torque_Nm",
    }
    templates = {
        "authoritative_total": "torque_product_{}_Nm",
        "left_wing": "left_wing_torque_product_{}_Nm",
        "right_wing": "right_wing_torque_product_{}_Nm",
        "wing_sum": "wing_sum_torque_product_{}_Nm",
        "nonwing_residual": "nonwing_residual_torque_product_{}_Nm",
        "left_wing_aerodynamic": "left_wing_aerodynamic_torque_product_{}_Nm",
        "right_wing_aerodynamic": "right_wing_aerodynamic_torque_product_{}_Nm",
    }
    output = {}
    for channel, array_name in sources.items():
        if array_name not in arrays:
            continue
        values = arrays[array_name]
        if channel in ("left_wing", "left_wing_aerodynamic"):
            values = values[:, :, 0]
        elif channel in ("right_wing", "right_wing_aerodynamic"):
            values = values[:, :, 1]
        if not np.all(np.isfinite(values)):
            continue
        channel_analysis = dict(analyze_phase_locked_trials(protocol, arrays["time_s"], values))
        channel_analysis["signal_products"] = {}
        for product in (
            "paper_comparison",
            "lowpass_10hz",
            "lowpass_25hz",
            "lowpass_50hz",
            "wingbeat_averaged",
        ):
            product_name = templates[channel].format(product)
            if product_name in arrays and np.all(np.isfinite(arrays[product_name])):
                channel_analysis["signal_products"][product] = dict(
                    analyze_phase_locked_trials(
                        protocol, arrays["time_s"], arrays[product_name]
                    )
                )
        output[channel] = channel_analysis
    analysis["mechanical_channels"] = output


def finalize_streamed_run(
    simulator: PaperOpenLoopSimulator,
    store: StreamingPaperRunStore,
    *,
    output: Path,
    convergence_references: Sequence[Path],
    web_replay_output: Optional[Path] = None,
) -> Mapping[str, Any]:
    protocol = simulator.config.protocol
    arrays, metadata = simulator.finalize_trials(store.trial_views(), store.receipts())
    analysis: Dict[str, Any] = dict(
        analyze_phase_locked_trials(
            protocol, arrays["time_s"], arrays["reported_yaw_torque_Nm"]
        )
    )
    _mechanical_analysis(protocol, arrays, analysis)
    products = (
        "paper_comparison",
        "lowpass_10hz",
        "lowpass_25hz",
        "lowpass_50hz",
        "wingbeat_averaged",
    )
    analysis["paper_figure3_comparisons"] = {
        product: compare_to_paper_figure3(
            protocol,
            arrays["time_s"],
            arrays["torque_product_{}_Nm".format(product)],
            signal_product=product,
        )
        for product in products
    }
    convergence_checks = []
    for supplied in convergence_references:
        reference_path = Path(supplied)
        if reference_path.is_dir():
            if (reference_path / "artifact/analysis.json").is_file():
                reference_path = reference_path / "artifact/analysis.json"
            else:
                reference_path = reference_path / "analysis.json"
        reference = json.loads(reference_path.read_text("utf-8"))
        total = compare_convergence(reference, analysis)
        channels = {}
        for channel in ("left_wing", "right_wing", "wing_sum"):
            reference_channel = reference.get("mechanical_channels", {}).get(channel)
            candidate = analysis["mechanical_channels"].get(channel)
            channels[channel] = (
                {"available": False, "passed": False}
                if reference_channel is None or candidate is None
                else {"available": True, **compare_convergence(reference_channel, candidate)}
            )
        comparison = {
            "schema_version": "paper_multichannel_convergence.v3",
            "authoritative_total": total,
            "mechanical_channels": channels,
            "passed": bool(total["passed"] and all(item["passed"] for item in channels.values())),
        }
        if not comparison["passed"]:
            raise PaperStreamingError("paper numerical convergence thresholds failed")
        convergence_checks.append(
            {"reference": str(reference_path.resolve()), "comparison": comparison}
        )
    if convergence_checks:
        analysis["convergence_against_reference"] = {
            "schema_version": "paper_convergence_collection.v1",
            "comparisons": convergence_checks,
            "passed": all(item["comparison"]["passed"] for item in convergence_checks),
        }
    calibration_passed = (metadata.get("torque_calibration") or {}).get("passed") is True
    convergence_passed = analysis.get("convergence_against_reference", {}).get("passed") is True
    manifest = write_paper_artifact(
        output,
        protocol,
        arrays,
        analysis,
        run_metadata=metadata,
        authoritative=(
            simulator.config.torque_meter_mode == "comparison"
            and calibration_passed
            and convergence_passed
        ),
        web_replay_output=web_replay_output,
    )
    store.mark_finalized(Path(output) / "manifest.json")
    return manifest


def run_streamed_simulation(
    simulator: PaperOpenLoopSimulator,
    *,
    run_root: Path,
    identity: Mapping[str, Any],
    resume: bool,
    convergence_references: Sequence[Path] = (),
    web_replay_output: Optional[Path] = None,
) -> Mapping[str, Any]:
    store = StreamingPaperRunStore(
        run_root,
        repetitions=simulator.config.repetitions,
        identity=identity,
        resume=resume,
    )
    artifact = Path(run_root) / "artifact"
    if artifact.is_dir():
        manifest_path = artifact / "manifest.json"
        if not manifest_path.is_file():
            raise PaperStreamingError("artifact directory is incomplete")
        expected = store.state.get("artifact_manifest_sha256")
        if expected is not None and _sha256_file(manifest_path) != expected:
            raise PaperStreamingError("final artifact manifest checksum mismatch")
        return json.loads(manifest_path.read_text("utf-8"))
    completed = set(store.completed_trial_ids)
    for trial_id in range(simulator.config.repetitions):
        if trial_id in completed:
            continue
        store.begin_trial(trial_id)
        trial, receipt = simulator.run_trial(trial_id)
        store.commit_trial(trial_id, trial, receipt)
        if int(store.state["peak_rss_bytes"]) > PAPER_STREAMING_MAX_RSS_BYTES:
            raise PaperStreamingError("peak RSS exceeded the 16 GiB release limit")
    return finalize_streamed_run(
        simulator,
        store,
        output=artifact,
        convergence_references=convergence_references,
        web_replay_output=web_replay_output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fly-s2b-paper-fgs-streaming")
    parser.add_argument("--protocol", choices=PAPER_FGS_PROTOCOL_IDS, default="R83_Fig3a_0_to_90")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=73)
    parser.add_argument("--circuit-rate-hz", type=int, choices=(400, 800), default=400)
    parser.add_argument("--physics-rate-hz", type=int, choices=(10000, 20000), default=10000)
    parser.add_argument("--torque-meter", choices=PAPER_TORQUE_METER_MODES, default="fixed-load-cell")
    parser.add_argument(
        "--wingbeat-phase-mode",
        choices=("uniformly_stratified", "fixed"),
        default="uniformly_stratified",
    )
    parser.add_argument(
        "--texture-mode",
        choices=("registered_copy", "independent_matched_statistics"),
        default="registered_copy",
    )
    parser.add_argument("--texture-seed", type=int, default=123456)
    parser.add_argument("--output", type=Path, required=True, help="resumable run directory")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-node-version", default="v22.22.1")
    parser.add_argument("--worker-manifest", type=Path, default=None)
    parser.add_argument("--convergence-reference", type=Path, action="append", default=[])
    parser.add_argument("--web-replay-output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.status:
        state_path = arguments.output.resolve() / "release_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        print(state_path.read_text("utf-8"), end="")
        return 0
    protocol = load_protocol(arguments.protocol)
    repetitions = protocol.repetitions if arguments.repetitions is None else arguments.repetitions
    config = PaperOpenLoopConfig(
        protocol=protocol,
        repetitions=repetitions,
        run_seed=arguments.run_seed,
        texture_relationship_mode=arguments.texture_mode,
        texture_seed=arguments.texture_seed,
        circuit_rate_hz=arguments.circuit_rate_hz,
        physics_rate_hz=arguments.physics_rate_hz,
        torque_meter_mode=arguments.torque_meter,
        wingbeat_phase_mode=arguments.wingbeat_phase_mode,
    )
    identity = build_run_identity(
        config,
        source_root=arguments.source_root.resolve(),
        worker_manifest=arguments.worker_manifest,
    )
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "identity_sha256": _sha256_bytes(_canonical_json_bytes(identity)),
                    "output": str(arguments.output.resolve()),
                    "repetitions": repetitions,
                    "raw_rate_hz": arguments.physics_rate_hz * 4,
                    "estimated_minimum_external_storage_bytes": 250 * 1024**3,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    worker = FlyBodyWorkerConfig(timestep_s=1.0 / arguments.physics_rate_hz)
    physics = make_paper_torque_meter(arguments.torque_meter, config=worker)

    def circuit_factory() -> NodePaperFGSCircuitRuntime:
        return NodePaperFGSCircuitRuntime(
            source_root=arguments.source_root.resolve(),
            expected_node_version=arguments.expected_node_version,
            circuit_dt_s=1.0 / arguments.circuit_rate_hz,
        )

    simulator = PaperOpenLoopSimulator(
        config, physics_adapter=physics, circuit_factory=circuit_factory
    )
    manifest = run_streamed_simulation(
        simulator,
        run_root=arguments.output.resolve(),
        identity=identity,
        resume=arguments.resume,
        convergence_references=arguments.convergence_reference,
        web_replay_output=arguments.web_replay_output,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "run_id": manifest["run_id"],
                "scientific_status": manifest["scientific_status"],
                "protocol_id": protocol.protocol_id,
                "repetitions": repetitions,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MemmapTrialSequence",
    "PAPER_STREAMING_ARRAY_SCHEMA_VERSION",
    "PAPER_STREAMING_BLOCK_SAMPLES",
    "PAPER_STREAMING_STATE_SCHEMA_VERSION",
    "PaperStreamingError",
    "StreamingPaperRunStore",
    "build_run_identity",
    "finalize_streamed_run",
    "run_streamed_simulation",
]
