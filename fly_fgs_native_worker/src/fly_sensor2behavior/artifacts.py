"""Scientific artifact and browser-replay exports.

The NumPy flight engine used here is a deterministic, reduced-order research
scaffold.  Its products are always labelled ``exploratory`` and
``calibration: none``.  In particular, these helpers never relabel that engine
as FlyBody or as an authoritative physical result.

Source-of-record arrays for each exploratory run are stored as independently
hashed ``.npy`` chunks. The small JSON replay is a decimated visualization
product derived from the same in-memory episode; it is not the run's scientific
source of record and is never described as authoritative biology.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import sysconfig
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .evidence import EvidenceGraph, default_seed_graph_path, load_evidence_graph
from .flight import (
    CausalBridgeResult,
    FlightEpisodeOutput,
    FlightEpisodeRunner,
    FlightSimulationConfig,
    MotorCommand,
    Perturbation,
    PerturbationMode,
    PerturbationTarget,
    ValidationStatus,
    default_motor_commands,
    quaternion_to_matrix,
)
from .schema import (
    REVIEWED_FLYBODY_WING_AXIS_ORDER,
    CircuitOutputTrace,
    CircuitSignalKind,
    RetinalFrame,
)


ARTIFACT_SCHEMA_VERSION = "2.0.0"
WEB_REPLAY_SCHEMA_VERSION = "1.0.0"
WEB_REPLAY_PROJECTION_SCHEMA_VERSION = "1.0.0"
WEB_REPLAY_PROJECTION_CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS: Tuple[str, ...] = (
    "source_artifact_manifest_sha256",
    "source_artifact_schema_version",
)
REDUCED_ORDER_SOURCE_KIND = "exploratory_reduced_order_numpy"
FLYBODY_WORKER_SOURCE_KIND = "exploratory_flybody_mujoco_worker"
WORKER_IMAGE_SOURCE_KIND = "image"
WORKER_IMAGE_SOURCE_NAME = "fly-s2b-worker-image"
WORKER_DEPENDENCY_LOCK_SOURCE_KIND = "dependency_lock"
WORKER_DEPENDENCY_LOCK_SOURCE_NAME = "requirements.lock"
GROUND_CONTACT_TELEMETRY_KIND = (
    "exact_mujoco_preintegration_transition_contact_points"
)
GROUND_CONTACT_SAMPLE_SEMANTICS = (
    "sample 0 is the reset-state mj_forward contact count; sample i>=1 is "
    "the pre-integration mjContact list used for the MuJoCo transition ending "
    "at that sample time, not a collision query at the displayed post-step "
    "pose and not a contact-force measurement"
)
AUTHORITY_NOTICE = (
    "These episodes come from the uncalibrated reduced-order NumPy scaffold, "
    "not FlyBody/MuJoCo and not an authoritative behavioral prediction. "
    "Connectome counts constrain topology only; all motor timing is explicitly "
    "identified as exact or inferred."
)
FLYBODY_AUTHORITY_NOTICE = (
    "This episode executed the pinned FlyGym/FlyBody MuJoCo worker, but its "
    "visual, neural, neuromuscular, and virtual-hinge parameters remain "
    "uncalibrated. Body position is the FlyBody root/thorax frame unless a "
    "separate whole-fly COM channel is present; desired hinge kinematics and "
    "measured MuJoCo wing joints are never treated as interchangeable."
)

SCENARIO_NAMES: Tuple[str, ...] = (
    "baseline",
    "vch_dch_ablation",
    "dng02_activation",
    "dna04_activation",
    "dnp26_activation",
    "dng32_activation",
    "gust",
)


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    label: str
    condition: str
    description: str
    color: str


SCENARIOS: Mapping[str, ScenarioDefinition] = {
    "baseline": ScenarioDefinition(
        "baseline",
        "Baseline flight",
        "baseline",
        "Bilaterally balanced, inferred tonic DN-to-wing-motor drive in the exploratory scaffold.",
        "#51d6c5",
    ),
    "vch_dch_ablation": ScenarioDefinition(
        "vch_dch_ablation",
        "vCH/DCH ablation",
        "pathway silencing",
        "The vCH/DCH pathway factor is set to zero for the full open-loop episode.",
        "#a78bfa",
    ),
    "dng02_activation": ScenarioDefinition(
        "dng02_activation",
        "DNg02 activation",
        "positive control",
        "Normalized DNg02 drive increases asynchronous power-muscle rate; the coefficient is uncalibrated.",
        "#fbbf24",
    ),
    "dna04_activation": ScenarioDefinition(
        "dna04_activation",
        "DNa04 activation",
        "DN activation",
        "Normalized DNa04 drive is routed to left b1 through an explicit exploratory rate-gain adapter.",
        "#fb7185",
    ),
    "dnp26_activation": ScenarioDefinition(
        "dnp26_activation",
        "DNp26 activation",
        "DN activation",
        "Normalized DNp26 drive is routed to right b1 through an explicit exploratory rate-gain adapter.",
        "#38bdf8",
    ),
    "dng32_activation": ScenarioDefinition(
        "dng32_activation",
        "DNg32 activation",
        "DN activation",
        "Normalized DNg32 drive is routed asymmetrically to b1 motor pools using uncalibrated demonstration gains.",
        "#f97316",
    ),
    "gust": ScenarioDefinition(
        "gust",
        "Wind-gust recovery",
        "environmental perturbation",
        "A bounded lateral/forward wind pulse tests open-loop physical sensitivity; no feedback controller is implied.",
        "#34d399",
    ),
}


_ARRAY_UNITS: Mapping[str, str] = {
    "time_s": "s",
    "position_world_m": "m",
    "velocity_world_m_s": "m s^-1",
    "quaternion_body_to_world": "1",
    "angular_velocity_body_rad_s": "rad s^-1",
    "wing_stroke_rad": "rad",
    "wing_angle_of_attack_rad": "rad",
    "aerodynamic_force_body_n": "N",
    "aerodynamic_torque_body_n_m": "N m",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _installed_data_path(*parts: str) -> Path:
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / Path(*parts)
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def web_replay_projection_sha256(replay: Mapping[str, Any]) -> str:
    """Hash the non-circular, canonical projection of one browser replay.

    The run-manifest checksum and schema are added only after the immutable run
    manifest exists, so those two top-level fields are the sole exclusions.
    Every scientific/display value and ``source_run_id`` remains covered.
    """

    if not isinstance(replay, Mapping):
        raise TypeError("web replay must be a mapping")
    projection = {
        key: value
        for key, value in replay.items()
        if key not in WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
    }
    encoded = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _write_web_episode_binding_receipts(
    output_dir: Path,
    *,
    episode: Mapping[str, Any],
    episode_relative_path: Path,
    artifact_manifest: Mapping[str, Any],
    artifact_manifest_path: Path,
) -> Mapping[str, str]:
    """Publish byte receipts and the artifact-owned projection preimage.

    The browser cannot safely reproduce Python's compact JSON encoding for all
    binary64 values.  The exact projection preimage is therefore served as a
    sidecar.  Its SHA-256 is already owned by the immutable artifact manifest,
    while the public summary separately binds the exact replay and artifact
    manifest response bytes.
    """

    episode_path = Path(output_dir) / episode_relative_path
    artifact_manifest_path = Path(artifact_manifest_path)
    replay_sha256 = sha256_file(episode_path)
    artifact_manifest_sha256 = sha256_file(artifact_manifest_path)
    if episode.get("source_artifact_manifest_sha256") != artifact_manifest_sha256:
        raise ValueError("web replay does not bind its exact artifact manifest bytes")
    receipt = artifact_manifest.get("web_replay_projection")
    if not isinstance(receipt, Mapping):
        raise ValueError("artifact manifest has no web replay projection receipt")
    if receipt.get("schema_version") != WEB_REPLAY_PROJECTION_SCHEMA_VERSION:
        raise ValueError("artifact web replay projection schema is unsupported")
    if receipt.get("canonicalization") != WEB_REPLAY_PROJECTION_CANONICALIZATION:
        raise ValueError("artifact web replay projection canonicalization is unsupported")
    if receipt.get("excluded_top_level_fields") != list(
        WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
    ):
        raise ValueError("artifact web replay projection exclusions are invalid")
    projection = {
        key: value
        for key, value in episode.items()
        if key not in WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
    }
    projection_bytes = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    projection_sha256 = sha256_bytes(projection_bytes)
    if receipt.get("sha256") != projection_sha256:
        raise ValueError("artifact manifest does not bind the web replay projection")
    episode_id = episode.get("id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("web replay id is required for its projection sidecar")
    projection_relative = Path("episodes") / (
        episode_id + ".artifact-projection.json"
    )
    _bytes_dump(Path(output_dir) / projection_relative, projection_bytes)
    return {
        "replay_sha256": replay_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "artifact_projection_url": "data/%s" % projection_relative.as_posix(),
    }


def _combined_source_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def model_hashes() -> Mapping[str, str]:
    """Hash every executable/model boundary used by published episodes."""

    package_root = Path(__file__).resolve().parent
    flight_sources = tuple((package_root / "flight").glob("*.py"))
    if not flight_sources:
        raise FileNotFoundError("reduced-order flight source files are unavailable")
    visual_pipeline_sources = (
        package_root / "pipeline.py",
        package_root / "nod1.py",
        package_root / "schema.py",
        *tuple((package_root / "vision").glob("*.py")),
    )
    if not visual_pipeline_sources or any(
        not path.is_file() for path in visual_pipeline_sources
    ):
        raise FileNotFoundError("visual/neural pipeline source files are unavailable")
    flybody_adapter = package_root / "flybody_adapter.py"
    if not flybody_adapter.is_file():
        raise FileNotFoundError("FlyBody adapter source file is unavailable")
    registry = _repository_root() / "data" / "models" / "flight_model_registry.json"
    if not registry.is_file():
        registry = _installed_data_path("models", "flight_model_registry.json")
    result = {
        "reduced_order_flight_source": _combined_source_hash(flight_sources),
        "visual_neural_pipeline_source": _combined_source_hash(
            visual_pipeline_sources
        ),
        "flybody_adapter_source": sha256_file(flybody_adapter),
        "artifact_contract_source": sha256_file(package_root / "artifacts.py"),
    }
    if registry.exists():
        result["flight_model_registry"] = sha256_file(registry)
    else:
        result["flight_model_registry"] = "unavailable"
    return result


def provisional_coverage(graph: EvidenceGraph) -> Tuple[Any, ...]:
    """Return only whole-pathway records that are explicitly provisional."""

    records = tuple(item for item in graph.coverage if "full_pathway_provisional" in item.branch)
    if not records:
        raise ValueError("evidence graph has no provisional full-pathway coverage records")
    return records


def audit_evidence(path: Optional[Path] = None) -> Mapping[str, Any]:
    graph_path = default_seed_graph_path() if path is None else Path(path)
    graph = load_evidence_graph(graph_path)
    provisional = provisional_coverage(graph)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "graph_id": graph.graph_id,
        "evidence_sha256": sha256_file(graph_path),
        "datasets": [dataset.to_dict() for dataset in graph.datasets],
        "edge_count": len(graph.edges),
        "structural_synapse_total": graph.structural_synapse_total(),
        "coverage_status": "provisional",
        "provisional_full_pathway_coverage": [
            {
                "branch": record.branch,
                "resolved_structural_synapses": record.resolved_structural_synapses,
                "unresolved_structural_synapses": record.unresolved_structural_synapses,
                "total_structural_synapses": record.total_structural_synapses,
                "resolved_fraction": record.resolved_fraction,
                "scope": record.scope,
                "confidence": record.confidence.to_dict(),
            }
            for record in provisional
        ],
        "warnings": [
            "Coverage is provisional until the source audit is reproduced from a versioned raw artifact.",
            "Structural synapse counts are not physiological weights, activations, or gains.",
            "Cross-atlas routes use cell-type crosswalks; neuron identifiers are never directly joined.",
        ],
    }


def _scenario_drives(name: str) -> Dict[str, float]:
    # These normalized values are intentionally exposed as uncalibrated model
    # inputs.  They are not computed from connectome counts.
    drives = {"DNa04": 0.10, "DNp26": 0.10, "DNg32": 0.0, "DNg02": 0.0}
    if name == "dng02_activation":
        drives["DNg02"] = 1.0
    elif name == "dna04_activation":
        drives["DNa04"] = 1.0
    elif name == "dnp26_activation":
        drives["DNp26"] = 1.0
    elif name == "dng32_activation":
        drives["DNg32"] = 1.0
    return drives


def build_scenario_config(
    scenario: str,
    *,
    duration_s: float = 0.200,
    physics_dt_s: float = 0.0001,
    logging_dt_s: float = 0.001,
    neural_dt_s: float = 0.005,
    seed: int = 0,
) -> FlightSimulationConfig:
    """Create one deterministic, explicitly uncalibrated scenario."""

    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario {!r}; choose from {}".format(scenario, ", ".join(SCENARIO_NAMES)))
    logging_steps = int(round(duration_s / logging_dt_s))
    if logging_steps < 1 or abs(logging_steps * logging_dt_s - duration_s) > 1.0e-10:
        raise ValueError("duration_s must be a positive integer multiple of logging_dt_s")
    perturbations: Tuple[Perturbation, ...] = ()
    pathway_factor = 0.0 if scenario == "vch_dch_ablation" else 1.0
    if scenario == "gust":
        start = 0.25 * duration_s
        end = 0.75 * duration_s
        perturbations = (
            Perturbation(
                target_type=PerturbationTarget.ENVIRONMENT,
                target="wind",
                mode=PerturbationMode.GUST,
                start_s=start,
                end_s=end,
                magnitude=1.0,
                vector=(1.0, 0.4, 0.0),
            ),
        )
    # These gains are named rate adapters and remain independent of anatomical
    # synapse counts.  Their uncalibrated status is repeated in every artifact.
    gains = {
        "DNa04": {"MN-b1-left": 180.0},
        "DNp26": {"MN-b1-right": 180.0},
        "DNg32": {"MN-b1-left": 160.0, "MN-b1-right": 30.0},
    }
    return FlightSimulationConfig(
        duration_s=duration_s,
        physics_dt_s=physics_dt_s,
        neural_dt_s=neural_dt_s,
        logging_dt_s=logging_dt_s,
        seed=seed,
        motor_commands=default_motor_commands(),
        dn_drives=_scenario_drives(scenario),
        dn_motor_rate_gains_hz=gains,
        perturbations=perturbations,
        vch_dch_factor=pathway_factor,
    )


def run_scenario(
    scenario: str,
    *,
    duration_s: float = 0.200,
    physics_dt_s: float = 0.0001,
    logging_dt_s: float = 0.001,
    neural_dt_s: float = 0.005,
    seed: int = 0,
) -> Tuple[FlightSimulationConfig, FlightEpisodeOutput]:
    config = build_scenario_config(
        scenario,
        duration_s=duration_s,
        physics_dt_s=physics_dt_s,
        logging_dt_s=logging_dt_s,
        neural_dt_s=neural_dt_s,
        seed=seed,
    )
    result = FlightEpisodeRunner().run(config)
    if result.diagnostics.status is not ValidationStatus.EXPLORATORY:
        raise RuntimeError("reduced-order runner returned a non-exploratory status")
    return config, result


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        allow_nan=False,
        separators=(",", ": "),
    ).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix=".%s." % path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _bytes_dump(path: Path, payload: bytes) -> None:
    """Atomically publish exact bytes whose digest is part of a web contract."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=".%s." % path.name, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _safe_array_path(name: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "array"
    suffix = sha256_bytes(name.encode("utf-8"))[:10]
    return "%s-%s" % (readable, suffix)


def _logical_array_hash(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return sha256_bytes(buffer.getvalue())


def _write_npy(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix=".%s." % path.name, delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    return sha256_file(path)


def _write_chunked_array(
    output_dir: Path,
    name: str,
    values: np.ndarray,
    unit: str,
    chunk_samples: int,
    provenance: str,
) -> Mapping[str, Any]:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("object arrays are forbidden in scientific artifacts")
    if array.dtype.kind not in "biufc":
        raise TypeError("scientific artifact arrays must have a numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("scientific artifact array %r contains NaN or infinity" % name)
    chunks: List[Mapping[str, Any]] = []
    sample_count = int(array.shape[0]) if array.ndim else 1
    starts = list(range(0, sample_count, chunk_samples)) or [0]
    stem = _safe_array_path(name)
    for index, start in enumerate(starts):
        stop = min(sample_count, start + chunk_samples)
        chunk = array[start:stop] if array.ndim else array.reshape(1)
        relative = Path("arrays") / stem / ("%06d.npy" % index)
        digest = _write_npy(output_dir / relative, chunk)
        chunks.append(
            {
                "path": relative.as_posix(),
                "start": start,
                "stop": stop,
                "shape": list(chunk.shape),
                "sha256": digest,
            }
        )
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "unit": unit,
        "provenance": provenance,
        "logical_npy_sha256": _logical_array_hash(array),
        "chunks": chunks,
    }


def _output_arrays(
    result: FlightEpisodeOutput, commands: Sequence[MotorCommand]
) -> Iterable[Tuple[str, np.ndarray, str, str]]:
    timing_provenance = {
        command.neuron_id: _event_timing_provenance(command)
        for command in commands
    }
    external = result.diagnostics.physics_backend == "flybody"
    for name in _ARRAY_UNITS:
        if not external:
            provenance = "reduced_order_model_output"
        elif name in ("wing_stroke_rad", "wing_angle_of_attack_rad"):
            provenance = "virtual_hinge_desired_kinematics_not_measured_flybody_state"
        elif name == "time_s":
            provenance = "shared_episode_clock"
        else:
            provenance = "flybody_mujoco_root_total_output"
        yield name, np.asarray(getattr(result, name)), _ARRAY_UNITS[name], provenance
    if result.measured_wing_joint_angle_rad is not None:
        yield (
            "measured_wing_joint_angle_rad",
            np.asarray(result.measured_wing_joint_angle_rad),
            "rad",
            "external_physics_measured_output",
        )
    if result.measured_wing_joint_velocity_rad_s is not None:
        yield (
            "measured_wing_joint_velocity_rad_s",
            np.asarray(result.measured_wing_joint_velocity_rad_s),
            "rad s^-1",
            "external_physics_measured_output",
        )
    if result.whole_fly_com_position_world_m is not None:
        yield (
            "whole_fly_com_position_world_m",
            np.asarray(result.whole_fly_com_position_world_m),
            "m",
            "flybody_mujoco_articulated_subtree_com",
        )
    if result.ground_contact_count is not None:
        yield (
            "ground_contact_count",
            np.asarray(result.ground_contact_count),
            "1",
            "decimated_projection_of_mujoco_transition_contact_points",
        )
    if result.external_actuator_torque_n_m is not None:
        yield (
            "external_actuator_torque_n_m",
            np.asarray(result.external_actuator_torque_n_m),
            "N m",
            "logged_projection_of_external_mujoco_wing_actuator_torque",
        )
    if result.physics_time_s is not None:
        yield (
            "physics_time_s",
            np.asarray(result.physics_time_s),
            "s",
            "authoritative_external_physics_clock",
        )
    if result.ground_contact_transition_point_count is not None:
        yield (
            "ground_contact_transition_point_count",
            np.asarray(result.ground_contact_transition_point_count),
            "1",
            GROUND_CONTACT_TELEMETRY_KIND,
        )
    if result.external_actuator_torque_physics_n_m is not None:
        yield (
            "external_actuator_torque_physics_n_m",
            np.asarray(result.external_actuator_torque_physics_n_m),
            "N m",
            "physics_rate_external_mujoco_wing_actuator_torque",
        )
    if result.measured_wing_joint_angle_physics_rad is not None:
        yield (
            "measured_wing_joint_angle_physics_rad",
            np.asarray(result.measured_wing_joint_angle_physics_rad),
            "rad",
            "physics_rate_external_measured_wing_output",
        )
    if result.measured_wing_joint_velocity_physics_rad_s is not None:
        yield (
            "measured_wing_joint_velocity_physics_rad_s",
            np.asarray(result.measured_wing_joint_velocity_physics_rad_s),
            "rad s^-1",
            "physics_rate_external_measured_wing_output",
        )
    if result.aerodynamic_force_body_physics_n is not None:
        yield (
            "aerodynamic_force_body_physics_n",
            np.asarray(result.aerodynamic_force_body_physics_n),
            "N",
            "physics_rate_flybody_root_total_fluid_force",
        )
    if result.aerodynamic_torque_body_physics_n_m is not None:
        yield (
            "aerodynamic_torque_body_physics_n_m",
            np.asarray(result.aerodynamic_torque_body_physics_n_m),
            "N m",
            "physics_rate_flybody_root_total_fluid_torque",
        )
    for neuron_id, values in sorted(result.motor_event_times_s.items()):
        yield (
            "motor_event_times_s/%s" % neuron_id,
            np.asarray(values),
            "s",
            timing_provenance[neuron_id],
        )
    for neuron_id, values in sorted(result.motor_event_phases_rad.items()):
        yield (
            "motor_event_phases_rad/%s" % neuron_id,
            np.asarray(values),
            "rad",
            timing_provenance[neuron_id],
        )
    for command in sorted(commands, key=lambda item: item.neuron_id):
        if command.signal_kind.value not in (
            "exact_spikes",
            "seeded_synthetic_spikes",
        ):
            continue
        measurement_provenance = (
            "exact_input_spike_measurement_times"
            if command.signal_kind.value == "exact_spikes"
            else "seeded_synthetic_spike_measurement_times"
        )
        yield (
            "motor_input_measurement_spike_times_s/%s" % command.neuron_id,
            np.asarray(command.measurement_spike_times_s),
            "s",
            measurement_provenance,
        )
    for muscle_id, values in sorted(result.muscle_activation.items()):
        yield "muscle_activation/%s" % muscle_id, np.asarray(values), "1", "model_inference"
    for muscle_id, values in sorted(result.muscle_force_n.items()):
        yield "muscle_force_n/%s" % muscle_id, np.asarray(values), "N", "model_inference"
    for muscle_id, values in sorted(result.muscle_phase_effect.items()):
        yield (
            "muscle_phase_effect/%s" % muscle_id,
            np.asarray(values),
            "1",
            "phase_dependent_model_inference",
        )
    for muscle_id, values in sorted(result.muscle_work_j.items()):
        yield (
            "muscle_work_j/%s" % muscle_id,
            np.asarray(values),
            "J",
            "virtual_hinge_work_estimate_model_inference",
        )


def _signal_provenance(commands: Sequence[MotorCommand]) -> Mapping[str, Any]:
    return {
        command.neuron_id: {
            "muscle": command.muscle,
            "side": command.side.value,
            "signal_kind": command.signal_kind.value,
            "event_timing": _event_timing_provenance(command),
            "generator_seed": command.generator_seed,
            "provenance": command.provenance,
        }
        for command in commands
    }


def _motor_command_payload(command: MotorCommand) -> Mapping[str, Any]:
    return {
        "neuron_id": command.neuron_id,
        "muscle": command.muscle,
        "side": command.side.value,
        "muscle_class": command.muscle_class.value,
        "signal_kind": command.signal_kind.value,
        "rate_hz": command.rate_hz,
        "spike_times_s": list(command.spike_times_s),
        "measurement_spike_times_s": list(command.measurement_spike_times_s),
        "motor_units": command.motor_units,
        "preferred_phase_rad": command.preferred_phase_rad,
        "provenance": command.provenance,
        "generator_seed": command.generator_seed,
    }


def _event_timing_provenance(command: MotorCommand) -> str:
    if command.signal_kind.value == "exact_spikes":
        return "exact_input_spike_runtime_availability_times"
    if command.signal_kind.value == "seeded_synthetic_spikes":
        return "seeded_synthetic_runtime_availability_times_from_inferred_rate"
    return "inferred_from_rate_by_physiology_scaffold"


def _configuration_payload(
    config: FlightSimulationConfig, commands: Sequence[MotorCommand]
) -> Mapping[str, Any]:
    return {
        "duration_s": config.duration_s,
        "physics_timestep_s": config.physics_dt_s,
        "neural_timestep_s": config.neural_dt_s,
        "logging_timestep_s": config.logging_dt_s,
        "seed": config.seed,
        "starts_airborne": True,
        "motor_commands": [_motor_command_payload(command) for command in commands],
        "dn_drives_normalized": dict(config.dn_drives),
        "dn_motor_rate_gains_hz": {
            key: dict(value) for key, value in config.dn_motor_rate_gains_hz.items()
        },
        "vch_dch_pathway_dn_names": list(config.pathway_dn_names),
        "dng02_power_rate_gain": config.dng02_power_rate_gain,
        "vch_dch_factor": config.vch_dch_factor,
        "perturbations": [
            {
                "target_type": item.target_type.value,
                "target": item.target,
                "mode": item.mode.value,
                "start_s": item.start_s,
                "end_s": item.end_s,
                "magnitude": item.magnitude,
                "vector": list(item.vector),
            }
            for item in config.perturbations
        ],
    }


def _canonical_identity_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def episode_run_id(
    scenario: str,
    config: FlightSimulationConfig,
    *,
    evidence_path: Optional[Path] = None,
    identity_context: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the content-addressed identity for one scientific run."""

    graph_path = default_seed_graph_path() if evidence_path is None else Path(evidence_path)
    commands = (
        default_motor_commands()
        if config.motor_commands is None
        else config.motor_commands
    )
    identity_payload = {
        "scenario": scenario,
        "configuration": _configuration_payload(config, commands),
        "model_hashes": dict(model_hashes()),
        "evidence_sha256": sha256_file(graph_path),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "identity_context": dict(identity_context or {}),
    }
    return "%s-%s" % (scenario, _canonical_identity_hash(identity_payload)[:20])


def _finite_numeric_array(name: str, values: Any) -> np.ndarray:
    """Return an array only when it is safe for a scientific artifact."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("%s must not use an object dtype" % name)
    if array.dtype.kind not in "biufc":
        raise TypeError("%s must have a numeric dtype" % name)
    if not np.all(np.isfinite(array)):
        raise ValueError("%s must contain only finite values" % name)
    return array


def _integer_clock_ratio(numerator_s: float, denominator_s: float, name: str) -> int:
    ratio = int(round(numerator_s / denominator_s))
    tolerance = max(1.0e-12, 1.0e-9 * abs(numerator_s))
    if ratio < 1 or abs(ratio * denominator_s - numerator_s) > tolerance:
        raise ValueError("%s must be an integer multiple of physics_dt_s" % name)
    return ratio


def _strict_json_preflight(name: str, payload: Any) -> None:
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be strict finite JSON: %s" % (name, exc)) from exc


def _validate_supplemental_arrays(
    supplemental_arrays: Sequence[Tuple[str, np.ndarray, str, str]],
    core_names: Sequence[str],
) -> Tuple[Tuple[str, np.ndarray, str, str], ...]:
    normalized: List[Tuple[str, np.ndarray, str, str]] = []
    by_name: Dict[str, np.ndarray] = {}
    reserved = set(core_names)
    for record in supplemental_arrays:
        if len(record) != 4:
            raise ValueError(
                "supplemental arrays require name, values, unit, and provenance"
            )
        name, values, unit, provenance = record
        if not isinstance(name, str) or not name:
            raise ValueError("supplemental array names must be non-empty strings")
        if name in reserved or name in by_name:
            raise ValueError("duplicate scientific array name %r" % name)
        if not isinstance(unit, str) or not unit:
            raise ValueError("supplemental array %r requires a non-empty unit" % name)
        if not isinstance(provenance, str) or not provenance:
            raise ValueError(
                "supplemental array %r requires non-empty provenance" % name
            )
        array = _finite_numeric_array("supplemental array %r" % name, values)
        if array.ndim == 0:
            raise ValueError("supplemental array %r must have a sample axis" % name)
        by_name[name] = array
        normalized.append((name, array, unit, provenance))

    retinal_names = {
        "retinal_normalized_luminance",
        "retinal_measurement_time_s",
        "retinal_availability_time_s",
        "retinal_exposure_interval_s",
    }
    present_retinal = retinal_names.intersection(by_name)
    if present_retinal:
        missing = retinal_names.difference(by_name)
        if missing:
            raise ValueError(
                "retinal supplemental arrays are incomplete: missing %s"
                % ", ".join(sorted(missing))
            )
        retinal_count = by_name["retinal_normalized_luminance"].shape[0]
        if by_name["retinal_normalized_luminance"].ndim != 2:
            raise ValueError("retinal_normalized_luminance must have shape (time, sample)")
        for time_name in (
            "retinal_measurement_time_s",
            "retinal_availability_time_s",
        ):
            if by_name[time_name].shape != (retinal_count,):
                raise ValueError("%s must align with retinal samples" % time_name)
        if by_name["retinal_exposure_interval_s"].shape != (retinal_count, 2):
            raise ValueError(
                "retinal_exposure_interval_s must have shape (time, 2)"
            )
        retinal_values = by_name["retinal_normalized_luminance"]
        measurements = by_name["retinal_measurement_time_s"]
        availability = by_name["retinal_availability_time_s"]
        exposures = by_name["retinal_exposure_interval_s"]
        if np.any(retinal_values < 0.0) or np.any(retinal_values > 1.0):
            raise ValueError("retinal normalized luminance must lie in [0, 1]")
        if (
            np.any(exposures[:, 0] < 0.0)
            or np.any(exposures[:, 1] <= exposures[:, 0])
            or np.any(measurements < exposures[:, 0])
            or np.any(measurements > exposures[:, 1])
            or np.any(availability < exposures[:, 1])
            or np.any(availability < measurements)
        ):
            raise ValueError(
                "retinal measurement, exposure, and availability times violate causality"
            )
        if retinal_count > 1 and np.any(np.diff(measurements) <= 0.0):
            raise ValueError("retinal measurement times must be strictly increasing")

    circuit_time = by_name.get("circuit_sample_time_s")
    circuit_values: Dict[str, np.ndarray] = {}
    circuit_availability: Dict[str, np.ndarray] = {}
    for name, array in by_name.items():
        if name.startswith("circuit_availability_time_s/"):
            circuit_availability[name.split("/", 1)[1]] = array
        elif (
            name.startswith("circuit_")
            and "/" in name
            and not name.startswith("circuit_spike_times_s/")
        ):
            circuit_values[name.split("/", 1)[1]] = array
    if circuit_values or circuit_availability:
        if circuit_time is None or circuit_time.ndim != 1:
            raise ValueError(
                "sampled circuit supplemental arrays require circuit_sample_time_s"
            )
        if set(circuit_values) != set(circuit_availability):
            raise ValueError(
                "circuit values and availability arrays must have matching neuron IDs"
            )
        for entity_id in sorted(circuit_values):
            expected = (len(circuit_time),)
            if circuit_values[entity_id].shape != expected:
                raise ValueError(
                    "circuit value array for %s must align with circuit_sample_time_s"
                    % entity_id
                )
            if circuit_availability[entity_id].shape != expected:
                raise ValueError(
                    "circuit availability array for %s must align with its values"
                    % entity_id
                )
            if np.any(circuit_availability[entity_id] < circuit_time):
                raise ValueError(
                    "circuit availability for %s cannot precede measurement"
                    % entity_id
                )
    if circuit_time is not None:
        if circuit_time.ndim != 1 or len(circuit_time) == 0:
            raise ValueError("circuit_sample_time_s must be a non-empty vector")
        if len(circuit_time) > 1 and np.any(np.diff(circuit_time) <= 0.0):
            raise ValueError("circuit_sample_time_s must be strictly increasing")

    def validate_triplets(prefixes: Tuple[str, str, str], label: str) -> None:
        grouped: Dict[str, Dict[str, np.ndarray]] = {}
        for prefix in prefixes:
            marker = prefix + "/"
            for name, array in by_name.items():
                if name.startswith(marker):
                    grouped.setdefault(name[len(marker) :], {})[prefix] = array
        for channel, fields in grouped.items():
            if set(fields) != set(prefixes):
                raise ValueError(
                    "%s supplemental channel %s is missing aligned arrays"
                    % (label, channel)
                )
            lengths = set()
            for prefix, array in fields.items():
                if array.ndim != 1:
                    raise ValueError("%s/%s must be one-dimensional" % (prefix, channel))
                lengths.add(len(array))
            if len(lengths) != 1:
                raise ValueError(
                    "%s supplemental channel %s arrays must have equal lengths"
                    % (label, channel)
                )

    validate_triplets(
        (
            "descending_rate_hz",
            "descending_measurement_time_s",
            "descending_availability_time_s",
        ),
        "descending",
    )
    validate_triplets(
        (
            "wing_motor_rate_hz",
            "wing_motor_rate_measurement_time_s",
            "wing_motor_rate_availability_time_s",
        ),
        "wing-motor rate",
    )
    validate_triplets(
        (
            "wing_motor_event_measurement_time_s",
            "wing_motor_event_availability_time_s",
            "wing_motor_event_phase_rad",
        ),
        "wing-motor event",
    )

    def validate_causal_channels(
        value_prefix: str,
        measurement_prefix: str,
        availability_prefix: str,
        *,
        value_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        marker = measurement_prefix + "/"
        for name, measurements in by_name.items():
            if not name.startswith(marker):
                continue
            channel = name[len(marker) :]
            availability = by_name[availability_prefix + "/" + channel]
            values = by_name[value_prefix + "/" + channel]
            if (
                np.any(measurements < 0.0)
                or np.any(availability < measurements)
                or (len(measurements) > 1 and np.any(np.diff(measurements) < 0.0))
            ):
                raise ValueError(
                    "%s/%s measurement and availability times violate causality"
                    % (measurement_prefix, channel)
                )
            if value_bounds is not None and (
                np.any(values < value_bounds[0])
                or np.any(values > value_bounds[1])
            ):
                raise ValueError(
                    "%s/%s values lie outside [%s, %s]"
                    % (
                        value_prefix,
                        channel,
                        value_bounds[0],
                        value_bounds[1],
                    )
                )

    validate_causal_channels(
        "descending_rate_hz",
        "descending_measurement_time_s",
        "descending_availability_time_s",
        value_bounds=(0.0, float("inf")),
    )
    validate_causal_channels(
        "wing_motor_rate_hz",
        "wing_motor_rate_measurement_time_s",
        "wing_motor_rate_availability_time_s",
        value_bounds=(0.0, float("inf")),
    )
    validate_causal_channels(
        "wing_motor_event_phase_rad",
        "wing_motor_event_measurement_time_s",
        "wing_motor_event_availability_time_s",
        value_bounds=(0.0, float(2.0 * np.pi)),
    )
    for name, values in by_name.items():
        if name.startswith("wing_motor_event_phase_rad/") and np.any(
            values >= 2.0 * np.pi
        ):
            raise ValueError("wing-motor event phase must lie in [0, 2*pi)")
    return tuple(normalized)


def _validate_episode_artifact_inputs(
    config: FlightSimulationConfig,
    result: FlightEpisodeOutput,
    commands: Sequence[MotorCommand],
    supplemental_arrays: Sequence[Tuple[str, np.ndarray, str, str]],
) -> Tuple[Tuple[str, np.ndarray, str, str], ...]:
    """Fail closed on an inconsistent scientific run before creating output."""

    clock_values = {
        "duration_s": config.duration_s,
        "physics_dt_s": config.physics_dt_s,
        "neural_dt_s": config.neural_dt_s,
        "logging_dt_s": config.logging_dt_s,
    }
    for name, raw_value in clock_values.items():
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be finite and positive" % name)
    duration_s = float(config.duration_s)
    physics_dt_s = float(config.physics_dt_s)
    physics_steps = _integer_clock_ratio(duration_s, physics_dt_s, "duration_s")
    logging_ratio = _integer_clock_ratio(
        float(config.logging_dt_s), physics_dt_s, "logging_dt_s"
    )
    neural_ratio = _integer_clock_ratio(
        float(config.neural_dt_s), physics_dt_s, "neural_dt_s"
    )
    if physics_steps % logging_ratio:
        raise ValueError("duration_s must end on a logging clock boundary")
    logging_steps = physics_steps // logging_ratio
    sample_count = logging_steps + 1
    tolerance = max(1.0e-12, 1.0e-9 * duration_s)

    core_shapes = {
        "time_s": (sample_count,),
        "position_world_m": (sample_count, 3),
        "velocity_world_m_s": (sample_count, 3),
        "quaternion_body_to_world": (sample_count, 4),
        "angular_velocity_body_rad_s": (sample_count, 3),
        "wing_stroke_rad": (sample_count, 2),
        "wing_angle_of_attack_rad": (sample_count, 2),
        "aerodynamic_force_body_n": (sample_count, 3),
        "aerodynamic_torque_body_n_m": (sample_count, 3),
    }
    core_arrays: Dict[str, np.ndarray] = {}
    for name, shape in core_shapes.items():
        array = _finite_numeric_array(name, getattr(result, name))
        if array.shape != shape:
            raise ValueError("%s must have shape %s" % (name, shape))
        core_arrays[name] = array

    time_s = core_arrays["time_s"]
    expected_time = np.arange(sample_count, dtype=float) * float(config.logging_dt_s)
    if not np.allclose(time_s, expected_time, rtol=0.0, atol=tolerance):
        raise ValueError("time_s must exactly follow the configured logging clock")
    if abs(float(time_s[0])) > tolerance or abs(float(time_s[-1]) - duration_s) > tolerance:
        raise ValueError("time_s must span the configured episode from 0 to duration_s")
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s must be strictly increasing")

    quaternion_norm = np.linalg.norm(
        core_arrays["quaternion_body_to_world"], axis=1
    )
    if not np.allclose(quaternion_norm, 1.0, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError("quaternion_body_to_world must contain unit quaternions")

    diagnostics = result.diagnostics
    if diagnostics.status is not ValidationStatus.EXPLORATORY:
        raise ValueError("artifact writer only accepts exploratory episode results")
    if int(diagnostics.physics_steps) != physics_steps:
        raise ValueError("diagnostic physics_steps does not match the configured clock")
    if not np.isfinite(float(diagnostics.final_time_s)) or abs(
        float(diagnostics.final_time_s) - duration_s
    ) > tolerance:
        raise ValueError("diagnostic final_time_s does not match duration_s")
    if not isinstance(diagnostics.physics_backend, str) or not diagnostics.physics_backend:
        raise ValueError("diagnostic physics_backend must be non-empty")
    expected_owner = {
        "reduced_order": "reduced_order_quasi_steady",
        "flybody": "flybody",
    }.get(diagnostics.physics_backend)
    if expected_owner is None:
        raise ValueError("unsupported artifact physics_backend %r" % diagnostics.physics_backend)
    if diagnostics.aerodynamic_owner != expected_owner:
        raise ValueError(
            "aerodynamic_owner is inconsistent with physics_backend %s"
            % diagnostics.physics_backend
        )
    impulse = _finite_numeric_array(
        "integrated_aerodynamic_impulse_n_s",
        diagnostics.integrated_aerodynamic_impulse_n_s,
    )
    if impulse.shape != (3,):
        raise ValueError("integrated_aerodynamic_impulse_n_s must have shape (3,)")
    for name, value in diagnostics.metrics.items():
        scalar = _finite_numeric_array("diagnostic metric %r" % name, value)
        if scalar.ndim != 0:
            raise ValueError("diagnostic metric %r must be scalar" % name)
    if "physics_dt_s" not in diagnostics.metrics or not np.isclose(
        float(diagnostics.metrics["physics_dt_s"]),
        physics_dt_s,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("diagnostic physics_dt_s does not match the configured clock")
    expected_neural_updates = (physics_steps + neural_ratio - 1) // neural_ratio
    if "neural_update_count" not in diagnostics.metrics or int(
        diagnostics.metrics["neural_update_count"]
    ) != expected_neural_updates:
        raise ValueError("diagnostic neural_update_count does not match the configured clock")

    command_ids = [command.neuron_id for command in commands]
    if len(set(command_ids)) != len(command_ids):
        raise ValueError("motor command neuron_id values must be unique")
    for command in commands:
        command_times = _finite_numeric_array(
            "motor command %s runtime spike_times_s" % command.neuron_id,
            command.spike_times_s,
        )
        measurement_times = _finite_numeric_array(
            "motor command %s measurement_spike_times_s" % command.neuron_id,
            command.measurement_spike_times_s,
        )
        if command_times.ndim != 1 or measurement_times.shape != command_times.shape:
            raise ValueError(
                "motor command %s measurement and runtime spike times must align"
                % command.neuron_id
            )
        if np.any(command_times < 0.0) or np.any(command_times >= duration_s):
            raise ValueError(
                "motor command %s contains an event outside [0, duration_s)"
                % command.neuron_id
            )
        if np.any(measurement_times < 0.0) or np.any(
            measurement_times >= duration_s
        ):
            raise ValueError(
                "motor command %s contains a measurement outside [0, duration_s)"
                % command.neuron_id
            )
        if np.any(measurement_times > command_times):
            raise ValueError(
                "motor command %s has a measurement after runtime availability"
                % command.neuron_id
            )
    expected_motor_keys = set(command_ids)
    actual_time_keys = set(result.motor_event_times_s)
    actual_phase_keys = set(result.motor_event_phases_rad)
    if actual_time_keys != expected_motor_keys or actual_phase_keys != expected_motor_keys:
        raise ValueError("motor event maps must contain exactly the configured neuron IDs")
    for neuron_id in command_ids:
        event_times = _finite_numeric_array(
            "motor_event_times_s/%s" % neuron_id,
            result.motor_event_times_s[neuron_id],
        )
        event_phases = _finite_numeric_array(
            "motor_event_phases_rad/%s" % neuron_id,
            result.motor_event_phases_rad[neuron_id],
        )
        if event_times.ndim != 1 or event_phases.shape != event_times.shape:
            raise ValueError(
                "motor event times and phases for %s must be aligned vectors"
                % neuron_id
            )
        if np.any(event_times < 0.0) or np.any(event_times >= duration_s):
            raise ValueError("motor events for %s must lie in [0, duration_s)" % neuron_id)
        if len(event_times) > 1 and np.any(np.diff(event_times) < 0.0):
            raise ValueError("motor events for %s must be sorted" % neuron_id)
        if np.any(event_phases < 0.0) or np.any(event_phases >= 2.0 * np.pi):
            raise ValueError("motor event phases for %s must lie in [0, 2*pi)" % neuron_id)

    expected_muscles = {
        "%s:%s" % (command.side.value, command.muscle) for command in commands
    }
    muscle_series = {
        "muscle_activation": result.muscle_activation,
        "muscle_force_n": result.muscle_force_n,
        "muscle_phase_effect": result.muscle_phase_effect,
        "muscle_work_j": result.muscle_work_j,
    }
    for group_name, series_by_muscle in muscle_series.items():
        if set(series_by_muscle) != expected_muscles:
            raise ValueError(
                "%s keys must match the configured side:muscle channels" % group_name
            )
        for muscle_id, values in series_by_muscle.items():
            values_array = _finite_numeric_array(
                "%s/%s" % (group_name, muscle_id), values
            )
            if values_array.shape != (sample_count,):
                raise ValueError(
                    "%s/%s must align with time_s" % (group_name, muscle_id)
                )
            if group_name in ("muscle_activation", "muscle_force_n") and np.any(
                values_array < 0.0
            ):
                raise ValueError("%s/%s must be non-negative" % (group_name, muscle_id))
            if group_name == "muscle_activation" and np.any(values_array > 1.5):
                raise ValueError(
                    "muscle_activation/%s must lie in [0, 1.5]" % muscle_id
                )
            if group_name == "muscle_phase_effect" and np.any(
                np.abs(values_array) > 1.5
            ):
                raise ValueError(
                    "muscle_phase_effect/%s must lie in [-1.5, 1.5]" % muscle_id
                )

    measured_angle = result.measured_wing_joint_angle_rad
    measured_velocity = result.measured_wing_joint_velocity_rad_s
    if (measured_angle is None) != (measured_velocity is None):
        raise ValueError("measured wing angle and velocity arrays must be supplied together")
    if measured_angle is not None:
        for name, values in (
            ("measured_wing_joint_angle_rad", measured_angle),
            ("measured_wing_joint_velocity_rad_s", measured_velocity),
        ):
            array = _finite_numeric_array(name, values)
            if array.shape != (sample_count, 6):
                raise ValueError("%s must have shape (time, 6)" % name)
        order = tuple(result.measured_wing_joint_order)
        if (
            len(order) != 6
            or len(set(order)) != 6
            or any(not isinstance(name, str) or not name for name in order)
        ):
            raise ValueError("measured_wing_joint_order must name six unique axes")
    elif result.measured_wing_joint_order:
        raise ValueError("measured_wing_joint_order requires measured wing arrays")

    measured_angle_physics = result.measured_wing_joint_angle_physics_rad
    measured_velocity_physics = result.measured_wing_joint_velocity_physics_rad_s
    if (measured_angle_physics is None) != (measured_velocity_physics is None):
        raise ValueError(
            "physics-rate measured wing angle and velocity must be supplied together"
        )
    if (measured_angle is None) != (measured_angle_physics is None):
        raise ValueError(
            "logged and physics-rate measured wing telemetry must be supplied together"
        )
    if measured_angle_physics is not None:
        physics_angle = _finite_numeric_array(
            "measured_wing_joint_angle_physics_rad", measured_angle_physics
        )
        physics_velocity = _finite_numeric_array(
            "measured_wing_joint_velocity_physics_rad_s",
            measured_velocity_physics,
        )
        expected_physics_shape = (physics_steps + 1, 6)
        if (
            physics_angle.shape != expected_physics_shape
            or physics_velocity.shape != expected_physics_shape
        ):
            raise ValueError(
                "physics-rate measured wing arrays must have shape (physics_time, 6)"
            )
        logged_indices = np.rint(time_s / physics_dt_s).astype(np.int64)
        if not np.array_equal(np.asarray(measured_angle), physics_angle[logged_indices]):
            raise ValueError(
                "logged wing angles do not match the physics-rate trace"
            )
        if not np.array_equal(
            np.asarray(measured_velocity), physics_velocity[logged_indices]
        ):
            raise ValueError(
                "logged wing velocities do not match the physics-rate trace"
            )
        expected_wing_excursion = float(np.max(np.ptp(physics_angle, axis=0)))
        reported_wing_excursion = diagnostics.metrics.get(
            "maximum_measured_wing_excursion_rad"
        )
        if reported_wing_excursion is None or not np.isclose(
            float(reported_wing_excursion),
            expected_wing_excursion,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "maximum measured wing excursion metric does not match the "
                "physics-rate trace"
            )

    aerodynamic_force_physics = result.aerodynamic_force_body_physics_n
    aerodynamic_torque_physics = result.aerodynamic_torque_body_physics_n_m
    if (aerodynamic_force_physics is None) != (aerodynamic_torque_physics is None):
        raise ValueError(
            "physics-rate aerodynamic force and torque must be supplied together"
        )
    if aerodynamic_force_physics is not None:
        physics_force = _finite_numeric_array(
            "aerodynamic_force_body_physics_n", aerodynamic_force_physics
        )
        physics_aerodynamic_torque = _finite_numeric_array(
            "aerodynamic_torque_body_physics_n_m", aerodynamic_torque_physics
        )
        expected_physics_wrench_shape = (physics_steps + 1, 3)
        if (
            physics_force.shape != expected_physics_wrench_shape
            or physics_aerodynamic_torque.shape != expected_physics_wrench_shape
        ):
            raise ValueError(
                "physics-rate aerodynamic arrays must have shape (physics_time, 3)"
            )
        logged_indices = np.rint(time_s / physics_dt_s).astype(np.int64)
        if not np.array_equal(
            core_arrays["aerodynamic_force_body_n"], physics_force[logged_indices]
        ) or not np.array_equal(
            core_arrays["aerodynamic_torque_body_n_m"],
            physics_aerodynamic_torque[logged_indices],
        ):
            raise ValueError(
                "logged aerodynamic wrench does not match the physics-rate trace"
            )
        reconstructed_impulse = physics_dt_s * np.sum(physics_force[1:], axis=0)
        if not np.allclose(
            reconstructed_impulse, impulse, rtol=1.0e-12, atol=1.0e-18
        ):
            raise ValueError(
                "integrated aerodynamic impulse does not match the physics-rate trace"
            )

    if result.whole_fly_com_position_world_m is not None:
        whole_fly_com = _finite_numeric_array(
            "whole_fly_com_position_world_m",
            result.whole_fly_com_position_world_m,
        )
        if whole_fly_com.shape != (sample_count, 3):
            raise ValueError("whole_fly_com_position_world_m must have shape (time, 3)")
    if result.ground_contact_count is not None:
        contact_count = _finite_numeric_array(
            "ground_contact_count", result.ground_contact_count
        )
        if contact_count.shape != (sample_count,):
            raise ValueError("ground_contact_count must have shape (time,)")
        if contact_count.dtype.kind not in "iu" or np.any(contact_count < 0):
            raise ValueError(
                "ground_contact_count must contain non-negative integers"
            )
        _ground_contact_summary(config, result)
    if result.external_actuator_torque_n_m is not None:
        if len(tuple(result.measured_wing_joint_order)) != 6:
            raise ValueError(
                "external actuator torque requires six measured wing-axis labels"
            )
        actuator_torque = _finite_numeric_array(
            "external_actuator_torque_n_m",
            result.external_actuator_torque_n_m,
        )
        if actuator_torque.shape != (sample_count, 6):
            raise ValueError(
                "external_actuator_torque_n_m must have shape (time, 6)"
            )
    physics_time_s = result.physics_time_s
    physics_actuator_torque = result.external_actuator_torque_physics_n_m
    if (result.external_actuator_torque_n_m is None) != (
        physics_actuator_torque is None
    ):
        raise ValueError(
            "logged and physics-rate external actuator torque must be supplied together"
        )
    requires_physics_clock = (
        result.ground_contact_transition_point_count is not None
        or physics_actuator_torque is not None
        or measured_angle_physics is not None
        or aerodynamic_force_physics is not None
    )
    if requires_physics_clock != (physics_time_s is not None):
        raise ValueError(
            "physics-rate external telemetry requires exactly one physics_time_s array"
        )
    if physics_time_s is not None:
        physics_time = _finite_numeric_array("physics_time_s", physics_time_s)
        expected_physics_time = (
            np.arange(physics_steps + 1, dtype=float) * physics_dt_s
        )
        if physics_time.shape != (physics_steps + 1,) or not np.allclose(
            physics_time, expected_physics_time, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                "physics_time_s must exactly follow the configured physics clock"
            )
    if physics_actuator_torque is not None:
        physics_torque = _finite_numeric_array(
            "external_actuator_torque_physics_n_m", physics_actuator_torque
        )
        if physics_torque.shape != (physics_steps + 1, 6):
            raise ValueError(
                "external_actuator_torque_physics_n_m must have shape "
                "(physics_time, 6)"
            )
        logged_indices = np.rint(time_s / physics_dt_s).astype(np.int64)
        if not np.array_equal(
            np.asarray(result.external_actuator_torque_n_m),
            physics_torque[logged_indices],
        ):
            raise ValueError(
                "logged actuator torque does not match the physics-rate trace"
            )
        expected_peak_torque = float(np.max(np.abs(physics_torque), initial=0.0))
        reported_peak_torque = diagnostics.metrics.get(
            "maximum_external_actuator_torque_n_m"
        )
        if reported_peak_torque is None or not np.isclose(
            float(reported_peak_torque),
            expected_peak_torque,
            rtol=0.0,
            atol=1.0e-18,
        ):
            raise ValueError(
                "maximum external actuator torque metric does not match the "
                "physics-rate trace"
            )

    core_names = tuple(name for name, _, _, _ in _output_arrays(result, commands))
    normalized_supplemental = _validate_supplemental_arrays(
        supplemental_arrays, core_names
    )
    _strict_json_preflight(
        "episode configuration", _configuration_payload(config, commands)
    )
    _strict_json_preflight(
        "episode diagnostics",
        {
            "warnings": list(diagnostics.warnings),
            "metrics": dict(diagnostics.metrics),
            "physics_provenance": dict(diagnostics.physics_provenance),
        },
    )
    return normalized_supplemental


def write_episode_artifact(
    output_dir: Path,
    scenario: str,
    config: FlightSimulationConfig,
    result: FlightEpisodeOutput,
    *,
    evidence_path: Optional[Path] = None,
    chunk_samples: int = 256,
    created_at_utc: Optional[str] = None,
    supplemental_arrays: Sequence[Tuple[str, np.ndarray, str, str]] = (),
    identity_context: Optional[Mapping[str, Any]] = None,
    pipeline_manifest: Optional[Mapping[str, Any]] = None,
    web_replay: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Validate, stage, and atomically commit one immutable run artifact."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                "immutable scientific artifact target is not empty at {}; choose a new run directory".format(
                    output_dir
                )
            )
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("scenario must be a non-empty string")
    if (
        isinstance(chunk_samples, bool)
        or not isinstance(chunk_samples, (int, np.integer))
        or int(chunk_samples) <= 0
    ):
        raise ValueError("chunk_samples must be a positive integer")
    chunk_samples = int(chunk_samples)
    commands = (
        default_motor_commands()
        if config.motor_commands is None
        else tuple(config.motor_commands)
    )
    normalized_supplemental = _validate_episode_artifact_inputs(
        config, result, commands, supplemental_arrays
    )
    _strict_json_preflight("identity_context", dict(identity_context or {}))
    pipeline_payload: Optional[Mapping[str, Any]] = None
    if pipeline_manifest is not None:
        _strict_json_preflight("pipeline_manifest", dict(pipeline_manifest))
        pipeline_payload = json.loads(
            json.dumps(dict(pipeline_manifest), sort_keys=True, allow_nan=False)
        )

    graph_path = default_seed_graph_path() if evidence_path is None else Path(evidence_path)
    graph = load_evidence_graph(graph_path)
    coverage = provisional_coverage(graph)
    configuration = _configuration_payload(config, commands)
    locked_model_hashes = dict(model_hashes())
    evidence_sha256 = sha256_file(graph_path)
    run_id = episode_run_id(
        scenario,
        config,
        evidence_path=graph_path,
        identity_context=identity_context,
    )
    web_replay_receipt: Optional[Mapping[str, Any]] = None
    if web_replay is not None:
        if web_replay.get("schema_version") != WEB_REPLAY_SCHEMA_VERSION:
            raise ValueError("web replay has an unsupported schema_version")
        if web_replay.get("source_run_id") != run_id:
            raise ValueError("web replay source_run_id does not match the run identity")
        circular_fields = tuple(
            field
            for field in WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
            if field in web_replay
        )
        if circular_fields:
            raise ValueError(
                "web replay artifact-reference fields must be added only after "
                "the immutable run manifest exists: %s"
                % ", ".join(circular_fields)
            )
        web_replay_receipt = {
            "schema_version": WEB_REPLAY_PROJECTION_SCHEMA_VERSION,
            "sha256": web_replay_projection_sha256(web_replay),
            "canonicalization": WEB_REPLAY_PROJECTION_CANONICALIZATION,
            "excluded_top_level_fields": list(
                WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
            ),
        }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=str(output_dir.parent), prefix=".%s.staging-" % output_dir.name
        )
    )
    try:
        arrays: Dict[str, Any] = {}
        for name, values, unit, provenance in _output_arrays(result, commands):
            arrays[name] = _write_chunked_array(
                staging_dir, name, values, unit, chunk_samples, provenance
            )
        for name, values, unit, provenance in normalized_supplemental:
            arrays[name] = _write_chunked_array(
                staging_dir,
                name,
                values,
                unit,
                chunk_samples,
                provenance,
            )

        wing_axis_labels = list(result.measured_wing_joint_order)
        for name in (
            "measured_wing_joint_angle_rad",
            "measured_wing_joint_velocity_rad_s",
            "external_actuator_torque_n_m",
            "external_actuator_torque_physics_n_m",
            "measured_wing_joint_angle_physics_rad",
            "measured_wing_joint_velocity_physics_rad_s",
        ):
            if name in arrays:
                arrays[name] = dict(
                    arrays[name], axis_labels=wing_axis_labels
                )
        if "external_actuator_torque_physics_n_m" in arrays:
            arrays["external_actuator_torque_physics_n_m"] = dict(
                arrays["external_actuator_torque_physics_n_m"],
                sample_semantics=(
                    "sample 0 is the reset value; sample i>=1 is the six-axis "
                    "MuJoCo actuator torque used for the physics transition "
                    "ending at physics_time_s[i]"
                ),
            )
        if "external_actuator_torque_n_m" in arrays:
            arrays["external_actuator_torque_n_m"] = dict(
                arrays["external_actuator_torque_n_m"],
                sample_semantics=(
                    "logging-clock projection of the physics-rate torque; each "
                    "sample is the final physics transition ending at time_s[i], "
                    "not a logging-bin mean or peak"
                ),
            )
        for name in (
            "measured_wing_joint_angle_physics_rad",
            "measured_wing_joint_velocity_physics_rad_s",
        ):
            if name in arrays:
                arrays[name] = dict(
                    arrays[name],
                    sample_semantics=(
                        "sample i is the measured post-transition MuJoCo wing "
                        "state at physics_time_s[i]"
                    ),
                )
        for name in (
            "aerodynamic_force_body_physics_n",
            "aerodynamic_torque_body_physics_n_m",
        ):
            if name in arrays:
                arrays[name] = dict(
                    arrays[name],
                    axis_labels=["body_x", "body_y", "body_z"],
                    sample_semantics=(
                        "sample 0 is zero at reset; sample i>=1 is the root-total "
                        "MuJoCo fluid wrench used for the transition ending at "
                        "physics_time_s[i]"
                    ),
                )
        if "ground_contact_transition_point_count" in arrays:
            arrays["ground_contact_transition_point_count"] = dict(
                arrays["ground_contact_transition_point_count"],
                sample_semantics=GROUND_CONTACT_SAMPLE_SEMANTICS,
            )
        if "ground_contact_count" in arrays:
            arrays["ground_contact_count"] = dict(
                arrays["ground_contact_count"],
                sample_semantics=(
                    "logging-clock projection of "
                    "ground_contact_transition_point_count; brief contact-bearing "
                    "transitions between logging samples can be absent"
                ),
            )

        external_flybody = result.diagnostics.physics_backend == "flybody"
        manifest: Dict[str, Any] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "scenario": scenario,
            "created_at_utc": created_at_utc or _utc_now(),
            "episode_status": "complete",
            "validation_status": "exploratory",
            "scientific_label": "exploratory; calibrated-by-none",
            "source_kind": (
                FLYBODY_WORKER_SOURCE_KIND
                if external_flybody
                else REDUCED_ORDER_SOURCE_KIND
            ),
            "authority_notice": (
                FLYBODY_AUTHORITY_NOTICE if external_flybody else AUTHORITY_NOTICE
            ),
            "calibration": {
                "status": "none",
                "calibrated_by": [],
                "datasets": [],
            },
            "configuration": configuration,
            "identity_context": dict(identity_context or {}),
            "units_policy": "SI internally; units are repeated on every array descriptor",
            "model_hashes": locked_model_hashes,
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "physics_backend": result.diagnostics.physics_backend,
                "physics_provenance": dict(result.diagnostics.physics_provenance),
            },
            "licenses": {
                "fly_sensor2behavior": "repository license not declared",
                "numpy": "BSD-3-Clause",
            },
            "evidence": {
                "graph_id": graph.graph_id,
                "sha256": evidence_sha256,
                "coverage_status": "provisional",
                "coverage": [record.to_dict() for record in coverage],
            },
            "signal_provenance": _signal_provenance(commands),
            "arrays": arrays,
            "diagnostics": {
                "status": result.diagnostics.status.value,
                "exact_spike_count": result.diagnostics.exact_spike_count,
                "inferred_spike_count": result.diagnostics.inferred_spike_count,
                "physics_steps": result.diagnostics.physics_steps,
                "final_time_s": result.diagnostics.final_time_s,
                "integrated_aerodynamic_impulse_n_s": list(
                    result.diagnostics.integrated_aerodynamic_impulse_n_s
                ),
                "metrics": dict(result.diagnostics.metrics),
                "warnings": list(result.diagnostics.warnings),
                "physics_backend": result.diagnostics.physics_backend,
                "aerodynamic_owner": result.diagnostics.aerodynamic_owner,
            },
        }
        external_licenses = result.diagnostics.physics_provenance.get("licenses")
        if external_flybody and isinstance(external_licenses, Mapping):
            manifest["licenses"]["external_physics"] = dict(external_licenses)
        if pipeline_payload is not None:
            manifest["pipeline"] = pipeline_payload
        if web_replay_receipt is not None:
            manifest["web_replay_projection"] = web_replay_receipt
        manifest_path = staging_dir / "manifest.json"
        _json_dump(manifest_path, manifest)
        digest = sha256_file(manifest_path)
        sidecar = staging_dir / "manifest.sha256"
        sidecar.write_text("%s  manifest.json\n" % digest, encoding="ascii")

        # Re-check immediately before the atomic directory rename so a target
        # populated concurrently is never replaced.
        if output_dir.exists() and (
            not output_dir.is_dir() or any(output_dir.iterdir())
        ):
            raise FileExistsError(
                "immutable scientific artifact target became non-empty at {}".format(
                    output_dir
                )
            )
        os.replace(str(staging_dir), str(output_dir))
        return manifest
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def read_chunked_array(run_dir: Path, descriptor: Mapping[str, Any]) -> np.ndarray:
    """Load and verify one array descriptor from a scientific artifact."""

    chunks: List[np.ndarray] = []
    for chunk in descriptor["chunks"]:
        path = Path(run_dir) / str(chunk["path"])
        actual = sha256_file(path)
        if actual != chunk["sha256"]:
            raise ValueError("chunk checksum mismatch for {}".format(path))
        with path.open("rb") as handle:
            values = np.load(handle, allow_pickle=False)
        chunks.append(values)
    if not chunks:
        raise ValueError("array descriptor has no chunks")
    result = np.concatenate(chunks, axis=0)
    expected_shape = tuple(int(value) for value in descriptor["shape"])
    result = result.reshape(expected_shape)
    if _logical_array_hash(result) != descriptor["logical_npy_sha256"]:
        raise ValueError("logical array checksum mismatch")
    return result


def _quaternion_to_euler(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=float)
    w, x, y, z = (values[:, index] for index in range(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.column_stack((roll, pitch, yaw))


def _aggregate_muscle(result: FlightEpisodeOutput, side: str, names: Sequence[str]) -> np.ndarray:
    selected = [
        np.asarray(result.muscle_activation["%s:%s" % (side, name)], dtype=float)
        for name in names
        if "%s:%s" % (side, name) in result.muscle_activation
    ]
    if not selected:
        return np.zeros_like(result.time_s)
    return np.mean(np.vstack(selected), axis=0)


def _rolling_wing_envelope(stroke_rad: np.ndarray, time_s: np.ndarray, wingbeat_hz: float) -> np.ndarray:
    count = len(time_s)
    if count < 2:
        return np.zeros((count, 4), dtype=float)
    dt = float(np.median(np.diff(time_s)))
    half_window = max(1, int(math.ceil(0.5 / max(wingbeat_hz * dt, 1.0e-12))))
    output = np.empty((count, 4), dtype=float)
    for index in range(count):
        start = max(0, index - half_window)
        stop = min(count, index + half_window + 1)
        local = stroke_rad[start:stop]
        minimum = np.min(local, axis=0)
        maximum = np.max(local, axis=0)
        output[index, :2] = np.rad2deg(0.5 * (maximum - minimum))
        output[index, 2:] = np.rad2deg(0.5 * (maximum + minimum))
    return output


def _decimation_indices(time_s: np.ndarray, target_sample_rate_hz: float) -> np.ndarray:
    if target_sample_rate_hz <= 0.0:
        raise ValueError("target_sample_rate_hz must be positive")
    target_count = max(2, int(math.ceil(float(time_s[-1]) * target_sample_rate_hz)) + 1)
    target_count = min(len(time_s), target_count)
    indices = np.unique(np.rint(np.linspace(0, len(time_s) - 1, target_count)).astype(int))
    if indices[0] != 0 or indices[-1] != len(time_s) - 1:
        indices = np.unique(np.concatenate(([0], indices, [len(time_s) - 1])))
    return indices


def _web_circuit_values(scenario: str, config: FlightSimulationConfig, count: int) -> np.ndarray:
    # Visualization-only normalized states. They are declared illustrative in
    # every episode and must not be interpreted as simulated or recorded voltage.
    vch = 0.62 * config.vch_dch_factor
    dn_peak = max(config.dn_drives.values()) if config.dn_drives else 0.0
    dn = min(1.0, 0.25 + 0.65 * dn_peak) * config.vch_dch_factor
    if scenario == "gust":
        dn = 0.25
    row = np.array(
        [0.68, vch, 0.55 * config.vch_dch_factor, 0.48 * config.vch_dch_factor, dn, 0.30, 0.35],
        dtype=float,
    )
    return np.repeat(row[None, :], count, axis=0)


def _web_motor_timing(command: MotorCommand) -> str:
    value = command.signal_kind.value
    if value in ("exact_spikes", "inferred_rate", "seeded_synthetic_spikes"):
        return value
    raise ValueError("unsupported web motor timing semantics %r" % value)


def _web_muscle_class(command: MotorCommand) -> str:
    return {
        "asynchronous_power": "power",
        "steering": "steering",
        "tension": "tension",
    }[command.muscle_class.value]


def _individual_muscle_contract(
    result: FlightEpisodeOutput,
    commands: Sequence[MotorCommand],
) -> Tuple[List[Mapping[str, Any]], Mapping[str, MotorCommand]]:
    by_muscle: Dict[str, MotorCommand] = {}
    for command in commands:
        key = "%s:%s" % (command.side.value, command.muscle)
        if key in by_muscle:
            raise ValueError("web replay has multiple commands for individual muscle %s" % key)
        by_muscle[key] = command
    channels = []
    for muscle_id in sorted(result.muscle_activation):
        command = by_muscle.get(muscle_id)
        side, _, muscle = muscle_id.partition(":")
        channels.append(
            {
                "id": muscle_id,
                "label": "%s %s" % (muscle, side.capitalize()),
                "side": side if side in ("left", "right") else "unknown",
                "muscle_class": (
                    "unknown" if command is None else _web_muscle_class(command)
                ),
                "motor_neuron": None if command is None else command.neuron_id,
                "timing_semantics": (
                    "unavailable" if command is None else _web_motor_timing(command)
                ),
                "origin": "inferred",
                "confidence": "low",
                "provenance": (
                    "uncalibrated individual muscle state from the reduced-order model"
                    if command is None
                    else command.provenance
                ),
            }
        )
        if channels[-1]["motor_neuron"] is None:
            del channels[-1]["motor_neuron"]
    return channels, by_muscle


def _causal_hold_value(
    availability_times_s: np.ndarray,
    values: np.ndarray,
    time_s: float,
) -> Optional[float]:
    """Return the latest value available at ``time_s`` without future reads."""

    if len(availability_times_s) != len(values):
        raise ValueError("trace availability and value arrays must have equal length")
    if len(values) == 0:
        return None
    index = int(np.searchsorted(availability_times_s, time_s + 1.0e-12, side="right") - 1)
    if index < 0:
        return None
    return float(values[index])


def _display_normalization(signal_kind: str, value: float) -> float:
    """Map a raw attached value to a bounded animation intensity.

    Raw values and their units remain present in ``neural_signals`` and
    ``pathway_values``.  This transform is only for the pathway glow.
    """

    if signal_kind == "voltage":
        return float(np.clip((value + 0.070) / 0.030, 0.0, 1.0))
    if signal_kind in ("firing_rate", "inferred_rate"):
        return float(np.clip(value / 200.0, 0.0, 1.0))
    if signal_kind == "cumulative_spike_count":
        return float(np.clip(value / 5.0, 0.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def _pipeline_trace_contract(
    circuit_trace: CircuitOutputTrace,
    bridge_result: CausalBridgeResult,
    retinal_frames: Sequence[RetinalFrame],
    result: FlightEpisodeOutput,
    frame_indices: np.ndarray,
) -> Mapping[str, Any]:
    """Build raw, causal replay channels from an executed NOD1 pipeline.

    The contract carries every exported NOD1 voltage plus downstream DN, MN,
    and individual-muscle activation.  It never substitutes unavailable
    LLPC1/T4/T5 signals.  ``circuit`` remains a seven-element normalized UI
    envelope for backwards compatibility; ``neural_signals`` is the raw,
    unit-annotated source for scientific inspection.
    """

    channels: List[Dict[str, Any]] = []
    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    bridge_muscles = {
        "%s:%s" % (channel.side.value, channel.muscle)
        for channel in bridge_result.wing_motor.channels
    }

    def add_channel(
        *,
        channel_id: str,
        label: str,
        stage: str,
        entity_id: str,
        cell_type: str,
        side: str,
        signal_kind: str,
        unit: str,
        origin: str,
        confidence: str,
        provenance: str,
        availability_times_s: Sequence[float],
        values: Sequence[float],
        causal_role: str,
    ) -> None:
        if channel_id in series:
            raise ValueError("duplicate neural replay channel id %s" % channel_id)
        availability = np.asarray(availability_times_s, dtype=float)
        data = np.asarray(values, dtype=float)
        if len(availability) != len(data):
            raise ValueError("neural replay channel %s has inconsistent timing" % channel_id)
        if len(availability) and np.any(np.diff(availability) < -1.0e-12):
            raise ValueError("neural replay channel %s availability must be monotonic" % channel_id)
        if not np.all(np.isfinite(availability)) or not np.all(np.isfinite(data)):
            raise ValueError("neural replay channel %s must be finite" % channel_id)
        channels.append(
            {
                "id": channel_id,
                "label": label,
                "stage": stage,
                "entity_id": entity_id,
                "cell_type": cell_type,
                "side": side,
                "signal_kind": signal_kind,
                "unit": unit,
                "origin": origin,
                "confidence": confidence,
                "provenance": provenance,
                "causal_role": causal_role,
            }
        )
        series[channel_id] = (availability, data)

    frames = tuple(retinal_frames)
    if frames:
        retinal_origin_s = float(frames[0].measurement_time_s)
        eye = frames[0].eye_side.value
        if any(frame.eye_side.value != eye for frame in frames):
            raise ValueError("one replay retinal stream cannot mix eye-side identities")
        add_channel(
            channel_id="retina:%s:mean" % eye,
            label="Retinal mean luminance (%s)" % eye,
            stage="retina",
            entity_id="retinal-sampler-%s" % eye,
            cell_type="retinal_sample",
            side=eye if eye in ("left", "right") else "bilateral",
            signal_kind="normalized_luminance",
            unit="1",
            origin="simulated",
            confidence="low",
            provenance=(
                "finite-exposure retinal samples supplied to the reduced NOD1 pipeline; "
                "not a T4/T5 neural recording"
            ),
            availability_times_s=[
                frame.availability_time_s - retinal_origin_s for frame in frames
            ],
            values=[float(np.mean(frame.samples)) for frame in frames],
            causal_role="visual_input",
        )

    for signal in circuit_trace.signals:
        channel_id = "circuit:%s" % signal.neuron.entity_id
        cell_type = signal.neuron.cell_type or "untyped"
        side = signal.neuron.anatomical_side.value
        if signal.signal_kind is CircuitSignalKind.SPIKE_EVENTS:
            availability = signal.availability_times_s or signal.spike_times_s
            values = tuple(float(index + 1) for index in range(len(signal.spike_times_s)))
            signal_kind = "cumulative_spike_count"
            unit = "1"
        else:
            availability = signal.availability_times_s or circuit_trace.sample_times_s
            values = signal.values
            signal_kind = signal.signal_kind.value
            unit = signal.unit
        add_channel(
            channel_id=channel_id,
            label="%s %s · %s" % (cell_type, side, signal.neuron.entity_id),
            stage="circuit",
            entity_id=signal.neuron.entity_id,
            cell_type=cell_type,
            side=side,
            signal_kind=signal_kind,
            unit=unit,
            origin=signal.origin.value,
            confidence=signal.confidence.level.value,
            provenance=signal.provenance.method,
            availability_times_s=availability,
            values=values,
            causal_role="circuit_output",
        )

    for channel in bridge_result.descending.channels:
        channel_id = "dn:%s:%s" % (channel.dn_type, channel.side.value)
        add_channel(
            channel_id=channel_id,
            label="%s %s" % (channel.dn_type, channel.side.value),
            stage="descending",
            entity_id="%s-%s" % (channel.dn_type, channel.side.value),
            cell_type=channel.dn_type,
            side=channel.side.value,
            signal_kind="inferred_rate",
            unit="Hz",
            origin="inferred",
            confidence="low",
            provenance=(
                "%s; functional gain %.6g Hz is a model parameter, not a synapse count"
                % (channel.mapping_method, channel.functional_gain_hz)
            ),
            availability_times_s=[sample.availability_time_s for sample in channel.samples],
            values=[sample.rate_hz for sample in channel.samples],
            causal_role="circuit_driven",
        )

    for channel in bridge_result.wing_motor.channels:
        channel_id = "mn:%s:%s" % (channel.motor_neuron, channel.side.value)
        if channel.rate_samples:
            availability = [sample.availability_time_s for sample in channel.rate_samples]
            values = [sample.rate_hz for sample in channel.rate_samples]
            signal_kind = "inferred_rate"
            unit = "Hz"
        else:
            availability = [event.availability_time_s for event in channel.events]
            values = [float(index + 1) for index in range(len(channel.events))]
            signal_kind = "cumulative_spike_count"
            unit = "1"
        add_channel(
            channel_id=channel_id,
            label="%s %s → %s" % (
                channel.motor_neuron,
                channel.side.value,
                channel.muscle,
            ),
            stage="motor",
            entity_id="%s-%s" % (channel.motor_neuron, channel.side.value),
            cell_type=channel.motor_neuron,
            side=channel.side.value,
            signal_kind=signal_kind,
            unit=unit,
            origin=("unspecified" if channel.signal_semantics.value == "exact_spikes" else "inferred"),
            confidence="low",
            provenance=(
                "causal exploratory VNC bridge; phase-tagged generated events retain "
                "seeded-synthetic semantics"
            ),
            availability_times_s=availability,
            values=values,
            causal_role="circuit_driven",
        )

    for muscle_id in sorted(result.muscle_activation):
        side, _, muscle = muscle_id.partition(":")
        circuit_driven = muscle_id in bridge_muscles
        add_channel(
            channel_id="muscle:%s" % muscle_id,
            label="%s %s activation" % (muscle, side),
            stage="muscle",
            entity_id=muscle_id,
            cell_type=muscle,
            side=side,
            signal_kind="activation",
            unit="1",
            origin="simulated",
            confidence="low",
            provenance=(
                "uncalibrated muscle state downstream of the NOD1/DNp26 bridge"
                if circuit_driven
                else "explicit airborne baseline drive; not inferred from the visual circuit"
            ),
            availability_times_s=result.time_s,
            values=result.muscle_activation[muscle_id],
            causal_role=("circuit_driven" if circuit_driven else "airborne_baseline_drive"),
        )

    stage_channel_ids: Tuple[Tuple[str, ...], ...] = (
        tuple(channel["id"] for channel in channels if channel["stage"] == "retina"),
        tuple(
            channel["id"]
            for channel in channels
            if channel["stage"] == "circuit" and channel["cell_type"] in ("vCH", "DCH")
        ),
        tuple(
            channel["id"]
            for channel in channels
            if channel["stage"] == "circuit" and channel["cell_type"] == "LLPC1"
        ),
        tuple(
            channel["id"]
            for channel in channels
            if channel["stage"] == "circuit" and channel["cell_type"] == "NOD1"
        ),
        tuple(channel["id"] for channel in channels if channel["stage"] == "descending"),
        tuple(channel["id"] for channel in channels if channel["stage"] == "motor"),
        tuple(
            channel["id"]
            for channel in channels
            if channel["stage"] == "muscle" and channel["causal_role"] == "circuit_driven"
        ),
    )
    stage_labels = ("retinal input", "vCH/DCH", "LLPC1", "NOD1", "DNp26", "wing MN", "wing muscle")
    by_id = {str(channel["id"]): channel for channel in channels}
    pathway_channels = []
    for index, (label, source_ids) in enumerate(zip(stage_labels, stage_channel_ids)):
        units = {str(by_id[channel_id]["unit"]) for channel_id in source_ids}
        pathway_channels.append(
            {
                "index": index,
                "label": label,
                "status": "attached" if source_ids else "unavailable",
                "source_channel_ids": list(source_ids),
                "aggregation": "mean of currently available raw channels",
                "raw_unit": next(iter(units)) if len(units) == 1 else None,
                "display_normalization": (
                    "voltage: clip((V + 0.070)/0.030); rate: clip(Hz/200); other: clip(value)"
                    if source_ids
                    else "none"
                ),
            }
        )

    sampled_frames = []
    normalized_pathway = []
    raw_pathway = []
    for result_index in frame_indices:
        time_s = float(result.time_s[int(result_index)])
        raw_values: Dict[str, float] = {}
        for channel_id, (availability, values) in series.items():
            value = _causal_hold_value(availability, values, time_s)
            if value is not None:
                raw_values[channel_id] = value
        stage_raw: List[Optional[float]] = []
        stage_normalized: List[float] = []
        for source_ids in stage_channel_ids:
            attached = [raw_values[channel_id] for channel_id in source_ids if channel_id in raw_values]
            if not attached:
                stage_raw.append(None)
                stage_normalized.append(0.0)
                continue
            raw = float(np.mean(attached))
            stage_raw.append(raw)
            signal_kinds = {
                str(by_id[channel_id]["signal_kind"])
                for channel_id in source_ids
                if channel_id in raw_values
            }
            kind = next(iter(signal_kinds)) if len(signal_kinds) == 1 else "activation"
            stage_normalized.append(_display_normalization(kind, raw))
        sampled_frames.append(raw_values)
        normalized_pathway.append(stage_normalized)
        raw_pathway.append(stage_raw)

    return {
        "channels": channels,
        "pathway_channels": pathway_channels,
        "frame_signals": sampled_frames,
        "normalized_pathway": np.asarray(normalized_pathway, dtype=float),
        "raw_pathway": raw_pathway,
    }


def _ground_contact_summary(
    config: FlightSimulationConfig,
    result: FlightEpisodeOutput,
) -> Optional[Mapping[str, Any]]:
    """Validate and summarize full-step external ground-contact telemetry."""

    logged_contact_count = result.ground_contact_count
    transition_contact_count = result.ground_contact_transition_point_count
    physics_time_s = result.physics_time_s
    if (
        logged_contact_count is None
        and transition_contact_count is None
        and physics_time_s is None
    ):
        return None
    if (
        logged_contact_count is None
        or transition_contact_count is None
        or physics_time_s is None
    ):
        raise ValueError(
            "ground-contact telemetry requires logged, transition-rate, and physics-time arrays"
        )
    logged_contact_count = np.asarray(logged_contact_count)
    transition_contact_count = np.asarray(transition_contact_count)
    physics_time_s = np.asarray(physics_time_s, dtype=float)
    expected_physics_samples = result.diagnostics.physics_steps + 1
    if (
        logged_contact_count.shape != (len(result.time_s),)
        or logged_contact_count.dtype.kind not in "iu"
        or np.any(logged_contact_count < 0)
        or transition_contact_count.shape != (expected_physics_samples,)
        or transition_contact_count.dtype.kind not in "iu"
        or np.any(transition_contact_count < 0)
        or physics_time_s.shape != (expected_physics_samples,)
        or not np.all(np.isfinite(physics_time_s))
    ):
        raise ValueError(
            "ground-contact telemetry arrays have invalid shape, dtype, or values"
        )
    expected_physics_time_s = np.arange(expected_physics_samples, dtype=float) * float(
        config.physics_dt_s
    )
    tolerance_s = max(1.0e-12, 1.0e-9 * float(config.duration_s))
    if not np.allclose(
        physics_time_s, expected_physics_time_s, rtol=0.0, atol=tolerance_s
    ):
        raise ValueError("physics_time_s does not follow the configured physics clock")
    logged_indices = np.rint(
        np.asarray(result.time_s, dtype=float) / float(config.physics_dt_s)
    ).astype(np.int64)
    if not np.array_equal(
        logged_contact_count, transition_contact_count[logged_indices]
    ):
        raise ValueError(
            "logged ground-contact projection does not match transition telemetry"
        )
    contact_metrics = result.diagnostics.metrics
    required_contact_metrics = (
        "initial_ground_contact_count",
        "ground_contact_transition_count",
        "maximum_ground_contact_count",
        "ground_contact_occurred",
        "first_ground_contact_transition_start_s_or_duration_s",
        "first_ground_contact_transition_end_s_or_duration_s",
    )
    if any(key not in contact_metrics for key in required_contact_metrics):
        raise ValueError(
            "ground-contact time series requires complete full-step diagnostics"
        )
    contact_count_scalars = tuple(
        float(contact_metrics[key])
        for key in (
            "initial_ground_contact_count",
            "ground_contact_transition_count",
            "maximum_ground_contact_count",
        )
    )
    contact_occurred_scalar = float(contact_metrics["ground_contact_occurred"])
    if (
        not all(
            np.isfinite(value) and value.is_integer()
            for value in contact_count_scalars
        )
        or contact_occurred_scalar not in (0.0, 1.0)
    ):
        raise ValueError(
            "ground-contact count and occurrence diagnostics must be integral"
        )
    initial_contact_count = int(contact_count_scalars[0])
    contact_transition_count = int(contact_count_scalars[1])
    maximum_contact_count = int(contact_count_scalars[2])
    contact_occurred = bool(contact_occurred_scalar)
    first_transition_start_or_duration_s = float(
        contact_metrics[
            "first_ground_contact_transition_start_s_or_duration_s"
        ]
    )
    first_transition_end_or_duration_s = float(
        contact_metrics["first_ground_contact_transition_end_s_or_duration_s"]
    )
    transition_indices = np.flatnonzero(transition_contact_count[1:]) + 1
    expected_initial_count = int(transition_contact_count[0])
    expected_transition_count = int(np.count_nonzero(transition_contact_count[1:]))
    expected_maximum_count = int(np.max(transition_contact_count, initial=0))
    expected_occurred = expected_maximum_count > 0
    if expected_initial_count > 0:
        expected_first_start_s = 0.0
        expected_first_end_s = 0.0
    elif len(transition_indices):
        first_index = int(transition_indices[0])
        expected_first_start_s = float(physics_time_s[first_index - 1])
        expected_first_end_s = float(physics_time_s[first_index])
    else:
        expected_first_start_s = float(config.duration_s)
        expected_first_end_s = float(config.duration_s)
    if (
        initial_contact_count < 0
        or contact_transition_count < 0
        or contact_transition_count > result.diagnostics.physics_steps
        or initial_contact_count != expected_initial_count
        or contact_transition_count != expected_transition_count
        or maximum_contact_count != expected_maximum_count
        or contact_occurred != expected_occurred
        or not np.isfinite(first_transition_start_or_duration_s)
        or not np.isfinite(first_transition_end_or_duration_s)
        or abs(first_transition_start_or_duration_s - expected_first_start_s)
        > tolerance_s
        or abs(first_transition_end_or_duration_s - expected_first_end_s)
        > tolerance_s
    ):
        raise ValueError("ground-contact full-step diagnostics are inconsistent")
    return {
        "telemetry": GROUND_CONTACT_TELEMETRY_KIND,
        "sample_semantics": GROUND_CONTACT_SAMPLE_SEMANTICS,
        "initial_count": initial_contact_count,
        "occurred": contact_occurred,
        "first_transition_start_s": (
            expected_first_start_s if contact_occurred else None
        ),
        "first_transition_end_s": (
            expected_first_end_s if contact_occurred else None
        ),
        "transition_count": contact_transition_count,
        "maximum_count": maximum_contact_count,
    }


def episode_to_web_replay(
    scenario: str,
    config: FlightSimulationConfig,
    result: FlightEpisodeOutput,
    *,
    target_sample_rate_hz: float = 200.0,
    evidence_graph: Optional[EvidenceGraph] = None,
    circuit_trace: Optional[CircuitOutputTrace] = None,
    bridge_result: Optional[CausalBridgeResult] = None,
    retinal_frames: Sequence[RetinalFrame] = (),
    neural_model_scope: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Build the exact browser contract declared in ``web/src/types.ts``."""

    definition = SCENARIOS[scenario]
    frequency_hz = float(result.diagnostics.metrics["mean_wingbeat_frequency_hz"])
    indices = _decimation_indices(result.time_s, target_sample_rate_hz)
    orientation = _quaternion_to_euler(result.quaternion_body_to_world)
    force_world = np.asarray(
        [
            quaternion_to_matrix(quaternion).dot(force)
            for quaternion, force in zip(
                result.quaternion_body_to_world,
                result.aerodynamic_force_body_n,
            )
        ],
        dtype=float,
    )
    moment_world = np.asarray(
        [
            quaternion_to_matrix(quaternion).dot(moment)
            for quaternion, moment in zip(
                result.quaternion_body_to_world,
                result.aerodynamic_torque_body_n_m,
            )
        ],
        dtype=float,
    )
    measured_wing = result.measured_wing_joint_angle_rad
    whole_fly_com = result.whole_fly_com_position_world_m
    ground_contact_count = result.ground_contact_count
    ground_contact_summary = _ground_contact_summary(config, result)
    wingbeat_inspection: Optional[Mapping[str, Any]] = None
    if (
        result.physics_time_s is not None
        and result.measured_wing_joint_angle_physics_rad is not None
        and result.external_actuator_torque_physics_n_m is not None
    ):
        inspection_time = np.asarray(result.physics_time_s, dtype=float)
        inspection_wing = np.asarray(
            result.measured_wing_joint_angle_physics_rad, dtype=float
        )
        inspection_torque = np.asarray(
            result.external_actuator_torque_physics_n_m, dtype=float
        )
        if (
            inspection_time.ndim != 1
            or inspection_wing.shape != (len(inspection_time), 6)
            or inspection_torque.shape != (len(inspection_time), 6)
            or len(inspection_time) < 2
            or not np.all(np.isfinite(inspection_time))
            or not np.all(np.isfinite(inspection_wing))
            or not np.all(np.isfinite(inspection_torque))
        ):
            raise ValueError("physics-rate wingbeat inspection arrays are invalid")
        inspection_end_s = min(0.060, float(config.duration_s))
        inspection_count = int(
            np.searchsorted(
                inspection_time, inspection_end_s + 1.0e-12, side="right"
            )
        )
        inspection_slice = slice(0, inspection_count)
        inspection_time = inspection_time[inspection_slice]
        inspection_wing = inspection_wing[inspection_slice]
        inspection_torque = inspection_torque[inspection_slice]
        wingbeat_inspection = {
            "sample_rate_hz": float(1.0 / np.median(np.diff(inspection_time))),
            "start_s": float(inspection_time[0]),
            "end_s": float(inspection_time[-1]),
            "time_s": [float(value) for value in inspection_time],
            "wing_joint_order": list(result.measured_wing_joint_order),
            "measured_wing_joint_angle_rad": [
                [float(value) for value in row] for row in inspection_wing
            ],
            "external_actuator_torque_n_m": [
                [float(value) for value in row] for row in inspection_torque
            ],
        }
    if whole_fly_com is not None:
        whole_fly_com = np.asarray(whole_fly_com, dtype=float)
        if whole_fly_com.shape != (len(result.time_s), 3):
            raise ValueError("whole-fly COM must have shape (time, 3)")
    if ground_contact_count is not None:
        ground_contact_count = np.asarray(ground_contact_count)
    if measured_wing is not None:
        measured_wing = np.asarray(measured_wing, dtype=float)
        if measured_wing.shape != (len(result.time_s), 6):
            raise ValueError("measured FlyBody wing angles must have shape (time, 6)")
        measured_order = tuple(result.measured_wing_joint_order)
        if measured_order != REVIEWED_FLYBODY_WING_AXIS_ORDER:
            raise ValueError(
                "measured FlyBody wing order must equal the reviewed FlyGym 2.1.0 "
                "yaw-roll-pitch axis order"
            )
        yaw_indices = (0, 3)
        envelope_source = measured_wing[:, yaw_indices]
        wing_kinematics_source = "measured_flybody_yaw_axes"
    else:
        envelope_source = result.wing_stroke_rad
        wing_kinematics_source = "virtual_hinge_desired_stroke"
    envelope = _rolling_wing_envelope(
        envelope_source, result.time_s, frequency_hz
    )
    power_left = _aggregate_muscle(result, "left", ("DLM", "DVM"))
    power_right = _aggregate_muscle(result, "right", ("DLM", "DVM"))
    steering_left = _aggregate_muscle(result, "left", ("b1",))
    steering_right = _aggregate_muscle(result, "right", ("b1",))
    tension = 0.5 * (
        _aggregate_muscle(result, "left", ("tp1",))
        + _aggregate_muscle(result, "right", ("tp1",))
    )
    muscles = np.column_stack((power_left, power_right, steering_left, steering_right, tension))
    commands = (
        default_motor_commands()
        if config.motor_commands is None
        else config.motor_commands
    )
    individual_channels, _ = _individual_muscle_contract(result, commands)
    if (circuit_trace is None) != (bridge_result is None):
        raise ValueError("circuit_trace and bridge_result must be supplied together")
    if circuit_trace is not None and neural_model_scope is None:
        raise ValueError("attached circuit traces require an explicit neural_model_scope")
    attached_trace = None
    if circuit_trace is None:
        circuit = _web_circuit_values(scenario, config, len(result.time_s))
    else:
        attached_trace = _pipeline_trace_contract(
            circuit_trace,
            bridge_result,  # type: ignore[arg-type]
            retinal_frames,
            result,
            indices,
        )
        circuit = attached_trace["normalized_pathway"]
    if evidence_graph is None:
        evidence_graph = load_evidence_graph(default_seed_graph_path())
    coverage_records = provisional_coverage(evidence_graph)
    unresolved = np.mean(
        tuple(1.0 - record.resolved_fraction for record in coverage_records)
    )
    evidence_incompleteness = min(1.0, 0.55 + 0.45 * float(unresolved))

    frames = []
    for frame_index, index in enumerate(indices):
        individual = {}
        for channel in individual_channels:
            muscle_id = str(channel["id"])
            state = {
                "activation": float(result.muscle_activation[muscle_id][index]),
            }
            if muscle_id in result.muscle_force_n:
                state["force_n"] = float(result.muscle_force_n[muscle_id][index])
            if muscle_id in result.muscle_phase_effect:
                state["phase_effect"] = float(
                    result.muscle_phase_effect[muscle_id][index]
                )
            if muscle_id in result.muscle_work_j:
                state["work_j"] = float(result.muscle_work_j[muscle_id][index])
            individual[muscle_id] = state
        frame = {
                "t": float(result.time_s[index]),
                "position_m": [
                    float(value)
                    for value in (
                        whole_fly_com[index]
                        if whole_fly_com is not None
                        else result.position_world_m[index]
                    )
                ],
                "orientation_rad": [float(value) for value in orientation[index]],
                "velocity_m_s": [float(value) for value in result.velocity_world_m_s[index]],
                "wing_envelope_deg": [float(value) for value in envelope[index]],
                "force_world_n": [float(value) for value in force_world[index]],
                "moment_world_n_m": [float(value) for value in moment_world[index]],
                "circuit": [
                    float(value)
                    for value in (
                        circuit[index] if attached_trace is None else circuit[frame_index]
                    )
                ],
                "muscles": [float(np.clip(value, 0.0, 1.0)) for value in muscles[index]],
                "evidence_incompleteness": evidence_incompleteness,
                "individual_muscles": individual,
            }
        if whole_fly_com is not None:
            frame["root_position_m"] = [
                float(value) for value in result.position_world_m[index]
            ]
        if ground_contact_count is not None:
            frame["ground_contact_count"] = int(ground_contact_count[index])
        if measured_wing is not None:
            assert result.measured_wing_joint_velocity_rad_s is not None
            frame["desired_wing_stroke_rad"] = [
                float(value) for value in result.wing_stroke_rad[index]
            ]
            frame["measured_wing_joint_angle_rad"] = [
                float(value) for value in measured_wing[index]
            ]
            frame["measured_wing_joint_velocity_rad_s"] = [
                float(value)
                for value in result.measured_wing_joint_velocity_rad_s[index]
            ]
        if attached_trace is not None:
            frame["neural_signals"] = attached_trace["frame_signals"][frame_index]
            frame["pathway_values"] = attached_trace["raw_pathway"][frame_index]
        frames.append(frame)
    actual_rate = 1.0 / float(np.median(np.diff(np.asarray([frame["t"] for frame in frames]))))
    perturbation = definition.condition
    replay: Dict[str, Any] = {
        "schema_version": WEB_REPLAY_SCHEMA_VERSION,
        "id": scenario,
        "label": definition.label,
        "status": "exploratory",
        "source_kind": (
            FLYBODY_WORKER_SOURCE_KIND
            if result.diagnostics.physics_backend == "flybody"
            else REDUCED_ORDER_SOURCE_KIND
        ),
        "authority_notice": (
            FLYBODY_AUTHORITY_NOTICE
            if result.diagnostics.physics_backend == "flybody"
            else AUTHORITY_NOTICE
        ),
        "duration_s": float(config.duration_s),
        "envelope_sample_rate_hz": actual_rate,
        "physics_timestep_s": float(config.physics_dt_s),
        "wingbeat_hz": frequency_hz,
        "stimulus": "declared open-loop figure-ground pathway demonstration",
        "perturbation": perturbation,
        "circuit_activity_status": (
            "illustrative pathway display; no upstream neural trace was ingested"
            if attached_trace is None
            else (
                "pipeline-derived raw circuit, DN, MN, and muscle channels attached; "
                "normalized pathway glow is display-only and unavailable stages remain empty"
            )
        ),
        "evidence_incompleteness_status": (
            "coverage-derived display proxy; not a trajectory posterior or model-reported uncertainty"
        ),
        "physics_backend": result.diagnostics.physics_backend,
        "body_state_reference": (
            "whole-fly articulated subtree COM for display; root/thorax retained separately"
            if result.diagnostics.physics_backend == "flybody"
            and whole_fly_com is not None
            else (
                "FlyBody root/thorax frame; whole-fly COM telemetry unavailable"
                if result.diagnostics.physics_backend == "flybody"
                else "reduced rigid-body reference"
            )
        ),
        "contact_telemetry_status": (
            GROUND_CONTACT_TELEMETRY_KIND
            if ground_contact_count is not None
            else "unavailable"
        ),
        "wing_kinematics_source": wing_kinematics_source,
        "measured_wing_joint_order": list(result.measured_wing_joint_order),
        "signal_semantics": {
            "neural_origin": "unavailable" if attached_trace is None else "simulated",
            "motor_timing": (
                "inferred_rate" if attached_trace is None else "seeded_synthetic_spikes"
            ),
            "mechanics_origin": "simulated",
            "confidence": "low",
            "provenance": (
                "No upstream circuit trace was ingested; motor, muscle, and mechanics "
                "signals are uncalibrated reduced-order model outputs."
                if attached_trace is None
                else (
                    "Raw SI circuit outputs were causally sampled at declared availability "
                    "times; DN/MN rates, generated motor events, muscles, and mechanics remain "
                    "uncalibrated model inference."
                )
            ),
        },
        "individual_muscle_channels": individual_channels,
        "units": {
            "position_m": "m",
            "root_position_m": "m",
            "ground_contact_count": "1",
            "orientation_rad": "rad",
            "velocity_m_s": "m s^-1",
            "wing_envelope_deg": "degree",
            "desired_wing_stroke_rad": "rad",
            "measured_wing_joint_angle_rad": "rad",
            "measured_wing_joint_velocity_rad_s": "rad s^-1",
            "force_world_n": "N",
            "moment_world_n_m": "N m",
            "circuit": (
                "normalized illustrative pathway state (1)"
                if attached_trace is None
                else "normalized display envelope derived from attached raw signals (1)"
            ),
            "muscles": "normalized modeled state (1)",
            "evidence_incompleteness": "fraction (1)",
        },
        "frames": frames,
    }
    if ground_contact_summary is not None:
        replay["ground_contact_summary"] = ground_contact_summary
    if wingbeat_inspection is not None:
        replay["wingbeat_inspection"] = wingbeat_inspection
    if attached_trace is not None:
        replay["neural_trace_channels"] = attached_trace["channels"]
        replay["pathway_channels"] = attached_trace["pathway_channels"]
        replay["neural_model_scope"] = dict(neural_model_scope or {})
    return replay


def write_web_replay_episode(
    output_path: Path,
    replay: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one validated-shape browser episode payload."""

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            "web replay episode already exists at {}; pass overwrite=True".format(path)
        )
    if replay.get("schema_version") != WEB_REPLAY_SCHEMA_VERSION:
        raise ValueError("web replay episode has an unsupported schema_version")
    if not isinstance(replay.get("frames"), list) or len(replay["frames"]) < 2:
        raise ValueError("web replay episode requires at least two frames")
    _json_dump(path, replay)
    return path


def _web_coverage(graph: EvidenceGraph) -> List[Mapping[str, Any]]:
    records = provisional_coverage(graph)
    labels = {
        "LLPC1_full_pathway_provisional": "LLPC1 → identified MN targets",
        "NOD1_full_pathway_provisional": "NOD1 → identified MN targets",
    }
    return [
        {
            "id": record.branch,
            "label": labels.get(record.branch, record.branch),
            "resolved_fraction": record.resolved_fraction,
            "resolved_synapses": record.resolved_structural_synapses,
            "total_synapses": record.total_structural_synapses,
            "note": "Provisional: %s" % record.scope,
        }
        for record in records
    ]


def export_web_replay(
    output_dir: Path,
    *,
    scenarios: Sequence[str] = SCENARIO_NAMES,
    duration_s: float = 0.200,
    physics_dt_s: float = 0.0001,
    logging_dt_s: float = 0.001,
    neural_dt_s: float = 0.005,
    seed: int = 0,
    target_sample_rate_hz: float = 200.0,
    evidence_path: Optional[Path] = None,
    validation_report_path: Optional[Path] = None,
    validation_registry_path: Optional[Path] = None,
    legacy_nod1_result_path: Optional[Path] = None,
    include_reduced_retinal_pipeline: bool = False,
    include_registered_nod1_fixture: bool = False,
    registered_nod1_fixture_path: Optional[Path] = None,
    include_registered_fly_fgs_fixture: bool = False,
    registered_fly_fgs_manifest_path: Optional[Path] = None,
    registered_fly_fgs_capture_path: Optional[Path] = None,
    pipeline_runner: Optional[Any] = None,
    retinal_receptor_count: int = 64,
    retinal_angular_velocity_rad_s: float = 1.0,
    retinal_sensor_latency_s: Optional[float] = None,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError("web replay already exists at {}; pass overwrite=True".format(manifest_path))
    if validation_registry_path is not None and validation_report_path is None:
        raise ValueError("validation_registry_path requires validation_report_path")
    if registered_nod1_fixture_path is not None and not include_registered_nod1_fixture:
        raise ValueError(
            "registered_nod1_fixture_path requires include_registered_nod1_fixture"
        )
    if (
        registered_fly_fgs_manifest_path is not None
        and not include_registered_fly_fgs_fixture
    ):
        raise ValueError(
            "registered_fly_fgs_manifest_path requires "
            "include_registered_fly_fgs_fixture"
        )
    if (
        registered_fly_fgs_capture_path is not None
        and not include_registered_fly_fgs_fixture
    ):
        raise ValueError(
            "registered_fly_fgs_capture_path requires "
            "include_registered_fly_fgs_fixture"
        )
    validation_attachment: Optional[Tuple[Any, bytes, Any, bytes, Any]] = None
    if validation_report_path is not None:
        from .validation import (
            BenchmarkRegistry,
            ValidationReport,
            default_validation_source_digests,
            default_benchmark_registry_path,
            validate_required_source_digests,
            verified_preregistered_protocol_files,
        )

        report_path = Path(validation_report_path)
        registry_path = (
            default_benchmark_registry_path()
            if validation_registry_path is None
            else Path(validation_registry_path)
        )
        report_bytes = report_path.read_bytes()
        registry_bytes = registry_path.read_bytes()
        registry = BenchmarkRegistry.from_json(
            registry_bytes.decode("utf-8", errors="strict")
        )
        report = ValidationReport.from_json(
            report_bytes.decode("utf-8"), registry=registry
        )
        protocol_files = verified_preregistered_protocol_files(
            registry, registry_path
        )
        validate_required_source_digests(
            report,
            default_validation_source_digests(registry, registry_path),
            exact_kinds=("registry", "code", "dependency_lock", "protocol"),
        )
        if not report.suite_complete:
            raise ValueError("web publication requires a complete validation suite")
        failed_case_ids = tuple(
            sorted(
                result.case_id
                for result in report.results
                if result.status.value == "fail"
            )
        )
        if failed_case_ids:
            raise ValueError(
                "web publication validation contains FAIL cases: %s"
                % ", ".join(failed_case_ids)
            )
        validation_attachment = (
            report,
            report_bytes,
            registry,
            registry_bytes,
            protocol_files,
        )
    graph_path = default_seed_graph_path() if evidence_path is None else Path(evidence_path)
    graph = load_evidence_graph(graph_path)

    def publish_reference(
        *,
        label: str,
        source: Path,
        relative: Path,
        description: str,
    ) -> Mapping[str, Any]:
        """Copy one exact local reference and return its browser provenance row."""

        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError("web provenance source is unavailable: %s" % source)
        payload = source.read_bytes()
        digest = sha256_bytes(payload)
        _bytes_dump(output_dir / relative, payload)
        return {
            "label": label,
            "url": "data/%s" % relative.as_posix(),
            "sha256": digest,
            "value": "%s · sha256:%s" % (description, digest),
        }

    model_registry_source = _repository_root() / "data" / "models" / "flight_model_registry.json"
    if not model_registry_source.is_file():
        model_registry_source = _installed_data_path("models", "flight_model_registry.json")
    banc_fanc_source = (
        _repository_root()
        / "data"
        / "reference"
        / "banc_fanc_wing_pathway_evidence.v1.json"
    )
    if not banc_fanc_source.is_file():
        banc_fanc_source = _installed_data_path(
            "reference", "banc_fanc_wing_pathway_evidence.v1.json"
        )
    evidence_provenance = publish_reference(
        label="Evidence graph",
        source=graph_path,
        relative=Path("evidence") / graph_path.name,
        description=graph.graph_id,
    )
    model_registry_provenance = publish_reference(
        label="Model registry",
        source=model_registry_source,
        relative=Path("models") / "flight_model_registry.json",
        description="flight model registry",
    )
    banc_fanc_provenance = publish_reference(
        label="BANC/FANC evidence",
        source=banc_fanc_source,
        relative=Path("evidence") / "banc_fanc_wing_pathway_evidence.v1.json",
        description="female structural pathway fixture v1",
    )
    requested = tuple(dict.fromkeys(scenarios))
    if not requested:
        raise ValueError("at least one scenario is required")
    if "baseline" not in requested:
        requested = ("baseline",) + requested
    episodes = []
    summaries = []
    attached_pipeline_scopes: List[str] = []
    referenced_flybody_worker_digests: List[Any] = []
    referenced_flybody_dependency_locks: List[Any] = []
    for scenario in requested:
        if scenario not in SCENARIOS:
            raise ValueError("unknown scenario {!r}".format(scenario))
        config, result = run_scenario(
            scenario,
            duration_s=duration_s,
            physics_dt_s=physics_dt_s,
            logging_dt_s=logging_dt_s,
            neural_dt_s=neural_dt_s,
            seed=seed,
        )
        episode = episode_to_web_replay(
            scenario,
            config,
            result,
            target_sample_rate_hz=target_sample_rate_hz,
            evidence_graph=graph,
        )
        run_id = episode_run_id(scenario, config, evidence_path=graph_path)
        episode["source_run_id"] = run_id
        run_dir = output_dir / "runs" / run_id
        run_manifest_path = run_dir / "manifest.json"
        if run_manifest_path.is_file():
            artifact_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            if artifact_manifest.get("run_id") != run_id:
                raise ValueError("existing scientific run directory has the wrong identity")
        else:
            artifact_manifest = write_episode_artifact(
                run_dir,
                scenario,
                config,
                result,
                evidence_path=graph_path,
                web_replay=episode,
            )
        expected_replay_sha = web_replay_projection_sha256(episode)
        replay_receipt = artifact_manifest.get("web_replay_projection")
        if not isinstance(replay_receipt, Mapping) or replay_receipt.get(
            "sha256"
        ) != expected_replay_sha:
            raise ValueError(
                "existing scientific run is not bound to this web replay projection"
            )
        episode["source_artifact_manifest_sha256"] = sha256_file(
            run_manifest_path
        )
        episode["source_artifact_schema_version"] = artifact_manifest.get(
            "schema_version"
        )
        if episode.get("physics_backend") == "flybody":
            runtime = artifact_manifest.get("runtime")
            physics_provenance = (
                runtime.get("physics_provenance")
                if isinstance(runtime, Mapping)
                else None
            )
            referenced_flybody_worker_digests.append(
                physics_provenance.get("worker_image_digest")
                if isinstance(physics_provenance, Mapping)
                else None
            )
            referenced_flybody_dependency_locks.append(
                physics_provenance.get("worker_dependency_lock")
                if isinstance(physics_provenance, Mapping)
                else None
            )
        definition = SCENARIOS[scenario]
        relative = Path("episodes") / (scenario + ".json")
        _json_dump(output_dir / relative, episode)
        episodes.append(episode)
        summary = {
            "id": scenario,
            "label": definition.label,
            "condition": definition.condition,
            "description": definition.description,
            "color": definition.color,
            "status": "exploratory",
            "data_url": "data/%s" % relative.as_posix(),
            "artifact_manifest_url": "data/runs/%s/manifest.json" % run_id,
        }
        summary.update(
            _write_web_episode_binding_receipts(
                output_dir,
                episode=episode,
                episode_relative_path=relative,
                artifact_manifest=artifact_manifest,
                artifact_manifest_path=run_manifest_path,
            )
        )
        summaries.append(summary)

    def attach_pipeline_run(
        run: Any,
        *,
        condition: str,
        description: str,
        color: str,
    ) -> None:
        # Lazy imports avoid a module cycle: pipeline owns execution and this
        # module owns the browser/artifact encoding boundary.
        from .pipeline import (
            nod1_run_identity_context,
            nod1_run_to_web_replay,
            write_nod1_flight_artifact,
        )

        episode = nod1_run_to_web_replay(
            run,
            target_sample_rate_hz=target_sample_rate_hz,
        )
        episode_id = str(episode["id"])
        if any(summary["id"] == episode_id for summary in summaries):
            raise ValueError("duplicate web episode id %r" % episode_id)
        run_id = episode_run_id(
            "nod1_visual_circuit_to_flight",
            run.flight_config,
            identity_context=nod1_run_identity_context(run),
        )
        run_dir = output_dir / "runs" / run_id
        run_manifest_path = run_dir / "manifest.json"
        if run_manifest_path.is_file():
            artifact_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            if artifact_manifest.get("run_id") != run_id:
                raise ValueError("existing pipeline run directory has the wrong identity")
        else:
            artifact_manifest = write_nod1_flight_artifact(
                run_dir,
                run,
                web_replay=episode,
            )
            if artifact_manifest["run_id"] != run_id:
                raise RuntimeError("pipeline replay and artifact identities diverged")
        if episode.get("source_run_id") != run_id:
            raise RuntimeError("pipeline replay source identity diverged from artifact")
        expected_replay_sha = web_replay_projection_sha256(episode)
        replay_receipt = artifact_manifest.get("web_replay_projection")
        if not isinstance(replay_receipt, Mapping) or replay_receipt.get(
            "sha256"
        ) != expected_replay_sha:
            raise ValueError(
                "existing pipeline run is not bound to this web replay projection"
            )
        episode["source_artifact_manifest_sha256"] = sha256_file(
            run_manifest_path
        )
        episode["source_artifact_schema_version"] = artifact_manifest.get(
            "schema_version"
        )
        if episode.get("physics_backend") == "flybody":
            runtime = artifact_manifest.get("runtime")
            physics_provenance = (
                runtime.get("physics_provenance")
                if isinstance(runtime, Mapping)
                else None
            )
            referenced_flybody_worker_digests.append(
                physics_provenance.get("worker_image_digest")
                if isinstance(physics_provenance, Mapping)
                else None
            )
            referenced_flybody_dependency_locks.append(
                physics_provenance.get("worker_dependency_lock")
                if isinstance(physics_provenance, Mapping)
                else None
            )
        relative = Path("episodes") / (episode_id + ".json")
        _json_dump(output_dir / relative, episode)
        episodes.append(episode)
        summary = {
            "id": episode_id,
            "label": episode["label"],
            "condition": condition,
            "description": description,
            "color": color,
            "status": episode["status"],
            "data_url": "data/%s" % relative.as_posix(),
            "artifact_manifest_url": "data/runs/%s/manifest.json" % run_id,
        }
        summary.update(
            _write_web_episode_binding_receipts(
                output_dir,
                episode=episode,
                episode_relative_path=relative,
                artifact_manifest=artifact_manifest,
                artifact_manifest_path=run_manifest_path,
            )
        )
        summaries.append(summary)
        attached_pipeline_scopes.append(str(episode["neural_model_scope"]["kind"]))

    if legacy_nod1_result_path is not None:
        from .pipeline import (
            NOD1FlightPipelineConfig,
            load_legacy_nod1_result,
            run_nod1_flight_pipeline,
        )

        legacy_run = run_nod1_flight_pipeline(
            load_legacy_nod1_result(Path(legacy_nod1_result_path)),
            config=NOD1FlightPipelineConfig(
                physics_dt_s=physics_dt_s,
                neural_dt_s=neural_dt_s,
                logging_dt_s=logging_dt_s,
                seed=seed,
            ),
            runner=pipeline_runner,
        )
        attach_pipeline_run(
            legacy_run,
            condition="saved circuit trace",
            description=(
                "Raw exported vCH/DCH/NOD1 voltages from the frozen 1,208-cell browser "
                "circuit drive the causal exploratory DNp26/VNC/muscle bridge."
            ),
            color="#22d3ee",
        )

    if include_reduced_retinal_pipeline:
        from .pipeline import NOD1FlightPipelineConfig, run_retinal_flight_pipeline
        from .vision import AnalyticGratingScene, PanoramicRetina

        exposure_steps = int(round(duration_s / neural_dt_s))
        if (
            exposure_steps < 2
            or abs(exposure_steps * neural_dt_s - duration_s) > 1.0e-12
        ):
            raise ValueError(
                "pipeline web export requires duration_s to be an integer multiple "
                "of neural_dt_s with at least two exposures"
            )
        retina = PanoramicRetina(
            receptor_count=retinal_receptor_count,
            sensor_latency_s=(
                min(0.0005, 0.5 * neural_dt_s)
                if retinal_sensor_latency_s is None
                else retinal_sensor_latency_s
            ),
        )
        scene = AnalyticGratingScene(
            angular_velocity_rad_s=retinal_angular_velocity_rad_s
        )
        retinal_frames = tuple(
            retina.sample(
                scene,
                exposure_start_s=index * neural_dt_s,
                exposure_end_s=(index + 1) * neural_dt_s,
            )
            for index in range(exposure_steps + 1)
        )
        retinal_run = run_retinal_flight_pipeline(
            retinal_frames,
            config=NOD1FlightPipelineConfig(
                physics_dt_s=physics_dt_s,
                neural_dt_s=neural_dt_s,
                logging_dt_s=logging_dt_s,
                seed=seed,
            ),
            runner=pipeline_runner,
        )
        attach_pipeline_run(
            retinal_run,
            condition="causal image-derived vertical slice",
            description=(
                "Finite-exposure retinal images drive the explicitly reduced NOD1 "
                "surrogate, inferred DNp26/MN rates, individual muscles, and body model."
            ),
            color="#2dd4bf",
        )

    if include_registered_fly_fgs_fixture:
        from .pipeline import (
            NOD1FlightPipelineConfig,
            run_registered_fly_fgs_flight_pipeline,
        )

        fly_fgs_run = run_registered_fly_fgs_flight_pipeline(
            manifest_path=registered_fly_fgs_manifest_path,
            capture_path=registered_fly_fgs_capture_path,
            config=NOD1FlightPipelineConfig(
                physics_dt_s=physics_dt_s,
                neural_dt_s=neural_dt_s,
                logging_dt_s=logging_dt_s,
                seed=seed,
            ),
            runner=pipeline_runner,
        )
        attach_pipeline_run(
            fly_fgs_run,
            condition="registered fixed-step fly-FGS visual circuit",
            description=(
                "The content-addressed fly-FGS visual/circuit replay drives the "
                "existing evidence-qualified DNp26/VNC/MN/muscle path using only "
                "four registered NOD1 voltages. Its toy muscle, wing, yaw, and body "
                "equations are excluded from the behavior simulation."
            ),
            color="#a78bfa",
        )

    if include_registered_nod1_fixture:
        from .pipeline import (
            NOD1FlightPipelineConfig,
            run_registered_nod1_browser_flight_pipeline,
        )

        registered_run = run_registered_nod1_browser_flight_pipeline(
            registry_path=validation_registry_path,
            fixture_path=registered_nod1_fixture_path,
            config=NOD1FlightPipelineConfig(
                physics_dt_s=physics_dt_s,
                neural_dt_s=neural_dt_s,
                logging_dt_s=logging_dt_s,
                seed=seed,
            ),
            runner=pipeline_runner,
        )
        attach_pipeline_run(
            registered_run,
            condition="historical registered frozen Chromium circuit output",
            description=(
                "Historical regression: four SHA-locked browser-voltage readouts "
                "from the frozen 1,208-cell "
                "NOD1 circuit drive the causal exploratory DNp26/VNC/muscle bridge. "
                "The fixture has no source retinal frames and is not biological validation."
            ),
            color="#06b6d4",
        )

    # The interface intentionally opens canonical fly-FGS FlyBody first when it
    # is attached, otherwise the first authoritative FlyBody episode, while
    # retaining baseline as index zero for comparison. This ordering is generated
    # from backend metadata, never hand-edited after publication.
    canonical_flybody_index = next(
        (
            index
            for index, episode in enumerate(episodes)
            if episode.get("id") == "fly_fgs_canonical"
            and episode.get("physics_backend") == "flybody"
        ),
        None,
    )
    first_flybody_index = next(
        (
            index
            for index, episode in enumerate(episodes)
            if episode.get("physics_backend") == "flybody"
        ),
        None,
    )
    preferred_flybody_index = (
        canonical_flybody_index
        if canonical_flybody_index is not None
        else first_flybody_index
    )
    if preferred_flybody_index is not None and preferred_flybody_index > 1:
        episodes.insert(1, episodes.pop(preferred_flybody_index))
        summaries.insert(1, summaries.pop(preferred_flybody_index))

    has_flybody = first_flybody_index is not None
    published_model_hashes = model_hashes()
    executable_model_provenance: List[Mapping[str, Any]] = [
        {
            "label": "Reduced-order executable model",
            "sha256": published_model_hashes["reduced_order_flight_source"],
            "value": "reduced-order source · sha256:%s"
            % published_model_hashes["reduced_order_flight_source"],
        }
    ]
    if has_flybody:
        executable_model_provenance.append(
            {
                "label": "FlyBody adapter executable model",
                "sha256": published_model_hashes["flybody_adapter_source"],
                "value": "FlyBody adapter source · sha256:%s"
                % published_model_hashes["flybody_adapter_source"],
            }
        )
    mixed_authority_notice = (
        "Episodes explicitly labelled FlyBody executed the pinned FlyGym/FlyBody "
        "MuJoCo worker; canonical online episodes use the content-addressed fly-FGS "
        "T4a circuit and expose only four NOD1 voltages to the motor bridge. The "
        "source has no T5, photoreceptor, or lamina model; other scenarios use the "
        "reduced-order NumPy scaffold. All visual, neural, neuromuscular, and "
        "virtual-hinge parameters remain uncalibrated, and no episode is an "
        "authoritative behavioral prediction."
    )
    manifest: Dict[str, Any] = {
        "schema_version": WEB_REPLAY_SCHEMA_VERSION,
        "title": "Causal fly-FGS visual-to-flight scientific replay",
        "created_at": _utc_now(),
        "status": "exploratory",
        "source_kind": (
            "mixed_exploratory_flybody_and_reduced_replay"
            if has_flybody
            else "mixed_exploratory_replay_with_pipeline_traces"
            if attached_pipeline_scopes
            else REDUCED_ORDER_SOURCE_KIND
        ),
        "authority_notice": (
            mixed_authority_notice if has_flybody else AUTHORITY_NOTICE
        ),
        "circuit": ["T4a", "vCH/DCH", "LLPC1", "NOD1", "DN", "MN", "Muscle"],
        "atlas_bridge": (
            "FAFB v783 and MANC v1.0 are connected only by explicit cell-type crosswalks; "
            "root/body identifiers and sexes/specimens are not treated as interchangeable."
        ),
        "coverage": _web_coverage(graph),
        "provenance": [
            evidence_provenance,
            {"label": "Coverage", "value": "Provisional whole-pathway audit; unresolved output is retained"},
            {
                "label": "Physics",
                "value": (
                    "Pinned FlyGym/FlyBody MuJoCo for explicitly labelled episodes; "
                    "reduced-order NumPy for the remaining scenarios"
                    if has_flybody
                    else "Reduced-order NumPy scaffold; FlyBody/MuJoCo not executed"
                ),
            },
            {"label": "Calibration", "value": "None — demonstration coefficients only"},
            {"label": "Motor timing", "value": "Rate-derived events are inferred, never presented as exact spikes"},
            {
                "label": "Circuit display",
                "value": (
                    "Per-episode registered online/frozen or reduced raw traces attached; "
                    "only each episode's declared motor-bound signals are eligible downstream"
                    if attached_pipeline_scopes
                    else "Illustrative pathway state; no upstream neural trace was ingested"
                ),
            },
            {"label": "Evidence gap", "value": "Coverage-derived display proxy; not trajectory uncertainty"},
            *executable_model_provenance,
            model_registry_provenance,
            banc_fanc_provenance,
        ],
        "episodes": summaries,
    }
    if attached_pipeline_scopes:
        manifest["provenance"].append(
            {
                "label": "Attached neural replay",
                "value": (
                    "%s; raw values sampled causally, unavailable stages retained"
                    % ", ".join(attached_pipeline_scopes)
                ),
            }
        )
    if validation_attachment is not None:
        (
            report,
            report_bytes,
            registry,
            registry_bytes,
            protocol_files,
        ) = validation_attachment
        worker_image_receipts = tuple(
            item
            for item in report.source_digests
            if item.kind == WORKER_IMAGE_SOURCE_KIND
            and item.name == WORKER_IMAGE_SOURCE_NAME
        )
        if len(worker_image_receipts) > 1:
            raise ValueError(
                "validation report contains ambiguous FlyBody worker image receipts"
            )
        if (
            report.suite_complete
            and referenced_flybody_worker_digests
            and len(worker_image_receipts) != 1
        ):
            raise ValueError(
                "a complete replay containing FlyBody requires one worker image receipt"
            )
        if worker_image_receipts:
            worker_image_sha = worker_image_receipts[0].sha256
            expected_worker_digest = "sha256:" + worker_image_sha
            if any(
                value != expected_worker_digest
                for value in referenced_flybody_worker_digests
            ):
                raise ValueError(
                    "FlyBody run worker image digest does not match the validation report"
                )
            manifest["provenance"].append(
                {
                    "label": "Worker image",
                    "sha256": worker_image_sha,
                    "value": "%s · sha256:%s"
                    % (WORKER_IMAGE_SOURCE_NAME, worker_image_sha),
                }
            )
        dependency_lock_receipts = tuple(
            item
            for item in report.source_digests
            if item.kind == WORKER_DEPENDENCY_LOCK_SOURCE_KIND
            and item.name == WORKER_DEPENDENCY_LOCK_SOURCE_NAME
        )
        if len(dependency_lock_receipts) > 1:
            raise ValueError(
                "validation report contains ambiguous worker dependency-lock receipts"
            )
        if (
            report.suite_complete
            and referenced_flybody_dependency_locks
            and len(dependency_lock_receipts) != 1
        ):
            raise ValueError(
                "a complete replay containing FlyBody requires one dependency-lock receipt"
            )
        if dependency_lock_receipts:
            expected_lock = {
                "name": WORKER_DEPENDENCY_LOCK_SOURCE_NAME,
                "sha256": "sha256:" + dependency_lock_receipts[0].sha256,
            }
            if any(
                value != expected_lock
                for value in referenced_flybody_dependency_locks
            ):
                raise ValueError(
                    "FlyBody run dependency lock does not match the validation report"
                )
        report_digest = sha256_bytes(report_bytes)
        registry_file_digest = sha256_bytes(registry_bytes)
        registry_major = registry.version.split(".", 1)[0]
        relative_report = Path("validation") / ("report.%s.json" % report_digest)
        relative_registry = Path("benchmarks") / (
            "registry.v%s.%s.json" % (registry_major, registry_file_digest)
        )
        _bytes_dump(output_dir / relative_report, report_bytes)
        _bytes_dump(output_dir / relative_registry, registry_bytes)
        manifest["validation_report_url"] = "data/%s" % relative_report.as_posix()
        manifest["validation_report_sha256"] = report_digest
        manifest["validation_registry_url"] = (
            "data/%s" % relative_registry.as_posix()
        )
        manifest["validation_registry_file_sha256"] = registry_file_digest
        manifest["validation_registry_canonical_sha256"] = (
            registry.content_sha256
        )
        manifest["provenance"].append(
            {
                "label": "Validation report",
                "url": manifest["validation_report_url"],
                "sha256": manifest["validation_report_sha256"],
                "value": "%s · sha256:%s"
                % (report.evaluation_id, manifest["validation_report_sha256"]),
            }
        )
        manifest["provenance"].append(
            {
                "label": "Evaluation registry",
                "url": manifest["validation_registry_url"],
                "sha256": manifest["validation_registry_file_sha256"],
                "value": "version %s · file sha256:%s · canonical digest %s"
                % (
                    registry.version,
                    manifest["validation_registry_file_sha256"],
                    manifest["validation_registry_canonical_sha256"],
                ),
            }
        )
        for source_uri, source_path, protocol_sha256, protocol_bytes in protocol_files:
            relative_protocol = Path("benchmarks") / "protocols" / source_path.name
            _bytes_dump(output_dir / relative_protocol, protocol_bytes)
            manifest["provenance"].append(
                {
                    "label": "Evaluation protocol · %s" % source_path.stem,
                    "url": "data/%s" % relative_protocol.as_posix(),
                    "sha256": protocol_sha256,
                    "value": "%s · sha256:%s" % (source_uri, protocol_sha256),
                }
            )
    _json_dump(manifest_path, manifest)
    return manifest


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "WEB_REPLAY_SCHEMA_VERSION",
    "WEB_REPLAY_PROJECTION_SCHEMA_VERSION",
    "WEB_REPLAY_PROJECTION_CANONICALIZATION",
    "WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS",
    "REDUCED_ORDER_SOURCE_KIND",
    "GROUND_CONTACT_TELEMETRY_KIND",
    "GROUND_CONTACT_SAMPLE_SEMANTICS",
    "AUTHORITY_NOTICE",
    "SCENARIO_NAMES",
    "SCENARIOS",
    "ScenarioDefinition",
    "audit_evidence",
    "build_scenario_config",
    "episode_run_id",
    "episode_to_web_replay",
    "export_web_replay",
    "model_hashes",
    "provisional_coverage",
    "read_chunked_array",
    "run_scenario",
    "sha256_file",
    "web_replay_projection_sha256",
    "write_episode_artifact",
    "write_web_replay_episode",
]
