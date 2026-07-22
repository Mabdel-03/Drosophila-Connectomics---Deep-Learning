#!/usr/bin/env python3
"""Fail-closed auditor for a built fly-sensor2behavior web release.

The static audit validates the manifest's local reference graph, immutable
scientific arrays, and the validation report's exact registry binding.  The
optional browser audit serves the same directory from an in-process HTTP
server and forces the application's pure-JavaScript SHA-256 fallback.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import math
import re
import sys
import threading
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlsplit

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fly_sensor2behavior.artifacts import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    FLYBODY_WORKER_SOURCE_KIND,
    GROUND_CONTACT_SAMPLE_SEMANTICS,
    GROUND_CONTACT_TELEMETRY_KIND,
    REDUCED_ORDER_SOURCE_KIND,
    WEB_REPLAY_PROJECTION_CANONICALIZATION,
    WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS,
    WEB_REPLAY_PROJECTION_SCHEMA_VERSION,
    model_hashes,
    read_chunked_array,
    web_replay_projection_sha256,
)
from fly_sensor2behavior.canonical_artifacts import (  # noqa: E402
    CANONICAL_ARTIFACT_SCHEMA_VERSION,
    CANONICAL_ARTIFACT_SOURCE_KIND,
    CANONICAL_WEB_PROJECTION_CANONICALIZATION,
    CANONICAL_WEB_SOURCE_KIND,
    ONLINE_CIRCUIT_REPLAY_SCHEMA_VERSION,
    _checkpoint_component_payload_sha256,
    canonical_web_projection_sha256,
)
from fly_sensor2behavior.flight.canonical_closed_loop import (  # noqa: E402
    CANONICAL_CLOSED_LOOP_RUNTIME_VERSION,
    CANONICAL_CLOSED_LOOP_SCHEMA_VERSION,
    CANONICAL_SOURCE_LIMITATIONS,
    CanonicalClosedLoopCheckpoint,
    CanonicalClosedLoopConfig,
)
from fly_sensor2behavior.flight.effector_mapping import (  # noqa: E402
    EFFECTOR_MAPPING_PROVENANCE,
    RAW_APP_LATERALITY_STATUS,
    SIGNED_BEHAVIOR_CLAIM_POLICY,
    EffectorMappingReceipt,
)
from fly_sensor2behavior.validation import (  # noqa: E402
    BenchmarkRegistry,
    GateStatus,
    ValidationReport,
    verified_preregistered_protocol_files,
)
from fly_sensor2behavior.schema import (  # noqa: E402
    REVIEWED_FLYBODY_WING_AXIS_ORDER,
)
from fly_sensor2behavior.fly_fgs import (  # noqa: E402
    FLY_FGS_NOD1_RAW_APP_SIDES,
    FLY_FGS_NOD1_ROOT_IDS,
    FLY_FGS_REGISTERED_CAPTURE_SHA256,
    FLY_FGS_REGISTERED_CAPTURE_URI,
    FLY_FGS_SCHEMA_VERSION,
    FLY_FGS_SNAPSHOT_ID,
    FLY_FGS_SOURCE_MANIFEST_SHA256,
    FLY_FGS_SOURCE_MANIFEST_URI,
    load_registered_fly_fgs_fixture,
)
from fly_sensor2behavior.fly_fgs_runtime import (  # noqa: E402
    FLY_FGS_RUNTIME_PROTOCOL_VERSION,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SHA256_IN_TEXT_PATTERN = re.compile(r"sha256:([0-9a-f]{64})(?:$|[^0-9a-f])")
WORKER_IMAGE_SOURCE_KIND = "image"
WORKER_IMAGE_SOURCE_NAME = "fly-s2b-worker-image"
DEPENDENCY_LOCK_SOURCE_KIND = "dependency_lock"
DEPENDENCY_LOCK_SOURCE_NAME = "requirements.lock"
FROZEN_BROWSER_EPISODE_ID = "frozen_browser_nod1"
FROZEN_BROWSER_PARITY_CASE_ID = "nod1.browser_python_parity"
FROZEN_BROWSER_FLYBODY_CASE_ID = (
    "pipeline.registered_nod1_to_flybody_vertical_slice"
)
FROZEN_BROWSER_DURATION_S = 0.5
FROZEN_BROWSER_PHYSICS_TIMESTEP_S = 1.0e-4
FLY_FGS_EPISODE_ID = "fly_fgs_canonical"
FLY_FGS_SOURCE_KIND = "exploratory_fly_fgs_circuit_flybody_pipeline"
FLY_FGS_PIPELINE_ID = "fly-fgs-retina-circuit-to-flight-v1"
FLY_FGS_INTEGRITY_CASE_ID = "fly_fgs.fixed_step_integrity"
FLY_FGS_INCREMENTAL_CASE_ID = "fly_fgs.incremental_runtime_parity"
FLY_FGS_CHECKPOINT_CASE_ID = "fly_fgs.checkpoint_reentry"
FLY_FGS_FLYBODY_CASE_ID = (
    "pipeline.registered_fly_fgs_to_flybody_vertical_slice"
)
FLY_FGS_REGISTRY_VERSION = "1.18.0"
FLY_FGS_TRACE_SHA256 = (
    "0db62b35fdcd5fd5ad615e8e52e8e390459c1d0d75ec90dfdfb3c8de5e5c1f75"
)
FLY_FGS_CELL_COUNT = 1684
FLY_FGS_MOTOR_CHANNEL_COUNT = 4
FLY_FGS_SOURCE_SAMPLE_COUNT = 100
FLY_FGS_RETINAL_INPUT_COUNT = 1441
FLY_FGS_CAUSAL_RETINAL_FRAME_COUNT = 99
FLY_FGS_SOURCE_DT_S = 0.005
FLY_FGS_DURATION_S = 0.5
FLY_FGS_PHYSICS_TIMESTEP_S = 1.0e-4
FLY_FGS_EXPECTED_INVARIANTS = {
    "nod1_voltage_excursion_v": 0.024822540388854418,
    "t4_activity_excursion": 0.007244982389009451,
}
FLY_FGS_ASSET_RECEIPTS = {
    "page_snapshot": (
        "f19434ee872984035fafbfb492538609cf458111ddb64ad9f998e6c371187932",
        "source_receipt_only",
        False,
    ),
    "circuit_engine": (
        "f6de20850d463399b308642222e0ef6dadf02129bf27440fd7f2d5cf7d45396f",
        "fixed_step_circuit_engine",
        True,
    ),
    "circuit_bundle": (
        "dc7ad294e54df7f4b8d4556f36e95b2676e19135d150557b8a16cfc17a4ac0dc",
        "fixed_step_circuit_bundle",
        True,
    ),
    "wing_dns_excluded": (
        "b3ae85e6151228919617a87907730705ec3acb8d8b441465ce182926d6c32390",
        "excluded_downstream_source_receipt_only",
        False,
    ),
    "source_manifest": (
        FLY_FGS_SOURCE_MANIFEST_SHA256,
        "strict_source_contract",
        False,
    ),
    "registered_capture": (
        FLY_FGS_REGISTERED_CAPTURE_SHA256,
        "registered_fixed_step_capture",
        False,
    ),
}
FLY_FGS_FORBIDDEN_DOWNSTREAM_FIELDS = frozenset(
    {
        "SCALE_M",
        "MUSCLE_META",
        "MN_PER_SIDE",
        "FG_DN",
        "DN_WMAX",
        "NT_SIGN",
        "computeControl",
        "dnGain",
        "yawToRate",
        "maxRate",
        "state.muscleAct",
        "state.ampL",
        "state.ampR",
        "state.aoaL",
        "state.aoaR",
        "state.yaw",
        "wingR",
        "wingL",
    }
)
FLYBODY_CONTACT_STATUS = GROUND_CONTACT_TELEMETRY_KIND
FLYBODY_CONTACT_SUMMARY_KIND = GROUND_CONTACT_TELEMETRY_KIND
WINGBEAT_INSPECTION_SOURCE_KINDS = frozenset(
    {
        "exploratory_frozen_browser_nod1_flybody_pipeline",
        "exploratory_retinal_nod1_flybody_pipeline",
        FLY_FGS_SOURCE_KIND,
    }
)
WINGBEAT_INSPECTION_TIMESTEP_S = 1.0e-4
WINGBEAT_INSPECTION_DURATION_S = 0.060
WINGBEAT_TORQUE_SAMPLE_SEMANTICS = (
    "sample 0 is the reset value; sample i>=1 is the six-axis MuJoCo "
    "actuator torque used for the physics transition ending at physics_time_s[i]"
)
WINGBEAT_ANGLE_SAMPLE_SEMANTICS = (
    "sample i is the measured post-transition MuJoCo wing state at "
    "physics_time_s[i]"
)
LOGICAL_EXECUTABLE_PROVENANCE_KEYS = {
    "Reduced-order executable model": "reduced_order_flight_source",
    "FlyBody adapter executable model": "flybody_adapter_source",
}
CANONICAL_ONLINE_VALIDATION_CASE_ID = (
    "pipeline.canonical_online_fly_fgs_to_flybody"
)
CANONICAL_ARRAY_DESCRIPTOR_BASE_FIELDS = frozenset(
    {"dtype", "shape", "unit", "provenance", "logical_npy_sha256", "chunks"}
)
CANONICAL_TABLE_IDS = (
    "motor_events",
    "rate_samples",
    "circuit_observations",
    "intervention_activity",
)
CANONICAL_RAW_LANES = ("L", "R")
CANONICAL_MOTORS = ("MN-iv2", "MN-i1", "MN-iv1", "MN-b3")
CANONICAL_MUSCLES = ("iv2", "i1", "iv1", "b3")
CANONICAL_DN_AXIS = tuple(
    "raw_app_{}:DNp26".format(lane) for lane in CANONICAL_RAW_LANES
)
CANONICAL_MN_AXIS = tuple(
    "raw_app_{}:{}".format(lane, motor)
    for lane in CANONICAL_RAW_LANES
    for motor in CANONICAL_MOTORS
)
CANONICAL_MUSCLE_AXIS = tuple(
    "raw_app_{}:{}".format(lane, muscle)
    for lane in CANONICAL_RAW_LANES
    for muscle in ("DLM", "DVM", "iv2", "i1", "iv1", "b3", "tp1")
)


class AuditError(RuntimeError):
    """Raised when a release artifact fails a binding or integrity check."""


@dataclass(frozen=True)
class ReleaseContext:
    root: Path
    manifest: Mapping[str, Any]
    episodes: Tuple[Mapping[str, Any], ...]
    report: ValidationReport
    registry: BenchmarkRegistry
    run_manifest_count: int
    array_count: int
    chunk_count: int
    provenance_reference_count: int
    provenance_hash_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditError("value is not finite canonical JSON: {}".format(exc)) from exc
    return hashlib.sha256(payload).hexdigest()


def _logical_npy_sha256(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _exact_fields(value: Any, expected: Sequence[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), "{} must be an object".format(label))
    _require(
        set(value) == set(expected),
        "{} fields do not match the exact contract".format(label),
    )
    return value


def _python_source_tree_sha256(package_root: Path) -> str:
    """Match the source-tree receipt emitted by ``fly-s2b validate``."""

    digest = hashlib.sha256()
    sources = sorted(Path(package_root).rglob("*.py"), key=lambda path: path.as_posix())
    _require(bool(sources), "Python package source tree is empty")
    for source in sources:
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        "{} must be a lowercase SHA-256 digest".format(label),
    )
    return value


def _reject_json_constant(value: str) -> None:
    raise AuditError("non-finite JSON constant {!r} is forbidden".format(value))


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError("duplicate JSON key {!r}".format(key))
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError("{} is not strict UTF-8 JSON: {}".format(label, exc)) from exc
    _require(isinstance(value, Mapping), "{} must contain a JSON object".format(label))
    return value


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AuditError("{} could not be read: {}".format(label, exc)) from exc
    return _load_json_bytes(payload, label)


def _safe_local_path(root: Path, url: Any, label: str) -> Path:
    """Resolve one browser-local URL while rejecting traversal and ambiguity."""

    _require(isinstance(url, str) and bool(url), "{} must be a non-empty string".format(label))
    _require(
        all(32 <= ord(character) < 127 for character in url),
        "{} contains a control or non-ASCII character".format(label),
    )
    parsed = urlsplit(url)
    _require(
        not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment,
        "{} must be a query-free, fragment-free local URL".format(label),
    )
    decoded = unquote(parsed.path)
    _require(
        decoded == parsed.path,
        "{} must use one unencoded canonical path spelling".format(label),
    )
    _require(
        unquote(decoded) == decoded,
        "{} contains nested percent encoding".format(label),
    )
    _require(decoded and not decoded.startswith("/"), "{} must be relative".format(label))
    _require("\\" not in decoded and "\x00" not in decoded, "{} contains an unsafe separator".format(label))
    raw_parts = decoded.split("/")
    _require(
        all(part not in ("", ".", "..") for part in raw_parts),
        "{} contains an empty or traversal component".format(label),
    )
    pure = PurePosixPath(decoded)
    _require(not pure.is_absolute(), "{} must be relative".format(label))
    root_resolved = root.resolve()
    lexical_candidate = root_resolved / Path(*pure.parts)
    current = root_resolved
    for part in pure.parts:
        current = current / part
        _require(
            not current.is_symlink(),
            "{} escapes the safe local-path policy through a symbolic link".format(
                label
            ),
        )
    candidate = lexical_candidate.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise AuditError("{} escapes the static release root".format(label)) from exc
    _require(candidate.is_file(), "{} does not exist: {}".format(label, candidate))
    return candidate


def _declared_provenance_sha256(entry: Mapping[str, Any], label: str) -> Optional[str]:
    explicit = entry.get("sha256")
    value = entry.get("value")
    textual = (
        tuple(SHA256_IN_TEXT_PATTERN.findall(value))
        if isinstance(value, str)
        else ()
    )
    _require(
        len(set(textual)) <= 1,
        "{} contains contradictory textual SHA-256 claims".format(label),
    )
    textual_digest = textual[0] if textual else None
    if explicit is None:
        return textual_digest
    explicit_digest = _require_sha256(explicit, "{}.sha256".format(label))
    _require(
        textual_digest is None or textual_digest == explicit_digest,
        "{} structured and textual SHA-256 claims disagree".format(label),
    )
    return explicit_digest


def _audit_logical_executable_provenance(
    entry_label: str,
    declared_sha: Optional[str],
    run_manifests: Sequence[Mapping[str, Any]],
) -> bool:
    """Bind non-file source digests to current code and every run manifest."""

    model_key = LOGICAL_EXECUTABLE_PROVENANCE_KEYS.get(entry_label)
    if model_key is None:
        return False
    expected_sha = model_hashes().get(model_key)
    _require(
        declared_sha == expected_sha,
        "{} provenance does not match the current executable source".format(
            entry_label
        ),
    )
    applicable_count = 0
    for index, run_manifest in enumerate(run_manifests):
        # Canonical online schema 1.1 has its own content-derived scientific
        # identity and nested runtime/source receipts, audited separately.  A
        # top-level provenance entry for the supporting legacy replay catalog
        # must not be projected onto that unrelated schema.
        if run_manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION:
            continue
        applicable_count += 1
        run_model_hashes = run_manifest.get("model_hashes")
        _require(
            isinstance(run_model_hashes, Mapping)
            and run_model_hashes.get(model_key) == expected_sha,
            "{} provenance is not bound by run manifest {}".format(
                entry_label, index
            ),
        )
    _require(
        applicable_count > 0,
        "{} provenance has no applicable legacy run manifest".format(entry_label),
    )
    return True


@functools.lru_cache(maxsize=1)
def _registered_online_display_contract() -> Mapping[str, Any]:
    fixture = load_registered_fly_fgs_fixture()
    cell_axis = list(fixture.circuit_topology["cell_axis"])
    retinal_axis = dict(fixture.circuit_topology["retinotopic_t4a"])
    cell_ids = [str(cell["id"]) for cell in cell_axis]
    _require(
        len(cell_ids) == FLY_FGS_CELL_COUNT
        and len(set(cell_ids)) == FLY_FGS_CELL_COUNT,
        "registered fly-FGS cell axis is invalid",
    )
    _require(
        retinal_axis.get("count") == FLY_FGS_RETINAL_INPUT_COUNT,
        "registered fly-FGS retinal axis is invalid",
    )
    assets = {
        receipt.asset_id: dict(receipt.to_dict())
        for receipt in fixture.asset_receipts
    }
    assets["source_manifest"] = {
        "asset_id": "source_manifest",
        "relative_path": fixture.source_manifest_uri,
        "sha256": fixture.source_manifest_sha256,
        "bytes": fixture.source_manifest_bytes,
        "media_type": "application/json",
        "role": "strict_source_contract",
        "eligible_circuit_input": False,
    }
    return {
        "fixture": fixture,
        "snapshot_id": fixture.snapshot_id,
        "source_manifest_sha256": fixture.source_manifest_sha256,
        "source_manifest_uri": fixture.source_manifest_uri,
        "dataset": dict(fixture.dataset_metadata),
        "cell_inventory": dict(fixture.circuit_inventory),
        "circuit_topology": dict(fixture.circuit_topology),
        "cell_axis": cell_axis,
        "retinal_axis": retinal_axis,
        "cell_ids": cell_ids,
        "nod1_cell_indices": {
            root_id: cell_ids.index("r{}".format(root_id))
            for root_id in FLY_FGS_NOD1_ROOT_IDS
        },
        "cell_axis_sha256": _canonical_json_sha256(cell_axis),
        "cell_ids_sha256": _canonical_json_sha256(cell_ids),
        "retinal_axis_sha256": _canonical_json_sha256(retinal_axis),
        "asset_receipts": assets,
        "rejected_downstream_fields": list(fixture.rejected_downstream_fields),
        "claim_scope": dict(fixture.claim_scope),
    }


def _canonical_expected_array_shapes(
    config: CanonicalClosedLoopConfig,
    event_count: int,
) -> Mapping[str, Tuple[int, ...]]:
    bridge_count = config.bridge_interval_count
    physics_count = bridge_count * 5
    circuit_count = config.circuit_sample_count
    shapes: Dict[str, Tuple[int, ...]] = {
        "physics_time_s": (physics_count + 1,),
        "body_position_world_m": (physics_count + 1, 3),
        "body_velocity_world_m_s": (physics_count + 1, 3),
        "body_quaternion_body_to_world": (physics_count + 1, 4),
        "body_angular_velocity_body_rad_s": (physics_count + 1, 3),
        "aerodynamic_force_body_n": (physics_count + 1, 3),
        "aerodynamic_torque_body_n_m": (physics_count + 1, 3),
        "aerodynamic_mechanical_power_w": (physics_count + 1,),
        "wing_phase_unwrapped_rad": (physics_count + 1,),
        "wing_frequency_hz": (physics_count + 1,),
        "muscle_natural_endpoint_time_s": (physics_count,),
        "circuit_measurement_time_s": (circuit_count,),
        "circuit_availability_time_s": (circuit_count,),
        "nod1_voltage_v": (circuit_count, 4),
        "retinal_input_luminance": (circuit_count, FLY_FGS_RETINAL_INPUT_COUNT),
        "full_circuit_voltage_v": (circuit_count, FLY_FGS_CELL_COUNT),
        "full_circuit_activity": (circuit_count, FLY_FGS_CELL_COUNT),
        "bridge_interval_start_s": (bridge_count,),
        "bridge_phase_path_unwrapped_rad": (bridge_count, 6),
        "motor_event_numeric": (event_count, 10),
    }
    for prefix in ("raw_app_actuation", "physical_actuation"):
        shapes["{}_phase_rad".format(prefix)] = (physics_count,)
        shapes["{}_frequency_hz".format(prefix)] = (physics_count,)
        for suffix in (
            "stroke_rad",
            "stroke_velocity_rad_s",
            "stroke_acceleration_rad_s2",
            "angle_of_attack_rad",
            "deviation_rad",
            "generalized_torque_n_m",
        ):
            shapes["{}_{}".format(prefix, suffix)] = (physics_count, 2)
        shapes["{}_wing_axis_torque_n_m".format(prefix)] = (physics_count, 6)
    for state, count in (("natural", physics_count), ("effective", physics_count + 1)):
        for field in ("activation", "force_n", "phase_effect", "resonance_scale"):
            shapes["muscle_{}_{}".format(state, field)] = (count, 14)
    for prefix in ("generated", "held"):
        for stage, width in (("dn", 2), ("mn", 8)):
            for suffix in (
                "rate_hz",
                "rate_measurement_time_s",
                "rate_availability_time_s",
            ):
                shapes["{}_{}_{}".format(prefix, stage, suffix)] = (
                    bridge_count,
                    width,
                )
    shapes["held_dn_rate_present"] = (bridge_count, 2)
    shapes["held_mn_rate_present"] = (bridge_count, 8)
    return shapes


def _canonical_expected_axis_metadata() -> Mapping[str, Mapping[str, Any]]:
    source = _registered_online_display_contract()
    retinal = source["retinal_axis"]
    metadata: Dict[str, Mapping[str, Any]] = {
        "body_position_world_m": {"axis_labels": ["world_x", "world_y", "world_z"]},
        "body_velocity_world_m_s": {"axis_labels": ["world_x", "world_y", "world_z"]},
        "body_angular_velocity_body_rad_s": {"axis_labels": ["body_x", "body_y", "body_z"]},
        "aerodynamic_force_body_n": {"axis_labels": ["body_x", "body_y", "body_z"]},
        "aerodynamic_torque_body_n_m": {"axis_labels": ["body_x", "body_y", "body_z"]},
        "nod1_voltage_v": {
            "axis_name": "motor_eligible_nod1_root",
            "axis_labels": list(FLY_FGS_NOD1_ROOT_IDS),
            "root_ids": list(FLY_FGS_NOD1_ROOT_IDS),
            "side_semantics": "raw application L/R only; anatomy unknown",
            "sample_axis": "circuit_measurement_time_s",
            "eligible_motor_input": True,
        },
        "retinal_input_luminance": {
            "axis_name": "registered_retinotopic_t4a_cell",
            "axis_labels": list(retinal["cell_ids"]),
            "parent_cell_indices": list(retinal["cell_indices"]),
            "azimuth_deg": list(retinal["azimuth_deg"]),
            "elevation_deg": list(retinal["elevation_deg"]),
            "axis_registration": "registered_fly_fgs_circuit_bundle_order",
            "axis_sha256": source["retinal_axis_sha256"],
            "source_manifest_sha256": FLY_FGS_SOURCE_MANIFEST_SHA256,
            "sample_axis": "circuit_measurement_time_s",
            "eligible_motor_input": False,
        },
    }
    full = {
        "axis_name": "registered_fly_fgs_circuit_cell",
        "axis_labels": list(source["cell_ids"]),
        "axis_registration": "registered_fly_fgs_circuit_bundle_order",
        "axis_sha256": source["cell_axis_sha256"],
        "axis_labels_sha256": source["cell_ids_sha256"],
        "source_manifest_sha256": FLY_FGS_SOURCE_MANIFEST_SHA256,
        "sample_axis": "circuit_measurement_time_s",
        "eligible_motor_input": False,
    }
    metadata["full_circuit_voltage_v"] = dict(full)
    metadata["full_circuit_activity"] = dict(full)
    for prefix in ("generated", "held"):
        for suffix in (
            "rate_hz",
            "rate_measurement_time_s",
            "rate_availability_time_s",
        ):
            metadata["{}_dn_{}".format(prefix, suffix)] = {
                "axis_labels": list(CANONICAL_DN_AXIS)
            }
            metadata["{}_mn_{}".format(prefix, suffix)] = {
                "axis_labels": list(CANONICAL_MN_AXIS)
            }
    metadata["held_dn_rate_present"] = {"axis_labels": list(CANONICAL_DN_AXIS)}
    metadata["held_mn_rate_present"] = {"axis_labels": list(CANONICAL_MN_AXIS)}
    for state in ("natural", "effective"):
        for field in ("activation", "force_n", "phase_effect", "resonance_scale"):
            metadata["muscle_{}_{}".format(state, field)] = {
                "axis_labels": list(CANONICAL_MUSCLE_AXIS),
                "anatomical_side": "unknown",
            }
    for prefix, labels in (
        ("raw_app_actuation", ["raw_app_L", "raw_app_R"]),
        ("physical_actuation", ["physical_left", "physical_right"]),
    ):
        for suffix in (
            "stroke_rad",
            "stroke_velocity_rad_s",
            "stroke_acceleration_rad_s2",
            "angle_of_attack_rad",
            "deviation_rad",
            "generalized_torque_n_m",
        ):
            metadata["{}_{}".format(prefix, suffix)] = {
                "axis_labels": list(labels),
                "sample_semantics": "left-boundary zero-order hold over the transition",
            }
        axis_labels = [
            "{}_{}_{}".format(
                "raw_app" if prefix.startswith("raw_app") else "physical",
                lane,
                axis,
            )
            for lane in (("L", "R") if prefix.startswith("raw_app") else ("left", "right"))
            for axis in ("x", "y", "z")
        ]
        metadata["{}_wing_axis_torque_n_m".format(prefix)] = {
            "axis_labels": axis_labels,
            "sample_semantics": "left-boundary zero-order hold over the transition",
        }
    return metadata


def _canonical_load_tables(
    path: Path,
    manifest: Mapping[str, Any],
) -> Tuple[Mapping[str, Tuple[Mapping[str, Any], ...]], Set[Tuple[int, int]]]:
    """Load all four authoritative ledgers through exact immutable receipts."""

    table_descriptors = _exact_fields(
        manifest.get("tables"), CANONICAL_TABLE_IDS, "canonical run tables"
    )
    contracts = {
        "motor_events": (
            "motor_events.json",
            "events",
            ("path", "sha256", "count", "timing_semantics"),
            {"timing_semantics": "discrete; never interpolate"},
            ("schema_version", "events"),
        ),
        "rate_samples": (
            "rate_samples.json",
            "samples",
            ("path", "sha256", "count", "timing_semantics"),
            {
                "timing_semantics": (
                    "separate measurement and causal availability clocks"
                )
            },
            ("schema_version", "samples"),
        ),
        "circuit_observations": (
            "circuit_observations.json",
            "observations",
            ("path", "sha256", "count"),
            {},
            ("schema_version", "observations"),
        ),
        "intervention_activity": (
            "intervention_activity.json",
            "records",
            (
                "path",
                "sha256",
                "count",
                "timing_semantics",
                "definitions",
            ),
            {"timing_semantics": "exact half-open component intervals"},
            ("schema_version", "records", "contract_notice"),
        ),
    }
    loaded: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
    paths: Set[Path] = set()
    identities: Set[Tuple[int, int]] = set()
    for table_id in CANONICAL_TABLE_IDS:
        filename, record_key, descriptor_fields, fixed, payload_fields = contracts[
            table_id
        ]
        descriptor = _exact_fields(
            table_descriptors[table_id],
            descriptor_fields,
            "canonical table {} descriptor".format(table_id),
        )
        _require(
            descriptor.get("path") == filename,
            "canonical table {} has a noncanonical path".format(table_id),
        )
        for field, expected in fixed.items():
            _require(
                descriptor.get(field) == expected,
                "canonical table {} {} is not exact".format(table_id, field),
            )
        table_path = _safe_local_path(
            path.parent,
            descriptor.get("path"),
            "canonical table {} path".format(table_id),
        )
        stat = table_path.stat()
        identity = (stat.st_dev, stat.st_ino)
        _require(
            table_path not in paths and identity not in identities,
            "canonical tables may not alias another authoritative file",
        )
        paths.add(table_path)
        identities.add(identity)
        expected_sha = _require_sha256(
            descriptor.get("sha256"),
            "canonical table {} sha256".format(table_id),
        )
        _require(
            _sha256_file(table_path) == expected_sha,
            "canonical table {} checksum mismatch".format(table_id),
        )
        payload = _exact_fields(
            _load_json(table_path, "canonical table {}".format(table_id)),
            payload_fields,
            "canonical table {} payload".format(table_id),
        )
        _require(
            payload.get("schema_version") == "1.0.0",
            "canonical table {} schema is unsupported".format(table_id),
        )
        if table_id == "intervention_activity":
            _require(
                payload.get("contract_notice")
                == (
                    "Active-ID receipts are complete; immutable definitions are in the "
                    "checkpoint receipt when an endpoint checkpoint was supplied."
                ),
                "canonical intervention table contract notice is not exact",
            )
        records = payload.get(record_key)
        _require(
            isinstance(records, list)
            and all(isinstance(record, Mapping) for record in records),
            "canonical table {} records must be objects".format(table_id),
        )
        count = descriptor.get("count")
        _require(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count == len(records),
            "canonical table {} count is inconsistent".format(table_id),
        )
        loaded[table_id] = tuple(records)
    return loaded, identities


def _canonical_load_arrays(
    path: Path,
    manifest: Mapping[str, Any],
    config: CanonicalClosedLoopConfig,
    event_count: int,
    reserved_identities: Set[Tuple[int, int]],
) -> Tuple[Mapping[str, np.ndarray], int]:
    """Verify every canonical chunk, logical array hash, shape, and axis."""

    expected_shapes = dict(_canonical_expected_array_shapes(config, event_count))
    if event_count == 0:
        expected_shapes.pop("motor_event_numeric")
    expected_metadata = dict(_canonical_expected_axis_metadata())
    if event_count:
        expected_metadata["motor_event_numeric"] = {
            "column_labels": [
                "event_time_s",
                "availability_time_s",
                "wingbeat_phase_rad",
                "rate_hz",
                "emission_probability",
                "source_measurement_time_s_or_zero",
                "phase_crossing_index",
                "delivered_to_mechanics",
                "applied_to_muscle_state",
                "suppressed_by_mechanics_intervention",
            ],
            "column_units": ["s", "s", "rad", "Hz", "1", "s", "1", "1", "1", "1"],
            # Filled from the authoritative event table by the caller below.
        }
    descriptors = _exact_fields(
        manifest.get("arrays"),
        tuple(expected_shapes),
        "canonical run arrays",
    )
    arrays: Dict[str, np.ndarray] = {}
    seen_paths: Set[Path] = set()
    seen_identities = set(reserved_identities)
    chunk_count = 0
    for array_id in sorted(expected_shapes):
        label = "canonical array {}".format(array_id)
        descriptor_value = descriptors[array_id]
        _require(
            isinstance(descriptor_value, Mapping),
            "{} descriptor must be an object".format(label),
        )
        descriptor = descriptor_value
        metadata = {
            key: value
            for key, value in descriptor.items()
            if key not in CANONICAL_ARRAY_DESCRIPTOR_BASE_FIELDS
        }
        if array_id == "motor_event_numeric":
            expected_event_metadata = expected_metadata[array_id]
            _require(
                set(metadata)
                == set(expected_event_metadata).union(
                    {"source_measurement_time_presence"}
                ),
                "{} axis metadata fields are not exact".format(label),
            )
            for key, expected in expected_event_metadata.items():
                _require(
                    metadata.get(key) == expected,
                    "{} {} is not exact".format(label, key),
                )
            presence = metadata.get("source_measurement_time_presence")
            _require(
                isinstance(presence, list)
                and len(presence) == event_count
                and all(isinstance(value, bool) for value in presence),
                "{} source-measurement presence mask is invalid".format(label),
            )
        else:
            _require(
                metadata == expected_metadata.get(array_id, {}),
                "{} axis metadata does not match the registered contract".format(
                    label
                ),
            )
        _require(
            set(descriptor)
            == set(CANONICAL_ARRAY_DESCRIPTOR_BASE_FIELDS).union(metadata),
            "{} descriptor fields are not exact".format(label),
        )
        expected_shape = expected_shapes[array_id]
        _require(
            descriptor.get("shape") == list(expected_shape),
            "{} shape is inconsistent with the canonical clocks".format(label),
        )
        expected_dtype_name = (
            "bool" if array_id in ("held_dn_rate_present", "held_mn_rate_present") else "float64"
        )
        _require(
            descriptor.get("dtype") == expected_dtype_name,
            "{} dtype must be {}".format(label, expected_dtype_name),
        )
        _require(
            isinstance(descriptor.get("unit"), str)
            and bool(descriptor.get("unit"))
            and isinstance(descriptor.get("provenance"), str)
            and bool(descriptor.get("provenance")),
            "{} unit/provenance must be non-empty".format(label),
        )
        _require_sha256(
            descriptor.get("logical_npy_sha256"),
            "{}.logical_npy_sha256".format(label),
        )
        chunks = descriptor.get("chunks")
        _require(
            isinstance(chunks, list) and bool(chunks),
            "{} requires at least one chunk".format(label),
        )
        cursor = 0
        for index, chunk_value in enumerate(chunks):
            chunk_label = "{}.chunks[{}]".format(label, index)
            chunk = _exact_fields(
                chunk_value,
                ("path", "start", "stop", "shape", "sha256"),
                chunk_label,
            )
            array_stem = "{}-{}".format(
                array_id, hashlib.sha256(array_id.encode("utf-8")).hexdigest()[:10]
            )
            _require(
                chunk.get("path")
                == "arrays/{}/{:06d}.npy".format(array_stem, index),
                "{} path is not canonical".format(chunk_label),
            )
            chunk_path = _safe_local_path(
                path.parent, chunk.get("path"), "{}.path".format(chunk_label)
            )
            stat = chunk_path.stat()
            identity = (stat.st_dev, stat.st_ino)
            _require(
                chunk_path not in seen_paths and identity not in seen_identities,
                "canonical authoritative files may not alias one another",
            )
            seen_paths.add(chunk_path)
            seen_identities.add(identity)
            expected_sha = _require_sha256(
                chunk.get("sha256"), "{}.sha256".format(chunk_label)
            )
            _require(
                _sha256_file(chunk_path) == expected_sha,
                "canonical chunk checksum mismatch: {}".format(chunk_path),
            )
            start = chunk.get("start")
            stop = chunk.get("stop")
            _require(
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(stop, int)
                and not isinstance(stop, bool)
                and start == cursor
                and stop > start,
                "{} interval is empty or non-contiguous".format(chunk_label),
            )
            chunk_shape = chunk.get("shape")
            _require(
                isinstance(chunk_shape, list)
                and chunk_shape == [stop - start, *expected_shape[1:]],
                "{} shape is inconsistent".format(chunk_label),
            )
            cursor = stop
            chunk_count += 1
        _require(
            cursor == expected_shape[0],
            "{} chunks do not cover the first axis".format(label),
        )
        try:
            reconstructed = read_chunked_array(path.parent, descriptor)
        except Exception as exc:
            raise AuditError("{} reconstruction failed: {}".format(label, exc)) from exc
        _require(
            tuple(reconstructed.shape) == expected_shape
            and reconstructed.dtype == np.dtype(expected_dtype_name),
            "{} reconstructed shape/dtype mismatch".format(label),
        )
        _require(
            bool(np.all(np.isfinite(reconstructed))),
            "{} contains NaN or infinity".format(label),
        )
        _require(
            _logical_npy_sha256(reconstructed)
            == descriptor.get("logical_npy_sha256"),
            "{} logical NPY checksum mismatch".format(label),
        )
        arrays[array_id] = reconstructed
    return arrays, chunk_count


def _canonical_bundle_sha256_from_files(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    tables: Mapping[str, Tuple[Mapping[str, Any], ...]],
) -> str:
    digest = hashlib.sha256()
    descriptors = manifest["arrays"]
    for name in sorted(arrays):
        descriptor = descriptors[name]
        array = np.asarray(arrays[name])
        canonical_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(
            array.astype(canonical_dtype, copy=False)
        )
        axis_metadata = {
            key: value
            for key, value in descriptor.items()
            if key not in CANONICAL_ARRAY_DESCRIPTOR_BASE_FIELDS
        }
        header = {
            "name": name,
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
            "unit": descriptor["unit"],
            "provenance": descriptor["provenance"],
            "axis_metadata": axis_metadata or None,
        }
        digest.update(
            json.dumps(
                header,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
        digest.update(b"\0")
    digest.update(
        json.dumps(
            {
                "event_records": tables["motor_events"],
                "rate_records": tables["rate_samples"],
                "circuit_records": tables["circuit_observations"],
                "intervention_records": tables["intervention_activity"],
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _finite_json_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(
                isinstance(key, str), "{} contains a non-string key".format(label)
            )
            _finite_json_tree(child, "{}.{}".format(label, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_json_tree(child, "{}[{}]".format(label, index))
    elif isinstance(value, float):
        _require(math.isfinite(value), "{} contains a non-finite value".format(label))
    else:
        _require(
            value is None or isinstance(value, (str, int, bool)),
            "{} contains unsupported JSON data".format(label),
        )


def _canonical_audit_circuit_table(
    config: CanonicalClosedLoopConfig,
    arrays: Mapping[str, np.ndarray],
    records: Sequence[Mapping[str, Any]],
) -> None:
    expected_fields = (
        "sample_index",
        "measurement_time_s",
        "availability_time_s",
        "initialization_mode",
        "requested_control",
        "wrapped_world_yaw_rad",
        "unwrapped_world_yaw_rad",
        "world_z_yaw_rate_rad_s",
        "nod1_voltage_v",
        "pooled_readout",
        "retinal_input_included",
        "full_cell_state_included",
        "full_cell_state_eligible_motor_input",
    )
    control_fields = (
        "heading_rad",
        "heading_velocity_rad_s",
        "figure_world_azimuth_rad",
        "figure_velocity_rad_s",
        "ground_velocity_rad_s",
    )
    _require(
        len(records) == config.circuit_sample_count,
        "canonical circuit observation count is incomplete",
    )
    contract = _registered_online_display_contract()
    cell_ids = contract["cell_ids"]
    nod_indices = {
        root_id: cell_ids.index("r{}".format(root_id))
        for root_id in FLY_FGS_NOD1_ROOT_IDS
    }
    for index, record_value in enumerate(records):
        record = _exact_fields(
            record_value,
            expected_fields,
            "canonical circuit observation {}".format(index),
        )
        expected_time = index * config.circuit_dt_s
        _require(
            record.get("sample_index") == index
            and record.get("measurement_time_s") == expected_time
            and record.get("availability_time_s") == expected_time
            and record.get("initialization_mode")
            == (
                "captured_static_preroll_gauge_matched"
                if index == 0
                else "body_scene_control_applied"
            ),
            "canonical circuit observation clock/reset semantics are inconsistent",
        )
        requested = _exact_fields(
            record.get("requested_control"),
            control_fields,
            "canonical circuit observation requested_control",
        )
        for field in control_fields:
            _finite_number(
                requested[field],
                "canonical circuit observation requested_control.{}".format(field),
            )
        for field in (
            "wrapped_world_yaw_rad",
            "unwrapped_world_yaw_rad",
            "world_z_yaw_rate_rad_s",
        ):
            _finite_number(record[field], "canonical circuit observation {}".format(field))
        nod = _exact_fields(
            record.get("nod1_voltage_v"),
            FLY_FGS_NOD1_ROOT_IDS,
            "canonical circuit NOD1 voltage",
        )
        for root_index, root_id in enumerate(FLY_FGS_NOD1_ROOT_IDS):
            voltage = _finite_number(nod[root_id], "canonical NOD1 voltage")
            _require(
                voltage == float(arrays["nod1_voltage_v"][index, root_index])
                and voltage
                == float(
                    arrays["full_circuit_voltage_v"][
                        index, nod_indices[root_id]
                    ]
                ),
                "canonical NOD1 table/full-state arrays disagree",
            )
        _require(
            record.get("retinal_input_included") is True
            and record.get("full_cell_state_included") is True
            and record.get("full_cell_state_eligible_motor_input") is False,
            "canonical online circuit display/motor eligibility flags are invalid",
        )
        _require(
            isinstance(record.get("pooled_readout"), Mapping),
            "canonical circuit pooled readout must be an object",
        )
        _finite_json_tree(
            record["pooled_readout"], "canonical circuit pooled readout"
        )
    expected_clock = np.arange(config.circuit_sample_count, dtype=float) * config.circuit_dt_s
    _require(
        np.array_equal(arrays["circuit_measurement_time_s"], expected_clock)
        and np.array_equal(arrays["circuit_availability_time_s"], expected_clock),
        "canonical circuit arrays do not follow the exact 5 ms clock",
    )


def _canonical_audit_rate_table(
    config: CanonicalClosedLoopConfig,
    arrays: Mapping[str, np.ndarray],
    records: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "record_kind",
        "bridge_tick_index",
        "stage",
        "target_name",
        "raw_app_side",
        "anatomical_side",
        "measurement_time_s",
        "availability_time_s",
        "rate_hz",
        "source_measurement_time_s",
        "semantics",
    )
    axes = {
        "dn": tuple((lane, "DNp26") for lane in CANONICAL_RAW_LANES),
        "mn": tuple(
            (lane, motor)
            for lane in CANONICAL_RAW_LANES
            for motor in CANONICAL_MOTORS
        ),
    }
    rows: Dict[Tuple[int, str, str, str, str], Mapping[str, Any]] = {}
    actual_sequence: List[Tuple[int, str, str, str, str]] = []
    generated_history: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for index, record_value in enumerate(records):
        record = _exact_fields(
            record_value, fields, "canonical rate record {}".format(index)
        )
        kind = record.get("record_kind")
        stage = record.get("stage")
        lane = record.get("raw_app_side")
        target = record.get("target_name")
        tick = record.get("bridge_tick_index")
        _require(
            kind in ("generated", "held_at_interval_start")
            and stage in axes
            and lane in CANONICAL_RAW_LANES
            and (lane, target) in axes[stage]
            and isinstance(tick, int)
            and not isinstance(tick, bool)
            and 0 <= tick < config.bridge_interval_count,
            "canonical rate record identity is invalid",
        )
        _require(
            record.get("anatomical_side") == "unknown"
            and record.get("semantics") == "inferred_rate",
            "canonical rate record laterality/semantics are invalid",
        )
        for field in ("measurement_time_s", "availability_time_s", "rate_hz"):
            _finite_number(record[field], "canonical rate {}".format(field))
        _require(
            float(record["rate_hz"]) >= 0.0
            and float(record["availability_time_s"])
            >= float(record["measurement_time_s"]),
            "canonical rate values violate causal/nonnegative constraints",
        )
        source_time = record.get("source_measurement_time_s")
        _require(
            source_time is None
            or (
                isinstance(source_time, (int, float))
                and not isinstance(source_time, bool)
                and math.isfinite(float(source_time))
            ),
            "canonical rate source measurement time is invalid",
        )
        key = (tick, stage, kind, lane, str(target))
        _require(key not in rows, "canonical rate table contains duplicate channels")
        rows[key] = record
        actual_sequence.append(key)
        if kind == "generated":
            _require(
                float(record["measurement_time_s"]) == tick * config.bridge_dt_s,
                "generated canonical rate is not measured on its bridge tick",
            )
            generated_history.setdefault((stage, lane, str(target)), []).append(record)

    expected_sequence: List[Tuple[int, str, str, str, str]] = []
    for tick in range(config.bridge_interval_count):
        start_s = tick * config.bridge_dt_s
        for stage in ("dn", "mn"):
            for lane, target in axes[stage]:
                key = (tick, stage, "generated", lane, target)
                _require(key in rows, "canonical generated-rate inventory is incomplete")
                expected_sequence.append(key)
            for column, (lane, target) in enumerate(axes[stage]):
                history = generated_history.get((stage, lane, target), [])
                causal = [
                    item
                    for item in history
                    if float(item["availability_time_s"]) <= start_s + 1.0e-12
                ]
                held_key = (tick, stage, "held_at_interval_start", lane, target)
                expected_held = causal[-1] if causal else None
                present_array = arrays["held_{}_rate_present".format(stage)]
                if expected_held is None:
                    _require(
                        held_key not in rows
                        and not bool(present_array[tick, column]),
                        "canonical held-rate presence predates causal availability",
                    )
                else:
                    _require(
                        held_key in rows and bool(present_array[tick, column]),
                        "canonical held-rate inventory is incomplete",
                    )
                    observed = rows[held_key]
                    for field in (
                        "measurement_time_s",
                        "availability_time_s",
                        "rate_hz",
                        "source_measurement_time_s",
                        "semantics",
                    ):
                        _require(
                            observed[field] == expected_held[field],
                            "canonical held rate is not the latest causal generated rate",
                        )
                for prefix, row in (
                    ("generated", rows[(tick, stage, "generated", lane, target)]),
                    ("held", rows.get(held_key)),
                ):
                    for suffix, record_field in (
                        ("rate_hz", "rate_hz"),
                        ("rate_measurement_time_s", "measurement_time_s"),
                        ("rate_availability_time_s", "availability_time_s"),
                    ):
                        array_value = float(
                            arrays["{}_{}_{}".format(prefix, stage, suffix)][
                                tick, column
                            ]
                        )
                        expected_value = 0.0 if row is None else float(row[record_field])
                        _require(
                            array_value == expected_value,
                            "canonical rate table and array values disagree",
                        )
            held_order = (
                axes[stage]
                if stage == "dn"
                else tuple(
                    (lane, motor)
                    for motor in CANONICAL_MOTORS
                    for lane in CANONICAL_RAW_LANES
                )
            )
            expected_sequence.extend(
                (tick, stage, "held_at_interval_start", lane, target)
                for lane, target in held_order
                if (tick, stage, "held_at_interval_start", lane, target) in rows
            )
    _require(
        actual_sequence == expected_sequence,
        "canonical rate table ordering is not the canonical bridge ordering",
    )


def _canonical_audit_event_table(
    config: CanonicalClosedLoopConfig,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    rate_records: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "generated_by_bridge",
        "event_id",
        "motor_neuron",
        "muscle",
        "raw_app_side",
        "anatomical_side",
        "event_time_s",
        "availability_time_s",
        "wingbeat_phase_rad",
        "rate_hz",
        "emission_probability",
        "source_measurement_time_s",
        "generator_seed",
        "phase_crossing_index",
        "semantics",
        "delivered_to_mechanics",
        "nmj_delivery_status",
        "mechanics_disposition",
        "applied_to_muscle_state",
        "suppressed_by_mechanics_intervention",
    )
    seen: Set[str] = set()
    previous_order: Optional[Tuple[float, str]] = None
    numeric_rows: List[List[float]] = []
    source_presence: List[bool] = []
    held_mn = {
        (
            record["bridge_tick_index"],
            record["raw_app_side"],
            record["target_name"],
        ): record
        for record in rate_records
        if record.get("record_kind") == "held_at_interval_start"
        and record.get("stage") == "mn"
    }
    for index, event_value in enumerate(events):
        event = _exact_fields(
            event_value, fields, "canonical motor event {}".format(index)
        )
        event_id = event.get("event_id")
        motor = event.get("motor_neuron")
        muscle = event.get("muscle")
        lane = event.get("raw_app_side")
        _require(
            isinstance(event_id, str)
            and bool(event_id)
            and event_id not in seen
            and motor in CANONICAL_MOTORS
            and muscle == motor.removeprefix("MN-")
            and lane in CANONICAL_RAW_LANES,
            "canonical motor event identity/pathway is invalid",
        )
        seen.add(event_id)
        _require(
            event.get("generated_by_bridge") is True
            and event.get("anatomical_side") == "unknown"
            and event.get("semantics") == "seeded_synthetic"
            and isinstance(event.get("generator_seed"), int)
            and not isinstance(event.get("generator_seed"), bool)
            and event["generator_seed"] >= 0
            and isinstance(event.get("phase_crossing_index"), int)
            and not isinstance(event.get("phase_crossing_index"), bool)
            and event["phase_crossing_index"] >= 0,
            "canonical motor event generator semantics are invalid",
        )
        for field in (
            "event_time_s",
            "availability_time_s",
            "wingbeat_phase_rad",
            "rate_hz",
            "emission_probability",
        ):
            _finite_number(event[field], "canonical motor event {}".format(field))
        event_time = float(event["event_time_s"])
        availability = float(event["availability_time_s"])
        _require(
            0.0 <= event_time < config.duration_s
            and availability >= event_time
            and float(event["rate_hz"]) >= 0.0
            and 0.0 <= float(event["emission_probability"]) <= 1.0,
            "canonical motor event timing/rate/probability is invalid",
        )
        order = (event_time, event_id)
        _require(
            previous_order is None or order >= previous_order,
            "canonical motor events are not in deterministic time/ID order",
        )
        previous_order = order
        tick = int(math.floor((event_time + 1.0e-15) / config.bridge_dt_s))
        held = held_mn.get((tick, lane, motor))
        _require(
            held is not None
            and event["rate_hz"] == held["rate_hz"]
            and event["source_measurement_time_s"]
            == held["source_measurement_time_s"],
            "canonical motor event is detached from its held MN source",
        )
        disposition = event.get("mechanics_disposition")
        delivered = disposition in ("applied", "suppressed")
        _require(
            disposition in ("applied", "suppressed", "pending_at_episode_end")
            and event.get("delivered_to_mechanics") is delivered
            and event.get("nmj_delivery_status")
            == (
                "delivered_to_mechanics"
                if delivered
                else "pending_at_episode_end"
            )
            and event.get("applied_to_muscle_state") is (disposition == "applied")
            and event.get("suppressed_by_mechanics_intervention")
            is (disposition == "suppressed"),
            "canonical motor-event delivery/disposition flags disagree",
        )
        source = event.get("source_measurement_time_s")
        _require(
            source is None
            or (
                isinstance(source, (int, float))
                and not isinstance(source, bool)
                and math.isfinite(float(source))
            ),
            "canonical event source measurement time is invalid",
        )
        source_presence.append(source is not None)
        numeric_rows.append(
            [
                event_time,
                availability,
                float(event["wingbeat_phase_rad"]),
                float(event["rate_hz"]),
                float(event["emission_probability"]),
                0.0 if source is None else float(source),
                float(event["phase_crossing_index"]),
                1.0 if delivered else 0.0,
                1.0 if disposition == "applied" else 0.0,
                1.0 if disposition == "suppressed" else 0.0,
            ]
        )
    descriptor = manifest["arrays"].get("motor_event_numeric")
    if events:
        _require(
            np.array_equal(
                arrays["motor_event_numeric"], np.asarray(numeric_rows, dtype=float)
            )
            and descriptor.get("source_measurement_time_presence")
            == source_presence,
            "canonical motor-event table and numeric array disagree",
        )
    else:
        _require(
            descriptor is None and "motor_event_numeric" not in arrays,
            "zero-event canonical run may not invent a motor-event array",
        )


def _canonical_audit_intervention_table(
    config: CanonicalClosedLoopConfig,
    records: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "stage",
        "tick_index",
        "interval_start_s",
        "interval_end_s",
        "active_intervention_ids",
        "schedule_sha256",
    )
    _require(
        len(records) == config.bridge_interval_count * 6,
        "canonical intervention activity ledger is incomplete",
    )
    mechanics_schedule: Optional[str] = None
    for bridge_tick in range(config.bridge_interval_count):
        base = bridge_tick * 6
        for offset in range(5):
            record = _exact_fields(
                records[base + offset],
                fields,
                "canonical mechanics intervention activity",
            )
            physics_tick = bridge_tick * 5 + offset
            schedule = _require_sha256(
                record.get("schedule_sha256"),
                "canonical mechanics intervention schedule",
            )
            if mechanics_schedule is None:
                mechanics_schedule = schedule
            _require(
                record.get("stage") == "mechanics_force"
                and record.get("tick_index") == physics_tick
                and _close(
                    record.get("interval_start_s"),
                    physics_tick * config.physics_dt_s,
                )
                and _close(
                    record.get("interval_end_s"),
                    (physics_tick + 1) * config.physics_dt_s,
                )
                and schedule == mechanics_schedule,
                "canonical mechanics intervention clock/schedule is inconsistent",
            )
            active = record.get("active_intervention_ids")
            _require(
                isinstance(active, list)
                and all(isinstance(item, str) and bool(item) for item in active)
                and len(active) == len(set(active)),
                "canonical mechanics active-intervention IDs are invalid",
            )
        bridge = _exact_fields(
            records[base + 5],
            fields,
            "canonical bridge intervention activity",
        )
        _require(
            bridge.get("stage") == "bridge_command_generation"
            and bridge.get("tick_index") == bridge_tick
            and _close(
                bridge.get("interval_start_s"),
                bridge_tick * config.bridge_dt_s,
            )
            and _close(
                bridge.get("interval_end_s"),
                (bridge_tick + 1) * config.bridge_dt_s,
            )
            and bridge.get("schedule_sha256") is None,
            "canonical bridge intervention clock/schedule is inconsistent",
        )
        active = bridge.get("active_intervention_ids")
        _require(
            isinstance(active, list)
            and all(isinstance(item, str) and bool(item) for item in active)
            and len(active) == len(set(active)),
            "canonical bridge active-intervention IDs are invalid",
        )


def _canonical_audit_clocks_and_tables(
    config: CanonicalClosedLoopConfig,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    tables: Mapping[str, Tuple[Mapping[str, Any], ...]],
) -> None:
    physics_clock = np.arange(
        config.bridge_interval_count * 5 + 1, dtype=float
    ) * config.physics_dt_s
    bridge_clock = np.arange(
        config.bridge_interval_count, dtype=float
    ) * config.bridge_dt_s
    _require(
        np.array_equal(arrays["physics_time_s"], physics_clock)
        and np.array_equal(arrays["bridge_interval_start_s"], bridge_clock)
        and np.array_equal(
            arrays["muscle_natural_endpoint_time_s"], physics_clock[1:]
        ),
        "canonical physics/bridge/muscle clocks are inconsistent",
    )
    phase = arrays["wing_phase_unwrapped_rad"]
    phase_path = arrays["bridge_phase_path_unwrapped_rad"]
    _require(
        bool(np.all(np.diff(phase) >= -1.0e-12))
        and np.array_equal(phase_path[:, 0], phase[:-1:5])
        and np.array_equal(phase_path[:, 1:].reshape(-1), phase[1:]),
        "canonical authoritative wing phase/path arrays disagree",
    )
    _canonical_audit_circuit_table(
        config, arrays, tables["circuit_observations"]
    )
    _canonical_audit_rate_table(config, arrays, tables["rate_samples"])
    _canonical_audit_event_table(
        config,
        arrays,
        manifest,
        tables["rate_samples"],
        tables["motor_events"],
    )
    _canonical_audit_intervention_table(
        config, tables["intervention_activity"]
    )


def _canonical_load_checkpoint(
    path: Path,
    manifest: Mapping[str, Any],
    config: CanonicalClosedLoopConfig,
) -> Tuple[Mapping[str, Any], Tuple[int, int]]:
    descriptor = _exact_fields(
        manifest.get("checkpoint"),
        (
            "schema_version",
            "runtime_version",
            "payload_sha256",
            "bridge_tick_index",
            "physics_tick_index",
            "component_payload_sha256",
            "intervention_contracts",
            "path",
            "file_sha256",
        ),
        "canonical checkpoint descriptor",
    )
    _require(
        descriptor.get("path") == "checkpoint.json",
        "canonical checkpoint path is not exact",
    )
    checkpoint_path = _safe_local_path(
        path.parent, descriptor.get("path"), "canonical checkpoint path"
    )
    _require(
        _sha256_file(checkpoint_path)
        == _require_sha256(
            descriptor.get("file_sha256"), "canonical checkpoint file_sha256"
        ),
        "canonical checkpoint file checksum mismatch",
    )
    payload = _load_json(checkpoint_path, "canonical checkpoint")
    try:
        validated = CanonicalClosedLoopCheckpoint(payload).to_dict()
    except Exception as exc:
        raise AuditError("canonical checkpoint contract failed: {}".format(exc)) from exc
    _require(validated == payload, "canonical checkpoint normalization changed its payload")
    components = _exact_fields(
        payload.get("components"),
        ("circuit", "bridge", "mechanics", "physics"),
        "canonical checkpoint components",
    )
    component_digests: Dict[str, str] = {}
    for name in ("circuit", "bridge", "mechanics", "physics"):
        child = components[name]
        _require(
            isinstance(child, Mapping),
            "canonical checkpoint component {} must be an object".format(name),
        )
        supplied_value = child.get("payload_sha256")
        _require(
            isinstance(supplied_value, str),
            "canonical checkpoint {} payload_sha256 must be a string".format(
                name
            ),
        )
        supplied = (
            supplied_value.removeprefix("sha256:")
            if name == "physics" and supplied_value.startswith("sha256:")
            else supplied_value
        )
        _require_sha256(
            supplied,
            "canonical checkpoint {} normalized payload_sha256".format(name),
        )
        _require(
            name == "physics" or supplied_value == supplied,
            "only the native FlyBody physics child may prefix its digest",
        )
        try:
            computed = _checkpoint_component_payload_sha256(name, child)
        except Exception as exc:
            raise AuditError(
                "canonical checkpoint {} source contract failed: {}".format(
                    name, exc
                )
            ) from exc
        _require(
            supplied == computed,
            "canonical checkpoint {} component checksum mismatch".format(name),
        )
        component_digests[name] = supplied
    bridge_contracts = components["bridge"].get("interventions")
    mechanics_config = components["mechanics"].get("config")
    mechanics_state = components["mechanics"].get("state")
    _require(
        isinstance(bridge_contracts, list)
        and isinstance(mechanics_config, Mapping)
        and isinstance(mechanics_config.get("muscle_interventions"), list)
        and isinstance(mechanics_state, Mapping),
        "canonical checkpoint intervention definitions are missing",
    )
    mechanics_contracts = mechanics_config["muscle_interventions"]
    expected_schedule_sha = _canonical_json_sha256(mechanics_contracts)
    _require(
        mechanics_state.get("intervention_schedule_sha256")
        == expected_schedule_sha,
        "canonical checkpoint mechanics schedule checksum mismatch",
    )
    intervention_contracts = {
        "bridge_command_generation": bridge_contracts,
        "mechanics_force": mechanics_contracts,
        "mechanics_schedule_sha256": expected_schedule_sha,
    }
    for field, expected in (
        ("schema_version", CANONICAL_CLOSED_LOOP_SCHEMA_VERSION),
        ("runtime_version", CANONICAL_CLOSED_LOOP_RUNTIME_VERSION),
        ("payload_sha256", payload["payload_sha256"]),
        ("bridge_tick_index", config.bridge_interval_count),
        ("physics_tick_index", config.bridge_interval_count * 5),
        ("component_payload_sha256", component_digests),
        ("intervention_contracts", intervention_contracts),
    ):
        _require(
            descriptor.get(field) == expected,
            "canonical checkpoint descriptor {} is inconsistent".format(field),
        )
    _require(
        payload.get("config") == dict(config.to_dict()),
        "canonical checkpoint configuration differs from its run manifest",
    )
    stat = checkpoint_path.stat()
    return payload, (stat.st_dev, stat.st_ino)


def _canonical_active_ids(
    contracts: Sequence[Mapping[str, Any]], time_s: float
) -> Tuple[str, ...]:
    result: List[str] = []
    seen: Set[str] = set()
    for index, contract in enumerate(contracts):
        _require(
            isinstance(contract, Mapping),
            "canonical intervention contract must be an object",
        )
        intervention_id = contract.get("intervention_id")
        start = _finite_number(
            contract.get("start_s"),
            "canonical intervention contract start_s",
        )
        end = _finite_number(
            contract.get("end_s"), "canonical intervention contract end_s"
        )
        _require(
            isinstance(intervention_id, str)
            and bool(intervention_id)
            and intervention_id not in seen
            and start >= 0.0
            and end > start,
            "canonical intervention contract identity/interval is invalid",
        )
        seen.add(intervention_id)
        if "contract_sha256" in contract:
            unsigned = {
                key: value
                for key, value in contract.items()
                if key != "contract_sha256"
            }
            _require(
                contract.get("contract_sha256") == _canonical_json_sha256(unsigned),
                "canonical mechanics intervention contract checksum mismatch",
            )
        if start <= time_s < end:
            result.append(intervention_id)
    return tuple(result)


def _canonical_event_checkpoint_record(
    event: Mapping[str, Any]
) -> Mapping[str, Any]:
    excluded = {
        "generated_by_bridge",
        "delivered_to_mechanics",
        "nmj_delivery_status",
        "mechanics_disposition",
        "applied_to_muscle_state",
        "suppressed_by_mechanics_intervention",
    }
    return {key: value for key, value in event.items() if key not in excluded}


def _canonical_audit_checkpoint_state(
    manifest: Mapping[str, Any],
    config: CanonicalClosedLoopConfig,
    checkpoint: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    tables: Mapping[str, Tuple[Mapping[str, Any], ...]],
) -> None:
    state = checkpoint["state"]
    _require(
        state.get("bridge_tick_index") == config.bridge_interval_count
        and state.get("physics_tick_index") == config.bridge_interval_count * 5
        and state.get("circuit_sample_count") == config.circuit_sample_count
        and state.get("last_circuit_sample_index")
        == config.circuit_sample_count - 1,
        "canonical checkpoint endpoint clocks are inconsistent",
    )
    body = _exact_fields(
        state.get("body_state"),
        (
            "position_world_m",
            "velocity_world_m_s",
            "quaternion_body_to_world",
            "angular_velocity_body_rad_s",
        ),
        "canonical checkpoint body state",
    )
    body_arrays = {
        "position_world_m": "body_position_world_m",
        "velocity_world_m_s": "body_velocity_world_m_s",
        "quaternion_body_to_world": "body_quaternion_body_to_world",
        "angular_velocity_body_rad_s": "body_angular_velocity_body_rad_s",
    }
    for state_field, array_id in body_arrays.items():
        _require(
            np.array_equal(
                np.asarray(body[state_field], dtype=float), arrays[array_id][-1]
            ),
            "canonical checkpoint body endpoint differs from scientific arrays",
        )
    runtime = manifest["runtime"]
    receipts = checkpoint["receipts"]
    _require(
        receipts.get("backend_name") == runtime["physics_backend"]
        and receipts.get("aerodynamic_owner") == runtime["aerodynamic_owner"]
        and receipts.get("articulated_physics_telemetry_available") is True
        and receipts.get("effector_mapping") == manifest["effector_mapping"]
        and receipts.get("source_limitations") == manifest["source_limitations"],
        "canonical checkpoint runtime receipts differ from the run manifest",
    )
    components = checkpoint["components"]
    physics = components["physics"]
    flybody = runtime["receipts"]["flybody"]
    compiled = flybody.get("compiled_model_fingerprint")
    _require(
        isinstance(compiled, Mapping)
        and physics.get("compiled_model_sha256") == compiled.get("sha256")
        and physics.get("worker_versions") == flybody.get("worker_versions")
        and physics.get("wing_dof_order") == flybody.get("wing_dof_order")
        and physics.get("physics_timestep_s") == config.physics_dt_s,
        "canonical checkpoint physics identity differs from nested FlyBody receipt",
    )
    bridge_contracts = components["bridge"]["interventions"]
    mechanics_contracts = components["mechanics"]["config"][
        "muscle_interventions"
    ]
    definitions = manifest["tables"]["intervention_activity"]["definitions"]
    _require(
        definitions
        == {
            "bridge_command_generation": bridge_contracts,
            "mechanics_force": mechanics_contracts,
            "mechanics_schedule_sha256": _canonical_json_sha256(
                mechanics_contracts
            ),
        },
        "canonical intervention table definitions differ from checkpoint",
    )
    for record in tables["intervention_activity"]:
        contracts = (
            bridge_contracts
            if record["stage"] == "bridge_command_generation"
            else mechanics_contracts
        )
        _require(
            tuple(record["active_intervention_ids"])
            == _canonical_active_ids(contracts, float(record["interval_start_s"])),
            "canonical intervention activity differs from immutable definitions",
        )

    events_by_id = {
        event["event_id"]: _canonical_event_checkpoint_record(event)
        for event in tables["motor_events"]
    }
    bridge_state = components["bridge"].get("state")
    mechanics_state = components["mechanics"].get("state")
    _require(
        isinstance(bridge_state, Mapping) and isinstance(mechanics_state, Mapping),
        "canonical checkpoint bridge/mechanics state is missing",
    )
    delivered = {
        item["event_id"]: item for item in bridge_state.get("delivered_events", [])
    }
    pending = {
        item["value"]["event_id"]: item["value"]
        for item in bridge_state.get("event_pending", [])
    }
    applied = {
        item["event_id"]: item for item in mechanics_state.get("applied_events", [])
    }
    suppressed = {
        item["event_id"]: item
        for item in mechanics_state.get("suppressed_events", [])
    }
    expected_delivered = {
        event["event_id"]: events_by_id[event["event_id"]]
        for event in tables["motor_events"]
        if event["delivered_to_mechanics"]
    }
    expected_pending = {
        event["event_id"]: events_by_id[event["event_id"]]
        for event in tables["motor_events"]
        if not event["delivered_to_mechanics"]
    }
    expected_applied = {
        event["event_id"]: events_by_id[event["event_id"]]
        for event in tables["motor_events"]
        if event["applied_to_muscle_state"]
    }
    expected_suppressed = {
        event["event_id"]: events_by_id[event["event_id"]]
        for event in tables["motor_events"]
        if event["suppressed_by_mechanics_intervention"]
    }
    _require(
        len(delivered) == len(bridge_state.get("delivered_events", []))
        and len(pending) == len(bridge_state.get("event_pending", []))
        and len(applied) == len(mechanics_state.get("applied_events", []))
        and len(suppressed) == len(mechanics_state.get("suppressed_events", []))
        and delivered == expected_delivered
        and pending == expected_pending
        and applied == expected_applied
        and suppressed == expected_suppressed
        and set(mechanics_state.get("applied_event_ids", []))
        == set(expected_applied)
        and set(mechanics_state.get("suppressed_event_ids", []))
        == set(expected_suppressed)
        and not mechanics_state.get("pending_events"),
        "canonical checkpoint event ledgers differ from authoritative motor events",
    )


def _audit_canonical_runtime_receipts(runtime: Mapping[str, Any]) -> None:
    receipts = _exact_fields(
        runtime.get("receipts"),
        ("fly_fgs_runtime", "flybody"),
        "canonical runtime receipts",
    )
    source = _exact_fields(
        receipts.get("fly_fgs_runtime"),
        (
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
        ),
        "canonical fly-FGS runtime receipt",
    )
    registered = _registered_online_display_contract()
    assets = registered["asset_receipts"]
    _require(
        source.get("event") == "ready"
        and source.get("protocol_version") == FLY_FGS_RUNTIME_PROTOCOL_VERSION
        and isinstance(source.get("node_version"), str)
        and bool(source.get("node_version"))
        and source.get("snapshot_id") == FLY_FGS_SNAPSHOT_ID
        and source.get("source_manifest_sha256")
        == FLY_FGS_SOURCE_MANIFEST_SHA256
        and source.get("circuit_engine_sha256")
        == assets["circuit_engine"]["sha256"]
        and source.get("circuit_bundle_sha256")
        == assets["circuit_bundle"]["sha256"]
        and source.get("dt_s") == 0.005
        and source.get("sample_count") == 100
        and source.get("pre_roll_steps") == 48
        and source.get("motor_input_policy")
        == "four individual NOD1 voltage channels only"
        and source.get("full_cell_state_eligible_motor_input") is False
        and source.get("executable_asset_ids")
        == ["circuit_engine", "circuit_bundle"],
        "canonical fly-FGS runtime receipt differs from the registered source",
    )
    flybody = _exact_fields(
        receipts.get("flybody"),
        (
            "schema_version",
            "engine",
            "worker_versions",
            "worker_image_digest",
            "worker_image_digest_status",
            "worker_dependency_lock",
            "dependency_record_sha256",
            "compiled_model_fingerprint",
            "worker_config",
            "wing_dof_order",
            "fluid_geom_names",
            "fluid_geoms_contact_disabled",
            "ground_contact_topology",
            "ground_contact_telemetry",
            "released_policy_topology_equivalent",
            "tendon_count",
            "source",
            "licenses",
            "body_state_reference",
            "root_fluid_wrench_scope",
            "checkpoint_contract",
            "public_units",
        ),
        "canonical nested FlyBody receipt",
    )
    versions = flybody.get("worker_versions")
    lock = flybody.get("worker_dependency_lock")
    compiled = flybody.get("compiled_model_fingerprint")
    worker_config = flybody.get("worker_config")
    source_info = flybody.get("source")
    checkpoint_contract = flybody.get("checkpoint_contract")
    worker_image_digest = flybody.get("worker_image_digest")
    lock_digest = lock.get("sha256") if isinstance(lock, Mapping) else None
    dependency_records = flybody.get("dependency_record_sha256")
    _require(
        flybody.get("schema_version") == "1.0.0"
        and flybody.get("engine") == "FlyGym/FlyBody with native MuJoCo"
        and isinstance(versions, Mapping)
        and versions.get("flygym") == "2.1.0"
        and isinstance(versions.get("mujoco"), str)
        and versions["mujoco"].startswith("3.9.")
        and isinstance(lock, Mapping)
        and lock.get("name") == DEPENDENCY_LOCK_SOURCE_NAME
        and isinstance(lock_digest, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", lock_digest) is not None
        and isinstance(worker_image_digest, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", worker_image_digest)
        is not None
        and flybody.get("worker_image_digest_status") == "declared_oci_digest"
        and isinstance(dependency_records, Mapping)
        and set(dependency_records) == {"flygym", "mujoco"}
        and all(
            isinstance(value, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
            for value in dependency_records.values()
        )
        and isinstance(compiled, Mapping)
        and isinstance(compiled.get("sha256"), str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", compiled["sha256"])
        is not None
        and isinstance(worker_config, Mapping)
        and worker_config.get("timestep_s") == 0.0001
        and flybody.get("wing_dof_order")
        == list(REVIEWED_FLYBODY_WING_AXIS_ORDER)
        and flybody.get("fluid_geoms_contact_disabled") is True
        and flybody.get("ground_contact_topology") == "legs_only"
        and flybody.get("released_policy_topology_equivalent") is False
        and isinstance(flybody.get("tendon_count"), int)
        and not isinstance(flybody.get("tendon_count"), bool)
        and flybody["tendon_count"] >= 0
        and isinstance(source_info, Mapping)
        and source_info.get("flygym_tag") == "v2.1.0"
        and source_info.get("flygym_commit")
        == "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"
        and checkpoint_contract
        == {
            "schema_version": "1.0.0",
            "state_spec": "mjSTATE_INTEGRATION",
            "float_encoding": "float64_le_base64",
            "compatibility_bound_to_compiled_model": True,
            "fresh_adapter_exact_reentry_required": True,
        }
        and flybody.get("public_units") == "SI"
        and flybody.get("root_fluid_wrench_scope")
        == "root-total; no reviewed per-wing decomposition",
        "canonical nested FlyBody receipt is not the reviewed native contract",
    )


def _audit_canonical_run_manifest_impl(
    path: Path, manifest: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], int, int]:
    """Audit schema 1.1 independently of the historical schema 2.0 path."""

    _exact_fields(
        manifest,
        (
            "schema_version",
            "run_id",
            "scenario_id",
            "created_at_utc",
            "episode_status",
            "validation_status",
            "scientific_content_sha256",
            "source_kind",
            "authority_notice",
            "configuration",
            "clocks",
            "runtime",
            "effector_mapping",
            "signed_behavior_claim_policy",
            "source_limitations",
            "units_policy",
            "arrays",
            "tables",
            "checkpoint",
            "web_replay_projection",
        ),
        "canonical run manifest",
    )
    _require(
        manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION
        and manifest.get("source_kind") == CANONICAL_ARTIFACT_SOURCE_KIND
        and manifest.get("episode_status") == "complete"
        and manifest.get("validation_status") == "exploratory",
        "canonical run identity/status is invalid",
    )
    _require(
        isinstance(manifest.get("scenario_id"), str)
        and bool(manifest.get("scenario_id"))
        and isinstance(manifest.get("created_at_utc"), str)
        and bool(manifest.get("created_at_utc"))
        and isinstance(manifest.get("authority_notice"), str)
        and bool(manifest.get("authority_notice")),
        "canonical run scenario/time/authority strings are required",
    )
    try:
        config = CanonicalClosedLoopConfig.from_dict(manifest.get("configuration"))
    except Exception as exc:
        raise AuditError("canonical run configuration failed: {}".format(exc)) from exc
    _require(
        manifest.get("configuration") == dict(config.to_dict())
        and config.include_retinal_input is True
        and config.include_full_cell_state is True,
        "public canonical online run requires exact registered retinal/full state",
    )
    clocks = _exact_fields(
        manifest.get("clocks"),
        (
            "circuit_dt_s",
            "bridge_dt_s",
            "physics_dt_s",
            "circuit_to_bridge_ratio",
            "bridge_to_physics_ratio",
        ),
        "canonical run clocks",
    )
    _require(
        clocks
        == {
            "circuit_dt_s": config.circuit_dt_s,
            "bridge_dt_s": config.bridge_dt_s,
            "physics_dt_s": config.physics_dt_s,
            "circuit_to_bridge_ratio": 10,
            "bridge_to_physics_ratio": 5,
        },
        "canonical run clocks do not match their locked ratios",
    )
    runtime = _exact_fields(
        manifest.get("runtime"),
        (
            "python",
            "python_implementation",
            "numpy",
            "platform",
            "canonical_schema_version",
            "canonical_runtime_version",
            "physics_backend",
            "aerodynamic_owner",
            "receipts",
        ),
        "canonical run runtime",
    )
    _require(
        all(
            isinstance(runtime.get(field), str) and bool(runtime.get(field))
            for field in ("python", "python_implementation", "numpy", "platform")
        )
        and runtime.get("canonical_schema_version")
        == CANONICAL_CLOSED_LOOP_SCHEMA_VERSION
        and runtime.get("canonical_runtime_version")
        == CANONICAL_CLOSED_LOOP_RUNTIME_VERSION
        and runtime.get("physics_backend") == "flybody"
        and runtime.get("aerodynamic_owner") == "flybody",
        "canonical online public run must use the canonical FlyBody runtime",
    )
    _audit_canonical_runtime_receipts(runtime)
    try:
        effector = EffectorMappingReceipt.from_dict(manifest.get("effector_mapping"))
    except Exception as exc:
        raise AuditError("canonical effector mapping receipt failed: {}".format(exc)) from exc
    _require(
        dict(effector.to_dict()) == manifest.get("effector_mapping")
        and effector.anatomical_status == RAW_APP_LATERALITY_STATUS
        and effector.provenance == EFFECTOR_MAPPING_PROVENANCE
        and manifest.get("signed_behavior_claim_policy")
        == SIGNED_BEHAVIOR_CLAIM_POLICY
        and manifest.get("signed_behavior_claim_policy")
        == config.effector_signed_behavior_claim_policy
        and manifest.get("source_limitations") == list(CANONICAL_SOURCE_LIMITATIONS)
        and manifest.get("units_policy")
        == "SI internally; every array descriptor declares its unit",
        "canonical laterality/claim/source-limitations contract is not locked",
    )
    _require(
        manifest.get("effector_mapping")
        == dict(config.effector_mapping_receipt.to_dict()),
        "canonical effector receipt differs from the selected configuration",
    )
    tables, table_identities = _canonical_load_tables(path, manifest)
    checkpoint, checkpoint_identity = _canonical_load_checkpoint(
        path, manifest, config
    )
    _require(
        checkpoint_identity not in table_identities,
        "canonical checkpoint may not alias an authoritative table",
    )
    arrays, chunk_count = _canonical_load_arrays(
        path,
        manifest,
        config,
        len(tables["motor_events"]),
        set(table_identities).union({checkpoint_identity}),
    )
    _canonical_audit_clocks_and_tables(config, manifest, arrays, tables)
    try:
        _canonical_audit_checkpoint_state(
            manifest, config, checkpoint, arrays, tables
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise AuditError(
            "canonical checkpoint semantic binding failed: {}".format(exc)
        ) from exc

    bundle_sha = _canonical_bundle_sha256_from_files(manifest, arrays, tables)
    scientific_sha = _canonical_json_sha256(
        {
            "bundle_sha256": bundle_sha,
            "validation_status": manifest["validation_status"],
            "source_limitations": manifest["source_limitations"],
            "backend_name": runtime["physics_backend"],
            "aerodynamic_owner": runtime["aerodynamic_owner"],
            "effector_mapping": manifest["effector_mapping"],
        }
    )
    _require(
        manifest.get("scientific_content_sha256") == scientific_sha,
        "canonical scientific-content checksum does not derive from its arrays/tables",
    )
    identity = {
        "scenario_id": manifest["scenario_id"],
        "config": manifest["configuration"],
        "backend_name": runtime["physics_backend"],
        "aerodynamic_owner": runtime["aerodynamic_owner"],
        "effector_mapping": manifest["effector_mapping"],
        "initial_bridge_tick_index": 0,
        "final_bridge_tick_index": config.bridge_interval_count,
        "scientific_content_sha256": scientific_sha,
        "checkpoint_payload_sha256": checkpoint["payload_sha256"],
    }
    expected_run_id = "{}-{}".format(
        manifest["scenario_id"], _canonical_json_sha256(identity)[:20]
    )
    _require(
        manifest.get("run_id") == expected_run_id and path.parent.name == expected_run_id,
        "canonical run_id/directory is not content-derived",
    )
    return manifest, len(arrays), chunk_count


def _audit_canonical_run_manifest(
    path: Path, manifest: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], int, int]:
    """Convert every malformed canonical payload into a fail-closed audit error."""

    try:
        return _audit_canonical_run_manifest_impl(path, manifest)
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(
            "canonical run manifest semantic audit failed: {}".format(exc)
        ) from exc


def _audit_run_manifest(path: Path) -> Tuple[Mapping[str, Any], int, int]:
    try:
        manifest_bytes = path.read_bytes()
    except OSError as exc:
        raise AuditError("run manifest could not be read: {}".format(exc)) from exc
    manifest = _load_json_bytes(manifest_bytes, "run manifest {}".format(path))
    sidecar = path.with_name("manifest.sha256")
    _require(sidecar.is_file(), "run manifest sidecar is missing: {}".format(sidecar))
    sidecar_lines = sidecar.read_text(encoding="ascii", errors="strict").splitlines()
    _require(len(sidecar_lines) == 1, "run manifest sidecar must contain exactly one line")
    sidecar_match = re.fullmatch(r"([0-9a-f]{64})  manifest\.json", sidecar_lines[0])
    _require(sidecar_match is not None, "run manifest sidecar has invalid syntax")
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert sidecar_match is not None
    _require(
        sidecar_match.group(1) == actual_manifest_sha,
        "run manifest sidecar checksum mismatch for {}".format(path),
    )

    run_id = manifest.get("run_id")
    _require(isinstance(run_id, str) and bool(run_id), "run manifest requires run_id")
    _require(path.name == "manifest.json", "run artifact URL must end in manifest.json")
    _require(path.parent.name == run_id, "run_id does not match its immutable directory")
    if manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION:
        return _audit_canonical_run_manifest(path, manifest)
    _require(
        manifest.get("schema_version") == ARTIFACT_SCHEMA_VERSION,
        "run manifest has an unsupported schema_version",
    )
    _require(
        manifest.get("episode_status") == "complete",
        "run manifest episode_status must be complete",
    )
    _require(
        manifest.get("validation_status") == "exploratory",
        "run manifest validation_status must be exploratory",
    )
    _require(
        isinstance(manifest.get("authority_notice"), str)
        and bool(manifest.get("authority_notice")),
        "run manifest requires an authority_notice",
    )
    configuration = manifest.get("configuration")
    _require(
        isinstance(configuration, Mapping),
        "run manifest configuration must be an object",
    )
    for clock_name in (
        "duration_s",
        "physics_timestep_s",
        "neural_timestep_s",
        "logging_timestep_s",
    ):
        clock_value = configuration.get(clock_name)
        _require(
            isinstance(clock_value, (int, float))
            and not isinstance(clock_value, bool)
            and math.isfinite(float(clock_value))
            and float(clock_value) > 0.0,
            "run manifest configuration {} must be finite and positive".format(
                clock_name
            ),
        )
    model_hashes = manifest.get("model_hashes")
    _require(
        isinstance(model_hashes, Mapping) and bool(model_hashes),
        "run manifest model_hashes must be a non-empty object",
    )
    for name, digest in model_hashes.items():
        _require(
            isinstance(name, str) and bool(name),
            "run manifest model hash names must be non-empty strings",
        )
        _require_sha256(digest, "run manifest model_hashes.{}".format(name))
    runtime = manifest.get("runtime")
    _require(isinstance(runtime, Mapping), "run manifest runtime must be an object")
    backend = runtime.get("physics_backend")
    expected_source_kind = {
        "reduced_order": REDUCED_ORDER_SOURCE_KIND,
        "flybody": FLYBODY_WORKER_SOURCE_KIND,
    }.get(backend)
    _require(expected_source_kind is not None, "run manifest physics backend is unsupported")
    _require(
        manifest.get("source_kind") == expected_source_kind,
        "run manifest source_kind is inconsistent with its physics backend",
    )
    diagnostics = manifest.get("diagnostics")
    _require(
        isinstance(diagnostics, Mapping),
        "run manifest diagnostics must be an object",
    )
    _require(
        diagnostics.get("physics_backend") == backend,
        "run manifest diagnostics physics backend is inconsistent",
    )
    arrays = manifest.get("arrays")
    _require(isinstance(arrays, Mapping) and bool(arrays), "run manifest arrays must be a non-empty object")

    array_count = 0
    chunk_count = 0
    seen_chunk_paths: Set[Path] = set()
    seen_chunk_identities: Set[Tuple[int, int]] = set()
    for array_id, descriptor_value in arrays.items():
        label = "run {}/array {}".format(run_id, array_id)
        _require(isinstance(array_id, str) and bool(array_id), "array IDs must be non-empty strings")
        _require(isinstance(descriptor_value, Mapping), "{} descriptor must be an object".format(label))
        descriptor = descriptor_value
        shape_value = descriptor.get("shape")
        _require(
            isinstance(shape_value, list)
            and bool(shape_value)
            and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape_value),
            "{} shape must contain non-negative integers".format(label),
        )
        expected_shape = tuple(shape_value)
        dtype_value = descriptor.get("dtype")
        _require(isinstance(dtype_value, str) and bool(dtype_value), "{} requires dtype".format(label))
        try:
            expected_dtype = np.dtype(dtype_value)
        except TypeError as exc:
            raise AuditError("{} has an invalid dtype".format(label)) from exc
        _require(expected_dtype.hasobject is False, "{} may not use an object dtype".format(label))
        _require(
            isinstance(descriptor.get("unit"), str) and bool(descriptor.get("unit")),
            "{} requires a non-empty unit".format(label),
        )
        _require(
            isinstance(descriptor.get("provenance"), str)
            and bool(descriptor.get("provenance")),
            "{} requires non-empty provenance".format(label),
        )
        _require_sha256(descriptor.get("logical_npy_sha256"), "{}.logical_npy_sha256".format(label))
        chunks = descriptor.get("chunks")
        _require(isinstance(chunks, list) and bool(chunks), "{} requires chunks".format(label))

        cursor = 0
        for index, chunk_value in enumerate(chunks):
            chunk_label = "{}.chunks[{}]".format(label, index)
            _require(isinstance(chunk_value, Mapping), "{} must be an object".format(chunk_label))
            chunk = chunk_value
            chunk_path = _safe_local_path(path.parent, chunk.get("path"), "{}.path".format(chunk_label))
            _require(chunk_path not in seen_chunk_paths, "chunk file is referenced more than once: {}".format(chunk_path))
            seen_chunk_paths.add(chunk_path)
            chunk_stat = chunk_path.stat()
            chunk_identity = (chunk_stat.st_dev, chunk_stat.st_ino)
            _require(
                chunk_identity not in seen_chunk_identities,
                "chunk file inode is referenced more than once: {}".format(chunk_path),
            )
            seen_chunk_identities.add(chunk_identity)
            expected_chunk_sha = _require_sha256(chunk.get("sha256"), "{}.sha256".format(chunk_label))
            _require(
                _sha256_file(chunk_path) == expected_chunk_sha,
                "chunk checksum mismatch for {}".format(chunk_path),
            )
            start = chunk.get("start")
            stop = chunk.get("stop")
            _require(
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(stop, int)
                and not isinstance(stop, bool)
                and start == cursor
                and stop >= start,
                "{} has a non-contiguous interval".format(chunk_label),
            )
            chunk_shape = chunk.get("shape")
            _require(
                isinstance(chunk_shape, list)
                and bool(chunk_shape)
                and len(chunk_shape) == len(expected_shape)
                and all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in chunk_shape
                )
                and tuple(chunk_shape[1:]) == expected_shape[1:]
                and chunk_shape[0] == stop - start,
                "{} shape does not match its interval/array shape".format(chunk_label),
            )
            cursor = stop
            chunk_count += 1
        _require(cursor == expected_shape[0], "{} chunks do not cover the logical first axis".format(label))

        try:
            reconstructed = read_chunked_array(path.parent, descriptor)
        except Exception as exc:
            raise AuditError("{} reconstruction failed: {}".format(label, exc)) from exc
        _require(tuple(reconstructed.shape) == expected_shape, "{} reconstructed shape mismatch".format(label))
        _require(reconstructed.dtype == expected_dtype, "{} reconstructed dtype mismatch".format(label))
        _require(
            _logical_npy_sha256(reconstructed)
            == descriptor.get("logical_npy_sha256"),
            "{} logical NPY checksum mismatch".format(label),
        )
        _require(
            np.issubdtype(reconstructed.dtype, np.number) or np.issubdtype(reconstructed.dtype, np.bool_),
            "{} must contain numeric or boolean scientific data".format(label),
        )
        _require(bool(np.all(np.isfinite(reconstructed))), "{} contains NaN or infinity".format(label))
        array_count += 1
    return manifest, array_count, chunk_count


def _audit_validation_attachment(
    root: Path, manifest: Mapping[str, Any]
) -> Tuple[ValidationReport, BenchmarkRegistry, bytes]:
    report_path = _safe_local_path(root, manifest.get("validation_report_url"), "validation_report_url")
    registry_path = _safe_local_path(root, manifest.get("validation_registry_url"), "validation_registry_url")
    expected_report_sha = _require_sha256(
        manifest.get("validation_report_sha256"), "validation_report_sha256"
    )
    expected_registry_file_sha = _require_sha256(
        manifest.get("validation_registry_file_sha256"),
        "validation_registry_file_sha256",
    )
    expected_registry_canonical_sha = _require_sha256(
        manifest.get("validation_registry_canonical_sha256"),
        "validation_registry_canonical_sha256",
    )
    try:
        report_bytes = report_path.read_bytes()
        registry_bytes = registry_path.read_bytes()
    except OSError as exc:
        raise AuditError("validation report or registry could not be read: {}".format(exc)) from exc
    _require(
        hashlib.sha256(report_bytes).hexdigest() == expected_report_sha,
        "validation report byte checksum mismatch",
    )
    _require(
        hashlib.sha256(registry_bytes).hexdigest() == expected_registry_file_sha,
        "validation registry byte checksum mismatch",
    )
    try:
        # Reject duplicate keys and non-finite JSON before the domain parsers
        # construct their typed contracts. Their canonical binding is not a
        # substitute for an unambiguous byte-level JSON representation.
        _load_json_bytes(registry_bytes, "validation registry")
        _load_json_bytes(report_bytes, "validation report")
        registry_text = registry_bytes.decode("utf-8", errors="strict")
        report_text = report_bytes.decode("utf-8", errors="strict")
        registry = BenchmarkRegistry.from_json(registry_text)
        report = ValidationReport.from_json(report_text, registry=registry)
    except Exception as exc:
        raise AuditError("validation report/registry canonical binding failed: {}".format(exc)) from exc
    _require(
        registry.content_sha256 == expected_registry_canonical_sha,
        "validation registry canonical checksum mismatch",
    )
    return report, registry, registry_bytes


def _audit_current_source_binding(
    report: ValidationReport,
    attached_registry_bytes: bytes,
) -> None:
    """Require the release report to have evaluated this exact source tree."""

    current_registry = REPOSITORY_ROOT / "data" / "benchmarks" / "registry.v1.json"
    current_evaluators = SOURCE_ROOT / "fly_sensor2behavior" / "evaluators.py"
    current_fly_fgs_runtime = REPOSITORY_ROOT / "scripts" / "fly_fgs_runtime_rpc.mjs"
    current_ordinary_flight_contract = (
        REPOSITORY_ROOT
        / "data"
        / "reference"
        / "flybody_ordinary_flight_release.expected.v1.json"
    )
    current_package = SOURCE_ROOT / "fly_sensor2behavior"
    current_dependency_lock = REPOSITORY_ROOT / DEPENDENCY_LOCK_SOURCE_NAME
    for path, label in (
        (current_registry, "source benchmark registry"),
        (current_evaluators, "source evaluator"),
        (current_fly_fgs_runtime, "source fly-FGS runtime sidecar"),
        (current_ordinary_flight_contract, "ordinary-flight reference contract"),
        (current_dependency_lock, "source dependency lock"),
    ):
        _require(path.is_file(), "{} is unavailable: {}".format(label, path))
    _require(
        attached_registry_bytes == current_registry.read_bytes(),
        "attached benchmark registry is stale relative to the source tree",
    )
    reported = {
        (item.kind, item.name): item.sha256 for item in report.source_digests
    }
    expected = {
        ("registry", current_registry.name): _sha256_file(current_registry),
        ("code", "built-in-evaluators.py"): _sha256_file(current_evaluators),
        ("code", "fly_sensor2behavior-python-tree"): _python_source_tree_sha256(
            current_package
        ),
        ("code", "fly_fgs_runtime_rpc.mjs"): _sha256_file(
            current_fly_fgs_runtime
        ),
        (
            DEPENDENCY_LOCK_SOURCE_KIND,
            DEPENDENCY_LOCK_SOURCE_NAME,
        ): _sha256_file(current_dependency_lock),
        (
            "reference_contract",
            current_ordinary_flight_contract.name,
        ): _sha256_file(current_ordinary_flight_contract),
    }
    expected.update(
        {
            ("protocol", source_uri): sha256
            for source_uri, _source_path, sha256, _payload in verified_preregistered_protocol_files(
                BenchmarkRegistry.from_json(current_registry.read_text(encoding="utf-8")),
                current_registry,
            )
        }
    )
    for key, digest in expected.items():
        _require(
            reported.get(key) == digest,
            "validation report source receipt {}:{} is stale or missing".format(*key),
        )
    release_critical_kinds = {
        "registry",
        "code",
        "dependency_lock",
        "protocol",
        "reference_contract",
    }
    unexpected = sorted(
        key
        for key in reported
        if key[0] in release_critical_kinds and key not in expected
    )
    _require(
        not unexpected,
        "validation report contains unexpected release-critical source receipts: {}".format(
            ", ".join("{}:{}".format(*key) for key in unexpected)
        ),
    )


def _audit_publication_gate_status(report: ValidationReport) -> None:
    """Reject partial or failing scientific reports from static publication."""

    _require(report.suite_complete, "public release requires a complete validation suite")
    failed_case_ids = tuple(
        sorted(
            result.case_id
            for result in report.results
            if result.status is GateStatus.FAIL
        )
    )
    _require(
        not failed_case_ids,
        "public release validation contains FAIL cases: %s"
        % ", ".join(failed_case_ids),
    )


def _audit_web_replay_projection(
    episode: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> None:
    """Require the immutable run to bind every non-circular replay field."""

    canonical = (
        run_manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION
    )
    expected_schema = (
        "1.0.0" if canonical else WEB_REPLAY_PROJECTION_SCHEMA_VERSION
    )
    expected_canonicalization = (
        CANONICAL_WEB_PROJECTION_CANONICALIZATION
        if canonical
        else WEB_REPLAY_PROJECTION_CANONICALIZATION
    )
    expected_exclusions = (
        ["source_artifact_manifest_sha256", "source_artifact_schema_version"]
        if canonical
        else list(WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS)
    )

    receipt = run_manifest.get("web_replay_projection")
    _require(
        isinstance(receipt, Mapping),
        "run manifest is missing its web replay projection receipt",
    )
    _require(
        receipt.get("schema_version") == expected_schema,
        "run manifest web replay projection schema is unsupported",
    )
    _require(
        receipt.get("canonicalization") == expected_canonicalization,
        "run manifest web replay projection canonicalization is unsupported",
    )
    _require(
        receipt.get("excluded_top_level_fields") == expected_exclusions,
        "run manifest web replay projection exclusions are not exact",
    )
    expected_sha = _require_sha256(
        receipt.get("sha256"), "run manifest web replay projection sha256"
    )
    try:
        actual_sha = (
            canonical_web_projection_sha256(episode)
            if canonical
            else web_replay_projection_sha256(episode)
        )
    except (TypeError, ValueError) as exc:
        raise AuditError("web replay projection is not canonical JSON: {}".format(exc)) from exc
    _require(
        actual_sha == expected_sha,
        "published web replay does not match its immutable run projection receipt",
    )


def _artifact_projection_preimage_bytes(
    episode: Mapping[str, Any]
) -> bytes:
    projected = {
        key: value
        for key, value in episode.items()
        if key not in WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
    }
    try:
        return json.dumps(
            projected,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditError(
            "episode artifact projection is not finite canonical JSON: {}".format(
                exc
            )
        ) from exc


def _audit_episode_content_binding(
    summary: Mapping[str, Any],
    episode_path: Path,
    episode: Mapping[str, Any],
    run_path: Path,
    run_manifest: Mapping[str, Any],
    projection_path: Path,
) -> None:
    """Bind exact public bytes to the immutable replay projection receipt."""

    episode_id = summary.get("id")
    replay_sha = _require_sha256(
        summary.get("replay_sha256"),
        "episode summary replay_sha256",
    )
    artifact_sha = _require_sha256(
        summary.get("artifact_manifest_sha256"),
        "episode summary artifact_manifest_sha256",
    )
    expected_projection_url = "data/episodes/{}.artifact-projection.json".format(
        episode_id
    )
    _require(
        summary.get("artifact_projection_url") == expected_projection_url,
        "episode summary artifact_projection_url is not the exact canonical path",
    )
    resolved_paths = (episode_path.resolve(), run_path.resolve(), projection_path.resolve())
    _require(
        len(set(resolved_paths)) == 3,
        "episode replay, artifact manifest, and projection paths must be distinct",
    )
    identities: Set[Tuple[int, int]] = set()
    for binding_path, label in (
        (episode_path, "episode replay"),
        (run_path, "artifact manifest"),
        (projection_path, "artifact projection"),
    ):
        _require(
            binding_path.is_file() and not binding_path.is_symlink(),
            "{} must be a regular non-symlink file".format(label),
        )
        stat = binding_path.stat()
        identity = (stat.st_dev, stat.st_ino)
        _require(
            identity not in identities,
            "episode binding files may not be hard-link aliases",
        )
        identities.add(identity)
    try:
        episode_bytes = episode_path.read_bytes()
        artifact_bytes = run_path.read_bytes()
        projection_bytes = projection_path.read_bytes()
    except OSError as exc:
        raise AuditError("episode binding bytes could not be read: {}".format(exc)) from exc
    _require(
        hashlib.sha256(episode_bytes).hexdigest() == replay_sha,
        "episode replay bytes do not match summary replay_sha256",
    )
    _require(
        hashlib.sha256(artifact_bytes).hexdigest() == artifact_sha,
        "artifact manifest bytes do not match summary artifact_manifest_sha256",
    )
    _require(
        episode.get("id") == episode_id
        and episode.get("label") == summary.get("label")
        and episode.get("status") == summary.get("status"),
        "episode replay identity/label/status differs from its public summary",
    )
    _require(
        episode.get("source_run_id") == run_manifest.get("run_id")
        and episode.get("source_artifact_manifest_sha256") == artifact_sha
        and episode.get("source_artifact_schema_version")
        == run_manifest.get("schema_version"),
        "episode replay does not bind the exact artifact manifest bytes/schema",
    )
    receipt = _exact_fields(
        run_manifest.get("web_replay_projection"),
        (
            "schema_version",
            "sha256",
            "canonicalization",
            "excluded_top_level_fields",
        ),
        "artifact web replay projection receipt",
    )
    canonical = (
        run_manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION
    )
    expected_canonicalization = (
        CANONICAL_WEB_PROJECTION_CANONICALIZATION
        if canonical
        else WEB_REPLAY_PROJECTION_CANONICALIZATION
    )
    _require(
        receipt.get("schema_version") == "1.0.0"
        and receipt.get("canonicalization") == expected_canonicalization
        and receipt.get("excluded_top_level_fields")
        == list(WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS),
        "artifact web replay projection receipt contract is not exact",
    )
    projection_sha = _require_sha256(
        receipt.get("sha256"), "artifact web replay projection sha256"
    )
    _require(
        hashlib.sha256(projection_bytes).hexdigest() == projection_sha,
        "artifact projection sidecar bytes do not match the immutable receipt",
    )
    expected_projection_bytes = _artifact_projection_preimage_bytes(episode)
    _require(
        projection_bytes == expected_projection_bytes,
        "artifact projection sidecar is not the exact replay projection preimage",
    )
    projection = _load_json_bytes(
        projection_bytes, "episode artifact projection sidecar"
    )
    expected_projection = {
        key: value
        for key, value in episode.items()
        if key not in WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
    }
    _require(
        projection == expected_projection,
        "artifact projection sidecar structure differs after circular-field exclusion",
    )


def _audit_worker_dependency_lock_binding(
    report: ValidationReport,
    flybody_run_manifests: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    """Bind FlyBody runs to the report's exact worker dependency inventory."""

    lock_receipts = tuple(
        item
        for item in report.source_digests
        if item.kind == DEPENDENCY_LOCK_SOURCE_KIND
        and item.name == DEPENDENCY_LOCK_SOURCE_NAME
    )
    _require(
        len(lock_receipts) <= 1,
        "validation report contains ambiguous worker dependency-lock receipts",
    )
    if report.suite_complete and flybody_run_manifests:
        _require(
            len(lock_receipts) == 1,
            "a complete release containing FlyBody requires one dependency-lock receipt",
        )
    if not lock_receipts:
        return None

    lock_sha = lock_receipts[0].sha256
    expected_lock_digest = "sha256:" + lock_sha
    for run_manifest in flybody_run_manifests:
        physics_provenance = _flybody_run_provenance(run_manifest)
        worker_lock = physics_provenance.get("worker_dependency_lock")
        _require(
            isinstance(worker_lock, Mapping),
            "FlyBody run worker_dependency_lock must be an object",
        )
        _require(
            worker_lock.get("name") == DEPENDENCY_LOCK_SOURCE_NAME
            and worker_lock.get("sha256") == expected_lock_digest,
            "FlyBody run dependency lock does not match the validation report",
        )
    return lock_sha


def _audit_worker_image_binding(
    report: ValidationReport,
    flybody_run_manifests: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    """Bind complete-release FlyBody runs to the report's OCI image receipt."""

    worker_receipts = tuple(
        item
        for item in report.source_digests
        if item.kind == WORKER_IMAGE_SOURCE_KIND
        and item.name == WORKER_IMAGE_SOURCE_NAME
    )
    _require(
        len(worker_receipts) <= 1,
        "validation report contains ambiguous FlyBody worker image receipts",
    )
    if report.suite_complete and flybody_run_manifests:
        _require(
            len(worker_receipts) == 1,
            "a complete release containing FlyBody requires one worker image receipt",
        )
    if not worker_receipts:
        return None

    worker_image_sha = worker_receipts[0].sha256
    expected_worker_image_digest = "sha256:" + worker_image_sha
    for run_manifest in flybody_run_manifests:
        physics_provenance = _flybody_run_provenance(run_manifest)
        _require(
            physics_provenance.get("worker_image_digest")
            == expected_worker_image_digest,
            "FlyBody run worker image digest does not match the validation report",
        )
    return worker_image_sha


def _flybody_run_provenance(
    run_manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Resolve the schema-specific authoritative FlyBody receipt."""

    runtime = run_manifest.get("runtime")
    _require(isinstance(runtime, Mapping), "FlyBody run runtime must be an object")
    if run_manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION:
        receipts = runtime.get("receipts")
        _require(
            isinstance(receipts, Mapping),
            "canonical FlyBody runtime receipts must be an object",
        )
        provenance = receipts.get("flybody")
        label = "canonical nested FlyBody receipt"
    else:
        provenance = runtime.get("physics_provenance")
        label = "FlyBody run physics_provenance"
    _require(isinstance(provenance, Mapping), "{} must be an object".format(label))
    return provenance


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        "{} must be finite numeric data".format(label),
    )
    return float(value)


def _nonnegative_integer(value: Any, label: str) -> int:
    numeric = _finite_number(value, label)
    _require(
        numeric >= 0.0 and numeric.is_integer(),
        "{} must be a non-negative integer".format(label),
    )
    return int(numeric)


def _close(actual: float, expected: float, *, atol: float = 1.0e-12) -> bool:
    return abs(float(actual) - float(expected)) <= atol


def _read_run_arrays(
    run_path: Path,
    run_manifest: Mapping[str, Any],
    array_ids: Sequence[str],
) -> Mapping[str, np.ndarray]:
    descriptors = run_manifest.get("arrays")
    _require(isinstance(descriptors, Mapping), "run manifest arrays must be an object")
    result: Dict[str, np.ndarray] = {}
    for array_id in array_ids:
        descriptor = descriptors.get(array_id)
        _require(
            isinstance(descriptor, Mapping),
            "run manifest is missing required scientific array {}".format(array_id),
        )
        try:
            result[array_id] = read_chunked_array(run_path.parent, descriptor)
        except Exception as exc:
            raise AuditError(
                "required scientific array {} could not be reconstructed: {}".format(
                    array_id, exc
                )
            ) from exc
    return result


def _audit_flybody_ground_contact_disclosure(
    episode: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Bind the replay disclosure to full-physics-step contact diagnostics."""

    _require(
        episode.get("physics_backend") == "flybody",
        "ground-contact disclosure is reserved for FlyBody episodes",
    )
    _require(
        episode.get("contact_telemetry_status") == FLYBODY_CONTACT_STATUS,
        "FlyBody episode must disclose exact MuJoCo ground-contact telemetry",
    )
    summary = episode.get("ground_contact_summary")
    _require(
        isinstance(summary, Mapping),
        "FlyBody episode requires authoritative full-step ground_contact_summary",
    )
    _require(
        summary.get("telemetry") == FLYBODY_CONTACT_SUMMARY_KIND,
        "FlyBody ground_contact_summary telemetry kind is unsupported",
    )
    _require(
        summary.get("sample_semantics") == GROUND_CONTACT_SAMPLE_SEMANTICS,
        "FlyBody ground_contact_summary must disclose the exact transition sample semantics",
    )

    initial_count = _nonnegative_integer(
        summary.get("initial_count"), "ground_contact_summary.initial_count"
    )
    transition_count = _nonnegative_integer(
        summary.get("transition_count"),
        "ground_contact_summary.transition_count",
    )
    maximum_count = _nonnegative_integer(
        summary.get("maximum_count"), "ground_contact_summary.maximum_count"
    )
    occurred = summary.get("occurred")
    _require(
        isinstance(occurred, bool),
        "ground_contact_summary.occurred must be boolean",
    )
    first_start_value = summary.get("first_transition_start_s")
    first_start_s = (
        None
        if first_start_value is None
        else _finite_number(
            first_start_value,
            "ground_contact_summary.first_transition_start_s",
        )
    )
    first_end_value = summary.get("first_transition_end_s")
    first_end_s = (
        None
        if first_end_value is None
        else _finite_number(
            first_end_value,
            "ground_contact_summary.first_transition_end_s",
        )
    )

    configuration = run_manifest.get("configuration")
    diagnostics = run_manifest.get("diagnostics")
    runtime = run_manifest.get("runtime")
    _require(isinstance(configuration, Mapping), "run configuration must be an object")
    _require(isinstance(diagnostics, Mapping), "run diagnostics must be an object")
    _require(isinstance(runtime, Mapping), "run runtime must be an object")
    _require(
        runtime.get("physics_backend") == "flybody",
        "ground-contact run must use FlyBody",
    )
    duration_s = _finite_number(configuration.get("duration_s"), "run duration_s")
    timestep_s = _finite_number(
        configuration.get("physics_timestep_s"), "run physics_timestep_s"
    )
    physics_steps = _nonnegative_integer(
        diagnostics.get("physics_steps"), "diagnostics.physics_steps"
    )
    _require(
        physics_steps > 0
        and _close(physics_steps * timestep_s, duration_s, atol=1.0e-10),
        "FlyBody duration, timestep, and physics-step count are inconsistent",
    )
    _require(
        transition_count <= physics_steps,
        "ground contact transition count exceeds the episode",
    )

    physics_provenance = runtime.get("physics_provenance")
    _require(
        isinstance(physics_provenance, Mapping),
        "FlyBody run physics_provenance must be an object",
    )
    contact_provenance = physics_provenance.get("ground_contact_telemetry")
    _require(
        isinstance(contact_provenance, Mapping)
        and contact_provenance.get("ground_geom_name") == "ground_plane"
        and _nonnegative_integer(
            contact_provenance.get("configured_pair_count"),
            "ground-contact configured_pair_count",
        )
        > 0,
        "FlyBody runtime lacks the reviewed ground-plane contact topology",
    )

    metrics = diagnostics.get("metrics")
    _require(isinstance(metrics, Mapping), "run diagnostics metrics must be an object")
    _require(
        _finite_number(
            metrics.get("ground_contact_telemetry_available"),
            "ground_contact_telemetry_available",
        )
        == 1.0,
        "FlyBody run does not declare ground-contact telemetry available",
    )
    metric_initial = _nonnegative_integer(
        metrics.get("initial_ground_contact_count"),
        "initial_ground_contact_count",
    )
    metric_transitions = _nonnegative_integer(
        metrics.get("ground_contact_transition_count"),
        "ground_contact_transition_count",
    )
    metric_maximum = _nonnegative_integer(
        metrics.get("maximum_ground_contact_count"),
        "maximum_ground_contact_count",
    )
    metric_occurred = _finite_number(
        metrics.get("ground_contact_occurred"), "ground_contact_occurred"
    )
    _require(
        metric_occurred in (0.0, 1.0),
        "ground_contact_occurred must be zero or one",
    )
    metric_first_start_s = _finite_number(
        metrics.get(
            "first_ground_contact_transition_start_s_or_duration_s"
        ),
        "first_ground_contact_transition_start_s_or_duration_s",
    )
    metric_first_end_s = _finite_number(
        metrics.get("first_ground_contact_transition_end_s_or_duration_s"),
        "first_ground_contact_transition_end_s_or_duration_s",
    )
    _require(
        (initial_count, transition_count, maximum_count, occurred)
        == (
            metric_initial,
            metric_transitions,
            metric_maximum,
            bool(metric_occurred),
        ),
        "published ground_contact_summary disagrees with full-step diagnostics",
    )

    physics_time_s = np.asarray(arrays.get("physics_time_s"), dtype=float)
    transition_contact = np.asarray(
        arrays.get("ground_contact_transition_point_count")
    )
    _require(
        physics_time_s.ndim == 1
        and len(physics_time_s) == physics_steps + 1
        and np.all(np.isfinite(physics_time_s))
        and _close(float(physics_time_s[0]), 0.0)
        and _close(float(physics_time_s[-1]), duration_s)
        and np.allclose(
            physics_time_s,
            np.arange(physics_steps + 1, dtype=float) * timestep_s,
            rtol=0.0,
            atol=1.0e-10,
        ),
        "physics_time_s is not the authoritative configured physics clock",
    )
    _require(
        transition_contact.ndim == 1
        and transition_contact.shape == physics_time_s.shape
        and transition_contact.dtype.kind in "iu"
        and np.all(transition_contact >= 0),
        "ground_contact_transition_point_count must contain one non-negative "
        "integer per physics sample",
    )
    transition_indices = np.flatnonzero(transition_contact[1:]) + 1
    expected_initial_count = int(transition_contact[0])
    expected_transition_count = int(len(transition_indices))
    expected_maximum_count = int(np.max(transition_contact, initial=0))
    expected_occurred = expected_maximum_count > 0
    if expected_initial_count > 0:
        expected_first_start_s: Optional[float] = 0.0
        expected_first_end_s: Optional[float] = 0.0
    elif len(transition_indices):
        first_transition_index = int(transition_indices[0])
        expected_first_start_s = float(
            physics_time_s[first_transition_index - 1]
        )
        expected_first_end_s = float(physics_time_s[first_transition_index])
    else:
        expected_first_start_s = None
        expected_first_end_s = None
    _require(
        (
            initial_count,
            transition_count,
            maximum_count,
            occurred,
        )
        == (
            expected_initial_count,
            expected_transition_count,
            expected_maximum_count,
            expected_occurred,
        ),
        "ground_contact_summary disagrees with authoritative full-step transition telemetry",
    )
    _require(
        (
            first_start_s is None
            and expected_first_start_s is None
            or first_start_s is not None
            and expected_first_start_s is not None
            and _close(first_start_s, expected_first_start_s)
        )
        and (
            first_end_s is None
            and expected_first_end_s is None
            or first_end_s is not None
            and expected_first_end_s is not None
            and _close(first_end_s, expected_first_end_s)
        ),
        "ground_contact_summary first transition interval disagrees with "
        "authoritative full-step telemetry",
    )
    expected_metric_start_s = (
        duration_s if expected_first_start_s is None else expected_first_start_s
    )
    expected_metric_end_s = (
        duration_s if expected_first_end_s is None else expected_first_end_s
    )
    _require(
        _close(metric_first_start_s, expected_metric_start_s)
        and _close(metric_first_end_s, expected_metric_end_s),
        "full-step first-transition diagnostics disagree with authoritative telemetry",
    )
    _require(
        maximum_count >= initial_count and occurred == (maximum_count > 0),
        "ground-contact occurrence and maximum count are inconsistent",
    )
    if not occurred:
        _require(
            first_start_s is None
            and first_end_s is None
            and transition_count == 0
            and maximum_count == 0
            and _close(metric_first_start_s, duration_s)
            and _close(metric_first_end_s, duration_s),
            "contact-free summary must use the exact no-contact sentinel",
        )
    elif initial_count > 0:
        _require(
            first_start_s is not None
            and first_end_s is not None
            and _close(first_start_s, 0.0)
            and _close(first_end_s, 0.0)
            and _close(metric_first_start_s, 0.0)
            and _close(metric_first_end_s, 0.0),
            "contact present at reset must have a zero-width transition interval at t=0",
        )
    else:
        _require(
            first_start_s is not None
            and first_end_s is not None
            and 0.0 <= first_start_s < first_end_s <= duration_s
            and _close(first_end_s - first_start_s, timestep_s)
            and transition_count > 0,
            "post-reset ground contact requires one bounded first transition interval",
        )
        start_tick = first_start_s / timestep_s
        end_tick = first_end_s / timestep_s
        _require(
            _close(start_tick, round(start_tick), atol=1.0e-8)
            and _close(end_tick, round(end_tick), atol=1.0e-8),
            "first ground-contact transition interval must lie on the physics clock",
        )

    contact = np.asarray(arrays.get("ground_contact_count"))
    time_s = np.asarray(arrays.get("time_s"), dtype=float)
    _require(
        contact.ndim == 1
        and contact.dtype.kind in "iu"
        and np.all(contact >= 0),
        "ground_contact_count array must contain non-negative integers",
    )
    _require(
        time_s.ndim == 1
        and len(time_s) == len(contact)
        and len(time_s) >= 2
        and np.all(np.isfinite(time_s))
        and np.all(np.diff(time_s) > 0.0)
        and _close(float(time_s[0]), 0.0)
        and _close(float(time_s[-1]), duration_s),
        "ground-contact and time arrays do not span the episode",
    )
    logged_indices = np.rint(time_s / timestep_s).astype(np.int64)
    _require(
        int(contact[0]) == initial_count
        and np.all(logged_indices >= 0)
        and np.all(logged_indices <= physics_steps)
        and np.allclose(
            time_s,
            physics_time_s[logged_indices],
            rtol=0.0,
            atol=1.0e-10,
        )
        and np.array_equal(contact, transition_contact[logged_indices]),
        "logged ground-contact array is not an exact projection of full-step telemetry",
    )

    descriptors = run_manifest.get("arrays")
    _require(isinstance(descriptors, Mapping), "run arrays must be an object")
    contact_descriptor = descriptors.get("ground_contact_count")
    physics_time_descriptor = descriptors.get("physics_time_s")
    transition_descriptor = descriptors.get(
        "ground_contact_transition_point_count"
    )
    _require(
        isinstance(contact_descriptor, Mapping)
        and contact_descriptor.get("unit") == "1"
        and contact_descriptor.get("provenance")
        == "decimated_projection_of_mujoco_transition_contact_points"
        and isinstance(physics_time_descriptor, Mapping)
        and physics_time_descriptor.get("unit") == "s"
        and physics_time_descriptor.get("provenance")
        == "authoritative_external_physics_clock"
        and isinstance(transition_descriptor, Mapping)
        and transition_descriptor.get("unit") == "1"
        and transition_descriptor.get("provenance")
        == GROUND_CONTACT_TELEMETRY_KIND
        and transition_descriptor.get("sample_semantics")
        == GROUND_CONTACT_SAMPLE_SEMANTICS,
        "ground-contact arrays lack authoritative transition telemetry provenance",
    )
    units = episode.get("units")
    _require(
        isinstance(units, Mapping) and units.get("ground_contact_count") == "1",
        "FlyBody replay must disclose the ground-contact count unit",
    )
    frames = episode.get("frames")
    _require(isinstance(frames, list) and bool(frames), "episode frames are required")
    for index, frame in enumerate(frames):
        _require(isinstance(frame, Mapping), "episode frame must be an object")
        frame_time_s = _finite_number(frame.get("t"), "frame[{}].t".format(index))
        frame_count = _nonnegative_integer(
            frame.get("ground_contact_count"),
            "frame[{}].ground_contact_count".format(index),
        )
        nearest = int(np.argmin(np.abs(time_s - frame_time_s)))
        _require(
            _close(float(time_s[nearest]), frame_time_s, atol=1.0e-10)
            and int(contact[nearest]) == frame_count,
            "replay frame ground-contact count is not a sample of the authoritative array",
        )


def _audit_flybody_wingbeat_inspection(
    episode: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Bind the browser's sub-wingbeat excerpt to physics-rate FlyBody arrays."""

    required = episode.get("source_kind") in WINGBEAT_INSPECTION_SOURCE_KINDS
    inspection = episode.get("wingbeat_inspection")
    if inspection is None:
        _require(
            not required,
            "frozen-browser and reduced-retinal FlyBody episodes require wingbeat_inspection",
        )
        return
    _require(
        episode.get("physics_backend") == "flybody",
        "wingbeat_inspection is reserved for FlyBody episodes",
    )
    _require(
        isinstance(inspection, Mapping),
        "wingbeat_inspection must be an object",
    )
    configuration = run_manifest.get("configuration")
    _require(
        isinstance(configuration, Mapping),
        "wingbeat_inspection requires a run configuration",
    )
    duration_s = _finite_number(
        configuration.get("duration_s"), "wingbeat run duration_s"
    )
    episode_duration_s = _finite_number(
        episode.get("duration_s"), "wingbeat episode duration_s"
    )
    physics_dt_s = _finite_number(
        configuration.get("physics_timestep_s"),
        "wingbeat run physics_timestep_s",
    )
    _require(
        _close(duration_s, episode_duration_s)
        and _close(
            physics_dt_s,
            WINGBEAT_INSPECTION_TIMESTEP_S,
            atol=1.0e-15,
        ),
        "wingbeat_inspection requires the authoritative 0.1 ms FlyBody clock",
    )
    duration_ticks = duration_s / physics_dt_s
    _require(
        _close(duration_ticks, round(duration_ticks), atol=1.0e-8),
        "wingbeat episode duration must contain an integral number of physics transitions",
    )
    expected_full_time = (
        np.arange(int(round(duration_ticks)) + 1, dtype=float) * physics_dt_s
    )

    try:
        physics_time_s = np.asarray(arrays.get("physics_time_s"), dtype=float)
        physics_wing = np.asarray(
            arrays.get("measured_wing_joint_angle_physics_rad"), dtype=float
        )
        physics_torque = np.asarray(
            arrays.get("external_actuator_torque_physics_n_m"), dtype=float
        )
        inspection_time = np.asarray(inspection.get("time_s"), dtype=float)
        inspection_wing = np.asarray(
            inspection.get("measured_wing_joint_angle_rad"), dtype=float
        )
        inspection_torque = np.asarray(
            inspection.get("external_actuator_torque_n_m"), dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise AuditError(
            "wingbeat_inspection contains non-numeric array data"
        ) from exc
    expected_end_s = min(WINGBEAT_INSPECTION_DURATION_S, duration_s)
    expected_inspection_count = int(round(expected_end_s / physics_dt_s)) + 1
    expected_inspection_time = expected_full_time[:expected_inspection_count]
    _require(
        physics_time_s.shape == expected_full_time.shape
        and np.all(np.isfinite(physics_time_s))
        and np.allclose(
            physics_time_s,
            expected_full_time,
            rtol=0.0,
            atol=1.0e-12,
        ),
        "wingbeat physics_time_s does not follow the complete 0.1 ms clock",
    )
    _require(
        physics_wing.shape == (len(physics_time_s), 6)
        and physics_torque.shape == (len(physics_time_s), 6)
        and np.all(np.isfinite(physics_wing))
        and np.all(np.isfinite(physics_torque)),
        "wingbeat authoritative wing or torque matrix is invalid",
    )
    _require(
        inspection_time.shape == expected_inspection_time.shape
        and np.all(np.isfinite(inspection_time))
        and np.allclose(
            inspection_time,
            expected_inspection_time,
            rtol=0.0,
            atol=1.0e-12,
        )
        and inspection_wing.shape == (expected_inspection_count, 6)
        and inspection_torque.shape == (expected_inspection_count, 6)
        and np.all(np.isfinite(inspection_wing))
        and np.all(np.isfinite(inspection_torque)),
        "wingbeat_inspection must contain finite time-aligned six-axis matrices at 0.1 ms spacing",
    )
    sample_rate_hz = _finite_number(
        inspection.get("sample_rate_hz"),
        "wingbeat_inspection.sample_rate_hz",
    )
    start_s = _finite_number(
        inspection.get("start_s"), "wingbeat_inspection.start_s"
    )
    end_s = _finite_number(
        inspection.get("end_s"), "wingbeat_inspection.end_s"
    )
    _require(
        _close(sample_rate_hz, 10_000.0, atol=1.0e-6)
        and _close(start_s, 0.0)
        and _close(end_s, expected_end_s),
        "wingbeat_inspection must span 0 to min(60 ms, duration) at exactly 10 kHz",
    )
    _require(
        np.array_equal(inspection_time, physics_time_s[:expected_inspection_count])
        and np.array_equal(
            inspection_wing, physics_wing[:expected_inspection_count]
        )
        and np.array_equal(
            inspection_torque, physics_torque[:expected_inspection_count]
        ),
        "wingbeat_inspection values do not match the authoritative physics-rate arrays",
    )

    wing_order = inspection.get("wing_joint_order")
    episode_wing_order = episode.get("measured_wing_joint_order")
    _require(
        isinstance(wing_order, list)
        and len(wing_order) == 6
        and all(isinstance(name, str) and bool(name) for name in wing_order)
        and len(set(wing_order)) == 6
        and wing_order == episode_wing_order,
        "wingbeat_inspection must preserve six unique named FlyBody axes",
    )
    _require(
        tuple(wing_order) == REVIEWED_FLYBODY_WING_AXIS_ORDER,
        "wingbeat_inspection must preserve the reviewed FlyGym 2.1.0 wing-axis order",
    )
    descriptors = run_manifest.get("arrays")
    _require(isinstance(descriptors, Mapping), "run arrays must be an object")
    physics_time_descriptor = descriptors.get("physics_time_s")
    wing_descriptor = descriptors.get("measured_wing_joint_angle_physics_rad")
    torque_descriptor = descriptors.get("external_actuator_torque_physics_n_m")
    _require(
        isinstance(physics_time_descriptor, Mapping)
        and physics_time_descriptor.get("unit") == "s"
        and physics_time_descriptor.get("provenance")
        == "authoritative_external_physics_clock"
        and isinstance(wing_descriptor, Mapping)
        and wing_descriptor.get("unit") == "rad"
        and wing_descriptor.get("provenance")
        == "physics_rate_external_measured_wing_output"
        and wing_descriptor.get("axis_labels") == wing_order
        and wing_descriptor.get("sample_semantics")
        == WINGBEAT_ANGLE_SAMPLE_SEMANTICS
        and isinstance(torque_descriptor, Mapping)
        and torque_descriptor.get("unit") == "N m"
        and torque_descriptor.get("provenance")
        == "physics_rate_external_mujoco_wing_actuator_torque"
        and torque_descriptor.get("axis_labels") == wing_order
        and torque_descriptor.get("sample_semantics")
        == WINGBEAT_TORQUE_SAMPLE_SEMANTICS,
        "wingbeat physics-rate arrays lack exact axis, unit, provenance, or sample semantics",
    )


def _audit_no_fly_fgs_downstream_mechanics(
    payload: Any,
    *,
    label: str,
) -> None:
    """Reject source-page mechanics outside explicit exclusion disclosures.

    Project-owned MN, muscle, hinge, and FlyBody fields are intentionally not
    forbidden.  This check targets only the exact legacy ``/fly-fgs`` field
    names that the intake contract rejects.  They may remain as literals in a
    ``rejected_downstream_fields`` list or explanatory notice, but never as a
    data-bearing key/value elsewhere in the published replay or run manifest.
    """

    narrative_keys = {
        "authority_notice",
        "does_not_establish",
        "notice",
        "notes",
        "rejected_downstream_fields",
    }
    violations: List[str] = []

    def visit(value: Any, path: Tuple[str, ...], exempt: bool) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = path + (key,)
                child_exempt = (
                    exempt or key in narrative_keys or key.endswith("_notice")
                )
                dotted = ".".join(child_path)
                if not child_exempt and any(
                    dotted == forbidden or dotted.endswith("." + forbidden)
                    for forbidden in FLY_FGS_FORBIDDEN_DOWNSTREAM_FIELDS
                ):
                    violations.append(dotted)
                visit(child, child_path, child_exempt)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, path + (str(index),), exempt)
        elif (
            not exempt
            and isinstance(value, str)
            and any(
                forbidden in value
                for forbidden in FLY_FGS_FORBIDDEN_DOWNSTREAM_FIELDS
            )
        ):
            violations.append(".".join(path))

    visit(payload, (), False)
    _require(
        not violations,
        "{} contains forbidden fly-FGS downstream mechanics outside an explicit "
        "rejected list/notice: {}".format(label, ", ".join(sorted(violations))),
    )


def _audit_fly_fgs_validation_inventory(
    report: ValidationReport,
    registry: BenchmarkRegistry,
    *,
    canonical_attached: bool,
) -> None:
    """Bind publication to the v1.13 fly-FGS runtime and worker gates."""

    _require(
        registry.version == FLY_FGS_REGISTRY_VERSION,
        "published validation registry must be version {}".format(
            FLY_FGS_REGISTRY_VERSION
        ),
    )
    required_case_ids = (
        FLY_FGS_INTEGRITY_CASE_ID,
        FLY_FGS_INCREMENTAL_CASE_ID,
        FLY_FGS_CHECKPOINT_CASE_ID,
        FLY_FGS_FLYBODY_CASE_ID,
    )
    for case_id in required_case_ids:
        try:
            case = registry.case(case_id)
        except KeyError as exc:
            raise AuditError(
                "required fly-FGS validation case is not registered: {}".format(
                    case_id
                )
            ) from exc
        _require(
            case.fixture_uri == FLY_FGS_REGISTERED_CAPTURE_URI
            and case.input_sha256 == FLY_FGS_REGISTERED_CAPTURE_SHA256,
            "{} does not bind the registered fly-FGS capture".format(case_id),
        )
        matches = tuple(
            result for result in report.results if result.case_id == case_id
        )
        _require(
            len(matches) == 1,
            "validation report must contain exactly one result for {}".format(
                case_id
            ),
        )
        _require(
            matches[0].case_version == case.version,
            "validation result version does not match registry case {}".format(
                case_id
            ),
        )

    integrity = next(
        result
        for result in report.results
        if result.case_id == FLY_FGS_INTEGRITY_CASE_ID
    )
    worker = next(
        result
        for result in report.results
        if result.case_id == FLY_FGS_FLYBODY_CASE_ID
    )
    _require(
        integrity.status is GateStatus.PASS,
        "published validation requires a PASS for {}".format(
            FLY_FGS_INTEGRITY_CASE_ID
        ),
    )
    for case_id in (FLY_FGS_INCREMENTAL_CASE_ID, FLY_FGS_CHECKPOINT_CASE_ID):
        runtime_result = next(
            result for result in report.results if result.case_id == case_id
        )
        _require(
            runtime_result.status is GateStatus.PASS,
            "published validation requires a PASS for {}".format(case_id),
        )
    if canonical_attached:
        _require(
            worker.status is GateStatus.PASS,
            "a published fly_fgs_canonical FlyBody replay requires a PASS for {}".format(
                FLY_FGS_FLYBODY_CASE_ID
            ),
        )
    else:
        _require(
            worker.status in (GateStatus.PASS, GateStatus.BLOCKED),
            "the unattached fly-FGS worker gate must be PASS or explicitly BLOCKED",
        )


def _fly_fgs_required_array_ids(
    run_manifest: Mapping[str, Any],
) -> Tuple[str, ...]:
    descriptors = run_manifest.get("arrays")
    _require(isinstance(descriptors, Mapping), "run arrays must be an object")
    exact = {
        "time_s",
        "circuit_sample_time_s",
        "retinal_normalized_luminance",
        "retinal_measurement_time_s",
        "retinal_availability_time_s",
        "retinal_exposure_interval_s",
        "fly_fgs_source_time_s",
        "fly_fgs_full_cell_voltage_v",
        "fly_fgs_full_cell_activity",
        "fly_fgs_t4a_input_luminance",
        "fly_fgs_t4a_input_azimuth_rad",
        "fly_fgs_t4a_input_elevation_rad",
        "fly_fgs_t4a_input_cell_index",
        "measured_wing_joint_angle_rad",
        "measured_wing_joint_velocity_rad_s",
        "whole_fly_com_position_world_m",
        "external_actuator_torque_n_m",
        "aerodynamic_force_body_n",
    }
    prefixes = (
        "fly_fgs_pooled/",
        "fly_fgs_stimulus/",
        "circuit_voltage/",
        "circuit_availability_time_s/",
        "descending_rate_hz/",
        "wing_motor_rate_hz/",
        "wing_motor_event_availability_time_s/",
        "muscle_activation/",
    )
    exact.update(
        array_id
        for array_id in descriptors
        if isinstance(array_id, str) and array_id.startswith(prefixes)
    )
    return tuple(sorted(exact))


def _audit_fly_fgs_canonical_episode(
    episode: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    registry: BenchmarkRegistry,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Audit the canonical source circuit and its motor-ineligible replay cut."""

    try:
        integrity_case = registry.case(FLY_FGS_INTEGRITY_CASE_ID)
        worker_case = registry.case(FLY_FGS_FLYBODY_CASE_ID)
    except KeyError as exc:
        raise AuditError("registered fly-FGS validation cases are missing") from exc
    _require(
        integrity_case.fixture_uri == FLY_FGS_REGISTERED_CAPTURE_URI
        and worker_case.fixture_uri == integrity_case.fixture_uri
        and integrity_case.input_sha256 == FLY_FGS_REGISTERED_CAPTURE_SHA256
        and worker_case.input_sha256 == integrity_case.input_sha256,
        "fly-FGS integrity and FlyBody cases do not bind one registered capture",
    )
    _require(
        episode.get("id") == FLY_FGS_EPISODE_ID,
        "fly-FGS episode ID is not canonical",
    )
    _require(
        episode.get("physics_backend") == "flybody"
        and episode.get("source_kind") == FLY_FGS_SOURCE_KIND,
        "fly_fgs_canonical must execute the FlyBody worker",
    )
    _require(
        episode.get("wing_kinematics_source") == "measured_flybody_yaw_axes"
        and episode.get("body_state_reference")
        == "whole-fly articulated subtree COM for display; root/thorax retained separately"
        and tuple(episode.get("measured_wing_joint_order", ()))
        == REVIEWED_FLYBODY_WING_AXIS_ORDER,
        "fly_fgs_canonical must identify measured FlyBody wing and whole-fly state",
    )
    _require(
        _close(
            _finite_number(episode.get("duration_s"), "fly-FGS duration_s"),
            FLY_FGS_DURATION_S,
        ),
        "fly_fgs_canonical must cover exactly 0.5 seconds",
    )
    scope = episode.get("neural_model_scope")
    _require(
        isinstance(scope, Mapping)
        and scope.get("kind") == "registered_fly_fgs_fixed_step_circuit"
        and scope.get("full_circuit_executed") is True
        and scope.get("circuit_cell_count") == FLY_FGS_CELL_COUNT
        and scope.get("exported_circuit_channel_count")
        == FLY_FGS_MOTOR_CHANNEL_COUNT,
        "fly_fgs_canonical neural scope is incomplete or substituted",
    )

    configuration = run_manifest.get("configuration")
    runtime = run_manifest.get("runtime")
    pipeline = run_manifest.get("pipeline")
    diagnostics = run_manifest.get("diagnostics")
    descriptors = run_manifest.get("arrays")
    _require(isinstance(configuration, Mapping), "run configuration is missing")
    _require(isinstance(runtime, Mapping), "run runtime is missing")
    _require(isinstance(pipeline, Mapping), "fly-FGS run pipeline is missing")
    _require(isinstance(diagnostics, Mapping), "run diagnostics are missing")
    _require(isinstance(descriptors, Mapping), "run arrays are missing")
    _require(
        runtime.get("physics_backend") == "flybody"
        and run_manifest.get("source_kind") == FLYBODY_WORKER_SOURCE_KIND,
        "fly_fgs_canonical run silently substituted a non-FlyBody backend",
    )
    _require(
        _close(
            _finite_number(configuration.get("duration_s"), "run duration_s"),
            FLY_FGS_DURATION_S,
        )
        and _close(
            _finite_number(
                configuration.get("physics_timestep_s"),
                "run physics_timestep_s",
            ),
            FLY_FGS_PHYSICS_TIMESTEP_S,
        ),
        "fly_fgs_canonical run duration or physics timestep is not authoritative",
    )
    _require(
        pipeline.get("pipeline_id") == FLY_FGS_PIPELINE_ID
        and pipeline.get("mode")
        == "open_loop_registered_fly_fgs_fixed_step_circuit_output",
        "fly_fgs_canonical run pipeline identity is incorrect",
    )
    visual_boundary = pipeline.get("visual_boundary")
    _require(
        isinstance(visual_boundary, Mapping)
        and visual_boundary.get("retinal_frames_present") is True
        and visual_boundary.get("status")
        == "registered_retinal_frames_and_circuit_replay",
        "fly_fgs_canonical must retain the registered causal retinal boundary",
    )
    replay_contract = pipeline.get("circuit_replay_contract")
    _require(
        isinstance(replay_contract, Mapping)
        and replay_contract.get("attachment") == "web_replay.circuit_replay"
        and replay_contract.get("scientific_array_prefix") == "fly_fgs_"
        and replay_contract.get("source_time_array") == "fly_fgs_source_time_s"
        and replay_contract.get("full_cell_voltage_array")
        == "fly_fgs_full_cell_voltage_v"
        and replay_contract.get("full_cell_activity_array")
        == "fly_fgs_full_cell_activity"
        and replay_contract.get("t4a_input_luminance_array")
        == "fly_fgs_t4a_input_luminance"
        and replay_contract.get("motor_input_policy")
        == "four CircuitOutputTrace NOD1 voltage channels only"
        and replay_contract.get("full_cell_state_eligible_motor_input") is False
        and replay_contract.get("raw_app_side_labels_are_anatomical") is False,
        "fly-FGS circuit replay contract widens or obscures the motor-input cut",
    )
    source = pipeline.get("source_metadata")
    _require(isinstance(source, Mapping), "fly-FGS source metadata is missing")
    expected_source = {
        "input_mode": "registered_fly_fgs_fixed_step_circuit",
        "visual_source": "fly_fgs_analytic_retinal_scene_and_circuit_bundle",
        "retinal_frames_present": True,
        "retinal_frame_count": FLY_FGS_CAUSAL_RETINAL_FRAME_COUNT,
        "source_manifest_uri": FLY_FGS_SOURCE_MANIFEST_URI,
        "source_manifest_sha256": FLY_FGS_SOURCE_MANIFEST_SHA256,
        "capture_uri": FLY_FGS_REGISTERED_CAPTURE_URI,
        "capture_sha256": FLY_FGS_REGISTERED_CAPTURE_SHA256,
        "source_snapshot_id": FLY_FGS_SNAPSHOT_ID,
        "source_sample_count": FLY_FGS_SOURCE_SAMPLE_COUNT,
        "pre_roll_steps": 48,
        "circuit_cell_count": FLY_FGS_CELL_COUNT,
        "exported_motor_bound_circuit_channels": FLY_FGS_MOTOR_CHANNEL_COUNT,
        "exported_circuit_channels": FLY_FGS_MOTOR_CHANNEL_COUNT,
        "motor_input_scope": "four_registered_nod1_voltage_channels_only",
        "circuit_replay_present": True,
        "full_cell_state_eligible_motor_input": False,
        "fly_fgs_downstream_mechanics_imported": False,
    }
    _require(
        all(source.get(key) == value for key, value in expected_source.items())
        and _close(
            _finite_number(source.get("source_duration_s"), "source_duration_s"),
            FLY_FGS_DURATION_S,
        )
        and _close(
            _finite_number(source.get("source_sample_dt_s"), "source_sample_dt_s"),
            FLY_FGS_SOURCE_DT_S,
        )
        and _close(
            _finite_number(source.get("pre_roll_duration_s"), "pre_roll_duration_s"),
            0.24,
        ),
        "fly_fgs_canonical does not bind the registered source scope/timebase",
    )
    inventory = source.get("circuit_inventory")
    _require(
        isinstance(inventory, Mapping)
        and inventory.get("cell_count") == FLY_FGS_CELL_COUNT
        and inventory.get("cell_type_counts")
        == {"DCH": 2, "LLPC1": 219, "Nod1": 4, "T4a": 1457, "VCH": 2}
        and inventory.get("t5_cell_count") == 0
        and inventory.get("event_count") == 61789
        and inventory.get("effective_runtime_event_count") == 60200
        and inventory.get("cable_count") == 8
        and inventory.get("retinotopic_t4a_input_count")
        == FLY_FGS_RETINAL_INPUT_COUNT
        and tuple(inventory.get("nod1_root_ids", ())) == FLY_FGS_NOD1_ROOT_IDS,
        "fly_fgs_canonical source circuit inventory is not the registered 1,684-cell circuit",
    )

    replay = episode.get("circuit_replay")
    _require(isinstance(replay, Mapping), "fly_fgs_canonical circuit replay is missing")
    _require(
        replay.get("schema_version") == FLY_FGS_SCHEMA_VERSION
        and replay.get("source_kind")
        == "registered_fly_fgs_fixed_step_circuit_replay"
        and replay.get("snapshot_id") == FLY_FGS_SNAPSHOT_ID,
        "fly-FGS circuit replay schema/source identity is incorrect",
    )
    receipts = replay.get("asset_receipts")
    _require(
        isinstance(receipts, Mapping)
        and set(receipts) == set(FLY_FGS_ASSET_RECEIPTS),
        "fly-FGS circuit replay source receipt inventory is incomplete",
    )
    assert isinstance(receipts, Mapping)
    for asset_id, (expected_sha, expected_role, eligible) in (
        FLY_FGS_ASSET_RECEIPTS.items()
    ):
        receipt = receipts.get(asset_id)
        _require(
            isinstance(receipt, Mapping)
            and receipt.get("asset_id") == asset_id
            and receipt.get("sha256") == expected_sha
            and receipt.get("role") == expected_role
            and receipt.get("eligible_circuit_input") is eligible,
            "fly-FGS source receipt {} is stale or has the wrong role".format(
                asset_id
            ),
        )
    fixed_step = replay.get("fixed_step")
    _require(isinstance(fixed_step, Mapping), "fly-FGS fixed-step record is missing")
    source_time = np.asarray(fixed_step.get("time_s"), dtype=float)
    expected_source_time = np.arange(FLY_FGS_SOURCE_SAMPLE_COUNT) * FLY_FGS_SOURCE_DT_S
    invariants = fixed_step.get("invariant_observations")
    _require(
        fixed_step.get("sample_count") == FLY_FGS_SOURCE_SAMPLE_COUNT
        and fixed_step.get("dt_s") == FLY_FGS_SOURCE_DT_S
        and fixed_step.get("duration_s") == FLY_FGS_DURATION_S
        and fixed_step.get("pre_roll_steps") == 48
        and fixed_step.get("pre_roll_duration_s") == 0.24
        and fixed_step.get("sample_interval") == "half-open [0, 0.5 s)"
        and isinstance(fixed_step.get("scheduler"), str)
        and "requestAnimationFrame" in fixed_step.get("scheduler", "")
        and source_time.shape == (FLY_FGS_SOURCE_SAMPLE_COUNT,)
        and np.array_equal(source_time, expected_source_time)
        and isinstance(invariants, Mapping)
        and all(
            _close(
                _finite_number(invariants.get(key), "fixed-step invariant {}".format(key)),
                value,
                atol=1.0e-15,
            )
            for key, value in FLY_FGS_EXPECTED_INVARIANTS.items()
        ),
        "fly-FGS fixed-step clock, pre-roll, or non-dead invariants changed",
    )
    replay_inventory = replay.get("cell_inventory")
    _require(
        replay_inventory == inventory,
        "fly-FGS replay and pipeline circuit inventories disagree",
    )
    side_semantics = replay.get("side_semantics")
    _require(
        isinstance(side_semantics, Mapping)
        and side_semantics.get("raw_bundle_labels")
        == dict(FLY_FGS_NOD1_RAW_APP_SIDES)
        and side_semantics.get("anatomical_side") == "unknown"
        and side_semantics.get("mapping_method") == "simulation_convention",
        "raw fly-FGS application L/R labels were promoted to anatomy",
    )
    _require(
        set(replay.get("rejected_downstream_fields", ()))
        == set(FLY_FGS_FORBIDDEN_DOWNSTREAM_FIELDS)
        and set(source.get("rejected_downstream_fields", ()))
        == set(FLY_FGS_FORBIDDEN_DOWNSTREAM_FIELDS),
        "fly-FGS rejected downstream field inventory is incomplete",
    )
    full_scope = replay.get("full_cell_state_scope")
    retinal_scope = replay.get("retinal_frame_scope")
    _require(
        isinstance(full_scope, Mapping)
        and full_scope.get("included") is True
        and full_scope.get("eligible_motor_input") is False,
        "fly-FGS full-cell display state became motor eligible",
    )
    _require(
        isinstance(retinal_scope, Mapping)
        and retinal_scope.get("count") == FLY_FGS_CAUSAL_RETINAL_FRAME_COUNT
        and retinal_scope.get("source_sample_indices")
        == list(range(1, FLY_FGS_SOURCE_SAMPLE_COUNT))
        and retinal_scope.get("sample_0_excluded") is True,
        "fly-FGS causal retinal-frame scope is incorrect",
    )

    full_state = replay.get("full_cell_state")
    topology = replay.get("circuit_topology")
    stimulus = replay.get("stimulus")
    _require(isinstance(full_state, Mapping), "fly-FGS full-cell state is missing")
    _require(isinstance(topology, Mapping), "fly-FGS circuit topology is missing")
    _require(isinstance(stimulus, Mapping), "fly-FGS stimulus replay is missing")
    cell_ids = full_state.get("cell_ids")
    voltage = np.asarray(full_state.get("voltage_v"), dtype=float)
    activity = np.asarray(full_state.get("activity"), dtype=float)
    cell_axis = topology.get("cell_axis")
    retinotopic = topology.get("retinotopic_t4a")
    _require(
        isinstance(cell_ids, list)
        and len(cell_ids) == FLY_FGS_CELL_COUNT
        and isinstance(cell_axis, list)
        and len(cell_axis) == FLY_FGS_CELL_COUNT
        and [cell.get("id") for cell in cell_axis] == cell_ids
        and voltage.shape == (FLY_FGS_SOURCE_SAMPLE_COUNT, FLY_FGS_CELL_COUNT)
        and activity.shape == voltage.shape
        and np.all(np.isfinite(voltage))
        and np.all(np.isfinite(activity))
        and isinstance(retinotopic, Mapping)
        and retinotopic.get("count") == FLY_FGS_RETINAL_INPUT_COUNT
        and len(retinotopic.get("cell_indices", ())) == FLY_FGS_RETINAL_INPUT_COUNT,
        "fly-FGS replay does not contain the registered 100x1,684 state/topology",
    )
    retinal_input = stimulus.get("retinal_input")
    _require(isinstance(retinal_input, Mapping), "fly-FGS retinal input is missing")
    retinal_luminance = np.asarray(retinal_input.get("luminance"), dtype=float)
    _require(
        retinal_luminance.shape
        == (FLY_FGS_SOURCE_SAMPLE_COUNT, FLY_FGS_RETINAL_INPUT_COUNT)
        and np.all(np.isfinite(retinal_luminance))
        and retinal_input.get("cell_ids") == retinotopic.get("cell_ids")
        and retinal_input.get("cell_indices") == retinotopic.get("cell_indices"),
        "fly-FGS replay does not contain the registered 100x1,441 retinal input",
    )
    pooled = replay.get("pooled_readout_traces")
    expected_pooled = {
        "dch_activity",
        "dch_voltage_v",
        "llpc1_activity",
        "nod1_activity",
        "nod1_voltage_v",
        "t4_activity",
        "vch_activity",
        "vch_voltage_v",
    }
    _require(
        isinstance(pooled, Mapping)
        and set(pooled) == expected_pooled
        and all(
            isinstance(channels, Mapping)
            and set(channels) == {"raw_app_L", "raw_app_R"}
            and all(
                np.asarray(values, dtype=float).shape
                == (FLY_FGS_SOURCE_SAMPLE_COUNT,)
                and np.all(np.isfinite(np.asarray(values, dtype=float)))
                for values in channels.values()
            )
            for channels in pooled.values()
        ),
        "fly-FGS pooled circuit display traces are incomplete or malformed",
    )

    circuit_contract = pipeline.get("circuit_output_contract")
    _require(
        isinstance(circuit_contract, Mapping)
        and circuit_contract.get("exact_timebase") is True
        and circuit_contract.get("sample_count") == FLY_FGS_SOURCE_SAMPLE_COUNT,
        "fly-FGS motor-bound circuit output has the wrong timebase",
    )
    signals = circuit_contract.get("signals")
    _require(
        isinstance(signals, list) and len(signals) == FLY_FGS_MOTOR_CHANNEL_COUNT,
        "fly-FGS circuit contract must expose exactly four NOD1 channels",
    )
    circuit_trace_sha = _require_sha256(
        pipeline.get("circuit_trace_sha256"), "pipeline.circuit_trace_sha256"
    )
    _require(
        circuit_trace_sha == FLY_FGS_TRACE_SHA256
        and episode.get("source_circuit_trace_sha256") == circuit_trace_sha,
        "fly-FGS motor-bound circuit trace digest is not the registered digest",
    )
    signal_by_root: Dict[str, Mapping[str, Any]] = {}
    for signal in signals:
        _require(isinstance(signal, Mapping), "fly-FGS circuit signal is not an object")
        neuron = signal.get("neuron")
        _require(isinstance(neuron, Mapping), "fly-FGS circuit neuron is missing")
        root_id = neuron.get("entity_id")
        side_context = neuron.get("side_context")
        _require(
            isinstance(root_id, str)
            and root_id in FLY_FGS_NOD1_ROOT_IDS
            and neuron.get("cell_type") == "NOD1"
            and neuron.get("anatomical_side") == "unknown"
            and isinstance(side_context, Mapping)
            and side_context.get("raw_dataset_side")
            == FLY_FGS_NOD1_RAW_APP_SIDES[root_id]
            and side_context.get("anatomical_side") == "unknown"
            and side_context.get("mapping_method") == "simulation_convention"
            and signal.get("signal_kind") == "voltage"
            and signal.get("unit") == "V",
            "fly-FGS NOD1 signal identity/laterality scope is incorrect",
        )
        _require(root_id not in signal_by_root, "duplicate fly-FGS NOD1 signal")
        signal_by_root[root_id] = signal
    _require(
        set(signal_by_root) == set(FLY_FGS_NOD1_ROOT_IDS),
        "fly-FGS motor-bound NOD1 root inventory is incomplete",
    )

    source_time_array = np.asarray(arrays.get("fly_fgs_source_time_s"), dtype=float)
    circuit_time_array = np.asarray(arrays.get("circuit_sample_time_s"), dtype=float)
    full_voltage_array = np.asarray(
        arrays.get("fly_fgs_full_cell_voltage_v"), dtype=float
    )
    full_activity_array = np.asarray(
        arrays.get("fly_fgs_full_cell_activity"), dtype=float
    )
    retinal_array = np.asarray(arrays.get("fly_fgs_t4a_input_luminance"), dtype=float)
    causal_retinal = np.asarray(arrays.get("retinal_normalized_luminance"), dtype=float)
    retinal_measurement = np.asarray(arrays.get("retinal_measurement_time_s"), dtype=float)
    retinal_availability = np.asarray(arrays.get("retinal_availability_time_s"), dtype=float)
    retinal_exposure = np.asarray(arrays.get("retinal_exposure_interval_s"), dtype=float)
    expected_exposure = np.column_stack(
        (expected_source_time[:-1], expected_source_time[1:])
    )
    _require(
        np.array_equal(source_time_array, expected_source_time)
        and np.array_equal(circuit_time_array, expected_source_time)
        and np.array_equal(full_voltage_array, voltage)
        and np.array_equal(full_activity_array, activity)
        and np.array_equal(retinal_array, retinal_luminance)
        and causal_retinal.shape
        == (FLY_FGS_CAUSAL_RETINAL_FRAME_COUNT, FLY_FGS_RETINAL_INPUT_COUNT)
        and np.array_equal(causal_retinal, retinal_luminance[1:])
        and np.array_equal(retinal_measurement, expected_source_time[1:])
        and np.array_equal(retinal_availability, expected_source_time[1:])
        and np.array_equal(retinal_exposure, expected_exposure),
        "fly-FGS chunked state/retinal arrays disagree with the causal replay",
    )
    exact_descriptor_contract = {
        "fly_fgs_source_time_s": ("s", "registered_fly_fgs_fixed_step_source_clock"),
        "fly_fgs_full_cell_voltage_v": (
            "V",
            "registered_fly_fgs_full_cell_display_state_not_motor_input",
        ),
        "fly_fgs_full_cell_activity": (
            "1",
            "registered_fly_fgs_full_cell_display_state_not_motor_input",
        ),
        "fly_fgs_t4a_input_luminance": (
            "1",
            "registered_fly_fgs_exact_t4a_analytic_input",
        ),
        "retinal_normalized_luminance": ("1", "causal_retinal_frame"),
        "retinal_measurement_time_s": ("s", "causal_retinal_frame_timing"),
        "retinal_availability_time_s": ("s", "causal_retinal_frame_timing"),
        "retinal_exposure_interval_s": ("s", "causal_retinal_frame_timing"),
        "time_s": ("s", "shared_episode_clock"),
        "measured_wing_joint_angle_rad": (
            "rad",
            "external_physics_measured_output",
        ),
        "measured_wing_joint_velocity_rad_s": (
            "rad s^-1",
            "external_physics_measured_output",
        ),
        "whole_fly_com_position_world_m": (
            "m",
            "flybody_mujoco_articulated_subtree_com",
        ),
        "external_actuator_torque_n_m": (
            "N m",
            "logged_projection_of_external_mujoco_wing_actuator_torque",
        ),
        "aerodynamic_force_body_n": (
            "N",
            "flybody_mujoco_root_total_output",
        ),
    }
    _require(
        all(
            isinstance(descriptors.get(array_id), Mapping)
            and descriptors[array_id].get("unit") == unit
            and descriptors[array_id].get("provenance") == provenance
            for array_id, (unit, provenance) in exact_descriptor_contract.items()
        )
        and all(
            descriptor.get("provenance")
            == "registered_fly_fgs_app_side_label_anatomy_unknown"
            for array_id, descriptor in descriptors.items()
            if isinstance(array_id, str)
            and array_id.startswith("fly_fgs_pooled/")
            and isinstance(descriptor, Mapping)
        ),
        "fly-FGS scientific arrays lack exact source/motor-ineligible provenance",
    )
    for root_id, signal in signal_by_root.items():
        value_id = signal.get("value_array")
        availability_id = signal.get("availability_time_array")
        _require(
            value_id == "circuit_voltage/{}".format(root_id)
            and availability_id
            == "circuit_availability_time_s/{}".format(root_id),
            "fly-FGS NOD1 circuit array IDs are incorrect",
        )
        value_descriptor = descriptors.get(value_id)
        availability_descriptor = descriptors.get(availability_id)
        values = np.asarray(arrays.get(value_id), dtype=float)
        availability = np.asarray(arrays.get(availability_id), dtype=float)
        column = cell_ids.index("r" + root_id)
        _require(
            isinstance(value_descriptor, Mapping)
            and value_descriptor.get("unit") == "V"
            and value_descriptor.get("provenance")
            == "registered_fly_fgs_fixed_step_voltage_v"
            and isinstance(availability_descriptor, Mapping)
            and availability_descriptor.get("unit") == "s"
            and availability_descriptor.get("provenance")
            == "registered_fly_fgs_fixed_step_timing"
            and np.array_equal(values, voltage[:, column])
            and np.array_equal(availability, expected_source_time),
            "fly-FGS NOD1 motor array is not an exact full-state projection",
        )

    time_s = np.asarray(arrays.get("time_s"), dtype=float)
    measured_angle = np.asarray(arrays.get("measured_wing_joint_angle_rad"), dtype=float)
    measured_velocity = np.asarray(
        arrays.get("measured_wing_joint_velocity_rad_s"), dtype=float
    )
    whole_fly_com = np.asarray(arrays.get("whole_fly_com_position_world_m"), dtype=float)
    actuator_torque = np.asarray(arrays.get("external_actuator_torque_n_m"), dtype=float)
    fluid_force = np.asarray(arrays.get("aerodynamic_force_body_n"), dtype=float)
    sample_count = len(time_s)
    _require(
        time_s.ndim == 1
        and sample_count >= 2
        and np.all(np.isfinite(time_s))
        and np.all(np.diff(time_s) > 0.0)
        and _close(float(time_s[0]), 0.0)
        and _close(float(time_s[-1]), FLY_FGS_DURATION_S)
        and measured_angle.shape == (sample_count, 6)
        and measured_velocity.shape == (sample_count, 6)
        and whole_fly_com.shape == (sample_count, 3)
        and actuator_torque.shape == (sample_count, 6)
        and fluid_force.shape == (sample_count, 3)
        and all(
            np.all(np.isfinite(value))
            for value in (
                measured_angle,
                measured_velocity,
                whole_fly_com,
                actuator_torque,
                fluid_force,
            )
        ),
        "fly_fgs_canonical is missing finite measured FlyBody telemetry",
    )
    motor_events = tuple(
        np.asarray(value, dtype=float)
        for key, value in arrays.items()
        if key.startswith("wing_motor_event_availability_time_s/")
    )
    steering_muscles = tuple(
        np.asarray(value, dtype=float)
        for key, value in arrays.items()
        if key.startswith("muscle_activation/")
        and key.rsplit(":", 1)[-1] in {"iv2", "i1", "iv1", "b3"}
    )
    _require(
        sum(int(value.size) for value in motor_events) > 0
        and steering_muscles
        and max(
            float(np.max(np.abs(value), initial=0.0)) for value in steering_muscles
        )
        > 0.0
        and float(np.max(np.abs(actuator_torque), initial=0.0)) > 0.0
        and float(np.max(np.linalg.norm(fluid_force, axis=1), initial=0.0)) > 0.0
        and float(np.max(np.ptp(measured_angle, axis=0), initial=0.0)) > 0.0,
        "fly_fgs_canonical circuit-to-FlyBody causal output is dead or incomplete",
    )
    _audit_no_fly_fgs_downstream_mechanics(episode, label="fly-FGS episode")
    _audit_no_fly_fgs_downstream_mechanics(
        run_manifest, label="fly-FGS run manifest"
    )


def _audit_frozen_browser_nod1_episode(
    episode: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    registry: BenchmarkRegistry,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Require the marquee replay to retain its exact circuit and worker scope."""

    try:
        parity_case = registry.case(FROZEN_BROWSER_PARITY_CASE_ID)
        flybody_case = registry.case(FROZEN_BROWSER_FLYBODY_CASE_ID)
    except KeyError as exc:
        raise AuditError("registered frozen-browser FlyBody cases are missing") from exc
    _require(
        parity_case.input_sha256 is not None
        and parity_case.fixture_uri is not None
        and flybody_case.input_sha256 == parity_case.input_sha256
        and flybody_case.fixture_uri == parity_case.fixture_uri,
        "frozen-browser parity and FlyBody cases do not bind one fixture",
    )
    fixture_sha256 = parity_case.input_sha256
    assert fixture_sha256 is not None

    _require(
        episode.get("id") == FROZEN_BROWSER_EPISODE_ID,
        "frozen-browser episode ID is not canonical",
    )
    _require(
        episode.get("physics_backend") == "flybody"
        and episode.get("source_kind")
        == "exploratory_frozen_browser_nod1_flybody_pipeline",
        "frozen-browser NOD1 episode must execute the FlyBody worker",
    )
    _require(
        episode.get("wing_kinematics_source") == "measured_flybody_yaw_axes"
        and episode.get("body_state_reference")
        == "whole-fly articulated subtree COM for display; root/thorax retained separately",
        "frozen-browser replay must identify measured wing and whole-fly COM state",
    )
    _require(
        _close(
            _finite_number(episode.get("duration_s"), "frozen episode duration_s"),
            FROZEN_BROWSER_DURATION_S,
        ),
        "frozen-browser episode must cover exactly 0.5 seconds",
    )
    scope = episode.get("neural_model_scope")
    _require(
        isinstance(scope, Mapping)
        and scope.get("kind") == "full_legacy_circuit_cable_export"
        and scope.get("historical") is True
        and scope.get("full_circuit_executed") is True
        and scope.get("circuit_cell_count") == 1208
        and scope.get("exported_circuit_channel_count") == 4,
        "frozen-browser replay must be explicitly historical and retain its scope",
    )
    wing_order = episode.get("measured_wing_joint_order")
    _require(
        isinstance(wing_order, list)
        and len(wing_order) == 6
        and len(set(wing_order)) == 6
        and all(isinstance(item, str) and bool(item) for item in wing_order),
        "frozen-browser FlyBody replay requires six measured wing axes",
    )
    _require(
        tuple(wing_order) == REVIEWED_FLYBODY_WING_AXIS_ORDER,
        "frozen-browser replay wing axes differ from the reviewed FlyGym 2.1.0 order",
    )

    configuration = run_manifest.get("configuration")
    runtime = run_manifest.get("runtime")
    pipeline = run_manifest.get("pipeline")
    diagnostics = run_manifest.get("diagnostics")
    _require(isinstance(configuration, Mapping), "run configuration must be an object")
    _require(isinstance(runtime, Mapping), "run runtime must be an object")
    _require(isinstance(pipeline, Mapping), "frozen-browser run pipeline is missing")
    _require(isinstance(diagnostics, Mapping), "run diagnostics must be an object")
    _require(
        runtime.get("physics_backend") == "flybody"
        and run_manifest.get("source_kind") == FLYBODY_WORKER_SOURCE_KIND,
        "frozen-browser run silently substituted a non-FlyBody backend",
    )
    _require(
        _close(
            _finite_number(configuration.get("duration_s"), "run duration_s"),
            FROZEN_BROWSER_DURATION_S,
        )
        and _close(
            _finite_number(
                configuration.get("physics_timestep_s"),
                "run physics_timestep_s",
            ),
            FROZEN_BROWSER_PHYSICS_TIMESTEP_S,
        ),
        "frozen-browser run duration or physics timestep is not authoritative",
    )
    _require(
        pipeline.get("pipeline_id")
        == "frozen-browser-nod1-parity-to-dnp26-to-flight-v1"
        and pipeline.get("mode")
        == "open_loop_frozen_browser_visual_circuit_output",
        "frozen-browser run pipeline identity is incorrect",
    )
    visual_boundary = pipeline.get("visual_boundary")
    _require(
        isinstance(visual_boundary, Mapping)
        and visual_boundary.get("retinal_frames_present") is False
        and visual_boundary.get("status")
        == "frozen_browser_output_no_retinal_frames",
        "frozen-browser run must disclose that source retinal frames are unavailable",
    )
    source = pipeline.get("source_metadata")
    _require(isinstance(source, Mapping), "frozen-browser source metadata is missing")
    expected_source = {
        "input_mode": "registered_frozen_browser_nod1_parity",
        "visual_source": "frozen_browser_visual_circuit_output",
        "retinal_frames_present": False,
        "retinal_frame_count": 0,
        "fixture_uri": parity_case.fixture_uri,
        "fixture_sha256": fixture_sha256,
        "benchmark_case_id": FROZEN_BROWSER_PARITY_CASE_ID,
        "benchmark_case_version": parity_case.version,
        "source_sample_count": 100,
        "circuit_cell_count": 1208,
        "exported_circuit_channels": 4,
        "circuit_model_scope": (
            "frozen_real_chromium_1208_cell_circuit_four_nod1_readouts"
        ),
    }
    _require(
        all(source.get(key) == value for key, value in expected_source.items())
        and _close(
            _finite_number(source.get("source_duration_s"), "source_duration_s"),
            FROZEN_BROWSER_DURATION_S,
        )
        and _close(
            _finite_number(source.get("source_sample_dt_s"), "source_sample_dt_s"),
            0.005,
        )
        and _close(
            _finite_number(
                source.get("source_sample_interval_end_s"),
                "source_sample_interval_end_s",
            ),
            FROZEN_BROWSER_DURATION_S,
        ),
        "frozen-browser run does not bind the registered fixture SHA/scope/timebase",
    )
    circuit_inventory = source.get("circuit_inventory")
    _require(
        isinstance(circuit_inventory, Mapping)
        and circuit_inventory.get("cells") == 1208,
        "frozen-browser source circuit inventory is not the 1,208-cell circuit",
    )
    circuit_contract = pipeline.get("circuit_output_contract")
    _require(
        isinstance(circuit_contract, Mapping)
        and circuit_contract.get("exact_timebase") is True
        and circuit_contract.get("sample_count") == 100,
        "frozen-browser circuit output contract has the wrong timebase",
    )
    signals = circuit_contract.get("signals")
    _require(
        isinstance(signals, list) and len(signals) == 4,
        "frozen-browser circuit contract must expose exactly four readouts",
    )
    circuit_trace_sha = _require_sha256(
        pipeline.get("circuit_trace_sha256"), "pipeline.circuit_trace_sha256"
    )
    _require(
        episode.get("source_circuit_trace_sha256") == circuit_trace_sha,
        "frozen-browser replay circuit digest disagrees with its run manifest",
    )

    descriptors = run_manifest.get("arrays")
    _require(isinstance(descriptors, Mapping), "run arrays must be an object")
    circuit_value_ids = []
    circuit_availability_ids = []
    for signal in signals:
        _require(isinstance(signal, Mapping), "circuit signal must be an object")
        value_id = signal.get("value_array")
        availability_id = signal.get("availability_time_array")
        _require(
            isinstance(value_id, str)
            and value_id.startswith("circuit_voltage/")
            and isinstance(availability_id, str)
            and availability_id.startswith("circuit_availability_time_s/"),
            "frozen-browser signal arrays are not registered voltage readouts",
        )
        value_descriptor = descriptors.get(value_id)
        availability_descriptor = descriptors.get(availability_id)
        _require(
            isinstance(value_descriptor, Mapping)
            and value_descriptor.get("unit") == "V"
            and value_descriptor.get("provenance")
            == "registered_frozen_chromium_browser_voltage_v"
            and isinstance(availability_descriptor, Mapping)
            and availability_descriptor.get("unit") == "s"
            and availability_descriptor.get("provenance")
            == "registered_frozen_chromium_readout_timing",
            "frozen-browser voltage or availability array lacks registered Chromium provenance",
        )
        circuit_value_ids.append(value_id)
        circuit_availability_ids.append(availability_id)
    _require(
        len(set(circuit_value_ids)) == 4
        and len(set(circuit_availability_ids)) == 4,
        "frozen-browser circuit signal arrays must be unique",
    )
    _require(
        not any(
            array_id.startswith("retinal_")
            or "python_voltage" in array_id
            or array_id in ("steering", "forward_looking_phase_T")
            for array_id in descriptors
        ),
        "frozen-browser run contains a forbidden substituted visual/motor array",
    )
    required_mechanics_descriptors = {
        "time_s": ("s", "shared_episode_clock"),
        "measured_wing_joint_angle_rad": (
            "rad",
            "external_physics_measured_output",
        ),
        "measured_wing_joint_velocity_rad_s": (
            "rad s^-1",
            "external_physics_measured_output",
        ),
        "whole_fly_com_position_world_m": (
            "m",
            "flybody_mujoco_articulated_subtree_com",
        ),
        "external_actuator_torque_n_m": (
            "N m",
            "logged_projection_of_external_mujoco_wing_actuator_torque",
        ),
        "aerodynamic_force_body_n": (
            "N",
            "flybody_mujoco_root_total_output",
        ),
    }
    _require(
        all(
            isinstance(descriptors.get(array_id), Mapping)
            and descriptors[array_id].get("unit") == expected_unit
            and descriptors[array_id].get("provenance") == expected_provenance
            for array_id, (
                expected_unit,
                expected_provenance,
            ) in required_mechanics_descriptors.items()
        ),
        "frozen-browser measured mechanics arrays lack authoritative units or provenance",
    )

    time_s = np.asarray(arrays.get("time_s"), dtype=float)
    circuit_time_s = np.asarray(arrays.get("circuit_sample_time_s"), dtype=float)
    measured_angle = np.asarray(arrays.get("measured_wing_joint_angle_rad"), dtype=float)
    measured_velocity = np.asarray(
        arrays.get("measured_wing_joint_velocity_rad_s"), dtype=float
    )
    whole_fly_com = np.asarray(
        arrays.get("whole_fly_com_position_world_m"), dtype=float
    )
    actuator_torque = np.asarray(
        arrays.get("external_actuator_torque_n_m"), dtype=float
    )
    fluid_force = np.asarray(arrays.get("aerodynamic_force_body_n"), dtype=float)
    _require(
        time_s.ndim == 1
        and len(time_s) >= 2
        and np.all(np.isfinite(time_s))
        and np.all(np.diff(time_s) > 0.0)
        and _close(float(time_s[0]), 0.0)
        and _close(float(time_s[-1]), FROZEN_BROWSER_DURATION_S)
        and circuit_time_s.shape == (100,)
        and np.all(np.isfinite(circuit_time_s))
        and np.allclose(
            circuit_time_s,
            np.arange(100, dtype=float) * 0.005,
            rtol=0.0,
            atol=1.0e-12,
        ),
        "frozen-browser run arrays do not preserve the 0.5-second timebase",
    )
    sample_count = len(time_s)
    _require(
        measured_angle.shape == (sample_count, 6)
        and measured_velocity.shape == (sample_count, 6)
        and whole_fly_com.shape == (sample_count, 3)
        and actuator_torque.shape == (sample_count, 6)
        and fluid_force.shape == (sample_count, 3)
        and all(
            np.all(np.isfinite(values))
            for values in (
                measured_angle,
                measured_velocity,
                whole_fly_com,
                actuator_torque,
                fluid_force,
            )
        ),
        "frozen-browser run is missing measured wing, actuator, force, or whole-COM telemetry",
    )
    for value_id, availability_id in zip(
        circuit_value_ids, circuit_availability_ids
    ):
        values = np.asarray(arrays.get(value_id), dtype=float)
        availability = np.asarray(arrays.get(availability_id), dtype=float)
        _require(
            values.shape == (100,)
            and np.all(np.isfinite(values))
            and availability.shape == (100,)
            and np.all(np.isfinite(availability))
            and np.allclose(
                availability,
                circuit_time_s,
                rtol=0.0,
                atol=1.0e-12,
            ),
            "frozen-browser circuit array {} has the wrong shape or timing".format(
                value_id
            ),
        )

    descending = tuple(
        np.asarray(value, dtype=float)
        for key, value in arrays.items()
        if key.startswith("descending_rate_hz/")
    )
    motor_rates = tuple(
        np.asarray(value, dtype=float)
        for key, value in arrays.items()
        if key.startswith("wing_motor_rate_hz/")
    )
    motor_events = tuple(
        np.asarray(value, dtype=float)
        for key, value in arrays.items()
        if key.startswith("wing_motor_event_availability_time_s/")
    )
    steering_muscles = tuple(
        np.asarray(value, dtype=float)
        for key, value in arrays.items()
        if key.startswith("muscle_activation/")
        and key.rsplit(":", 1)[-1] in {"iv2", "i1", "iv1", "b3"}
    )
    _require(
        all(np.all(np.isfinite(value)) for value in descending)
        and all(np.all(np.isfinite(value)) for value in motor_rates)
        and all(
            np.all(np.isfinite(value))
            and np.all(value >= 0.0)
            and np.all(value <= FROZEN_BROWSER_DURATION_S)
            for value in motor_events
        )
        and all(np.all(np.isfinite(value)) for value in steering_muscles)
        and np.max(np.abs(whole_fly_com), initial=0.0) > 0.0
        and
        descending
        and max(float(np.max(np.abs(value), initial=0.0)) for value in descending)
        > 0.0
        and motor_rates
        and max(float(np.max(np.abs(value), initial=0.0)) for value in motor_rates)
        > 0.0
        and sum(int(value.size) for value in motor_events) > 0
        and steering_muscles
        and max(
            float(np.max(np.abs(value), initial=0.0)) for value in steering_muscles
        )
        > 0.0
        and float(np.max(np.abs(actuator_torque), initial=0.0)) > 0.0
        and float(np.max(np.linalg.norm(fluid_force, axis=1), initial=0.0)) > 0.0
        and float(np.max(np.ptp(measured_angle, axis=0), initial=0.0)) > 0.0,
        "frozen-browser circuit-to-FlyBody causal output is dead or incomplete",
    )
    metrics = diagnostics.get("metrics")
    _require(
        isinstance(metrics, Mapping)
        and _finite_number(
            metrics.get("maximum_external_actuator_torque_n_m"),
            "maximum_external_actuator_torque_n_m",
        )
        > 0.0
        and _finite_number(
            metrics.get("maximum_measured_wing_excursion_rad"),
            "maximum_measured_wing_excursion_rad",
        )
        > 0.0,
        "frozen-browser diagnostics do not disclose nonzero measured actuation",
    )
    frames = episode.get("frames")
    _require(isinstance(frames, list) and bool(frames), "frozen episode frames are missing")
    for frame in frames:
        _require(
            isinstance(frame, Mapping)
            and isinstance(frame.get("root_position_m"), list)
            and len(frame["root_position_m"]) == 3
            and isinstance(frame.get("position_m"), list)
            and len(frame["position_m"]) == 3
            and isinstance(frame.get("measured_wing_joint_angle_rad"), list)
            and len(frame["measured_wing_joint_angle_rad"]) == 6
            and isinstance(frame.get("measured_wing_joint_velocity_rad_s"), list)
            and len(frame["measured_wing_joint_velocity_rad_s"]) == 6,
            "frozen-browser replay frame omits measured wing or root/COM state",
        )


def _audit_required_validation_pass(
    report: ValidationReport,
    registry: BenchmarkRegistry,
    case_id: str,
) -> None:
    try:
        case = registry.case(case_id)
    except KeyError as exc:
        raise AuditError("required validation case is not registered: {}".format(case_id)) from exc
    matches = tuple(result for result in report.results if result.case_id == case_id)
    _require(
        len(matches) == 1,
        "validation report must contain exactly one result for {}".format(case_id),
    )
    result = matches[0]
    _require(
        result.case_version == case.version and result.status is GateStatus.PASS,
        "public release requires matching validation PASS for {}".format(case_id),
    )


def _canonical_checkpoint_receipt_from_manifest(
    run_manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    descriptor = run_manifest.get("checkpoint")
    _require(
        isinstance(descriptor, Mapping),
        "canonical run is missing its checkpoint receipt",
    )
    return {
        key: value
        for key, value in descriptor.items()
        if key not in ("path", "file_sha256")
    }


def _audit_online_circuit_replay_attachment(
    attachment_value: Any,
    config: CanonicalClosedLoopConfig,
    arrays: Mapping[str, np.ndarray],
    circuit_records: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "schema_version",
        "source_kind",
        "snapshot_id",
        "display_only",
        "eligible_motor_input",
        "source_receipts",
        "dataset",
        "cell_inventory",
        "circuit_topology",
        "axis_registration",
        "clocks",
        "causal_samples",
        "retinal_input",
        "full_cell_state",
        "motor_boundary",
        "relation_to_frozen_replay",
        "rejected_downstream_fields",
        "claim_scope",
        "content_canonicalization",
        "attachment_sha256",
    )
    attachment = _exact_fields(
        attachment_value, fields, "canonical online circuit replay attachment"
    )
    contract = _registered_online_display_contract()
    expected_receipts = {
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "source_manifest_uri": contract["source_manifest_uri"],
        "asset_receipts": contract["asset_receipts"],
    }
    expected_axis = {
        "cell_axis_sha256": contract["cell_axis_sha256"],
        "cell_ids_sha256": contract["cell_ids_sha256"],
        "retinal_axis_sha256": contract["retinal_axis_sha256"],
        "cell_axis_source": "registered content-addressed fly-FGS circuit bundle",
        "full_state_axis_labels": contract["cell_ids"],
        "retinal_axis_labels": contract["retinal_axis"]["cell_ids"],
    }
    measurement = arrays["circuit_measurement_time_s"].tolist()
    availability = arrays["circuit_availability_time_s"].tolist()
    expected_clocks = {
        "circuit_dt_s": config.circuit_dt_s,
        "sample_count": len(circuit_records),
        "measurement_time_s": measurement,
        "availability_time_s": availability,
        "episode_interval_end_s": config.duration_s,
        "sample_semantics": (
            "sample 0 is the registered static pre-roll gauge; sample i>=1 is the "
            "exact online state after the causal body/scene control; the last sample "
            "is held to the episode boundary"
        ),
    }
    _require(
        attachment.get("schema_version") == ONLINE_CIRCUIT_REPLAY_SCHEMA_VERSION
        and attachment.get("source_kind")
        == "registered_fly_fgs_online_circuit_replay"
        and attachment.get("snapshot_id") == FLY_FGS_SNAPSHOT_ID
        and attachment.get("display_only") is True
        and attachment.get("eligible_motor_input") is False
        and attachment.get("source_receipts") == expected_receipts
        and attachment.get("dataset") == contract["dataset"]
        and attachment.get("cell_inventory") == contract["cell_inventory"]
        and attachment.get("circuit_topology") == contract["circuit_topology"]
        and attachment.get("axis_registration") == expected_axis
        and attachment.get("clocks") == expected_clocks,
        "canonical online circuit attachment source/topology/clock receipt differs",
    )
    retinal = _exact_fields(
        attachment.get("retinal_input"),
        (
            "cell_indices",
            "cell_ids",
            "azimuth_deg",
            "elevation_deg",
            "luminance",
            "unit",
            "sampling_semantics",
            "eligible_motor_input",
        ),
        "canonical online retinal attachment",
    )
    retinal_axis = contract["retinal_axis"]
    _require(
        retinal.get("cell_indices") == retinal_axis["cell_indices"]
        and retinal.get("cell_ids") == retinal_axis["cell_ids"]
        and retinal.get("azimuth_deg") == retinal_axis["azimuth_deg"]
        and retinal.get("elevation_deg") == retinal_axis["elevation_deg"]
        and retinal.get("luminance") == arrays["retinal_input_luminance"].tolist()
        and retinal.get("unit") == "1"
        and retinal.get("sampling_semantics")
        == (
            "online analytic endpoint luminance in exact registered T4a order; "
            "not calibrated compound-eye optics or radiometry"
        )
        and retinal.get("eligible_motor_input") is False,
        "canonical online retinal attachment differs from registered arrays",
    )
    full_state = _exact_fields(
        attachment.get("full_cell_state"),
        ("cell_ids", "voltage_v", "activity", "eligible_motor_input"),
        "canonical online full-cell attachment",
    )
    _require(
        full_state.get("cell_ids") == contract["cell_ids"]
        and full_state.get("voltage_v")
        == arrays["full_circuit_voltage_v"].tolist()
        and full_state.get("activity")
        == arrays["full_circuit_activity"].tolist()
        and full_state.get("eligible_motor_input") is False,
        "canonical online full-cell attachment differs from immutable arrays",
    )
    _require(
        attachment.get("motor_boundary")
        == {
            "eligible_signal": "nod1_voltage_v",
            "eligible_root_ids": list(FLY_FGS_NOD1_ROOT_IDS),
            "retinal_input_eligible": False,
            "pooled_readout_eligible": False,
            "full_cell_state_eligible": False,
            "notice": (
                "This attachment is visualization/audit data. It cannot be consumed by "
                "the motor bridge; only the four exact NOD1 voltage channels cross that boundary."
            ),
        }
        and attachment.get("relation_to_frozen_replay")
        == {
            "is_frozen_registered_capture": False,
            "notice": (
                "These are exact states from this online closed-loop run, not rows from "
                "the separately registered fixed-stimulus circuit_replay capture."
            ),
        }
        and attachment.get("rejected_downstream_fields")
        == contract["rejected_downstream_fields"]
        and attachment.get("claim_scope") == contract["claim_scope"]
        and attachment.get("content_canonicalization")
        == (
            "SHA-256 of UTF-8 Python json.dumps(sort_keys=True,"
            "separators=(',',':'),allow_nan=False,ensure_ascii=True); "
            "attachment_sha256 excluded; browser integrity relies on the enclosing "
            "artifact web-projection receipt because JavaScript number spelling can differ"
        ),
        "canonical online circuit motor boundary/claim contract differs",
    )
    causal = attachment.get("causal_samples")
    _require(
        isinstance(causal, list) and len(causal) == len(circuit_records),
        "canonical online causal-sample inventory is incomplete",
    )
    causal_fields = (
        "sample_index",
        "measurement_time_s",
        "availability_time_s",
        "initialization_mode",
        "requested_control",
        "applied_circuit_control",
        "control_semantics",
        "wrapped_world_yaw_rad",
        "unwrapped_world_yaw_rad",
        "world_z_yaw_rate_rad_s",
        "nod1_voltage_v",
        "pooled_readout",
    )
    zero_control = {
        "heading_rad": 0.0,
        "heading_velocity_rad_s": 0.0,
        "figure_world_azimuth_rad": 0.0,
        "figure_velocity_rad_s": 0.0,
        "ground_velocity_rad_s": 0.0,
    }
    shared = set(causal_fields).difference(
        {"applied_circuit_control", "control_semantics"}
    )
    for index, (sample_value, record) in enumerate(zip(causal, circuit_records)):
        sample = _exact_fields(
            sample_value,
            causal_fields,
            "canonical online causal sample {}".format(index),
        )
        _require(
            all(sample[field] == record[field] for field in shared)
            and sample.get("applied_circuit_control")
            == (zero_control if index == 0 else record["requested_control"])
            and sample.get("control_semantics")
            == (
                "captured static-scene pre-roll gauge at reset"
                if index == 0
                else "body/scene control causally applied for this 5 ms circuit step"
            ),
            "canonical online causal sample differs from authoritative circuit table",
        )
    expected_attachment_sha = _canonical_json_sha256(
        {
            key: value
            for key, value in attachment.items()
            if key != "attachment_sha256"
        }
    )
    _require(
        attachment.get("attachment_sha256") == expected_attachment_sha,
        "canonical online circuit attachment checksum mismatch",
    )


def _audit_canonical_frame_causality(
    frames: Sequence[Mapping[str, Any]],
    config: CanonicalClosedLoopConfig,
    arrays: Mapping[str, np.ndarray],
    tables: Mapping[str, Tuple[Mapping[str, Any], ...]],
) -> None:
    """Bind display frames to the outgoing bridge/mechanics boundary state."""

    rate_axes = {"dn": CANONICAL_DN_AXIS, "mn": CANONICAL_MN_AXIS}
    mechanics_activity = {
        int(record["tick_index"]): tuple(record["active_intervention_ids"])
        for record in tables["intervention_activity"]
        if record["stage"] == "mechanics_force"
    }
    _require(
        set(mechanics_activity) == set(range(config.bridge_interval_count * 5)),
        "canonical mechanics activity is incomplete for frame binding",
    )
    expected_applied: Dict[int, Set[str]] = {}
    expected_suppressed: Dict[int, Set[str]] = {}
    mechanics_records = tuple(
        record
        for record in tables["intervention_activity"]
        if record["stage"] == "mechanics_force"
    )
    for event in tables["motor_events"]:
        disposition = event["mechanics_disposition"]
        if disposition == "pending_at_episode_end":
            continue
        availability = float(event["availability_time_s"])
        matching_ticks = [
            int(record["tick_index"])
            for record in mechanics_records
            if availability >= float(record["interval_start_s"]) - 1.0e-12
            and availability < float(record["interval_end_s"]) - 1.0e-12
        ]
        _require(
            len(matching_ticks) == 1,
            "delivered canonical event does not select one completed physics transition",
        )
        completed_frame_index = matching_ticks[0] + 1
        target = (
            expected_applied if disposition == "applied" else expected_suppressed
        )
        target.setdefault(completed_frame_index, set()).add(event["event_id"])

    observed_applied: Set[str] = set()
    observed_suppressed: Set[str] = set()
    final_physics_index = config.bridge_interval_count * 5
    for frame_index, frame in enumerate(frames):
        bridge_index = min(frame_index // 5, config.bridge_interval_count - 1)
        signals = frame.get("neural_signals")
        _require(
            isinstance(signals, Mapping),
            "canonical frame neural_signals must be an object",
        )
        for stage, axis in rate_axes.items():
            present = arrays["held_{}_rate_present".format(stage)][bridge_index]
            rates = arrays["held_{}_rate_hz".format(stage)][bridge_index]
            expected = {
                "{}:{}".format(stage, channel): float(rates[column])
                for column, channel in enumerate(axis)
                if bool(present[column])
            }
            observed = {
                key: value
                for key, value in signals.items()
                if isinstance(key, str) and key.startswith(stage + ":")
            }
            _require(
                observed == expected,
                "canonical frame uses stale or non-authoritative held {} rates".format(
                    stage.upper()
                ),
            )
            expected_mean = (
                None
                if not expected
                else float(np.mean(np.asarray(tuple(expected.values()), dtype=float)))
            )
            pathway = frame.get("pathway_values")
            normalized = frame.get("circuit")
            pathway_index = 4 if stage == "dn" else 5
            expected_normalized = (
                0.0
                if expected_mean is None
                else float(np.clip(abs(expected_mean) / 200.0, 0.0, 1.0))
            )
            pathway_matches = bool(
                expected_mean is None
                and isinstance(pathway, list)
                and len(pathway) == 7
                and pathway[pathway_index] is None
            )
            if expected_mean is not None and isinstance(pathway, list) and len(pathway) == 7:
                pathway_matches = _close(
                    _finite_number(
                        pathway[pathway_index],
                        "canonical frame pathway held-rate mean",
                    ),
                    expected_mean,
                    atol=1.0e-12,
                )
            normalized_matches = bool(
                isinstance(normalized, list)
                and len(normalized) == 7
                and _close(
                    _finite_number(
                        normalized[pathway_index],
                        "canonical frame normalized held-rate mean",
                    ),
                    expected_normalized,
                    atol=1.0e-12,
                )
            )
            _require(
                isinstance(pathway, list)
                and len(pathway) == 7
                and pathway_matches
                and isinstance(normalized, list)
                and len(normalized) == 7
                and normalized_matches,
                "canonical frame pathway display is detached from held {} rates".format(
                    stage.upper()
                ),
            )

        active = frame.get("active_intervention_ids")
        expected_active = (
            ()
            if frame_index == final_physics_index
            else mechanics_activity[frame_index]
        )
        _require(
            isinstance(active, list)
            and tuple(active) == expected_active,
            "canonical frame active interventions do not govern its outgoing transition",
        )
        applied = frame.get("applied_motor_event_ids")
        suppressed = frame.get("suppressed_motor_event_ids")
        _require(
            isinstance(applied, list)
            and isinstance(suppressed, list)
            and len(applied) == len(set(applied))
            and len(suppressed) == len(set(suppressed))
            and not set(applied).intersection(suppressed)
            and set(applied) == expected_applied.get(frame_index, set())
            and set(suppressed) == expected_suppressed.get(frame_index, set()),
            "canonical frame completed event receipts are on the wrong endpoint",
        )
        observed_applied.update(applied)
        observed_suppressed.update(suppressed)
    _require(
        observed_applied
        == set().union(*expected_applied.values())
        if expected_applied
        else not observed_applied,
        "canonical applied-event frame ledger is incomplete",
    )
    _require(
        observed_suppressed
        == set().union(*expected_suppressed.values())
        if expected_suppressed
        else not observed_suppressed,
        "canonical suppressed-event frame ledger is incomplete",
    )


def _audit_canonical_online_episode_impl(
    episode: Mapping[str, Any],
    run_path: Path,
    run_manifest: Mapping[str, Any],
) -> None:
    """Bind one public online replay exactly to its authoritative run."""

    _require(
        run_manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION
        and run_manifest.get("source_kind") == CANONICAL_ARTIFACT_SOURCE_KIND
        and episode.get("source_kind") == CANONICAL_WEB_SOURCE_KIND,
        "canonical online source kinds may not be aliased or attached to legacy runs",
    )
    _require(
        "circuit_replay" not in episode,
        "canonical online replay may not masquerade as the frozen circuit replay",
    )
    try:
        config = CanonicalClosedLoopConfig.from_dict(
            run_manifest.get("configuration")
        )
    except Exception as exc:
        raise AuditError("canonical online replay configuration failed: {}".format(exc)) from exc
    tables, _identities = _canonical_load_tables(run_path, run_manifest)
    arrays = _read_run_arrays(
        run_path, run_manifest, tuple(sorted(run_manifest["arrays"]))
    )
    scope = _exact_fields(
        episode.get("neural_model_scope"),
        (
            "kind",
            "label",
            "full_circuit_executed",
            "circuit_cell_count",
            "exported_circuit_channel_count",
            "notice",
        ),
        "canonical online neural_model_scope",
    )
    _require(
        scope
        == {
            "kind": "online_registered_fly_fgs_closed_loop",
            "label": "Canonical online fly-FGS fixed-step runtime",
            "full_circuit_executed": True,
            "circuit_cell_count": FLY_FGS_CELL_COUNT,
            "exported_circuit_channel_count": 4,
            "notice": (
                "The full captured circuit executes online; only four exact NOD1 voltage channels cross the motor boundary."
            ),
        },
        "canonical online neural-model scope is not exact",
    )
    trace = _exact_fields(
        episode.get("online_closed_loop"),
        (
            "schema_version",
            "mode",
            "clocks",
            "circuit_samples",
            "rate_samples",
            "motor_events",
            "intervention_activity",
            "event_timing_semantics",
            "frame_event_receipt_semantics",
            "wing_phase_semantics",
            "effector_mapping",
            "anatomical_laterality",
            "signed_behavior_claim_policy",
            "checkpoint_receipt",
            "source_limitations",
        ),
        "canonical online closed-loop trace",
    )
    _require(
        trace.get("schema_version") == "1.0.0"
        and trace.get("mode")
        == "causal_online_fly_fgs_to_streaming_muscle_to_external_physics"
        and trace.get("clocks")
        == {
            "circuit_dt_s": config.circuit_dt_s,
            "bridge_dt_s": config.bridge_dt_s,
            "physics_dt_s": config.physics_dt_s,
            "circuit_sample_count": len(tables["circuit_observations"]),
            "bridge_interval_count": config.bridge_interval_count,
            "physics_transition_count": config.bridge_interval_count * 5,
        }
        and trace.get("circuit_samples")
        == list(tables["circuit_observations"])
        and trace.get("rate_samples") == list(tables["rate_samples"])
        and trace.get("motor_events") == list(tables["motor_events"])
        and trace.get("intervention_activity")
        == list(tables["intervention_activity"])
        and trace.get("event_timing_semantics")
        == "discrete generation and NMJ availability; never interpolate"
        and trace.get("frame_event_receipt_semantics")
        == (
            "applied/suppressed IDs are reported at the completed physics right endpoint; "
            "active intervention IDs govern the outgoing half-open transition"
        )
        and trace.get("wing_phase_semantics")
        == "exported model-owned unwrapped oscillator endpoints; not measured"
        and trace.get("effector_mapping") == run_manifest["effector_mapping"]
        and trace.get("anatomical_laterality") == "unknown"
        and trace.get("signed_behavior_claim_policy")
        == SIGNED_BEHAVIOR_CLAIM_POLICY
        and trace.get("checkpoint_receipt")
        == _canonical_checkpoint_receipt_from_manifest(run_manifest)
        and trace.get("source_limitations") == list(CANONICAL_SOURCE_LIMITATIONS),
        "canonical online trace differs from authoritative tables/receipts",
    )
    _audit_online_circuit_replay_attachment(
        episode.get("online_circuit_replay"),
        config,
        arrays,
        tables["circuit_observations"],
    )
    frames = episode.get("frames")
    expected_time = arrays["physics_time_s"]
    _require(
        isinstance(frames, list) and len(frames) == len(expected_time),
        "canonical online web replay frame inventory is incomplete",
    )
    for index, frame in enumerate(frames):
        _require(
            isinstance(frame, Mapping)
            and frame.get("t") == float(expected_time[index])
            and frame.get("wing_phase_unwrapped_rad")
            == float(arrays["wing_phase_unwrapped_rad"][index])
            and frame.get("root_position_m")
            == arrays["body_position_world_m"][index].tolist()
            and isinstance(frame.get("measured_wing_joint_angle_rad"), list)
            and len(frame["measured_wing_joint_angle_rad"]) == 6
            and isinstance(frame.get("measured_wing_joint_velocity_rad_s"), list)
            and len(frame["measured_wing_joint_velocity_rad_s"]) == 6,
            "canonical online replay frame differs from authoritative clock/root/phase",
        )
    _audit_canonical_frame_causality(frames, config, arrays, tables)


def _audit_canonical_online_episode(
    episode: Mapping[str, Any],
    run_path: Path,
    run_manifest: Mapping[str, Any],
) -> None:
    try:
        _audit_canonical_online_episode_impl(episode, run_path, run_manifest)
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(
            "canonical online replay semantic audit failed: {}".format(exc)
        ) from exc


def _is_canonical_online_declaration(
    episode: Mapping[str, Any], run_manifest: Mapping[str, Any]
) -> bool:
    episode_source = episode.get("source_kind")
    artifact_source = run_manifest.get("source_kind")
    scope = episode.get("neural_model_scope")
    scope_kind = scope.get("kind") if isinstance(scope, Mapping) else None
    markers = (
        episode_source == CANONICAL_WEB_SOURCE_KIND,
        artifact_source == CANONICAL_ARTIFACT_SOURCE_KIND,
        run_manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION,
        "online_circuit_replay" in episode,
        "online_closed_loop" in episode,
        scope_kind == "online_registered_fly_fgs_closed_loop",
        isinstance(episode_source, str) and "canonical_online" in episode_source,
        isinstance(artifact_source, str) and "canonical_online" in artifact_source,
    )
    return any(markers)


def _frozen_browser_required_array_ids(
    run_manifest: Mapping[str, Any],
) -> Tuple[str, ...]:
    descriptors = run_manifest.get("arrays")
    _require(isinstance(descriptors, Mapping), "run arrays must be an object")
    exact = {
        "time_s",
        "circuit_sample_time_s",
        "measured_wing_joint_angle_rad",
        "measured_wing_joint_velocity_rad_s",
        "whole_fly_com_position_world_m",
        "external_actuator_torque_n_m",
        "aerodynamic_force_body_n",
    }
    prefixes = (
        "circuit_voltage/",
        "circuit_availability_time_s/",
        "descending_rate_hz/",
        "wing_motor_rate_hz/",
        "wing_motor_event_availability_time_s/",
        "muscle_activation/",
    )
    exact.update(
        array_id
        for array_id in descriptors
        if isinstance(array_id, str) and array_id.startswith(prefixes)
    )
    return tuple(sorted(exact))


def _canonical_fly_fgs_binding(
    summaries: Sequence[Mapping[str, Any]],
    episode_run_bindings: Sequence[
        Tuple[Mapping[str, Any], Path, Mapping[str, Any]]
    ],
) -> Optional[Tuple[Mapping[str, Any], Path, Mapping[str, Any]]]:
    """Resolve an optional canonical attachment while rejecting aliases/copies."""

    declarations = tuple(
        binding
        for binding in episode_run_bindings
        if binding[0].get("id") == FLY_FGS_EPISODE_ID
        or (
            isinstance(binding[0].get("neural_model_scope"), Mapping)
            and binding[0]["neural_model_scope"].get("kind")
            == "registered_fly_fgs_fixed_step_circuit"
        )
        or (
            isinstance(binding[2].get("pipeline"), Mapping)
            and binding[2]["pipeline"].get("pipeline_id")
            == FLY_FGS_PIPELINE_ID
        )
    )
    if not declarations:
        return None
    canonical_bindings = tuple(
        binding
        for binding in episode_run_bindings
        if binding[0].get("id") == FLY_FGS_EPISODE_ID
    )
    canonical_summaries = tuple(
        summary for summary in summaries if summary.get("id") == FLY_FGS_EPISODE_ID
    )
    _require(
        len(canonical_summaries) == 1
        and len(canonical_bindings) == 1
        and len(declarations) == 1
        and declarations[0] == canonical_bindings[0],
        "a release declaring the canonical fly-FGS pipeline requires exactly "
        "one fly_fgs_canonical episode",
    )
    return canonical_bindings[0]


def audit_release(root: Path) -> ReleaseContext:
    root = Path(root).resolve()
    _require(root.is_dir(), "static release root does not exist: {}".format(root))
    manifest_path = root / "data" / "manifest.json"
    _require(manifest_path.is_file(), "release manifest is missing: {}".format(manifest_path))
    manifest = _load_json(manifest_path, "release manifest")

    summaries = manifest.get("episodes")
    _require(isinstance(summaries, list) and bool(summaries), "release manifest requires episodes")
    episode_ids: Set[str] = set()
    episode_paths: Set[Path] = set()
    episode_identities: Set[Tuple[int, int]] = set()
    run_paths: Set[Path] = set()
    run_identities: Set[Tuple[int, int]] = set()
    binding_paths: Set[Path] = set()
    binding_identities: Set[Tuple[int, int]] = set()
    episodes: List[Mapping[str, Any]] = []
    run_records: Dict[Path, Tuple[Mapping[str, Any], int, int]] = {}
    flybody_run_paths: Set[Path] = set()
    episode_run_bindings: List[
        Tuple[Mapping[str, Any], Path, Mapping[str, Any]]
    ] = []
    canonical_online_bindings: List[
        Tuple[Mapping[str, Any], Path, Mapping[str, Any]]
    ] = []
    for index, summary_value in enumerate(summaries):
        label = "episodes[{}]".format(index)
        _require(isinstance(summary_value, Mapping), "{} must be an object".format(label))
        summary = summary_value
        summary_id = summary.get("id")
        _require(isinstance(summary_id, str) and bool(summary_id), "{}.id is required".format(label))
        _require(summary_id not in episode_ids, "duplicate episode id {!r}".format(summary_id))
        episode_ids.add(summary_id)
        episode_path = _safe_local_path(root, summary.get("data_url"), "{}.data_url".format(label))
        _require(episode_path not in episode_paths, "episode JSON is referenced more than once")
        episode_paths.add(episode_path)
        episode_stat = episode_path.stat()
        episode_identity = (episode_stat.st_dev, episode_stat.st_ino)
        _require(
            episode_identity not in episode_identities,
            "episode JSON inode is referenced more than once",
        )
        episode_identities.add(episode_identity)
        _require(
            episode_path not in binding_paths
            and episode_identity not in binding_identities,
            "public episode binding path/inode is referenced more than once",
        )
        binding_paths.add(episode_path)
        binding_identities.add(episode_identity)
        episode = _load_json(episode_path, "episode {}".format(summary_id))
        _require(episode.get("id") == summary_id, "episode JSON id does not match manifest id {!r}".format(summary_id))
        episode_duration = episode.get("duration_s")
        _require(
            isinstance(episode_duration, (int, float))
            and not isinstance(episode_duration, bool)
            and math.isfinite(float(episode_duration))
            and float(episode_duration) > 0.0,
            "episode duration_s must be finite and positive",
        )
        _require(
            isinstance(episode.get("frames"), list)
            and len(episode["frames"]) >= 2,
            "episode requires at least two replay frames",
        )
        episodes.append(episode)

        projection_path = _safe_local_path(
            root,
            summary.get("artifact_projection_url"),
            "{}.artifact_projection_url".format(label),
        )
        projection_stat = projection_path.stat()
        projection_identity = (projection_stat.st_dev, projection_stat.st_ino)
        _require(
            projection_path not in binding_paths
            and projection_identity not in binding_identities,
            "public artifact projection path/inode is referenced more than once",
        )
        binding_paths.add(projection_path)
        binding_identities.add(projection_identity)

        run_path = _safe_local_path(
            root,
            summary.get("artifact_manifest_url"),
            "{}.artifact_manifest_url".format(label),
        )
        run_stat = run_path.stat()
        run_identity = (run_stat.st_dev, run_stat.st_ino)
        _require(
            run_path not in run_paths and run_identity not in run_identities,
            "run manifest path/inode is referenced more than once",
        )
        run_paths.add(run_path)
        run_identities.add(run_identity)
        _require(
            run_path not in binding_paths and run_identity not in binding_identities,
            "public artifact manifest binding path/inode is referenced more than once",
        )
        binding_paths.add(run_path)
        binding_identities.add(run_identity)
        run_records[run_path] = _audit_run_manifest(run_path)
        run_manifest = run_records[run_path][0]
        _audit_episode_content_binding(
            summary,
            episode_path,
            episode,
            run_path,
            run_manifest,
            projection_path,
        )
        source_run_id = episode.get("source_run_id")
        _require(
            isinstance(source_run_id, str) and bool(source_run_id),
            "episode source_run_id is required",
        )
        _require(source_run_id == run_manifest.get("run_id"), "episode source_run_id does not match run manifest")
        source_manifest_sha = episode.get("source_artifact_manifest_sha256")
        _require_sha256(source_manifest_sha, "episode source_artifact_manifest_sha256")
        _require(
            source_manifest_sha == _sha256_file(run_path),
            "episode source artifact checksum does not match its run manifest",
        )
        _require(
            episode.get("source_artifact_schema_version")
            == run_manifest.get("schema_version"),
            "episode source artifact schema does not match its run manifest",
        )
        _audit_web_replay_projection(episode, run_manifest)
        runtime = run_manifest.get("runtime")
        _require(isinstance(runtime, Mapping), "run manifest runtime must be an object")
        episode_backend = episode.get("physics_backend")
        run_backend = runtime.get("physics_backend")
        _require(
            episode_backend == run_backend,
            "episode physics_backend does not match its run manifest",
        )
        episode_run_bindings.append((episode, run_path, run_manifest))
        if _is_canonical_online_declaration(episode, run_manifest):
            canonical_online_bindings.append((episode, run_path, run_manifest))
        if run_backend == "flybody":
            flybody_run_paths.add(run_path)

    report, registry, attached_registry_bytes = _audit_validation_attachment(
        root, manifest
    )
    _audit_publication_gate_status(report)
    _audit_current_source_binding(report, attached_registry_bytes)

    for online_episode, online_run_path, online_run_manifest in canonical_online_bindings:
        _audit_canonical_online_episode(
            online_episode, online_run_path, online_run_manifest
        )
    if canonical_online_bindings:
        _audit_required_validation_pass(
            report, registry, CANONICAL_ONLINE_VALIDATION_CASE_ID
        )

    canonical_binding = _canonical_fly_fgs_binding(
        summaries, episode_run_bindings
    )
    canonical_attached = canonical_binding is not None
    _audit_fly_fgs_validation_inventory(
        report,
        registry,
        canonical_attached=canonical_attached,
    )

    frozen_bindings = tuple(
        binding
        for binding in episode_run_bindings
        if binding[0].get("id") == FROZEN_BROWSER_EPISODE_ID
    )
    _require(
        len(frozen_bindings) <= 1,
        "frozen_browser_nod1 may appear at most once as a historical replay",
    )
    for episode, run_path, run_manifest in episode_run_bindings:
        if run_manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION:
            continue
        inspect_wingbeat = (
            episode.get("source_kind") in WINGBEAT_INSPECTION_SOURCE_KINDS
            or "wingbeat_inspection" in episode
        )
        if episode.get("physics_backend") != "flybody":
            _require(
                not inspect_wingbeat,
                "wingbeat_inspection source kinds and payloads require FlyBody",
            )
            continue
        required_array_ids = [
            "time_s",
            "ground_contact_count",
            "physics_time_s",
            "ground_contact_transition_point_count",
        ]
        if inspect_wingbeat:
            required_array_ids.extend(
                (
                    "measured_wing_joint_angle_physics_rad",
                    "external_actuator_torque_physics_n_m",
                )
            )
        contact_arrays = _read_run_arrays(
            run_path,
            run_manifest,
            required_array_ids,
        )
        _audit_flybody_ground_contact_disclosure(
            episode,
            run_manifest,
            contact_arrays,
        )
        _audit_flybody_wingbeat_inspection(
            episode,
            run_manifest,
            contact_arrays,
        )
    if canonical_binding is not None:
        canonical_episode, canonical_run_path, canonical_run_manifest = (
            canonical_binding
        )
        canonical_arrays = _read_run_arrays(
            canonical_run_path,
            canonical_run_manifest,
            _fly_fgs_required_array_ids(canonical_run_manifest),
        )
        _audit_fly_fgs_canonical_episode(
            canonical_episode,
            canonical_run_manifest,
            registry,
            canonical_arrays,
        )
    if frozen_bindings:
        frozen_episode, frozen_run_path, frozen_run_manifest = frozen_bindings[0]
        frozen_arrays = _read_run_arrays(
            frozen_run_path,
            frozen_run_manifest,
            _frozen_browser_required_array_ids(frozen_run_manifest),
        )
        _audit_frozen_browser_nod1_episode(
            frozen_episode,
            frozen_run_manifest,
            registry,
            frozen_arrays,
        )
        _audit_required_validation_pass(
            report,
            registry,
            FROZEN_BROWSER_FLYBODY_CASE_ID,
        )

    worker_image_sha = _audit_worker_image_binding(
        report,
        tuple(run_records[path][0] for path in sorted(flybody_run_paths)),
    )
    _audit_worker_dependency_lock_binding(
        report,
        tuple(run_records[path][0] for path in sorted(flybody_run_paths)),
    )

    provenance = manifest.get("provenance")
    _require(isinstance(provenance, list), "release manifest provenance must be an array")
    provenance_reference_count = 0
    provenance_hash_count = 0
    worker_image_provenance_count = 0
    provenance_labels: Set[str] = set()
    all_run_manifests = tuple(
        run_records[path][0] for path in sorted(run_records)
    )
    for index, entry_value in enumerate(provenance):
        label = "provenance[{}]".format(index)
        _require(isinstance(entry_value, Mapping), "{} must be an object".format(label))
        entry = entry_value
        entry_label = entry.get("label")
        _require(
            isinstance(entry_label, str) and bool(entry_label),
            "{}.label must be a non-empty string".format(label),
        )
        _require(
            entry_label not in provenance_labels,
            "duplicate provenance label {!r}".format(entry_label),
        )
        provenance_labels.add(entry_label)
        url = entry.get("url")
        declared_sha = _declared_provenance_sha256(entry, label)
        if url is None:
            if _audit_logical_executable_provenance(
                entry_label, declared_sha, all_run_manifests
            ):
                provenance_hash_count += 1
                continue
            if entry.get("label") == "Worker image":
                _require(
                    worker_image_sha is not None,
                    "Worker image provenance has no matching report source digest",
                )
                _require(
                    declared_sha == worker_image_sha,
                    "Worker image provenance does not match the report source digest",
                )
                worker_image_provenance_count += 1
                provenance_hash_count += 1
                continue
            _require(
                declared_sha is None,
                "{} declares a file checksum without a local URL".format(label),
            )
            continue
        referenced_path = _safe_local_path(root, url, "{}.url".format(label))
        provenance_reference_count += 1
        _require(
            declared_sha is not None,
            "{} references a local provenance file without a full SHA-256".format(label),
        )
        _require(
            _sha256_file(referenced_path) == declared_sha,
            "{} declared checksum does not match {}".format(label, referenced_path),
        )
        provenance_hash_count += 1

    if worker_image_sha is not None:
        _require(
            worker_image_provenance_count == 1,
            "release provenance must contain exactly one bound Worker image entry",
        )

    current_reference_sources = {
        "Evidence graph": REPOSITORY_ROOT / "data" / "evidence" / "seed_graph.v1.json",
        "Model registry": REPOSITORY_ROOT / "data" / "models" / "flight_model_registry.json",
        "BANC/FANC evidence": (
            REPOSITORY_ROOT
            / "data"
            / "reference"
            / "banc_fanc_wing_pathway_evidence.v1.json"
        ),
    }
    by_label = {
        entry.get("label"): entry
        for entry in provenance
        if isinstance(entry.get("label"), str)
    }
    for label, source in current_reference_sources.items():
        _require(source.is_file(), "source provenance file is unavailable: {}".format(source))
        entry = by_label.get(label)
        _require(isinstance(entry, Mapping), "release provenance is missing {}".format(label))
        local = _safe_local_path(root, entry.get("url"), "provenance {}.url".format(label))
        _require(
            local.read_bytes() == source.read_bytes(),
            "release provenance {} is stale relative to the source tree".format(label),
        )
    current_registry = REPOSITORY_ROOT / "data" / "benchmarks" / "registry.v1.json"
    for source_uri, source, protocol_sha256, _payload in verified_preregistered_protocol_files(
        registry, current_registry
    ):
        label = "Evaluation protocol · {}".format(source.stem)
        entry = by_label.get(label)
        _require(
            isinstance(entry, Mapping),
            "release provenance is missing {}".format(label),
        )
        _require(
            entry.get("sha256") == protocol_sha256,
            "release provenance {} has the wrong protocol digest".format(label),
        )
        local = _safe_local_path(
            root, entry.get("url"), "provenance {}.url".format(label)
        )
        _require(
            local.read_bytes() == source.read_bytes(),
            "release provenance {} is stale relative to the source tree".format(label),
        )
        _require(
            source_uri in str(entry.get("value", "")),
            "release provenance {} omits its registered source URI".format(label),
        )

    return ReleaseContext(
        root=root,
        manifest=manifest,
        episodes=tuple(episodes),
        report=report,
        registry=registry,
        run_manifest_count=len(run_records),
        array_count=sum(record[1] for record in run_records.values()),
        chunk_count=sum(record[2] for record in run_records.values()),
        provenance_reference_count=provenance_reference_count,
        provenance_hash_count=provenance_hash_count,
    )


class _QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format_string: str, *args: Any) -> None:
        del format_string, args


def _preferred_episode_index(episodes: Sequence[Mapping[str, Any]]) -> int:
    """Mirror the browser's baseline/initial selector contract exactly."""

    _require(bool(episodes), "release requires at least one episode")
    # main.ts reserves entry zero as the comparison baseline and initially
    # selects entry one when present.  The canonical publisher deliberately
    # places the online baseline first and its paired intervention second.
    return 1 if len(episodes) > 1 else 0


def _browser_audit(context: ReleaseContext, screenshot: Optional[Path]) -> Mapping[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise AuditError("--browser requires the Python playwright package") from exc

    summaries = context.manifest["episodes"]
    initial_index = _preferred_episode_index(context.episodes)
    initial_summary = summaries[initial_index]
    initial_episode = context.episodes[initial_index]
    baseline_episode = context.episodes[0]
    initially_comparable = bool(
        initial_index != 0
        and (initial_episode.get("physics_backend") or "reduced_order")
        == (baseline_episode.get("physics_backend") or "reduced_order")
        and (initial_episode.get("body_state_reference") or "reduced_order_body_state")
        == (baseline_episode.get("body_state_reference") or "reduced_order_body_state")
    )
    _require(initial_episode.get("physics_backend") == "flybody", "initial browser episode must use FlyBody")
    _require(context.report.suite_complete, "browser release requires a complete validation suite")

    expected_neural = len(initial_episode.get("neural_trace_channels", ()))
    expected_muscles = len(initial_episode.get("individual_muscle_channels", ()))
    duration_s = initial_episode.get("duration_s")
    _require(
        isinstance(duration_s, (int, float))
        and not isinstance(duration_s, bool)
        and math.isfinite(float(duration_s))
        and float(duration_s) > 0.0,
        "initial episode duration is invalid",
    )
    if screenshot is not None:
        screenshot = screenshot.resolve()
        try:
            screenshot.relative_to(context.root)
        except ValueError:
            pass
        else:
            raise AuditError("browser screenshot must be outside the audited release root")
    expected_window_ms = int(round(min(0.03, float(duration_s)) * 1000.0))
    status_counts = {
        status.value: sum(1 for result in context.report.results if result.status is status)
        for status in GateStatus
    }

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        functools.partial(_QuietStaticHandler, directory=str(context.root)),
    )
    server_thread = threading.Thread(target=server.serve_forever, name="release-audit-http")
    server_thread.daemon = True
    server_thread.start()
    errors: List[str] = []
    request_failures: List[str] = []
    http_errors: List[str] = []
    mobile_overflow_px = 0
    focusable_control_count = 0
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
            )
            try:
                browser_context = browser.new_context(viewport={"width": 1440, "height": 1000})
                browser_context.add_init_script(
                    "Object.defineProperty(Crypto.prototype, 'subtle', "
                    "{ configurable: true, get: () => undefined });"
                )
                page = browser_context.new_page()
                # Each episode switch loads and verifies a content-addressed
                # manifest plus its replay projections before clearing
                # aria-busy.  The nine-episode scientific release can exceed
                # 15 seconds on the shared AWS host even when every request and
                # UI transition is healthy.  Keep Playwright's actionability
                # checks, but give the authoritative transition one bounded
                # 30-second window rather than forcing DOM events.
                page.set_default_timeout(30000)
                page.on(
                    "console",
                    lambda message: errors.append("console:{}:{}".format(message.type, message.text))
                    if message.type == "error"
                    else None,
                )
                page.on("pageerror", lambda error: errors.append("pageerror:{}".format(error)))
                page.on(
                    "requestfailed",
                    lambda request: request_failures.append(
                        "{}: {}".format(request.url, request.failure)
                    ),
                )
                page.on(
                    "response",
                    lambda response: http_errors.append(
                        "{}: HTTP {}".format(response.url, response.status)
                    )
                    if response.url.startswith("http://127.0.0.1:")
                    and response.status >= 400
                    else None,
                )
                url = "http://127.0.0.1:{}/".format(server.server_address[1])
                response = page.goto(url, wait_until="domcontentloaded", timeout=15000)
                _require(response is not None and response.ok, "browser release root did not return HTTP 2xx")
                page.locator("#episode-select").wait_for(state="visible", timeout=15000)
                page.locator(".validation-summary").wait_for(state="visible", timeout=15000)
                _require(
                    page.locator("#episode-title").inner_text() == initial_summary.get("label"),
                    "browser did not load the manifest's initial FlyBody episode",
                )
                option_rows = page.locator("#episode-select option").evaluate_all(
                    "options => options.map(option => [option.value, option.textContent.trim()])"
                )
                _require(
                    option_rows
                    == [[item["id"], item["label"]] for item in summaries],
                    "browser episode options do not match the release manifest",
                )
                validation_text = page.locator(".validation-summary").inner_text()
                _require(context.report.evaluation_id in validation_text, "browser did not render the bound evaluation id")
                _require("complete suite" in validation_text, "browser did not render complete-suite status")
                for status in GateStatus:
                    observed = page.locator(".validation-case .gate-{}".format(status.value)).count()
                    _require(
                        observed == status_counts[status.value],
                        "browser {} result count {} does not match report {}".format(
                            status.value, observed, status_counts[status.value]
                        ),
                    )
                _require(
                    page.locator("#wing-window-label").inner_text()
                    == "{} ms WINDOW".format(expected_window_ms),
                    "browser wing window does not match episode duration",
                )
                if initially_comparable:
                    _require(
                        not page.locator("#compare-toggle").is_disabled()
                        and page.locator("#compare-toggle").is_checked()
                        and page.locator("#compare-label").inner_text()
                        == "Overlay baseline",
                        "compatible paired replay comparison is not enabled",
                    )
                else:
                    _require(
                        page.locator("#compare-toggle").is_disabled(),
                        "incompatible replay comparison is enabled",
                    )
                    _require(
                        page.locator("#compare-label").inner_text()
                        == "Matched replay unavailable",
                        "browser compare label does not disclose the missing matched replay",
                    )
                observed_neural = page.locator("[data-neural-channel-id]").count()
                observed_muscles = page.locator("[data-individual-muscle-index]").count()
                _require(observed_neural == expected_neural, "browser neural channel count mismatch")
                _require(observed_muscles == expected_muscles, "browser muscle channel count mismatch")
                _require(
                    bool(page.evaluate("globalThis.crypto.subtle === undefined")),
                    "browser SHA-256 fallback was not forced",
                )
                # Exercise every lazy episode URL and verify that the latest
                # selected condition, rendered title, trace counts, and run
                # download remain mutually consistent.
                for summary, episode in zip(summaries, context.episodes):
                    page.select_option("#episode-select", str(summary["id"]))
                    page.wait_for_function(
                        "expected => {"
                        " const bar = document.querySelector('#experiment-bar');"
                        " const select = document.querySelector('#episode-select');"
                        " const title = document.querySelector('#episode-title');"
                        " return bar?.getAttribute('aria-busy') === 'false'"
                        "   && select?.value === expected.id"
                        "   && title?.textContent === expected.label;"
                        "}",
                        arg={"id": summary["id"], "label": summary["label"]},
                    )
                    _require(
                        page.locator("#episode-load-error").is_hidden(),
                        "browser reported an episode load error for {}".format(summary["id"]),
                    )
                    _require(
                        page.locator("[data-neural-channel-id]").count()
                        == len(episode.get("neural_trace_channels", ())),
                        "browser neural rows mismatch episode {}".format(summary["id"]),
                    )
                    _require(
                        page.locator("[data-individual-muscle-index]").count()
                        == len(episode.get("individual_muscle_channels", ())),
                        "browser muscle rows mismatch episode {}".format(summary["id"]),
                    )
                    _require(
                        page.locator("#download-link").get_attribute("href").endswith(
                            str(summary["artifact_manifest_url"])
                        ),
                        "browser run download is not bound to episode {}".format(summary["id"]),
                    )
                    _require(
                        page.locator("#current-time").inner_text() == "0.000 s"
                        and page.locator("#playback-state").inner_text() == "PAUSED",
                        "episode selection did not reset replay time/state",
                    )
                    if summary["id"] == summaries[0]["id"]:
                        _require(
                            page.locator("#compare-toggle").is_disabled()
                            and page.locator("#compare-label").inner_text()
                            == "Baseline selected",
                            "baseline self-comparison is not disabled",
                        )

                # Return to the intended authoritative episode for transport,
                # window, provenance, and download checks.
                page.select_option("#episode-select", str(initial_summary["id"]))
                page.wait_for_function(
                    "expected => document.querySelector('#experiment-bar')?.getAttribute('aria-busy') === 'false'"
                    " && document.querySelector('#episode-title')?.textContent === expected",
                    arg=initial_summary["label"],
                )
                page.click("#reset-button")
                page.click("#step-button")
                page.wait_for_function(
                    "document.querySelector('#current-time')?.textContent === '0.001 s'"
                )
                page.select_option("#speed-select", "0.1")
                page.click("#play-button")
                page.wait_for_function(
                    "document.querySelector('#play-button')?.getAttribute('aria-label') === 'Pause replay'"
                )
                page.wait_for_timeout(40)
                _require(
                    float(page.locator("#timeline").input_value()) > 0.001,
                    "play control did not advance replay time",
                )
                page.click("#play-button")
                page.click("#reset-button")
                midpoint = float(duration_s) / 2.0
                page.locator("#timeline").evaluate(
                    "(element, value) => { element.value = String(value);"
                    " element.dispatchEvent(new Event('input', {bubbles: true})); }",
                    midpoint,
                )
                page.wait_for_function(
                    "expected => document.querySelector('#current-time')?.textContent === expected",
                    arg="{:.3f} s".format(midpoint),
                )
                window_length = min(0.03, float(duration_s))
                window_start = max(
                    0.0,
                    min(float(duration_s) - window_length, midpoint - window_length / 2.0),
                )
                _require(
                    page.locator("#window-start").inner_text()
                    == "{} ms".format(round(window_start * 1000.0))
                    and page.locator("#window-end").inner_text()
                    == "{} ms".format(round((window_start + window_length) * 1000.0)),
                    "wingbeat inspection window did not follow the timeline",
                )

                page.click("#provenance-toggle")
                _require(
                    page.locator("#provenance-toggle").get_attribute("aria-expanded")
                    == "true"
                    and page.locator("#provenance-list").is_visible(),
                    "provenance disclosure did not open accessibly",
                )
                _require(
                    page.locator("#provenance-list .provenance-row").count()
                    == len(context.manifest.get("provenance", ())),
                    "browser provenance rows do not match the manifest",
                )
                for entry in context.manifest.get("provenance", ()):
                    if entry.get("url") is None:
                        continue
                    provenance_response = browser_context.request.get(
                        new_url := "http://127.0.0.1:{}/{}".format(
                            server.server_address[1], entry["url"]
                        )
                    )
                    _require(
                        provenance_response.ok,
                        "browser provenance URL returned non-2xx: {}".format(new_url),
                    )
                    _require(
                        hashlib.sha256(provenance_response.body()).hexdigest()
                        == entry["sha256"],
                        "browser provenance bytes failed SHA-256: {}".format(new_url),
                    )

                # Collapse the disclosure again before testing the sticky
                # download dock.  Besides covering both disclosure states,
                # this prevents scroll anchoring from moving the link between
                # Playwright's actionability frames on the shared host.
                page.click("#provenance-toggle")
                _require(
                    page.locator("#provenance-toggle").get_attribute("aria-expanded")
                    == "false"
                    and page.locator("#provenance-list").is_hidden(),
                    "provenance disclosure did not close accessibly",
                )
                download_link = page.locator("#download-link")
                download_link.scroll_into_view_if_needed()
                with page.expect_download(timeout=30000) as download_info:
                    download_link.click()
                downloaded_path = download_info.value.path()
                _require(downloaded_path is not None, "run manifest download has no local path")
                _require(
                    _sha256_file(Path(downloaded_path))
                    == initial_episode["source_artifact_manifest_sha256"],
                    "downloaded run manifest checksum does not match the episode",
                )

                # Validation cases live inside collapsed native <details>
                # disclosures.  innerText is deliberately empty for those
                # hidden descendants; textContent still proves that every
                # report case was materialized in the disclosure DOM.
                rendered_case_ids = [
                    value.strip()
                    for value in page.locator(
                        ".validation-case > div > strong"
                    ).all_text_contents()
                ]
                rendered_case_id_set = set(rendered_case_ids)
                expected_case_id_set = {
                    result.case_id for result in context.report.results
                }
                _require(
                    rendered_case_id_set == expected_case_id_set,
                    "browser validation case inventory does not match the report; "
                    f"missing={sorted(expected_case_id_set - rendered_case_id_set)!r}; "
                    f"unexpected={sorted(rendered_case_id_set - expected_case_id_set)!r}",
                )

                if screenshot is not None:
                    screenshot.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(
                        path=str(screenshot),
                        full_page=False,
                        animations="disabled",
                        caret="hide",
                        # WebGL/canvas framebuffer readback on the shared AWS
                        # host can be slower than ordinary DOM actionability.
                        # Keep capture bounded without weakening any UI gate.
                        timeout=60000,
                    )

                primary_controls = (
                    "#about-button",
                    "#episode-select",
                    "#reset-button",
                    "#play-button",
                    "#step-button",
                    "#timeline",
                    "#speed-select",
                    "#download-link",
                )
                for selector in primary_controls:
                    locator = page.locator(selector)
                    _require(locator.count() == 1 and locator.is_visible(), "primary control is unavailable: {}".format(selector))
                    accessible_name = locator.evaluate(
                        "element => {"
                        " const aria = (element.getAttribute('aria-label') || '').trim();"
                        " const labelled = element.labels && element.labels.length"
                        "   ? (element.labels[0].innerText || '').trim() : '';"
                        " return aria || labelled || (element.innerText || '').trim();"
                        "}"
                    )
                    _require(bool(accessible_name), "primary control has no accessible label: {}".format(selector))
                    locator.focus(timeout=15000)
                    _require(
                        bool(locator.evaluate("element => document.activeElement === element")),
                        "primary control cannot receive keyboard focus: {}".format(selector),
                    )
                    focusable_control_count += 1

                page.set_viewport_size({"width": 390, "height": 844})
                core_mobile_selectors = (
                    "#about-button",
                    "#episode-select",
                    "#reset-button",
                    "#step-button",
                    "#download-link",
                    "#flight-scene",
                    ".pathway-panel",
                    ".neural-trace-panel",
                    ".validation-panel",
                )
                for selector in core_mobile_selectors:
                    locator = page.locator(selector)
                    _require(locator.count() == 1 and locator.is_visible(), "mobile core section is unavailable: {}".format(selector))
                    locator.evaluate(
                        "element => element.scrollIntoView({block: 'center', inline: 'nearest'})"
                    )
                mobile_overflow_px = int(
                    page.evaluate(
                        "Math.max(0, document.documentElement.scrollWidth "
                        "- document.documentElement.clientWidth)"
                    )
                )
                _require(
                    mobile_overflow_px <= 1,
                    "mobile document has {} px horizontal overflow".format(mobile_overflow_px),
                )
                _require(not errors, "browser emitted errors: {}".format(errors))
                _require(
                    not request_failures,
                    "browser emitted request failures: {}".format(request_failures),
                )
                _require(
                    not http_errors,
                    "browser received non-2xx local responses: {}".format(http_errors),
                )
            finally:
                browser.close()
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError("browser audit failed: {}".format(exc)) from exc
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)

    return {
        "console_page_errors": len(errors),
        "evaluation_id": context.report.evaluation_id,
        "initial_episode_id": initial_episode.get("id"),
        "muscle_channels": expected_muscles,
        "mobile_horizontal_overflow_px": mobile_overflow_px,
        "mobile_viewport_width_px": 390,
        "neural_channels": expected_neural,
        "primary_focusable_controls": focusable_control_count,
        "request_failures": len(request_failures),
        "http_errors": len(http_errors),
        "episodes_exercised": len(summaries),
        "sha256_fallback": True,
        "window_ms": expected_window_ms,
    }


def _check_optional_expected_counts(context: ReleaseContext, args: argparse.Namespace) -> None:
    initial_index = _preferred_episode_index(context.episodes)
    initial = context.episodes[initial_index]
    actual = {
        "episodes": len(context.episodes),
        "neural_channels": len(initial.get("neural_trace_channels", ())),
        "muscle_channels": len(initial.get("individual_muscle_channels", ())),
        "pass": sum(1 for result in context.report.results if result.status is GateStatus.PASS),
        "blocked": sum(1 for result in context.report.results if result.status is GateStatus.BLOCKED),
        "fail": sum(1 for result in context.report.results if result.status is GateStatus.FAIL),
    }
    for key in actual:
        expected = getattr(args, "expected_{}".format(key))
        if expected is not None:
            _require(actual[key] == expected, "expected {}={}, observed {}".format(key, expected, actual[key]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=REPOSITORY_ROOT / "web" / "dist",
        help="built/static release root (default: web/dist)",
    )
    parser.add_argument("--browser", action="store_true", help="also run the plain-HTTP Playwright smoke")
    parser.add_argument("--screenshot", type=Path, help="optional browser screenshot output path")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-neural-channels", type=int)
    parser.add_argument("--expected-muscle-channels", type=int)
    parser.add_argument("--expected-pass", type=int)
    parser.add_argument("--expected-blocked", type=int)
    parser.add_argument("--expected-fail", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _require(args.browser or args.screenshot is None, "--screenshot requires --browser")
        context = audit_release(args.root)
        _check_optional_expected_counts(context, args)
        statuses = {
            status.value: sum(1 for result in context.report.results if result.status is status)
            for status in GateStatus
        }
        summary: Dict[str, Any] = {
            "arrays": context.array_count,
            "chunks": context.chunk_count,
            "episodes": len(context.episodes),
            "ok": True,
            "provenance_hashes": context.provenance_hash_count,
            "provenance_references": context.provenance_reference_count,
            "registry_version": context.registry.version,
            "report": {
                "evaluation_id": context.report.evaluation_id,
                "statuses": statuses,
                "suite_complete": context.report.suite_complete,
            },
            "root": str(context.root),
            "run_manifests": context.run_manifest_count,
        }
        if args.browser:
            summary["browser"] = _browser_audit(context, args.screenshot)
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0
    except (AuditError, OSError, UnicodeError) as exc:
        print(
            json.dumps({"error": str(exc), "ok": False}, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
