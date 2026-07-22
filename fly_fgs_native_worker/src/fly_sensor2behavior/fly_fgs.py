"""Strict, fixed-step intake for the upstream ``/fly-fgs`` circuit.

Only the content-addressed ``fgmodel.js`` engine and ``bundle_fg.json`` circuit
bundle are executable inputs.  The captured HTML and wing-DN JSON are retained
as receipts, but the importer cannot ingest their DN, motor, muscle, wing, yaw,
or toy-body equations.  The public :class:`CircuitOutputTrace` contains only
the four exact NOD1 voltage channels.  Rich, bounded circuit-display state is
available separately through :meth:`RegisteredFlyFGSFixture.circuit_replay_attachment`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sysconfig
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .schema import (
    AnatomicalSide,
    CircuitOutputTrace,
    CircuitSignal,
    CircuitSignalKind,
    Confidence,
    ConfidenceLevel,
    DatasetNamespace,
    DatasetRef,
    EntityKind,
    EntityRef,
    EvidenceTier,
    EyeSide,
    Provenance,
    RetinalFrame,
    RetinalSignalKind,
    SideContext,
    SideMappingMethod,
    SignalOrigin,
)


FLY_FGS_SCHEMA_VERSION = "1.0.0"
FLY_FGS_FIXTURE_KIND = "fly_fgs_fixed_step_circuit_capture"
FLY_FGS_SNAPSHOT_ID = "fly-fgs-live-2026-07-18"
FLY_FGS_SOURCE_MANIFEST_SHA256 = (
    "fef59e4b8abe5216081b6d64decd0d1775454d06e7631bb9af8ae409c1cd05c3"
)
FLY_FGS_REGISTERED_CAPTURE_SHA256 = (
    "5612cc9c2218a917bf2ad939d6aaa2a8402db5f11fd2de36cd3168d9198cbd83"
)
FLY_FGS_SOURCE_MANIFEST_URI = "data/reference/fly_fgs/source_manifest.v1.json"
FLY_FGS_REGISTERED_CAPTURE_URI = (
    "data/reference/fly_fgs/fixed_step_capture.v1.json.gz"
)
FLY_FGS_NOD1_ROOT_IDS: Tuple[str, ...] = (
    "720575940628438427",
    "720575940625528556",
    "720575940623997949",
    "720575940629456860",
)
FLY_FGS_NOD1_RAW_APP_SIDES: Mapping[str, str] = {
    "720575940628438427": "L",
    "720575940625528556": "L",
    "720575940623997949": "R",
    "720575940629456860": "R",
}
FLY_FGS_NOD1_APP_RENDERING_SIDES: Mapping[str, AnatomicalSide] = {
    root_id: AnatomicalSide.LEFT if raw_side == "L" else AnatomicalSide.RIGHT
    for root_id, raw_side in FLY_FGS_NOD1_RAW_APP_SIDES.items()
}
FLY_FGS_POOLED_TRACE_IDS: Tuple[str, ...] = (
    "dch_activity",
    "dch_voltage_v",
    "llpc1_activity",
    "nod1_activity",
    "nod1_voltage_v",
    "t4_activity",
    "vch_activity",
    "vch_voltage_v",
)
_EXPECTED_MANIFEST_FIELDS = {
    "schema_version",
    "snapshot_id",
    "source",
    "dataset",
    "assets",
    "circuit_inventory",
    "execution_contract",
    "rejected_downstream_fields",
    "claim_scope",
}
_EXPECTED_CAPTURE_FIELDS = {
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
_EXPECTED_ASSET_FIELDS = {
    "asset_id",
    "path",
    "sha256",
    "bytes",
    "media_type",
    "role",
    "eligible_circuit_input",
}
_EXPECTED_CELL_FIELDS = {
    "id",
    "type",
    "side",
    "nt",
    "precision",
    "retinotopic",
    "retinoAzimuth",
    "retinoElevation",
    "preferredDirection",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), "%s must be an object" % label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(isinstance(value, list), "%s must be an array" % label)
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    _require(set(value) == set(expected), "%s fields do not match contract" % label)


def _finite(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), "%s must be numeric" % label)
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % label) from exc
    _require(math.isfinite(converted), "%s must be finite" % label)
    return converted


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        "%s must be an integer" % label,
    )
    return int(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_object(raw: bytes, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError("%s contains forbidden non-finite JSON constant %s" % (label, value))

    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("%s contains duplicate JSON key %r" % (label, key))
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("%s is not strict UTF-8 JSON" % label) from exc
    _require(isinstance(value, Mapping), "%s top level must be an object" % label)
    return value


def default_fly_fgs_source_manifest_path() -> Path:
    source_tree = Path(__file__).resolve().parents[2] / FLY_FGS_SOURCE_MANIFEST_URI
    if source_tree.is_file():
        return source_tree
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / "reference"
        / "fly_fgs"
        / "source_manifest.v1.json"
    )


def default_fly_fgs_registered_capture_path() -> Path:
    source_tree = Path(__file__).resolve().parents[2] / FLY_FGS_REGISTERED_CAPTURE_URI
    if source_tree.is_file():
        return source_tree
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / "reference"
        / "fly_fgs"
        / "fixed_step_capture.v1.json.gz"
    )


def _read_regular_file(path: Path, label: str) -> bytes:
    resolved = Path(path)
    _require(resolved.exists(), "%s is unavailable" % label)
    _require(resolved.is_file() and not resolved.is_symlink(), "%s must be a regular non-symlink file" % label)
    return resolved.read_bytes()


@dataclass(frozen=True)
class FlyFGSAssetReceipt:
    asset_id: str
    relative_path: str
    sha256: str
    bytes: int
    media_type: str
    role: str
    eligible_circuit_input: bool

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "media_type": self.media_type,
            "role": self.role,
            "eligible_circuit_input": self.eligible_circuit_input,
        }


@dataclass(frozen=True)
class RegisteredFlyFGSFixture:
    """Validated upstream circuit trace and bounded display attachment."""

    circuit_trace: CircuitOutputTrace
    source_manifest_sha256: str
    capture_sha256: str
    source_manifest_bytes: int
    capture_bytes: int
    source_manifest_uri: str
    capture_uri: str
    snapshot_id: str
    asset_receipts: Tuple[FlyFGSAssetReceipt, ...]
    fixed_step: Mapping[str, Any]
    stimulus: Mapping[str, Any]
    pooled_readout_traces: Mapping[str, Any]
    circuit_inventory: Mapping[str, Any]
    rejected_downstream_fields: Tuple[str, ...]
    retinal_frames: Tuple[RetinalFrame, ...]
    full_cell_state: Optional[Mapping[str, Any]]
    circuit_topology: Mapping[str, Any]
    dataset_metadata: Mapping[str, Any]
    claim_scope: Mapping[str, Any]

    @property
    def dt_s(self) -> float:
        return float(self.fixed_step["dt_s"])

    @property
    def duration_s(self) -> float:
        return float(self.fixed_step["duration_s"])

    @property
    def circuit_replay(self) -> Mapping[str, Any]:
        return self.circuit_replay_attachment()

    def pipeline_source_metadata(self) -> Mapping[str, Any]:
        metadata = {
            "input_mode": "registered_fly_fgs_fixed_step_circuit",
            "visual_source": "fly_fgs_analytic_retinal_scene_and_circuit_bundle",
            "retinal_frames_present": True,
            "retinal_frame_count": len(self.retinal_frames),
            "retinal_display_present": True,
            "source_manifest_uri": self.source_manifest_uri,
            "source_manifest_sha256": self.source_manifest_sha256,
            "capture_uri": self.capture_uri,
            "capture_sha256": self.capture_sha256,
            "source_snapshot_id": self.snapshot_id,
            "source_duration_s": self.duration_s,
            "source_sample_dt_s": self.dt_s,
            "source_sample_count": len(self.circuit_trace.sample_times_s),
            "source_sample_interval": self.fixed_step["sample_interval"],
            "source_sample_interval_end_s": self.duration_s,
            "pre_roll_steps": self.fixed_step["pre_roll_steps"],
            "pre_roll_duration_s": self.fixed_step["pre_roll_duration_s"],
            "pre_roll_mode": self.fixed_step["pre_roll_mode"],
            "source_availability_semantics": (
                "offline_fixed_step_at_sample_time; no streaming availability or "
                "physiological latency measurement is claimed"
            ),
            "circuit_model_scope": "fly_fgs_1684_cell_fixed_step_circuit_four_motor_bound_nod1_readouts",
            "circuit_cell_count": int(self.circuit_inventory["cell_count"]),
            "exported_motor_bound_circuit_channels": len(
                self.circuit_trace.signals
            ),
            "circuit_replay_full_cell_state_present": self.full_cell_state
            is not None,
            "dataset": deepcopy(dict(self.dataset_metadata)),
            "stimulus": deepcopy(dict(self.stimulus["configuration"])),
            "rejected_downstream_fields": list(
                self.rejected_downstream_fields
            ),
            "scientific_confidence": "low_exploratory",
            "retinal_boundary_notice": (
                "Frames 1..99 expose exact endpoint lumNow samples at all 1,441 "
                "retinotopic T4a circuit coordinates. Sample 0 is intentionally omitted "
                "because its source interval lies in pre-roll. These are uncalibrated "
                "analytic scene samples, not compound-eye optics or biological ommatidia."
            ),
        }
        json.dumps(metadata, allow_nan=False, sort_keys=True)
        return metadata

    def circuit_replay_attachment(self) -> Mapping[str, Any]:
        """Return a JSON-safe, bounded attachment for synchronized circuit UI.

        The full-cell matrices are exactly 100 samples by 1,684 cells in the
        registered fixture.  They are display data only and never become motor
        inputs; the motor-bound trace remains the four NOD1 voltage channels.
        """

        assets = {
            receipt.asset_id: deepcopy(dict(receipt.to_dict()))
            for receipt in self.asset_receipts
        }
        assets["source_manifest"] = {
            "asset_id": "source_manifest",
            "relative_path": self.source_manifest_uri,
            "sha256": self.source_manifest_sha256,
            "bytes": self.source_manifest_bytes,
            "media_type": "application/json",
            "role": "strict_source_contract",
            "eligible_circuit_input": False,
        }
        assets["registered_capture"] = {
            "asset_id": "registered_capture",
            "relative_path": self.capture_uri,
            "sha256": self.capture_sha256,
            "bytes": self.capture_bytes,
            "media_type": "application/gzip",
            "role": "registered_fixed_step_capture",
            "eligible_circuit_input": False,
        }
        attachment: Dict[str, Any] = {
            "schema_version": FLY_FGS_SCHEMA_VERSION,
            "source_kind": "registered_fly_fgs_fixed_step_circuit_replay",
            "snapshot_id": self.snapshot_id,
            "asset_receipts": assets,
            "fixed_step": deepcopy(dict(self.fixed_step)),
            "stimulus": deepcopy(dict(self.stimulus)),
            "pooled_readout_traces": deepcopy(dict(self.pooled_readout_traces)),
            "cell_inventory": deepcopy(dict(self.circuit_inventory)),
            "circuit_topology": deepcopy(dict(self.circuit_topology)),
            "motor_bound_nod1_root_ids": list(FLY_FGS_NOD1_ROOT_IDS),
            "side_semantics": {
                "raw_bundle_labels": deepcopy(dict(FLY_FGS_NOD1_RAW_APP_SIDES)),
                "anatomical_side": "unknown",
                "mapping_method": "simulation_convention",
                "notice": (
                    "Raw bundle L/R is retained only as an application rendering "
                    "label. No soma-x evidence is present, so it is not anatomical laterality."
                ),
            },
            "rejected_downstream_fields": list(self.rejected_downstream_fields),
            "full_cell_state_scope": {
                "included": self.full_cell_state is not None,
                "sample_axis": "fixed_step.time_s",
                "cell_axis": "full_cell_state.cell_ids in exact circuit-bundle order",
                "eligible_motor_input": False,
                "purpose": "T4a/LLPC1 and cable-neuron circuit visualization only",
            },
            "retinal_frame_scope": {
                "count": len(self.retinal_frames),
                "source_sample_indices": list(
                    range(1, len(self.circuit_trace.sample_times_s))
                ),
                "sample_0_excluded": True,
                "sample_0_exclusion_reason": (
                    "source exposure lies in pre-roll and cannot fit a nonnegative "
                    "RetinalFrame exposure interval"
                ),
                "coordinate_axis": "stimulus.retinal_input exact T4a circuit order",
                "direction_frame": (
                    "+x ahead, +y positive azimuth, +z positive elevation; spherical "
                    "unit vectors derived from exact T4a azimuth and elevation"
                ),
                "semantics": (
                    "analytic endpoint luminance; not calibrated compound-eye optics "
                    "or biological ommatidia"
                ),
            },
            "full_cell_state": (
                None
                if self.full_cell_state is None
                else deepcopy(dict(self.full_cell_state))
            ),
            "claim_scope": deepcopy(dict(self.claim_scope)),
        }
        # This is both a safety assertion and a stable public contract: callers
        # never receive enum objects, paths, NaN, or other non-JSON values.
        json.dumps(attachment, allow_nan=False, sort_keys=True)
        return attachment


def _load_and_validate_source_manifest(
    manifest_path: Path,
) -> Tuple[
    Mapping[str, Any],
    Tuple[FlyFGSAssetReceipt, ...],
    Mapping[str, Mapping[str, Any]],
    int,
]:
    manifest_bytes = _read_regular_file(manifest_path, "fly-FGS source manifest")
    _require(
        _sha256_bytes(manifest_bytes) == FLY_FGS_SOURCE_MANIFEST_SHA256,
        "fly-FGS source manifest SHA-256 mismatch",
    )
    manifest = _strict_json_object(manifest_bytes, "fly-FGS source manifest")
    _exact_fields(manifest, _EXPECTED_MANIFEST_FIELDS, "fly-FGS source manifest")
    _require(manifest.get("schema_version") == FLY_FGS_SCHEMA_VERSION, "unsupported fly-FGS source-manifest schema")
    _require(manifest.get("snapshot_id") == FLY_FGS_SNAPSHOT_ID, "unexpected fly-FGS snapshot ID")
    dataset = _mapping(manifest.get("dataset"), "source manifest dataset")
    _require(dataset.get("family") == "FlyWire FAFB", "fly-FGS dataset must be FlyWire FAFB")
    _require(dataset.get("materialization") == 783, "fly-FGS dataset must use FAFB materialization 783")
    _require(dataset.get("root_id_encoding") == "decimal strings", "fly-FGS root IDs must remain strings")
    _require(dataset.get("bundle_side_semantics") == "raw application compatibility label only", "fly-FGS side semantics mismatch")
    _require(dataset.get("structural_counts_are_physiological_weights") is False, "structural counts cannot be physiological weights")

    assets_raw = _sequence(manifest.get("assets"), "source manifest assets")
    manifest_directory = manifest_path.resolve().parent
    receipts = []
    assets_by_id: Dict[str, Mapping[str, Any]] = {}
    for index, raw_entry in enumerate(assets_raw):
        entry = _mapping(raw_entry, "source manifest asset %d" % index)
        _exact_fields(entry, _EXPECTED_ASSET_FIELDS, "source manifest asset %d" % index)
        asset_id = entry.get("asset_id")
        _require(isinstance(asset_id, str) and asset_id and asset_id not in assets_by_id, "source asset IDs must be unique")
        relative_path = entry.get("path")
        _require(isinstance(relative_path, str) and relative_path, "source asset path is required")
        pure_parts = Path(relative_path).parts
        _require(not Path(relative_path).is_absolute() and ".." not in pure_parts, "source asset path is unsafe")
        asset_path = manifest_directory / relative_path
        asset_bytes = _read_regular_file(asset_path, "fly-FGS asset %s" % asset_id)
        expected_bytes = _integer(entry.get("bytes"), "source asset bytes")
        expected_sha256 = entry.get("sha256")
        _require(len(asset_bytes) == expected_bytes, "fly-FGS asset %s byte count mismatch" % asset_id)
        _require(isinstance(expected_sha256, str) and _sha256_bytes(asset_bytes) == expected_sha256, "fly-FGS asset %s SHA-256 mismatch" % asset_id)
        assets_by_id[asset_id] = {"entry": entry, "bytes": asset_bytes, "path": asset_path}
        receipts.append(
            FlyFGSAssetReceipt(
                asset_id=asset_id,
                relative_path=relative_path,
                sha256=str(expected_sha256),
                bytes=expected_bytes,
                media_type=str(entry.get("media_type")),
                role=str(entry.get("role")),
                eligible_circuit_input=bool(entry.get("eligible_circuit_input")),
            )
        )
    _require(
        set(assets_by_id)
        == {"page_snapshot", "circuit_engine", "circuit_bundle", "wing_dns_excluded"},
        "fly-FGS source asset inventory mismatch",
    )
    eligible = {
        asset_id
        for asset_id, record in assets_by_id.items()
        if record["entry"].get("eligible_circuit_input") is True
    }
    _require(eligible == {"circuit_engine", "circuit_bundle"}, "only engine and circuit bundle may be circuit inputs")
    return manifest, tuple(receipts), assets_by_id, len(manifest_bytes)


def _validate_bundle_axis(
    bundle_bytes: bytes, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    bundle = _strict_json_object(bundle_bytes, "fly-FGS circuit bundle")
    _exact_fields(
        bundle,
        ("cableIndices", "cables", "cells", "dt", "events", "ids", "meta", "modelParams"),
        "fly-FGS circuit bundle",
    )
    inventory = _mapping(manifest.get("circuit_inventory"), "source circuit inventory")
    expected_count = _integer(inventory.get("cell_count"), "source cell count")
    ids = tuple(str(value) for value in _sequence(bundle.get("ids"), "bundle ids"))
    raw_cells = _sequence(bundle.get("cells"), "bundle cells")
    cells = tuple(
        _mapping(cell, "bundle cell %d" % index)
        for index, cell in enumerate(raw_cells)
    )
    _require(len(ids) == expected_count and len(cells) == expected_count, "fly-FGS bundle cell axis mismatch")
    for index, cell in enumerate(cells):
        _exact_fields(cell, _EXPECTED_CELL_FIELDS, "bundle cell %d" % index)
        _require(cell.get("id") == ids[index], "fly-FGS bundle cell IDs are not aligned")
    _require(all(value.startswith("r") and value[1:].isdigit() for value in ids), "fly-FGS bundle IDs must be r-prefixed decimal roots")
    observed_id_digest = _sha256_bytes(
        json.dumps(list(ids), separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    _require(
        observed_id_digest == inventory.get("cell_ids_canonical_sha256"),
        "fly-FGS bundle cell-ID digest mismatch",
    )
    type_counts: Dict[str, int] = {}
    for cell in cells:
        cell_type = str(cell.get("type"))
        type_counts[cell_type] = type_counts.get(cell_type, 0) + 1
    _require(type_counts == dict(inventory.get("cell_type_counts", {})), "fly-FGS bundle cell-type inventory mismatch")
    _require(type_counts.get("T5", 0) == 0, "registered fly-FGS bundle unexpectedly contains T5")

    bundle_dt = _finite(bundle.get("dt"), "bundle dt")
    _require(bundle_dt == manifest.get("execution_contract", {}).get("dt_s"), "fly-FGS bundle timestep mismatch")
    model_parameters = _mapping(bundle.get("modelParams"), "bundle model parameters")
    _require(
        model_parameters == inventory.get("model_parameters"),
        "fly-FGS bundle model-parameter mapping mismatch",
    )
    python_parameter_digest = _sha256_bytes(
        json.dumps(
            model_parameters,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    _require(
        python_parameter_digest
        == inventory.get("model_parameters_python_sorted_json_sha256"),
        "fly-FGS bundle Python-sorted model-parameter digest mismatch",
    )

    meta = _mapping(bundle.get("meta"), "bundle meta")
    raw_event_count = _integer(
        inventory.get("raw_event_count"), "source raw event count"
    )
    _require(
        raw_event_count == inventory.get("event_count") == 61789,
        "fly-FGS raw-event inventory mismatch",
    )
    _require(meta.get("nCells") == expected_count, "fly-FGS bundle meta cell count mismatch")
    _require(meta.get("nEvents") == raw_event_count, "fly-FGS bundle meta raw-event count mismatch")
    cable_indices = tuple(
        _integer(value, "bundle cable index")
        for value in _sequence(bundle.get("cableIndices"), "bundle cable indices")
    )
    _require(cable_indices == tuple(inventory.get("cable_indices", ())), "fly-FGS bundle cable-index inventory mismatch")
    _require(meta.get("nCables") == inventory.get("cable_count") == len(cable_indices), "fly-FGS bundle cable count mismatch")

    events = _mapping(bundle.get("events"), "bundle events")
    _exact_fields(events, ("gunit", "post", "postc", "pre", "prec", "sign"), "bundle events")
    for name, vector in events.items():
        _require(
            len(_sequence(vector, "bundle event vector %s" % name))
            == raw_event_count,
            "fly-FGS bundle event vector %s length mismatch" % name,
        )
    pre = _sequence(events.get("pre"), "bundle event pre")
    post = _sequence(events.get("post"), "bundle event post")
    llpc_indices = {
        index for index, cell in enumerate(cells) if "LLPC1" in str(cell.get("type", "")).upper()
    }
    removed_count = sum(
        1
        for pre_index, post_index in zip(pre, post)
        if pre_index in llpc_indices and post_index in llpc_indices
    )
    _require(
        removed_count
        == inventory.get("removed_llpc1_to_llpc1_event_count")
        == 1589,
        "fly-FGS removed LLPC1-to-LLPC1 event count mismatch",
    )
    _require(
        raw_event_count - removed_count
        == inventory.get("effective_runtime_event_count")
        == 60200,
        "fly-FGS effective runtime-event count mismatch",
    )

    retinotopic_t4a = tuple(
        (index, cell)
        for index, cell in enumerate(cells)
        if cell.get("type") == "T4a"
        and cell.get("retinotopic") is True
        and isinstance(cell.get("retinoAzimuth"), (int, float))
        and not isinstance(cell.get("retinoAzimuth"), bool)
        and math.isfinite(float(cell.get("retinoAzimuth")))
        and isinstance(cell.get("retinoElevation"), (int, float))
        and not isinstance(cell.get("retinoElevation"), bool)
        and math.isfinite(float(cell.get("retinoElevation")))
    )
    _require(
        len(retinotopic_t4a)
        == inventory.get("retinotopic_t4a_input_count")
        == 1441,
        "fly-FGS retinotopic T4a input-grid count mismatch",
    )
    return {
        "cell_ids": ids,
        "cells": tuple(deepcopy(dict(cell)) for cell in cells),
        "cable_indices": cable_indices,
        "retinotopic_t4a_indices": tuple(index for index, _ in retinotopic_t4a),
        "retinotopic_t4a_cell_ids": tuple(
            str(cell["id"]) for _, cell in retinotopic_t4a
        ),
        "retinotopic_t4a_azimuth_deg": tuple(
            float(cell["retinoAzimuth"]) for _, cell in retinotopic_t4a
        ),
        "retinotopic_t4a_elevation_deg": tuple(
            float(cell["retinoElevation"]) for _, cell in retinotopic_t4a
        ),
    }


def _finite_series(value: Any, count: int, label: str) -> Tuple[float, ...]:
    raw = _sequence(value, label)
    _require(len(raw) == count, "%s must contain %d samples" % (label, count))
    return tuple(_finite(item, "%s[%d]" % (label, index)) for index, item in enumerate(raw))


def _validate_capture_payload(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    asset_receipts: Tuple[FlyFGSAssetReceipt, ...],
    bundle_topology: Mapping[str, Any],
    capture_sha256: str,
    source_manifest_bytes: int,
    capture_bytes: int,
    capture_uri: str,
) -> RegisteredFlyFGSFixture:
    _exact_fields(payload, _EXPECTED_CAPTURE_FIELDS, "fly-FGS capture")
    _require(payload.get("schema_version") == FLY_FGS_SCHEMA_VERSION, "unsupported fly-FGS capture schema")
    _require(payload.get("fixture_kind") == FLY_FGS_FIXTURE_KIND, "unexpected fly-FGS fixture kind")
    _require(payload.get("snapshot_id") == FLY_FGS_SNAPSHOT_ID, "fly-FGS capture snapshot mismatch")

    receipts_by_id = {receipt.asset_id: receipt for receipt in asset_receipts}
    bundle_cell_ids = tuple(bundle_topology["cell_ids"])
    bundle_cells = tuple(bundle_topology["cells"])
    source_receipt = _mapping(payload.get("source_receipt"), "capture source receipt")
    _exact_fields(
        source_receipt,
        ("source_manifest_sha256", "circuit_engine_sha256", "circuit_bundle_sha256"),
        "capture source receipt",
    )
    _require(source_receipt.get("source_manifest_sha256") == FLY_FGS_SOURCE_MANIFEST_SHA256, "capture source-manifest receipt mismatch")
    _require(source_receipt.get("circuit_engine_sha256") == receipts_by_id["circuit_engine"].sha256, "capture engine receipt mismatch")
    _require(source_receipt.get("circuit_bundle_sha256") == receipts_by_id["circuit_bundle"].sha256, "capture bundle receipt mismatch")

    execution = _mapping(payload.get("execution"), "capture execution")
    _exact_fields(
        execution,
        (
            "dt_s",
            "duration_s",
            "invariant_observations",
            "pre_roll_duration_s",
            "pre_roll_mode",
            "pre_roll_steps",
            "sample_count",
            "sample_interval",
            "sample_semantics",
            "scheduler",
            "time_s",
        ),
        "capture execution",
    )
    contract = _mapping(manifest.get("execution_contract"), "source execution contract")
    dt_s = _finite(execution.get("dt_s"), "capture dt_s")
    duration_s = _finite(execution.get("duration_s"), "capture duration_s")
    sample_count = _integer(execution.get("sample_count"), "capture sample_count")
    _require(dt_s == contract.get("dt_s") == 0.005, "fly-FGS capture timestep mismatch")
    _require(duration_s == contract.get("duration_s") == 0.5, "fly-FGS capture duration mismatch")
    _require(sample_count == contract.get("sample_count") == 100, "fly-FGS capture sample count mismatch")
    _require(execution.get("scheduler") == contract.get("scheduler"), "fly-FGS capture scheduler mismatch")
    _require(execution.get("sample_semantics") == contract.get("sample_semantics"), "fly-FGS capture sample semantics mismatch")
    pre_roll_steps = _integer(execution.get("pre_roll_steps"), "capture pre_roll_steps")
    pre_roll_duration_s = _finite(execution.get("pre_roll_duration_s"), "capture pre_roll_duration_s")
    _require(pre_roll_steps == contract.get("pre_roll_steps") == 48, "fly-FGS pre-roll step count mismatch")
    _require(pre_roll_duration_s == contract.get("pre_roll_duration_s") == 0.24, "fly-FGS pre-roll duration mismatch")
    _require(math.isclose(pre_roll_steps * dt_s, pre_roll_duration_s, rel_tol=0.0, abs_tol=1e-15), "fly-FGS pre-roll clock mismatch")
    _require(execution.get("pre_roll_mode") == contract.get("pre_roll_mode"), "fly-FGS pre-roll mode mismatch")
    raw_times = _finite_series(execution.get("time_s"), sample_count, "capture time_s")
    for index, observed in enumerate(raw_times):
        _require(math.isclose(observed, index * dt_s, rel_tol=0.0, abs_tol=1e-15), "fly-FGS time grid mismatch")
    _require(math.isclose(raw_times[-1] + dt_s, duration_s, rel_tol=0.0, abs_tol=1e-15), "fly-FGS time grid is not half-open")
    invariant_observations = _mapping(
        execution.get("invariant_observations"),
        "capture invariant observations",
    )
    _exact_fields(
        invariant_observations,
        ("t4_activity_excursion", "nod1_voltage_excursion_v"),
        "capture invariant observations",
    )
    recorded_t4_excursion = _finite(
        invariant_observations.get("t4_activity_excursion"),
        "recorded T4 activity excursion",
    )
    recorded_nod1_excursion = _finite(
        invariant_observations.get("nod1_voltage_excursion_v"),
        "recorded NOD1 voltage excursion",
    )

    inventory = _mapping(payload.get("circuit_inventory"), "capture circuit inventory")
    _require(inventory == manifest.get("circuit_inventory"), "capture circuit inventory differs from source manifest")
    _require(tuple(inventory.get("nod1_root_ids", ())) == FLY_FGS_NOD1_ROOT_IDS, "capture NOD1 root inventory mismatch")
    _require(dict(inventory.get("nod1_raw_app_sides", {})) == dict(FLY_FGS_NOD1_RAW_APP_SIDES), "capture NOD1 app-side inventory mismatch")
    _require(
        inventory.get("raw_event_count") == inventory.get("event_count") == 61789,
        "capture raw-event count mismatch",
    )
    _require(
        inventory.get("removed_llpc1_to_llpc1_event_count") == 1589,
        "capture removed-event count mismatch",
    )
    _require(
        inventory.get("effective_runtime_event_count") == 60200,
        "capture effective runtime-event count mismatch",
    )

    rejected = tuple(_sequence(payload.get("rejected_downstream_fields"), "rejected downstream fields"))
    _require(rejected == tuple(manifest.get("rejected_downstream_fields", ())), "capture downstream rejection boundary mismatch")
    _require(len(rejected) == len(set(rejected)), "rejected downstream fields must be unique")

    nod1_voltage = _mapping(payload.get("nod1_voltage_v"), "capture NOD1 voltages")
    _require(set(nod1_voltage) == set(FLY_FGS_NOD1_ROOT_IDS), "capture must contain exactly four NOD1 roots")
    nod1_values = {
        root_id: _finite_series(
            nod1_voltage.get(root_id), sample_count, "NOD1 voltage %s" % root_id
        )
        for root_id in FLY_FGS_NOD1_ROOT_IDS
    }
    _require(
        all(-1.0 < value < 1.0 for values in nod1_values.values() for value in values),
        "capture NOD1 voltage is not in SI volts",
    )

    pooled = _mapping(payload.get("pooled_readout_traces"), "capture pooled traces")
    _require(set(pooled) == set(FLY_FGS_POOLED_TRACE_IDS), "capture pooled trace inventory mismatch")
    validated_pooled: Dict[str, Any] = {}
    for trace_id in FLY_FGS_POOLED_TRACE_IDS:
        sides = _mapping(pooled.get(trace_id), "pooled trace %s" % trace_id)
        _exact_fields(sides, ("raw_app_L", "raw_app_R"), "pooled trace %s" % trace_id)
        validated_sides = {
            side: list(_finite_series(sides.get(side), sample_count, "%s.%s" % (trace_id, side)))
            for side in ("raw_app_L", "raw_app_R")
        }
        if trace_id.endswith("_activity"):
            _require(all(0.0 <= item <= 1.0 for values in validated_sides.values() for item in values), "%s activity must be in [0, 1]" % trace_id)
        else:
            _require(all(-1.0 < item < 1.0 for values in validated_sides.values() for item in values), "%s voltage must be SI volts" % trace_id)
        validated_pooled[trace_id] = validated_sides

    t4_values = tuple(
        validated_pooled["t4_activity"][side][index]
        for side in ("raw_app_L", "raw_app_R")
        for index in range(sample_count)
    )
    nod1_flat_values = tuple(
        nod1_values[root_id][index]
        for root_id in FLY_FGS_NOD1_ROOT_IDS
        for index in range(sample_count)
    )
    recomputed_t4_excursion = max(t4_values) - min(t4_values)
    recomputed_nod1_excursion = max(nod1_flat_values) - min(nod1_flat_values)
    _require(
        recomputed_t4_excursion > 1e-12
        and math.isclose(
            recorded_t4_excursion,
            recomputed_t4_excursion,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "fly-FGS T4 activity trace is dead or its invariant is inconsistent",
    )
    _require(
        recomputed_nod1_excursion > 1e-12
        and math.isclose(
            recorded_nod1_excursion,
            recomputed_nod1_excursion,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "fly-FGS NOD1 voltage trace is dead or its invariant is inconsistent",
    )

    stimulus = _mapping(payload.get("stimulus"), "capture stimulus")
    _exact_fields(
        stimulus,
        ("configuration", "schedule", "retinal_display", "retinal_input"),
        "capture stimulus",
    )
    _require(stimulus.get("configuration") == contract.get("options"), "capture stimulus configuration mismatch")
    schedule = _mapping(stimulus.get("schedule"), "capture stimulus schedule")
    _exact_fields(
        schedule,
        (
            "figure_world_azimuth_deg",
            "figure_velocity_deg_s",
            "figure_retinal_azimuth_deg",
            "heading_deg",
            "heading_velocity_deg_s",
            "grating_velocity_deg_s",
            "grating_world_displacement_deg",
        ),
        "capture stimulus schedule",
    )
    for name, values in schedule.items():
        raw_values = _sequence(values, "stimulus schedule %s" % name)
        _require(len(raw_values) == sample_count, "stimulus schedule %s length mismatch" % name)
        for index, value in enumerate(raw_values):
            _require(value is None or math.isfinite(float(value)), "stimulus schedule %s[%d] is invalid" % (name, index))
    retinal = _mapping(stimulus.get("retinal_display"), "capture retinal display")
    _exact_fields(retinal, ("azimuth_deg", "luminance", "unit"), "capture retinal display")
    azimuth = _finite_series(retinal.get("azimuth_deg"), 180, "retinal azimuth")
    _require(all(azimuth[index] > azimuth[index - 1] for index in range(1, len(azimuth))), "retinal azimuth axis must increase")
    luminance = _sequence(retinal.get("luminance"), "retinal luminance")
    _require(len(luminance) == sample_count, "retinal luminance time axis mismatch")
    for index, row in enumerate(luminance):
        values = _finite_series(row, len(azimuth), "retinal luminance[%d]" % index)
        _require(all(0.0 <= item <= 1.0 for item in values), "retinal luminance must be normalized")
    _require(retinal.get("unit") == "normalized luminance (1)", "retinal luminance unit mismatch")

    retinal_input = _mapping(
        stimulus.get("retinal_input"), "capture exact retinal input"
    )
    _exact_fields(
        retinal_input,
        (
            "cell_indices",
            "cell_ids",
            "azimuth_deg",
            "elevation_deg",
            "luminance",
            "unit",
            "sampling_semantics",
        ),
        "capture exact retinal input",
    )
    retinal_input_count = _integer(
        inventory.get("retinotopic_t4a_input_count"),
        "retinotopic T4a input count",
    )
    input_indices = tuple(
        _integer(item, "retinal input cell index")
        for item in _sequence(
            retinal_input.get("cell_indices"), "retinal input cell indices"
        )
    )
    input_cell_ids = tuple(
        str(item)
        for item in _sequence(retinal_input.get("cell_ids"), "retinal input cell IDs")
    )
    input_azimuth_deg = _finite_series(
        retinal_input.get("azimuth_deg"),
        retinal_input_count,
        "retinal input azimuth",
    )
    input_elevation_deg = _finite_series(
        retinal_input.get("elevation_deg"),
        retinal_input_count,
        "retinal input elevation",
    )
    _require(
        input_indices == tuple(bundle_topology["retinotopic_t4a_indices"]),
        "retinal input indices differ from the exact circuit-bundle axis",
    )
    _require(
        input_cell_ids == tuple(bundle_topology["retinotopic_t4a_cell_ids"]),
        "retinal input cell IDs differ from the exact circuit-bundle axis",
    )
    _require(
        input_azimuth_deg
        == tuple(bundle_topology["retinotopic_t4a_azimuth_deg"])
        and input_elevation_deg
        == tuple(bundle_topology["retinotopic_t4a_elevation_deg"]),
        "retinal input coordinates differ from the exact circuit bundle",
    )
    _require(retinal_input.get("unit") == "1", "exact retinal input unit mismatch")
    expected_sampling_semantics = (
        "exact endpoint lumNow value at each retinotopic T4a circuit coordinate; "
        "analytic scene input, not calibrated compound-eye optics or biological ommatidia"
    )
    _require(
        retinal_input.get("sampling_semantics") == expected_sampling_semantics,
        "exact retinal input sampling semantics mismatch",
    )
    retinal_input_rows_raw = _sequence(
        retinal_input.get("luminance"), "exact retinal input luminance"
    )
    _require(
        len(retinal_input_rows_raw) == sample_count,
        "exact retinal input time axis mismatch",
    )
    retinal_input_rows = tuple(
        _finite_series(
            row,
            retinal_input_count,
            "exact retinal input luminance[%d]" % index,
        )
        for index, row in enumerate(retinal_input_rows_raw)
    )
    _require(
        all(
            0.0 <= item <= 1.0
            for row in retinal_input_rows
            for item in row
        ),
        "exact retinal input luminance must be normalized",
    )
    configuration = _mapping(stimulus.get("configuration"), "stimulus configuration")
    _require(
        configuration.get("figure_enabled") is True
        and configuration.get("grating_enabled") is False,
        "registered exact retinal-input validator expects the canonical figure-only scene",
    )
    figure_width_deg = _finite(
        configuration.get("figure_width_deg"), "figure width"
    )
    figure_height_deg = _finite(
        configuration.get("figure_height_deg"), "figure height"
    )
    figure_centres = schedule.get("figure_retinal_azimuth_deg")
    for time_index, (centre, row) in enumerate(
        zip(figure_centres, retinal_input_rows)
    ):
        _require(centre is not None, "canonical figure centre cannot be absent")
        centre_deg = float(centre)
        for cell_index, observed in enumerate(row):
            angular_distance = abs(
                ((input_azimuth_deg[cell_index] - centre_deg + 180.0) % 360.0)
                - 180.0
            )
            expected = (
                1.0
                if angular_distance <= 0.5 * figure_width_deg
                and abs(input_elevation_deg[cell_index])
                <= 0.5 * figure_height_deg
                else 0.0
            )
            _require(
                observed == expected,
                "exact retinal input disagrees with canonical analytic scene at sample %d cell %d"
                % (time_index, cell_index),
            )

    raw_full_state = payload.get("full_cell_state")
    _require(raw_full_state is not None, "registered fly-FGS fixture requires full_cell_state")
    full_state = _mapping(raw_full_state, "capture full_cell_state")
    _exact_fields(full_state, ("cell_ids", "voltage_v", "activity"), "capture full_cell_state")
    cell_ids = tuple(_sequence(full_state.get("cell_ids"), "full-state cell_ids"))
    _require(cell_ids == bundle_cell_ids, "full-cell state axis differs from exact circuit-bundle order")
    cell_count = len(bundle_cell_ids)
    voltage_rows = _sequence(full_state.get("voltage_v"), "full-state voltage_v")
    activity_rows = _sequence(full_state.get("activity"), "full-state activity")
    _require(len(voltage_rows) == sample_count and len(activity_rows) == sample_count, "full-cell time axis mismatch")
    validated_voltage_rows = tuple(
        _finite_series(row, cell_count, "full-state voltage_v[%d]" % index)
        for index, row in enumerate(voltage_rows)
    )
    validated_activity_rows = tuple(
        _finite_series(row, cell_count, "full-state activity[%d]" % index)
        for index, row in enumerate(activity_rows)
    )
    for voltages, activities in zip(
        validated_voltage_rows, validated_activity_rows
    ):
        _require(all(-1.0 < item < 1.0 for item in voltages), "full-cell voltage must be SI volts")
        _require(all(0.0 <= item <= 1.0 for item in activities), "full-cell activity must be in [0, 1]")

    cell_id_to_index = {
        cell_id: index for index, cell_id in enumerate(bundle_cell_ids)
    }
    for root_id in FLY_FGS_NOD1_ROOT_IDS:
        state_index = cell_id_to_index["r%s" % root_id]
        for sample_index, expected in enumerate(nod1_values[root_id]):
            _require(
                validated_voltage_rows[sample_index][state_index] == expected,
                "NOD1 motor-bound voltage differs from full-cell state",
            )

    def group_indices(
        cell_type: str, side: Optional[str] = None
    ) -> Tuple[int, ...]:
        return tuple(
            index
            for index, cell in enumerate(bundle_cells)
            if cell.get("type") == cell_type
            and (side is None or cell.get("side") == side)
        )

    def mean_series(
        rows: Sequence[Sequence[float]], indices: Sequence[int]
    ) -> Tuple[float, ...]:
        _require(bool(indices), "pooled consistency group cannot be empty")
        return tuple(
            sum(row[index] for index in indices) / len(indices) for row in rows
        )

    def require_pooled_match(
        trace_id: str,
        side: str,
        rows: Sequence[Sequence[float]],
        indices: Sequence[int],
    ) -> None:
        expected_series = mean_series(rows, indices)
        observed_series = validated_pooled[trace_id][side]
        _require(
            all(
                math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-14)
                for observed, expected in zip(observed_series, expected_series)
            ),
            "%s.%s differs from the full-cell state" % (trace_id, side),
        )

    for trace_prefix, cell_type in (
        ("nod1", "Nod1"),
        ("vch", "VCH"),
        ("dch", "DCH"),
    ):
        for side_label, cell_side in (("raw_app_L", "L"), ("raw_app_R", "R")):
            indices = group_indices(cell_type, cell_side)
            require_pooled_match(
                "%s_activity" % trace_prefix,
                side_label,
                validated_activity_rows,
                indices,
            )
            require_pooled_match(
                "%s_voltage_v" % trace_prefix,
                side_label,
                validated_voltage_rows,
                indices,
            )
    for side_label, cell_side in (("raw_app_L", "L"), ("raw_app_R", "R")):
        require_pooled_match(
            "llpc1_activity",
            side_label,
            validated_activity_rows,
            group_indices("LLPC1", cell_side),
        )
    t4_left_indices = tuple(
        index
        for index in bundle_topology["retinotopic_t4a_indices"]
        if float(bundle_cells[index]["retinoAzimuth"]) < 0.0
    )
    t4_right_indices = tuple(
        index
        for index in bundle_topology["retinotopic_t4a_indices"]
        if float(bundle_cells[index]["retinoAzimuth"]) >= 0.0
    )
    require_pooled_match(
        "t4_activity",
        "raw_app_L",
        validated_activity_rows,
        t4_left_indices,
    )
    require_pooled_match(
        "t4_activity",
        "raw_app_R",
        validated_activity_rows,
        t4_right_indices,
    )

    dataset_metadata = _mapping(manifest.get("dataset"), "source dataset")
    dataset = DatasetRef(
        namespace=DatasetNamespace.FLYWIRE_FAFB,
        release="FAFB",
        materialization=783,
        source_uri="http://54.160.228.98/drosophila/api/manifest",
        neuron_universe=str(dataset_metadata.get("neuron_universe")),
        coordinate_units="nm",
    )
    confidence = Confidence(
        tier=EvidenceTier.MODEL_INFERENCE,
        level=ConfidenceLevel.LOW,
        score=0.1,
        basis=(
            "content-addressed fly-FGS fixed-step circuit execution; passive/reduced "
            "cable physiology, synthetic-fallback policy, and gains remain uncalibrated"
        ),
    )
    provenance = Provenance(
        source_uri=capture_uri,
        method=(
            "compressed capture SHA-256 verification before gzip; strict fixed-step "
            "import of four NOD1 voltage channels only"
        ),
        accessed_at_utc=str(manifest.get("source", {}).get("checked_at_utc")),
        dataset_identity=dataset.identity_space,
        artifact_hash="sha256:%s" % capture_sha256,
        source_run_id="%s@%s" % (FLY_FGS_FIXTURE_KIND, FLY_FGS_SNAPSHOT_ID),
        filters={
            "source_manifest_sha256": FLY_FGS_SOURCE_MANIFEST_SHA256,
            "circuit_engine_sha256": receipts_by_id["circuit_engine"].sha256,
            "circuit_bundle_sha256": receipts_by_id["circuit_bundle"].sha256,
            "included_voltage_root_ids": list(FLY_FGS_NOD1_ROOT_IDS),
            "excluded_display_state_from_motor_trace": True,
            "rejected_downstream_fields": list(rejected),
            "sample_interval": execution.get("sample_interval"),
            "pre_roll_steps": pre_roll_steps,
            "pre_roll_duration_s": pre_roll_duration_s,
            "availability_schedule": "offline_fixed_step_at_sample_time",
            "captured_streaming_availability": False,
        },
        notes=(
            "Upstream visual-circuit execution only. Raw bundle L/R labels are retained "
            "as low-confidence application-rendering conventions; without soma-x they "
            "are not anatomical laterality. No DN, MN, muscle, wing, yaw, aerodynamic, "
            "or body-motion value from the legacy page is imported."
        ),
    )

    retinal_provenance = Provenance(
        source_uri=capture_uri,
        method=(
            "exact endpoint fgmodel.lumNow sampling at every retinotopic T4a "
            "circuit coordinate, associated causally with the preceding fixed step"
        ),
        accessed_at_utc=str(manifest.get("source", {}).get("checked_at_utc")),
        dataset_identity=dataset.identity_space,
        artifact_hash="sha256:%s" % capture_sha256,
        source_run_id="%s@%s" % (FLY_FGS_FIXTURE_KIND, FLY_FGS_SNAPSHOT_ID),
        filters={
            "retinotopic_t4a_cell_count": retinal_input_count,
            "excluded_sample_indices": [0],
            "excluded_sample_0_reason": (
                "its source exposure lies in the pre-roll interval and cannot be "
                "represented by a nonnegative RetinalFrame exposure"
            ),
            "frame_sample_indices": list(range(1, sample_count)),
            "exposure_semantics": "[t_i-dt,t_i] with endpoint luminance at t_i",
            "availability_semantics": "available_at_t_i_offline_fixed_step",
            "eye_side": "binocular",
            "direction_frame": (
                "repository analytic body frame: +x ahead, +y positive azimuth, "
                "+z positive elevation"
            ),
        },
        notes=(
            "The samples are the engine's uncalibrated analytic scene values at T4a "
            "retinotopic coordinates. Spherical unit view vectors use the repository's "
            "analytic convention (+x ahead, +y positive azimuth, +z positive elevation), "
            "consistent with its PanoramicRetina schema adapter. The upstream page "
            "establishes head-forward +X and signed azimuth/elevation display coordinates, "
            "but these samples do not constitute calibrated biological ommatidia, "
            "compound-eye optics, photoreceptor output, or radiometry."
        ),
    )
    retinal_directions_body = tuple(
        (
            math.cos(math.radians(elevation_deg))
            * math.cos(math.radians(azimuth_deg)),
            math.cos(math.radians(elevation_deg))
            * math.sin(math.radians(azimuth_deg)),
            math.sin(math.radians(elevation_deg)),
        )
        for azimuth_deg, elevation_deg in zip(
            input_azimuth_deg, input_elevation_deg
        )
    )
    retinal_frames = tuple(
        RetinalFrame(
            measurement_time_s=raw_times[index],
            availability_time_s=raw_times[index],
            # Reuse the registered clock value rather than recomputing
            # ``t_i - dt``.  They are mathematically identical, but selecting
            # the previous source sample preserves byte-exact clock adjacency
            # in scientific artifacts (for example 0.145 rather than a nearby
            # floating subtraction result).
            exposure_start_s=raw_times[index - 1],
            exposure_end_s=raw_times[index],
            eye_side=EyeSide.BINOCULAR,
            signal_kind=RetinalSignalKind.NORMALIZED_LUMINANCE,
            unit="1",
            samples=retinal_input_rows[index],
            ommatidial_directions_body=retinal_directions_body,
            provenance=retinal_provenance,
            confidence=confidence,
            calibration_id="fly-fgs-analytic-t4a-grid-v1-uncalibrated",
        )
        for index in range(1, sample_count)
    )

    signals = []
    for root_id in FLY_FGS_NOD1_ROOT_IDS:
        raw_side = FLY_FGS_NOD1_RAW_APP_SIDES[root_id]
        side_context = SideContext(
            dataset=dataset,
            raw_dataset_side=raw_side,
            anatomical_side=AnatomicalSide.UNKNOWN,
            visual_field_side=AnatomicalSide.UNKNOWN,
            app_rendering_side=FLY_FGS_NOD1_APP_RENDERING_SIDES[root_id],
            eye_side=EyeSide.UNKNOWN,
            effector_side=AnatomicalSide.UNKNOWN,
            mapping_method=SideMappingMethod.SIMULATION_CONVENTION,
            confidence=confidence,
            notes=(
                "Captured fly-FGS compatibility label only; no soma-x coordinate is "
                "present, so this cannot be promoted as anatomical laterality."
            ),
        )
        signals.append(
            CircuitSignal(
                neuron=EntityRef(
                    dataset=dataset,
                    entity_id=root_id,
                    kind=EntityKind.NEURON,
                    cell_type="NOD1",
                    anatomical_side=AnatomicalSide.UNKNOWN,
                    side_context=side_context,
                ),
                signal_kind=CircuitSignalKind.VOLTAGE,
                unit="V",
                values=nod1_values[root_id],
                availability_times_s=raw_times,
                provenance=provenance,
                confidence=confidence,
                origin=SignalOrigin.SIMULATED,
            )
        )
    trace = CircuitOutputTrace(
        dataset=dataset,
        sample_times_s=raw_times,
        signals=tuple(signals),
        provenance=provenance,
        confidence=confidence,
        exact_timebase=True,
        sample_interval_end_s=duration_s,
    )

    validated_stimulus = deepcopy(dict(stimulus))
    validated_stimulus["retinal_display"] = {
        "azimuth_deg": list(azimuth),
        "luminance": [list(map(float, row)) for row in luminance],
        "unit": retinal.get("unit"),
    }
    validated_stimulus["retinal_input"] = {
        "cell_indices": list(input_indices),
        "cell_ids": list(input_cell_ids),
        "azimuth_deg": list(input_azimuth_deg),
        "elevation_deg": list(input_elevation_deg),
        "luminance": [list(row) for row in retinal_input_rows],
        "unit": "1",
        "sampling_semantics": expected_sampling_semantics,
    }
    validated_full_state = {
        "cell_ids": list(cell_ids),
        "voltage_v": [list(row) for row in validated_voltage_rows],
        "activity": [list(row) for row in validated_activity_rows],
    }
    circuit_topology = {
        "cell_axis": [deepcopy(dict(cell)) for cell in bundle_cells],
        "cable_indices": list(bundle_topology["cable_indices"]),
        "retinotopic_t4a": {
            "cell_indices": list(bundle_topology["retinotopic_t4a_indices"]),
            "cell_ids": list(bundle_topology["retinotopic_t4a_cell_ids"]),
            "azimuth_deg": list(
                bundle_topology["retinotopic_t4a_azimuth_deg"]
            ),
            "elevation_deg": list(
                bundle_topology["retinotopic_t4a_elevation_deg"]
            ),
            "count": retinal_input_count,
        },
    }
    return RegisteredFlyFGSFixture(
        circuit_trace=trace,
        source_manifest_sha256=FLY_FGS_SOURCE_MANIFEST_SHA256,
        capture_sha256=capture_sha256,
        source_manifest_bytes=source_manifest_bytes,
        capture_bytes=capture_bytes,
        source_manifest_uri=FLY_FGS_SOURCE_MANIFEST_URI,
        capture_uri=capture_uri,
        snapshot_id=FLY_FGS_SNAPSHOT_ID,
        asset_receipts=asset_receipts,
        fixed_step=deepcopy(dict(execution)),
        stimulus=validated_stimulus,
        pooled_readout_traces=validated_pooled,
        circuit_inventory=deepcopy(dict(inventory)),
        rejected_downstream_fields=rejected,
        retinal_frames=retinal_frames,
        full_cell_state=validated_full_state,
        circuit_topology=circuit_topology,
        dataset_metadata=deepcopy(dict(dataset_metadata)),
        claim_scope=deepcopy(dict(manifest.get("claim_scope", {}))),
    )


def parse_fly_fgs_fixed_step_capture_bytes(
    compressed_capture_bytes: bytes,
    *,
    expected_capture_sha256: str = FLY_FGS_REGISTERED_CAPTURE_SHA256,
    capture_uri: str = FLY_FGS_REGISTERED_CAPTURE_URI,
    manifest_path: Optional[Path] = None,
) -> RegisteredFlyFGSFixture:
    """Verify, decompress, strictly validate, and adapt a fixed-step capture."""

    _require(isinstance(compressed_capture_bytes, bytes), "compressed fly-FGS capture must be bytes")
    _require(
        isinstance(expected_capture_sha256, str)
        and len(expected_capture_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_capture_sha256),
        "expected fly-FGS capture SHA-256 is invalid",
    )
    observed_sha256 = _sha256_bytes(compressed_capture_bytes)
    _require(observed_sha256 == expected_capture_sha256, "compressed fly-FGS capture SHA-256 mismatch")
    try:
        raw_capture = gzip.decompress(compressed_capture_bytes)
    except (OSError, EOFError) as exc:
        raise ValueError("registered fly-FGS capture is not valid gzip data") from exc
    payload = _strict_json_object(raw_capture, "fly-FGS fixed-step capture")
    resolved_manifest = (
        default_fly_fgs_source_manifest_path()
        if manifest_path is None
        else Path(manifest_path)
    )
    (
        manifest,
        asset_receipts,
        assets_by_id,
        source_manifest_bytes,
    ) = _load_and_validate_source_manifest(
        resolved_manifest
    )
    bundle_topology = _validate_bundle_axis(
        assets_by_id["circuit_bundle"]["bytes"], manifest
    )
    return _validate_capture_payload(
        payload,
        manifest=manifest,
        asset_receipts=asset_receipts,
        bundle_topology=bundle_topology,
        capture_sha256=observed_sha256,
        source_manifest_bytes=source_manifest_bytes,
        capture_bytes=len(compressed_capture_bytes),
        capture_uri=capture_uri,
    )


def load_registered_fly_fgs_fixture(
    *,
    manifest_path: Optional[Path] = None,
    capture_path: Optional[Path] = None,
) -> RegisteredFlyFGSFixture:
    """Load the exact registered fly-FGS fixed-step circuit fixture."""

    resolved_capture = (
        default_fly_fgs_registered_capture_path()
        if capture_path is None
        else Path(capture_path)
    )
    capture_bytes = _read_regular_file(
        resolved_capture, "registered fly-FGS fixed-step capture"
    )
    return parse_fly_fgs_fixed_step_capture_bytes(
        capture_bytes,
        manifest_path=manifest_path,
    )


def load_registered_fly_fgs_circuit_trace(
    *,
    manifest_path: Optional[Path] = None,
    capture_path: Optional[Path] = None,
) -> CircuitOutputTrace:
    return load_registered_fly_fgs_fixture(
        manifest_path=manifest_path,
        capture_path=capture_path,
    ).circuit_trace


__all__ = [
    "FLY_FGS_FIXTURE_KIND",
    "FLY_FGS_NOD1_APP_RENDERING_SIDES",
    "FLY_FGS_NOD1_RAW_APP_SIDES",
    "FLY_FGS_NOD1_ROOT_IDS",
    "FLY_FGS_REGISTERED_CAPTURE_SHA256",
    "FLY_FGS_SCHEMA_VERSION",
    "FLY_FGS_SNAPSHOT_ID",
    "FLY_FGS_SOURCE_MANIFEST_SHA256",
    "FlyFGSAssetReceipt",
    "RegisteredFlyFGSFixture",
    "default_fly_fgs_registered_capture_path",
    "default_fly_fgs_source_manifest_path",
    "load_registered_fly_fgs_circuit_trace",
    "load_registered_fly_fgs_fixture",
    "parse_fly_fgs_fixed_step_capture_bytes",
]
