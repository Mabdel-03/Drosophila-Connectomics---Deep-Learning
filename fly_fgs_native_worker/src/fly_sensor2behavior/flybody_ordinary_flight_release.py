"""Fail-closed, read-only validation of the official FlyBody flight release.

This module deliberately has no download or extraction function.  It validates
an expected-artifact contract, hashes an already staged tree, and checks a
separately produced runtime/signature audit.  Unknown official archive hashes
or inventories remain blockers; observing local bytes never promotes those
bytes into an expected receipt.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import stat
import sysconfig
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0.0"
CONTRACT_ID = "flybody-official-ordinary-flight-release"
SCOPE = "official_ordinary_flight_only"
UPSTREAM_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
FIGSHARE_DOI = "10.25378/janelia.25309105"
FIGSHARE_VERSION = 4

FLIGHT_DATA_FILE_ID = "51196859"
ORDINARY_POLICY_FILE_ID = "44815195"
CONTROLLER_REUSE_FILE_ID = "51196886"
REQUIRED_ARCHIVE_FILE_IDS = (ORDINARY_POLICY_FILE_ID, FLIGHT_DATA_FILE_ID)

WPG_PATH = "datasets_flight-imitation/wing_pattern_fmech.npy"
REFERENCE_PATH = (
    "datasets_flight-imitation/flight-dataset_saccade-evasion_augmented.hdf5"
)
POLICY_ROOT = "trained-fly-policies/flight"
SAVED_MODEL_PATH = POLICY_ROOT + "/saved_model.pb"
VARIABLES_INDEX_PATH = POLICY_ROOT + "/variables/variables.index"
VARIABLES_PATTERN = POLICY_ROOT + "/variables/variables.data-*-of-*"
EXPECTED_MANIFEST_FILENAME = "flybody_ordinary_flight_release.expected.v1.json"

PHYSICS_TIMESTEP_S = 0.00005
CONTROL_TIMESTEP_S = 0.0002
WPG_BASE_FREQUENCY_HZ = 218.0
WPG_RELATIVE_FREQUENCY_RANGE = 0.05
WPG_DISCRETE_FREQUENCY_COUNT = 201

EXPECTED_INFERENCE_STAGE_IDS = (
    "construct_environment",
    "single_precision_wrapper",
    "canonical_action_wrapper",
    "load_saved_model",
    "mean_policy_wrapper",
    "reset_environment",
    "infer_distribution_mean_action",
    "step_environment",
)
EXPECTED_WRAPPER_ORDER = (
    "dm_env_wrappers.SinglePrecisionWrapper",
    "dm_env_wrappers.CanonicalSpecWrapper(clip=True)",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VARIABLE_SHARD_RE = re.compile(
    r"^trained-fly-policies/flight/variables/"
    r"variables\.data-([0-9]{5})-of-([0-9]{5})$"
)

_TOP_LEVEL_FIELDS = {
    "archive_policy",
    "claim_boundary",
    "contract_id",
    "expected_receipts",
    "original_inference_sequence",
    "required_audit_fields",
    "required_internal_paths",
    "runtime_facts",
    "schema_version",
    "scope",
    "source",
}

_AUDIT_FIELDS = {
    "archive_receipt": (
        "file_id",
        "source_uri",
        "bytes",
        "sha256",
        "verification_method",
    ),
    "inventory_entry": ("path", "bytes", "sha256"),
    "checkpoint_audit": (
        "policy_root",
        "format",
        "saved_model_load_succeeded",
        "callable_succeeded",
    ),
    "saved_model_signature_audit": (
        "policy_root",
        "signature_inventory_complete",
        "signature_names",
        "callable_python_type",
        "callable_input_signature",
        "callable_output_signature",
        "policy_return_type",
        "policy_output_semantics",
        "mean_action_shape",
        "mean_action_dtype",
        "external_observation_normalization",
        "layer_norm_variables_in_saved_model",
        "observation_spec_sha256",
        "action_spec_sha256",
    ),
    "runtime_audit": (
        "upstream_commit",
        "python_version",
        "numpy_version",
        "tensorflow_version",
        "tensorflow_probability_version",
        "acme_version",
        "sonnet_version",
        "dm_control_version",
        "mujoco_version",
        "platform_tag",
        "dependency_lock_sha256",
        "container_image_digest",
        "physics_timestep_s",
        "control_timestep_s",
    ),
    "action_wrapper_audit": (
        "environment_factory",
        "wrapper_order",
        "canonical_clip",
        "policy_action_selection",
        "canonical_action_bounds",
        "unwrapped_action_spec_sha256",
        "wrapped_action_spec_sha256",
        "action_vector_shape",
        "wing_action_indices",
        "user_frequency_action_index",
        "wpg_frequency_mapping_verified",
    ),
    "inference_sequence_audit": (
        "executed_stage_ids",
        "observation_precedes_action",
        "one_policy_call_per_control_step",
        "environment_step_receives_wrapped_action",
    ),
}


class OrdinaryFlightReleaseError(ValueError):
    """The expected contract or staged artifact tree is malformed."""


@dataclass(frozen=True, order=True)
class ReleaseBlocker:
    """One independently actionable reason a release intake is not ready."""

    code: str
    subject: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "subject": self.subject, "detail": self.detail}


@dataclass(frozen=True, order=True)
class InventoryEntry:
    """A byte-precise member of a locally staged extracted tree."""

    path: str
    bytes: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class OrdinaryFlightReleaseReport:
    """Validation result; this intake never promotes a scientific gate."""

    ready: bool
    blockers: Tuple[ReleaseBlocker, ...]
    observed_inventory: Tuple[InventoryEntry, ...]
    contract_id: str = CONTRACT_ID
    schema_version: str = SCHEMA_VERSION
    physics_timestep_convergence_unblocked: bool = False

    @property
    def blocker_codes(self) -> Tuple[str, ...]:
        return tuple(blocker.code for blocker in self.blockers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "ready": self.ready,
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "observed_inventory": [entry.to_dict() for entry in self.observed_inventory],
            "physics_timestep_convergence_unblocked": (
                self.physics_timestep_convergence_unblocked
            ),
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrdinaryFlightReleaseError("%s must be a JSON object" % label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise OrdinaryFlightReleaseError("%s must be a JSON array" % label)
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise OrdinaryFlightReleaseError("%s must be a non-empty trimmed string" % label)
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if _SHA256_RE.fullmatch(text) is None:
        raise OrdinaryFlightReleaseError("%s must be a lowercase SHA-256" % label)
    return text


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OrdinaryFlightReleaseError("%s must be a positive integer" % label)
    return value


def _relative_path(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if "\\" in text or text.startswith("/"):
        raise OrdinaryFlightReleaseError("%s must be a relative POSIX path" % label)
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise OrdinaryFlightReleaseError("%s must be a normalized relative path" % label)
    path = PurePosixPath(text)
    if path.is_absolute():
        raise OrdinaryFlightReleaseError("%s must be relative" % label)
    return path.as_posix()


def _require_exact_fields(
    value: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    expected_set = set(expected)
    actual = set(value)
    missing = sorted(expected_set - actual)
    unknown = sorted(actual - expected_set)
    if missing:
        raise OrdinaryFlightReleaseError(
            "%s is missing fields: %s" % (label, ", ".join(missing))
        )
    if unknown:
        raise OrdinaryFlightReleaseError(
            "%s has unknown fields: %s" % (label, ", ".join(unknown))
        )


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OrdinaryFlightReleaseError("duplicate JSON key %r" % key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise OrdinaryFlightReleaseError("non-finite JSON constant %r" % value)


def canonical_inventory_sha256(entries: Iterable[Mapping[str, Any]]) -> str:
    """Return the contract digest for an exact, sorted extracted inventory."""

    normalized = _validate_inventory_entries(entries, "inventory digest input")
    payload = json.dumps(
        [entry.to_dict() for entry in normalized],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_inventory_entries(
    values: Iterable[Mapping[str, Any]], label: str
) -> Tuple[InventoryEntry, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise OrdinaryFlightReleaseError("%s must be an array" % label)
    records: List[InventoryEntry] = []
    for index, raw in enumerate(values):
        item = _mapping(raw, "%s[%d]" % (label, index))
        _require_exact_fields(item, _AUDIT_FIELDS["inventory_entry"], "%s[%d]" % (label, index))
        records.append(
            InventoryEntry(
                path=_relative_path(item["path"], "%s[%d].path" % (label, index)),
                bytes=_positive_integer(item["bytes"], "%s[%d].bytes" % (label, index)),
                sha256=_sha256(item["sha256"], "%s[%d].sha256" % (label, index)),
            )
        )
    if not records:
        raise OrdinaryFlightReleaseError("%s must not be empty" % label)
    paths = [record.path for record in records]
    if paths != sorted(paths):
        raise OrdinaryFlightReleaseError("%s must be sorted by path" % label)
    if len(paths) != len(set(paths)):
        raise OrdinaryFlightReleaseError("%s contains duplicate paths" % label)
    if len({path.casefold() for path in paths}) != len(paths):
        raise OrdinaryFlightReleaseError("%s contains case-colliding paths" % label)
    return tuple(records)


def _validate_expected_receipt(receipt: Mapping[str, Any], label: str) -> None:
    _require_exact_fields(
        receipt, ("bytes", "file_id", "reason", "sha256", "status"), label
    )
    _nonempty_string(receipt["file_id"], label + ".file_id")
    _nonempty_string(receipt["reason"], label + ".reason")
    status_value = receipt["status"]
    if status_value == "unresolved_not_downloaded":
        if receipt["bytes"] is not None or receipt["sha256"] is not None:
            raise OrdinaryFlightReleaseError(
                "%s unresolved receipt must use null bytes and sha256" % label
            )
    elif status_value == "resolved_reviewed":
        _positive_integer(receipt["bytes"], label + ".bytes")
        _sha256(receipt["sha256"], label + ".sha256")
    else:
        raise OrdinaryFlightReleaseError("%s has unsupported status" % label)


def validate_expected_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return an expected-artifact manifest without mutating it."""

    root = _mapping(manifest, "expected manifest")
    _require_exact_fields(root, _TOP_LEVEL_FIELDS, "expected manifest")
    if root["schema_version"] != SCHEMA_VERSION:
        raise OrdinaryFlightReleaseError("unsupported expected manifest schema_version")
    if root["contract_id"] != CONTRACT_ID or root["scope"] != SCOPE:
        raise OrdinaryFlightReleaseError("expected manifest scope or contract_id is wrong")

    source = _mapping(root["source"], "source")
    _require_exact_fields(
        source,
        (
            "figshare_doi",
            "figshare_version",
            "repository_uri",
            "upstream_commit",
            "upstream_fact_locations",
        ),
        "source",
    )
    if (
        source["figshare_doi"] != FIGSHARE_DOI
        or source["figshare_version"] != FIGSHARE_VERSION
        or source["upstream_commit"] != UPSTREAM_COMMIT
        or source["repository_uri"] != "https://github.com/TuragaLab/flybody"
    ):
        raise OrdinaryFlightReleaseError("expected manifest source is not the pinned release")
    facts = tuple(_sequence(source["upstream_fact_locations"], "source.upstream_fact_locations"))
    if not facts or any(not isinstance(item, str) or not item for item in facts):
        raise OrdinaryFlightReleaseError("source.upstream_fact_locations must be non-empty paths")

    archive_policy = _mapping(root["archive_policy"], "archive_policy")
    _require_exact_fields(archive_policy, ("excluded", "required"), "archive_policy")
    required_archives = _sequence(archive_policy["required"], "archive_policy.required")
    required_ids: List[str] = []
    for index, raw in enumerate(required_archives):
        item = _mapping(raw, "archive_policy.required[%d]" % index)
        _require_exact_fields(
            item, ("file_id", "name", "purpose", "source_uri"),
            "archive_policy.required[%d]" % index,
        )
        required_ids.append(_nonempty_string(item["file_id"], "required archive id"))
        _nonempty_string(item["name"], "required archive name")
        _nonempty_string(item["purpose"], "required archive purpose")
        expected_uri = "https://janelia.figshare.com/ndownloader/files/%s" % item["file_id"]
        if item["source_uri"] != expected_uri:
            raise OrdinaryFlightReleaseError("required archive source_uri is not pinned")
    if tuple(sorted(required_ids)) != REQUIRED_ARCHIVE_FILE_IDS:
        raise OrdinaryFlightReleaseError("required archives must be exactly 44815195 and 51196859")

    excluded = _sequence(archive_policy["excluded"], "archive_policy.excluded")
    if len(excluded) != 1:
        raise OrdinaryFlightReleaseError("exactly one non-interchangeable archive is required")
    excluded_item = _mapping(excluded[0], "archive_policy.excluded[0]")
    _require_exact_fields(
        excluded_item, ("file_id", "name", "reason", "status"),
        "archive_policy.excluded[0]",
    )
    if (
        excluded_item["file_id"] != CONTROLLER_REUSE_FILE_ID
        or excluded_item["status"] != "excluded_non_interchangeable"
    ):
        raise OrdinaryFlightReleaseError("file 51196886 must be explicitly excluded")
    _nonempty_string(excluded_item["reason"], "excluded archive reason")

    paths = _mapping(root["required_internal_paths"], "required_internal_paths")
    _require_exact_fields(paths, ("exact", "patterns"), "required_internal_paths")
    exact = _sequence(paths["exact"], "required_internal_paths.exact")
    exact_by_id: Dict[str, str] = {}
    for index, raw in enumerate(exact):
        item = _mapping(raw, "required_internal_paths.exact[%d]" % index)
        _require_exact_fields(
            item, ("archive_file_id", "artifact_id", "path"),
            "required_internal_paths.exact[%d]" % index,
        )
        artifact_id = _nonempty_string(item["artifact_id"], "artifact_id")
        if artifact_id in exact_by_id:
            raise OrdinaryFlightReleaseError("duplicate required artifact_id")
        if item["archive_file_id"] not in REQUIRED_ARCHIVE_FILE_IDS:
            raise OrdinaryFlightReleaseError("required path references a non-required archive")
        exact_by_id[artifact_id] = _relative_path(item["path"], "required path")
    expected_exact = {
        "flight_reference_dataset": REFERENCE_PATH,
        "measured_wing_pattern": WPG_PATH,
        "ordinary_flight_saved_model": SAVED_MODEL_PATH,
        "ordinary_flight_variables_index": VARIABLES_INDEX_PATH,
    }
    if exact_by_id != expected_exact:
        raise OrdinaryFlightReleaseError("required exact paths do not match the official release")

    patterns = _sequence(paths["patterns"], "required_internal_paths.patterns")
    if len(patterns) != 1:
        raise OrdinaryFlightReleaseError("one TensorFlow variable-shard pattern is required")
    pattern = _mapping(patterns[0], "required_internal_paths.patterns[0]")
    _require_exact_fields(
        pattern,
        ("archive_file_id", "artifact_id", "minimum_matches", "pattern", "validation"),
        "required_internal_paths.patterns[0]",
    )
    if (
        pattern["archive_file_id"] != ORDINARY_POLICY_FILE_ID
        or pattern["artifact_id"] != "ordinary_flight_variable_shards"
        or pattern["minimum_matches"] != 1
        or pattern["pattern"] != VARIABLES_PATTERN
        or pattern["validation"] != "tensorflow_contiguous_shard_set"
    ):
        raise OrdinaryFlightReleaseError("ordinary-flight variable-shard contract is wrong")

    runtime = _mapping(root["runtime_facts"], "runtime_facts")
    _require_exact_fields(
        runtime, ("control_timestep_s", "physics_timestep_s", "wing_pattern_generator"),
        "runtime_facts",
    )
    if (
        runtime["physics_timestep_s"] != PHYSICS_TIMESTEP_S
        or runtime["control_timestep_s"] != CONTROL_TIMESTEP_S
    ):
        raise OrdinaryFlightReleaseError("ordinary-flight clocks are wrong")
    wpg = _mapping(runtime["wing_pattern_generator"], "runtime_facts.wing_pattern_generator")
    _require_exact_fields(
        wpg,
        (
            "base_frequency_hz",
            "control_action_mapping",
            "discrete_frequency_count",
            "relative_frequency_range",
        ),
        "runtime_facts.wing_pattern_generator",
    )
    if (
        wpg["base_frequency_hz"] != WPG_BASE_FREQUENCY_HZ
        or wpg["relative_frequency_range"] != WPG_RELATIVE_FREQUENCY_RANGE
        or wpg["discrete_frequency_count"] != WPG_DISCRETE_FREQUENCY_COUNT
        or wpg["control_action_mapping"]
        != "base_frequency_hz * (1 + relative_frequency_range * canonical_user_action)"
    ):
        raise OrdinaryFlightReleaseError("ordinary-flight WPG facts are wrong")

    sequence = _sequence(root["original_inference_sequence"], "original_inference_sequence")
    stage_ids: List[str] = []
    for index, raw in enumerate(sequence):
        item = _mapping(raw, "original_inference_sequence[%d]" % index)
        _require_exact_fields(item, ("operation", "stage_id"), "inference stage")
        stage_ids.append(_nonempty_string(item["stage_id"], "inference stage_id"))
        _nonempty_string(item["operation"], "inference operation")
    if tuple(stage_ids) != EXPECTED_INFERENCE_STAGE_IDS:
        raise OrdinaryFlightReleaseError("original inference sequence has changed")

    required_audits = _mapping(root["required_audit_fields"], "required_audit_fields")
    _require_exact_fields(required_audits, _AUDIT_FIELDS, "required_audit_fields")
    for name, expected_fields in _AUDIT_FIELDS.items():
        values = tuple(_sequence(required_audits[name], "required_audit_fields.%s" % name))
        if values != expected_fields:
            raise OrdinaryFlightReleaseError(
                "required_audit_fields.%s cannot be weakened or reordered" % name
            )

    receipts = _mapping(root["expected_receipts"], "expected_receipts")
    _require_exact_fields(receipts, ("archives", "extracted_inventory"), "expected_receipts")
    archive_receipts = _sequence(receipts["archives"], "expected_receipts.archives")
    receipt_ids: List[str] = []
    for index, raw in enumerate(archive_receipts):
        item = _mapping(raw, "expected_receipts.archives[%d]" % index)
        _validate_expected_receipt(item, "expected_receipts.archives[%d]" % index)
        receipt_ids.append(item["file_id"])
    if tuple(receipt_ids) != REQUIRED_ARCHIVE_FILE_IDS:
        raise OrdinaryFlightReleaseError("archive expected receipts must be sorted and complete")

    inventory_receipt = _mapping(
        receipts["extracted_inventory"], "expected_receipts.extracted_inventory"
    )
    _require_exact_fields(
        inventory_receipt,
        ("inventory_sha256", "member_count", "members", "reason", "status"),
        "expected_receipts.extracted_inventory",
    )
    _nonempty_string(inventory_receipt["reason"], "inventory receipt reason")
    if inventory_receipt["status"] == "unresolved_not_downloaded":
        if any(
            inventory_receipt[name] is not None
            for name in ("inventory_sha256", "member_count", "members")
        ):
            raise OrdinaryFlightReleaseError(
                "unresolved extracted inventory must use null receipt fields"
            )
    elif inventory_receipt["status"] == "resolved_reviewed":
        members = _validate_inventory_entries(
            inventory_receipt["members"], "expected extracted inventory members"
        )
        if inventory_receipt["member_count"] != len(members):
            raise OrdinaryFlightReleaseError("expected inventory member_count is wrong")
        expected_digest = canonical_inventory_sha256(
            entry.to_dict() for entry in members
        )
        if inventory_receipt["inventory_sha256"] != expected_digest:
            raise OrdinaryFlightReleaseError("expected inventory digest is wrong")
    else:
        raise OrdinaryFlightReleaseError("unsupported extracted inventory status")

    boundary = _mapping(root["claim_boundary"], "claim_boundary")
    _require_exact_fields(boundary, ("allowed", "forbidden"), "claim_boundary")
    _nonempty_string(boundary["allowed"], "claim_boundary.allowed")
    forbidden = _sequence(boundary["forbidden"], "claim_boundary.forbidden")
    if len(forbidden) < 3 or any(not isinstance(item, str) or not item for item in forbidden):
        raise OrdinaryFlightReleaseError("claim_boundary.forbidden is incomplete")
    return root


def load_expected_manifest(path: Path) -> Mapping[str, Any]:
    """Load a UTF-8 expected-artifact manifest with duplicate-key rejection."""

    try:
        payload = Path(path).read_bytes().decode("utf-8", errors="strict")
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OrdinaryFlightReleaseError("cannot load expected manifest: %s" % exc) from exc
    return validate_expected_manifest(_mapping(value, "expected manifest"))


def default_expected_manifest_path() -> Path:
    """Locate the immutable contract in a checkout or installed worker."""

    candidates = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "reference"
        / EXPECTED_MANIFEST_FILENAME,
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / "reference"
        / EXPECTED_MANIFEST_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise OrdinaryFlightReleaseError(
        "official ordinary-flight expected manifest is unavailable"
    )


def load_runtime_audit(path: Path) -> Mapping[str, Any]:
    """Load a candidate audit without accepting duplicate or non-finite JSON."""

    try:
        payload = Path(path).read_bytes().decode("utf-8", errors="strict")
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OrdinaryFlightReleaseError("cannot load runtime audit: %s" % exc) from exc
    return _mapping(value, "runtime audit")


def scan_staged_tree(root: Path) -> Tuple[InventoryEntry, ...]:
    """Hash an existing extracted tree without following links or writing files."""

    root = Path(root)
    if root.is_symlink():
        raise OrdinaryFlightReleaseError("staged root must not be a symlink")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise OrdinaryFlightReleaseError("staged root is missing") from exc
    if not resolved_root.is_dir():
        raise OrdinaryFlightReleaseError("staged root must be a directory")
    records: List[InventoryEntry] = []
    inodes: Dict[Tuple[int, int], str] = {}
    folded_paths: Dict[str, str] = {}

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(str(directory)), key=lambda item: item.name)
        except OSError as exc:
            raise OrdinaryFlightReleaseError("cannot scan staged tree: %s" % exc) from exc
        for directory_entry in entries:
            path = Path(directory_entry.path)
            relative = _relative_path(
                path.relative_to(resolved_root).as_posix(), "staged member path"
            )
            try:
                info = directory_entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise OrdinaryFlightReleaseError(
                    "cannot stat staged member %s: %s" % (relative, exc)
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise OrdinaryFlightReleaseError("staged tree contains a symlink: %s" % relative)
            if stat.S_ISDIR(info.st_mode):
                visit(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise OrdinaryFlightReleaseError(
                    "staged tree contains a non-regular member: %s" % relative
                )
            if info.st_size <= 0:
                raise OrdinaryFlightReleaseError("staged file is empty: %s" % relative)
            inode = (info.st_dev, info.st_ino)
            if inode in inodes:
                raise OrdinaryFlightReleaseError(
                    "staged tree contains duplicate hard links: %s and %s"
                    % (inodes[inode], relative)
                )
            folded = relative.casefold()
            if folded in folded_paths:
                raise OrdinaryFlightReleaseError(
                    "staged tree contains case-colliding paths: %s and %s"
                    % (folded_paths[folded], relative)
                )
            digest = hashlib.sha256()
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(str(path), flags)
            except OSError as exc:
                raise OrdinaryFlightReleaseError(
                    "cannot open staged member %s: %s" % (relative, exc)
                ) from exc
            try:
                before = os.fstat(descriptor)
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if before_identity != after_identity:
                raise OrdinaryFlightReleaseError("staged file changed while hashing: %s" % relative)
            inodes[inode] = relative
            folded_paths[folded] = relative
            records.append(
                InventoryEntry(path=relative, bytes=before.st_size, sha256=digest.hexdigest())
            )

    visit(resolved_root)
    if not records:
        raise OrdinaryFlightReleaseError("staged tree contains no files")
    return tuple(sorted(records, key=lambda item: item.path))


def _add_blocker(
    blockers: List[ReleaseBlocker], code: str, subject: str, detail: str
) -> None:
    blockers.append(ReleaseBlocker(code=code, subject=subject, detail=detail))


def _missing_fields(
    section: Mapping[str, Any], required: Sequence[str]
) -> Tuple[str, ...]:
    return tuple(sorted(set(required) - set(section)))


def _is_json_signature(value: Any) -> bool:
    return isinstance(value, (Mapping, list)) and bool(value)


def _validate_shards(paths: Sequence[str]) -> Optional[str]:
    matches = []
    for path in paths:
        match = _VARIABLE_SHARD_RE.fullmatch(path)
        if match is None:
            return "variable shard name does not match TensorFlow's indexed shard form"
        matches.append((int(match.group(1)), int(match.group(2))))
    totals = {total for _index, total in matches}
    if len(totals) != 1 or next(iter(totals)) <= 0:
        return "variable shards declare inconsistent totals"
    total = next(iter(totals))
    if len(matches) != total or {index for index, _total in matches} != set(range(total)):
        return "variable shard set is not contiguous and complete"
    return None


def validate_staged_release(
    manifest: Mapping[str, Any],
    staged_root: Path,
    *,
    audit: Optional[Mapping[str, Any]] = None,
) -> OrdinaryFlightReleaseReport:
    """Validate staged bytes and audit metadata against an expected contract.

    A locally observed digest is never substituted for an unresolved expected
    digest.  Consequently the checked-in official contract remains not-ready
    until independent reviewed receipts replace its explicit null values.
    """

    contract = validate_expected_manifest(manifest)
    blockers: List[ReleaseBlocker] = []
    try:
        observed = scan_staged_tree(staged_root)
    except OrdinaryFlightReleaseError as exc:
        observed = ()
        _add_blocker(blockers, "staged_inventory_invalid", "staged_root", str(exc))

    observed_by_path = {entry.path: entry for entry in observed}
    exact_artifacts = contract["required_internal_paths"]["exact"]
    for artifact in exact_artifacts:
        path = artifact["path"]
        if path not in observed_by_path:
            artifact_id = artifact["artifact_id"]
            if artifact_id == "ordinary_flight_saved_model":
                code = "checkpoint_descriptor_missing"
            elif artifact_id == "ordinary_flight_variables_index":
                code = "checkpoint_variables_index_missing"
            else:
                code = "required_artifact_missing"
            _add_blocker(blockers, code, artifact_id, "missing staged path %s" % path)

    shard_paths = sorted(
        path for path in observed_by_path if fnmatch.fnmatchcase(path, VARIABLES_PATTERN)
    )
    all_variable_data_paths = sorted(
        path
        for path in observed_by_path
        if path.startswith(POLICY_ROOT + "/variables/variables.data-")
    )
    if not shard_paths:
        if all_variable_data_paths:
            _add_blocker(
                blockers,
                "checkpoint_variable_shards_invalid",
                "ordinary_flight_variable_shards",
                "variable-data files are present but none has a valid TensorFlow shard name",
            )
        else:
            _add_blocker(
                blockers,
                "checkpoint_variable_shards_missing",
                "ordinary_flight_variable_shards",
                "no staged TensorFlow variable shard matches %s" % VARIABLES_PATTERN,
            )
    elif shard_paths != all_variable_data_paths:
        _add_blocker(
            blockers,
            "checkpoint_variable_shards_invalid",
            "ordinary_flight_variable_shards",
            "one or more variable-data files has an invalid shard name",
        )
    else:
        shard_error = _validate_shards(shard_paths)
        if shard_error is not None:
            _add_blocker(
                blockers,
                "checkpoint_variable_shards_invalid",
                "ordinary_flight_variable_shards",
                shard_error,
            )

    expected_receipts = contract["expected_receipts"]
    expected_archive_by_id = {
        item["file_id"]: item for item in expected_receipts["archives"]
    }
    for file_id in REQUIRED_ARCHIVE_FILE_IDS:
        if expected_archive_by_id[file_id]["status"] != "resolved_reviewed":
            _add_blocker(
                blockers,
                "expected_archive_receipt_unresolved",
                file_id,
                "the expected archive SHA-256 and byte count are explicitly unresolved",
            )
    if expected_receipts["extracted_inventory"]["status"] != "resolved_reviewed":
        _add_blocker(
            blockers,
            "expected_inventory_receipt_unresolved",
            "extracted_inventory",
            "the reviewed complete extracted-member inventory is explicitly unresolved",
        )

    audit_root: Mapping[str, Any]
    if audit is None:
        audit_root = {}
    elif isinstance(audit, Mapping):
        audit_root = audit
        if audit_root.get("schema_version") != SCHEMA_VERSION:
            _add_blocker(
                blockers, "audit_schema_mismatch", "audit", "audit schema_version is missing or wrong"
            )
        if audit_root.get("contract_id") != CONTRACT_ID:
            _add_blocker(
                blockers, "audit_contract_mismatch", "audit", "audit contract_id is missing or wrong"
            )
    else:
        audit_root = {}
        _add_blocker(blockers, "audit_invalid", "audit", "audit must be a JSON object")

    required_archive_policy = {
        item["file_id"]: item for item in contract["archive_policy"]["required"]
    }
    raw_archive_receipts = audit_root.get("archive_receipts")
    candidate_archive_by_id: Dict[str, Mapping[str, Any]] = {}
    if not isinstance(raw_archive_receipts, list):
        for file_id in REQUIRED_ARCHIVE_FILE_IDS:
            _add_blocker(
                blockers, "archive_receipt_missing", file_id, "archive_receipts audit is absent"
            )
    else:
        for index, raw in enumerate(raw_archive_receipts):
            if not isinstance(raw, Mapping):
                _add_blocker(
                    blockers, "archive_receipt_invalid", str(index), "archive receipt is not an object"
                )
                continue
            missing = _missing_fields(raw, _AUDIT_FIELDS["archive_receipt"])
            file_id = str(raw.get("file_id", index))
            if missing:
                _add_blocker(
                    blockers,
                    "archive_receipt_incomplete",
                    file_id,
                    "missing fields: %s" % ", ".join(missing),
                )
                continue
            if file_id in candidate_archive_by_id:
                _add_blocker(
                    blockers, "archive_receipt_duplicate", file_id, "duplicate archive receipt"
                )
                continue
            candidate_archive_by_id[file_id] = raw
        unexpected_ids = sorted(
            set(candidate_archive_by_id)
            - set(REQUIRED_ARCHIVE_FILE_IDS)
            - {CONTROLLER_REUSE_FILE_ID}
        )
        for file_id in unexpected_ids:
            _add_blocker(
                blockers,
                "archive_receipt_unexpected",
                file_id,
                "archive is outside the ordinary-flight contract",
            )
        if CONTROLLER_REUSE_FILE_ID in candidate_archive_by_id:
            _add_blocker(
                blockers,
                "excluded_archive_present",
                CONTROLLER_REUSE_FILE_ID,
                "controller-reuse artifacts cannot satisfy the ordinary-flight contract",
            )
        for file_id in REQUIRED_ARCHIVE_FILE_IDS:
            candidate = candidate_archive_by_id.get(file_id)
            if candidate is None:
                _add_blocker(
                    blockers, "archive_receipt_missing", file_id, "required archive receipt is absent"
                )
                continue
            try:
                _positive_integer(candidate["bytes"], "archive receipt bytes")
                _sha256(candidate["sha256"], "archive receipt sha256")
                _nonempty_string(candidate["verification_method"], "verification_method")
            except OrdinaryFlightReleaseError as exc:
                _add_blocker(blockers, "archive_receipt_invalid", file_id, str(exc))
                continue
            if candidate["source_uri"] != required_archive_policy[file_id]["source_uri"]:
                _add_blocker(
                    blockers, "archive_receipt_source_mismatch", file_id, "source_uri is not pinned"
                )
            expected = expected_archive_by_id[file_id]
            if expected["status"] == "resolved_reviewed" and (
                candidate["bytes"] != expected["bytes"]
                or candidate["sha256"] != expected["sha256"]
            ):
                _add_blocker(
                    blockers,
                    "archive_receipt_mismatch",
                    file_id,
                    "observed archive receipt does not match the reviewed expectation",
                )

    raw_inventory = audit_root.get("inventory")
    audited_inventory: Tuple[InventoryEntry, ...] = ()
    if raw_inventory is None:
        _add_blocker(
            blockers, "inventory_audit_missing", "extracted_inventory", "inventory audit is absent"
        )
    else:
        try:
            audited_inventory = _validate_inventory_entries(raw_inventory, "audit.inventory")
        except OrdinaryFlightReleaseError as exc:
            _add_blocker(blockers, "inventory_audit_invalid", "extracted_inventory", str(exc))
        else:
            if audited_inventory != observed:
                _add_blocker(
                    blockers,
                    "inventory_audit_mismatch",
                    "extracted_inventory",
                    "audited inventory does not exactly match the read-only staged-tree scan",
                )
    expected_inventory = expected_receipts["extracted_inventory"]
    if expected_inventory["status"] == "resolved_reviewed" and observed:
        reviewed_inventory = _validate_inventory_entries(
            expected_inventory["members"], "expected extracted inventory members"
        )
        if reviewed_inventory != observed:
            _add_blocker(
                blockers,
                "expected_inventory_mismatch",
                "extracted_inventory",
                "staged inventory does not match the reviewed expected inventory",
            )

    checkpoint = audit_root.get("checkpoint_audit")
    if not isinstance(checkpoint, Mapping):
        _add_blocker(
            blockers, "checkpoint_audit_missing", POLICY_ROOT, "checkpoint audit is absent"
        )
    else:
        missing = _missing_fields(checkpoint, _AUDIT_FIELDS["checkpoint_audit"])
        if missing:
            _add_blocker(
                blockers,
                "checkpoint_audit_incomplete",
                POLICY_ROOT,
                "missing fields: %s" % ", ".join(missing),
            )
        else:
            if checkpoint["policy_root"] != POLICY_ROOT or checkpoint["format"] != "tensorflow_saved_model":
                _add_blocker(
                    blockers, "checkpoint_audit_invalid", POLICY_ROOT, "checkpoint identity or format is wrong"
                )
            if checkpoint["saved_model_load_succeeded"] is not True:
                _add_blocker(
                    blockers, "checkpoint_load_failed", POLICY_ROOT, "tf.saved_model.load did not succeed"
                )
            if checkpoint["callable_succeeded"] is not True:
                _add_blocker(
                    blockers, "checkpoint_callable_failed", POLICY_ROOT, "loaded policy call did not succeed"
                )

    signature = audit_root.get("saved_model_signature_audit")
    if not isinstance(signature, Mapping):
        _add_blocker(
            blockers, "saved_model_signature_audit_missing", POLICY_ROOT, "signature audit is absent"
        )
    else:
        missing = _missing_fields(signature, _AUDIT_FIELDS["saved_model_signature_audit"])
        if missing:
            _add_blocker(
                blockers,
                "saved_model_signature_audit_incomplete",
                POLICY_ROOT,
                "missing fields: %s" % ", ".join(missing),
            )
        else:
            invalid_reasons: List[str] = []
            if signature["policy_root"] != POLICY_ROOT:
                invalid_reasons.append("wrong policy_root")
            if signature["signature_inventory_complete"] is not True:
                invalid_reasons.append("signature inventory is not complete")
            names = signature["signature_names"]
            if not isinstance(names, list) or any(not isinstance(item, str) for item in names):
                invalid_reasons.append("signature_names is not an explicit string list")
            for name in ("callable_python_type", "policy_return_type", "mean_action_dtype"):
                if not isinstance(signature[name], str) or not signature[name]:
                    invalid_reasons.append("%s is empty" % name)
            if not _is_json_signature(signature["callable_input_signature"]):
                invalid_reasons.append("callable_input_signature is empty")
            if not _is_json_signature(signature["callable_output_signature"]):
                invalid_reasons.append("callable_output_signature is empty")
            if signature["policy_output_semantics"] != "distribution":
                invalid_reasons.append("policy output is not audited as a distribution")
            if signature["external_observation_normalization"] != "none_documented":
                invalid_reasons.append(
                    "external observation normalization is not audited as none documented"
                )
            if signature["layer_norm_variables_in_saved_model"] is not True:
                invalid_reasons.append("embedded LayerNorm variables were not verified")
            shape = signature["mean_action_shape"]
            if (
                not isinstance(shape, list)
                or not shape
                or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
            ):
                invalid_reasons.append("mean_action_shape is invalid")
            for name in ("observation_spec_sha256", "action_spec_sha256"):
                try:
                    _sha256(signature[name], name)
                except OrdinaryFlightReleaseError:
                    invalid_reasons.append("%s is invalid" % name)
            if invalid_reasons:
                _add_blocker(
                    blockers,
                    "saved_model_signature_audit_invalid",
                    POLICY_ROOT,
                    "; ".join(invalid_reasons),
                )

    runtime_audit = audit_root.get("runtime_audit")
    if not isinstance(runtime_audit, Mapping):
        _add_blocker(blockers, "runtime_audit_missing", "runtime", "runtime audit is absent")
    else:
        missing = _missing_fields(runtime_audit, _AUDIT_FIELDS["runtime_audit"])
        if missing:
            _add_blocker(
                blockers,
                "runtime_audit_incomplete",
                "runtime",
                "missing fields: %s" % ", ".join(missing),
            )
        else:
            invalid_reasons = []
            if runtime_audit["upstream_commit"] != UPSTREAM_COMMIT:
                invalid_reasons.append("wrong upstream commit")
            for name in (
                "python_version",
                "numpy_version",
                "tensorflow_version",
                "tensorflow_probability_version",
                "acme_version",
                "sonnet_version",
                "dm_control_version",
                "mujoco_version",
                "platform_tag",
            ):
                if not isinstance(runtime_audit[name], str) or not runtime_audit[name]:
                    invalid_reasons.append("%s is empty" % name)
            try:
                _sha256(runtime_audit["dependency_lock_sha256"], "dependency_lock_sha256")
            except OrdinaryFlightReleaseError:
                invalid_reasons.append("dependency_lock_sha256 is invalid")
            image_digest = runtime_audit["container_image_digest"]
            if not isinstance(image_digest, str) or _IMAGE_DIGEST_RE.fullmatch(image_digest) is None:
                invalid_reasons.append("container_image_digest is invalid")
            for name, expected in (
                ("physics_timestep_s", PHYSICS_TIMESTEP_S),
                ("control_timestep_s", CONTROL_TIMESTEP_S),
            ):
                value = runtime_audit[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15)
                ):
                    invalid_reasons.append("%s is wrong" % name)
            if invalid_reasons:
                _add_blocker(
                    blockers, "runtime_audit_invalid", "runtime", "; ".join(invalid_reasons)
                )

    action_audit = audit_root.get("action_wrapper_audit")
    if not isinstance(action_audit, Mapping):
        _add_blocker(
            blockers, "action_wrapper_audit_missing", "action_wrapper", "action wrapper audit is absent"
        )
    else:
        missing = _missing_fields(action_audit, _AUDIT_FIELDS["action_wrapper_audit"])
        if missing:
            _add_blocker(
                blockers,
                "action_wrapper_audit_incomplete",
                "action_wrapper",
                "missing fields: %s" % ", ".join(missing),
            )
        else:
            invalid_reasons = []
            if action_audit["environment_factory"] != "flybody.fly_envs.flight_imitation":
                invalid_reasons.append("wrong environment factory")
            wrapper_order = action_audit["wrapper_order"]
            if (
                not isinstance(wrapper_order, list)
                or tuple(wrapper_order) != EXPECTED_WRAPPER_ORDER
            ):
                invalid_reasons.append("wrapper order differs from the official sequence")
            if action_audit["canonical_clip"] is not True:
                invalid_reasons.append("canonical clipping is disabled")
            if action_audit["policy_action_selection"] != "distribution_mean":
                invalid_reasons.append("policy action is not the distribution mean")
            if action_audit["canonical_action_bounds"] != [-1.0, 1.0]:
                invalid_reasons.append("canonical action bounds are not [-1, 1]")
            for name in ("unwrapped_action_spec_sha256", "wrapped_action_spec_sha256"):
                try:
                    _sha256(action_audit[name], name)
                except OrdinaryFlightReleaseError:
                    invalid_reasons.append("%s is invalid" % name)
            shape = action_audit["action_vector_shape"]
            action_size = None
            if (
                not isinstance(shape, list)
                or len(shape) != 1
                or isinstance(shape[0], bool)
                or not isinstance(shape[0], int)
                or shape[0] <= 0
            ):
                invalid_reasons.append("action_vector_shape is invalid")
            else:
                action_size = shape[0]
            wing_indices = action_audit["wing_action_indices"]
            if (
                not isinstance(wing_indices, list)
                or len(wing_indices) != 6
                or len(set(wing_indices)) != 6
                or any(isinstance(item, bool) or not isinstance(item, int) for item in wing_indices)
            ):
                invalid_reasons.append("six unique integer wing_action_indices are required")
            elif action_size is not None and any(item < 0 or item >= action_size for item in wing_indices):
                invalid_reasons.append("wing_action_indices lie outside the action vector")
            user_index = action_audit["user_frequency_action_index"]
            if isinstance(user_index, bool) or not isinstance(user_index, int):
                invalid_reasons.append("user_frequency_action_index is invalid")
            elif action_size is not None and (user_index < 0 or user_index >= action_size):
                invalid_reasons.append("user_frequency_action_index lies outside the action vector")
            elif isinstance(wing_indices, list) and user_index in wing_indices:
                invalid_reasons.append("frequency and wing action indices overlap")
            if action_audit["wpg_frequency_mapping_verified"] is not True:
                invalid_reasons.append("WPG frequency mapping was not verified")
            if invalid_reasons:
                _add_blocker(
                    blockers,
                    "action_wrapper_audit_invalid",
                    "action_wrapper",
                    "; ".join(invalid_reasons),
                )

    sequence_audit = audit_root.get("inference_sequence_audit")
    if not isinstance(sequence_audit, Mapping):
        _add_blocker(
            blockers,
            "inference_sequence_audit_missing",
            "inference_sequence",
            "inference sequence audit is absent",
        )
    else:
        missing = _missing_fields(sequence_audit, _AUDIT_FIELDS["inference_sequence_audit"])
        if missing:
            _add_blocker(
                blockers,
                "inference_sequence_audit_incomplete",
                "inference_sequence",
                "missing fields: %s" % ", ".join(missing),
            )
        else:
            invalid_reasons = []
            executed_stage_ids = sequence_audit["executed_stage_ids"]
            if (
                not isinstance(executed_stage_ids, list)
                or tuple(executed_stage_ids) != EXPECTED_INFERENCE_STAGE_IDS
            ):
                invalid_reasons.append("executed stage order differs from the official sequence")
            for name in (
                "observation_precedes_action",
                "one_policy_call_per_control_step",
                "environment_step_receives_wrapped_action",
            ):
                if sequence_audit[name] is not True:
                    invalid_reasons.append("%s is not true" % name)
            if invalid_reasons:
                _add_blocker(
                    blockers,
                    "inference_sequence_audit_invalid",
                    "inference_sequence",
                    "; ".join(invalid_reasons),
                )

    unique_blockers = tuple(sorted(set(blockers)))
    return OrdinaryFlightReleaseReport(
        ready=not unique_blockers,
        blockers=unique_blockers,
        observed_inventory=observed,
    )


__all__ = [
    "CONTRACT_ID",
    "CONTROLLER_REUSE_FILE_ID",
    "CONTROL_TIMESTEP_S",
    "EXPECTED_INFERENCE_STAGE_IDS",
    "EXPECTED_WRAPPER_ORDER",
    "FIGSHARE_DOI",
    "FLIGHT_DATA_FILE_ID",
    "InventoryEntry",
    "ORDINARY_POLICY_FILE_ID",
    "OrdinaryFlightReleaseError",
    "OrdinaryFlightReleaseReport",
    "PHYSICS_TIMESTEP_S",
    "POLICY_ROOT",
    "REFERENCE_PATH",
    "ReleaseBlocker",
    "SAVED_MODEL_PATH",
    "SCHEMA_VERSION",
    "UPSTREAM_COMMIT",
    "VARIABLES_INDEX_PATH",
    "VARIABLES_PATTERN",
    "WPG_BASE_FREQUENCY_HZ",
    "WPG_DISCRETE_FREQUENCY_COUNT",
    "WPG_PATH",
    "WPG_RELATIVE_FREQUENCY_RANGE",
    "canonical_inventory_sha256",
    "load_expected_manifest",
    "scan_staged_tree",
    "validate_expected_manifest",
    "validate_staged_release",
]
