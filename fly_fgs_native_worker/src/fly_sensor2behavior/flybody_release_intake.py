"""Non-downloading intake and attestation for the official FlyBody flight bundle.

The intake hashes a user-supplied extracted tree.  It never downloads data and
never treats an observed digest as an expected digest.  A candidate receipt is
always ``candidate_unreviewed``.  Readiness additionally requires a separate,
reviewed expected receipt with an exact full inventory and explicit inference,
normalization, and runtime metadata.

Even a ready bundle does not change or unblock ``physics.timestep_convergence``;
that remains a benchmark-registry decision outside this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from datetime import datetime
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit


RELEASE_INTAKE_SCHEMA_VERSION = "1.0.0"
CANDIDATE_STATUS = "candidate_unreviewed"
REVIEWED_EXPECTED_STATUS = "reviewed_expected"

FIGSHARE_DOI = "10.25378/janelia.25309105"
FIGSHARE_VERSION = 4
FIGSHARE_SOURCE_URI = "https://doi.org/10.25378/janelia.25309105"
FLIGHT_IMITATION_ARCHIVE_FILE_ID = "51196859"
TRAINED_POLICIES_ARCHIVE_FILE_ID = "44815195"
UPSTREAM_CODE_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
RELEASE_LICENSE = "GPL-3.0+"

WPG_SENTINEL = "datasets_flight-imitation/wing_pattern_fmech.npy"
REFERENCE_SENTINEL = "flight-dataset_saccade-evasion_augmented.hdf5"
POLICY_ROOT = "trained-fly-policies/flight"
REQUIRED_SENTINELS = (WPG_SENTINEL, REFERENCE_SENTINEL)


class ReleaseIntakeError(ValueError):
    """Raised when a bundle or attestation violates the intake contract."""


class BundleRole(str, Enum):
    """Scientific role of a file; supporting files make no runtime claim."""

    WPG = "wpg"
    REFERENCE = "reference"
    POLICY = "policy"
    NORMALIZATION = "normalization"
    INFERENCE = "inference"
    RUNTIME = "runtime"
    SUPPORTING = "supporting"


class NormalizationMode(str, Enum):
    ARTIFACT_FILES = "artifact_files"
    EMBEDDED_SAVED_MODEL = "embedded_saved_model"
    IDENTITY = "identity"


_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAVED_MODEL_DATA_SHARD_PATTERN = re.compile(
    r"^variables/variables\.data-([0-9]{5})-of-([0-9]{5})$"
)
_UTC_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseIntakeError("%s must be a non-empty string" % label)
    if value != value.strip():
        raise ReleaseIntakeError("%s must not contain surrounding whitespace" % label)
    return value


def _identifier(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if not _ID_PATTERN.fullmatch(text):
        raise ReleaseIntakeError("%s must be a lowercase identifier" % label)
    return text


def _version(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if not _VERSION_PATTERN.fullmatch(text):
        raise ReleaseIntakeError("%s must be a semantic version" % label)
    return text


def _sha256(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ReleaseIntakeError("%s must be a lowercase SHA-256 digest" % label)
    return text


def _image_digest(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if not _IMAGE_DIGEST_PATTERN.fullmatch(text):
        raise ReleaseIntakeError("%s must be a sha256:<64 lowercase hex> digest" % label)
    return text


def _uri(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if any(character.isspace() for character in text) or not urlsplit(text).scheme:
        raise ReleaseIntakeError("%s must be an absolute URI without whitespace" % label)
    return text


def _enum(enum_type: Any, value: Any, label: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ReleaseIntakeError("invalid %s: %r" % (label, value)) from exc


def _tuple(values: Iterable[Any], label: str) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ReleaseIntakeError("%s must be a sequence" % label)
    try:
        return tuple(values)
    except TypeError as exc:
        raise ReleaseIntakeError("%s must be a sequence" % label) from exc


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ReleaseIntakeError("%s must be numeric" % label)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReleaseIntakeError("%s must be numeric" % label) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ReleaseIntakeError("%s must be finite and positive" % label)
    return number


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReleaseIntakeError("%s must be a positive integer" % label)
    return value


def _relative_path(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if "\\" in text or any(ord(character) < 32 for character in text):
        raise ReleaseIntakeError("%s must be a printable POSIX path" % label)
    parts = text.split("/")
    if text.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ReleaseIntakeError("%s must be a normalized relative path" % label)
    path = PurePosixPath(text)
    if path.is_absolute():
        raise ReleaseIntakeError("%s must be relative" % label)
    return path.as_posix()


def _unique_paths(values: Iterable[Any], label: str, *, allow_empty: bool = False) -> Tuple[str, ...]:
    paths = tuple(
        _relative_path(value, "%s[%d]" % (label, index))
        for index, value in enumerate(_tuple(values, label))
    )
    if not paths and not allow_empty:
        raise ReleaseIntakeError("%s must not be empty" % label)
    if len(paths) != len(set(paths)):
        raise ReleaseIntakeError("%s contain duplicate paths" % label)
    if len({path.casefold() for path in paths}) != len(paths):
        raise ReleaseIntakeError("%s contain case-colliding paths" % label)
    return tuple(sorted(paths))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseIntakeError("%s must be a JSON object" % label)
    return value


def _require_keys(
    data: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    label: str,
) -> None:
    required_set = set(required)
    optional_set = set(optional)
    actual = set(data)
    missing = sorted(required_set - actual)
    unknown = sorted(actual - required_set - optional_set)
    if missing:
        raise ReleaseIntakeError("%s missing fields: %s" % (label, ", ".join(missing)))
    if unknown:
        raise ReleaseIntakeError("%s has unknown fields: %s" % (label, ", ".join(unknown)))


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseIntakeError("duplicate JSON key %r" % key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ReleaseIntakeError("non-finite JSON constant %r is forbidden" % value)


def _load_json_text(text: str, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseIntakeError("invalid %s JSON: %s" % (label, exc)) from exc
    return _mapping(value, label)


class CanonicalJsonMixin:
    def to_dict(self) -> Dict[str, Any]:
        return _primitive(self)

    def to_json(self, *, indent: Optional[int] = None) -> str:
        if indent is None:
            return _canonical_json(self)
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent, allow_nan=False)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OfficialReleaseSource(CanonicalJsonMixin):
    source_uri: str = FIGSHARE_SOURCE_URI
    doi: str = FIGSHARE_DOI
    version: int = FIGSHARE_VERSION
    flight_imitation_archive_file_id: str = FLIGHT_IMITATION_ARCHIVE_FILE_ID
    trained_policies_archive_file_id: str = TRAINED_POLICIES_ARCHIVE_FILE_ID
    upstream_code_commit: str = UPSTREAM_CODE_COMMIT
    license: str = RELEASE_LICENSE

    def __post_init__(self) -> None:
        expected = {
            "source_uri": FIGSHARE_SOURCE_URI,
            "doi": FIGSHARE_DOI,
            "version": FIGSHARE_VERSION,
            "flight_imitation_archive_file_id": FLIGHT_IMITATION_ARCHIVE_FILE_ID,
            "trained_policies_archive_file_id": TRAINED_POLICIES_ARCHIVE_FILE_ID,
            "upstream_code_commit": UPSTREAM_CODE_COMMIT,
            "license": RELEASE_LICENSE,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ReleaseIntakeError(
                    "official release %s must equal %r" % (name, value)
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialReleaseSource":
        data = _mapping(data, "official source")
        names = tuple(field.name for field in fields(cls))
        _require_keys(data, required=names, label="official source")
        return cls(**dict(data))


@dataclass(frozen=True)
class BundleFile(CanonicalJsonMixin):
    path: str
    bytes: int
    sha256: str
    role: BundleRole

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, "bundle file path"))
        object.__setattr__(self, "bytes", _positive_integer(self.bytes, "bundle file bytes"))
        _sha256(self.sha256, "bundle file sha256")
        object.__setattr__(self, "role", _enum(BundleRole, self.role, "bundle role"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BundleFile":
        data = _mapping(data, "bundle file")
        _require_keys(data, required=("path", "bytes", "sha256", "role"), label="bundle file")
        return cls(**dict(data))


def _validate_file_inventory(files: Iterable[BundleFile]) -> Tuple[BundleFile, ...]:
    records = _tuple(files, "files")
    if not records or not all(isinstance(item, BundleFile) for item in records):
        raise ReleaseIntakeError("files must contain at least one BundleFile")
    paths = tuple(item.path for item in records)
    if paths != tuple(sorted(paths)):
        raise ReleaseIntakeError("file inventory must be sorted by path")
    if len(paths) != len(set(paths)):
        raise ReleaseIntakeError("file inventory contains duplicate paths")
    if len({path.casefold() for path in paths}) != len(paths):
        raise ReleaseIntakeError("file inventory contains case-colliding paths")
    return records


def _discover_saved_model_roots(files: Iterable[BundleFile]) -> Tuple[str, ...]:
    suffix = "/saved_model.pb"
    return tuple(
        sorted(
            item.path[: -len(suffix)]
            for item in files
            if item.path.startswith(POLICY_ROOT + "/") and item.path.endswith(suffix)
        )
    )


def _validate_bundle_shape(
    files: Tuple[BundleFile, ...], saved_model_roots: Tuple[str, ...]
) -> None:
    by_path = {item.path: item for item in files}
    for path, role in (
        (WPG_SENTINEL, BundleRole.WPG),
        (REFERENCE_SENTINEL, BundleRole.REFERENCE),
    ):
        if path not in by_path:
            raise ReleaseIntakeError("required sentinel is missing: %s" % path)
        if by_path[path].role is not role:
            raise ReleaseIntakeError("required sentinel %s must have role %s" % (path, role.value))

    discovered = _discover_saved_model_roots(files)
    if not discovered:
        raise ReleaseIntakeError("trained-fly-policies/flight has no SavedModel inventory")
    if discovered != saved_model_roots:
        raise ReleaseIntakeError("saved_model_roots do not match the complete SavedModel inventory")
    for root in saved_model_roots:
        if root != POLICY_ROOT and not root.startswith(POLICY_ROOT + "/"):
            raise ReleaseIntakeError("SavedModel root lies outside trained-fly-policies/flight")
        model_file = root + "/saved_model.pb"
        index_file = root + "/variables/variables.index"
        data_prefix = root + "/variables/variables.data-"
        if by_path[model_file].role is not BundleRole.POLICY:
            raise ReleaseIntakeError("SavedModel descriptor must have policy role: %s" % model_file)
        if index_file not in by_path:
            raise ReleaseIntakeError("SavedModel variables index is missing: %s" % index_file)
        if by_path[index_file].role is not BundleRole.POLICY:
            raise ReleaseIntakeError("SavedModel variables index must have policy role: %s" % index_file)
        data_files = tuple(path for path in by_path if path.startswith(data_prefix))
        if not data_files:
            raise ReleaseIntakeError("SavedModel variables data are missing under %s" % root)
        if any(by_path[path].role is not BundleRole.POLICY for path in data_files):
            raise ReleaseIntakeError("SavedModel variables data must have policy role under %s" % root)
        shard_records = []
        for path in data_files:
            relative = path[len(root) + 1 :]
            match = _SAVED_MODEL_DATA_SHARD_PATTERN.fullmatch(relative)
            if match is None:
                raise ReleaseIntakeError(
                    "SavedModel variables data has an invalid shard name: %s" % path
                )
            shard_records.append((int(match.group(1)), int(match.group(2))))
        totals = {total for _index, total in shard_records}
        if len(totals) != 1 or 0 in totals:
            raise ReleaseIntakeError(
                "SavedModel variables data shards declare inconsistent totals under %s"
                % root
            )
        total = next(iter(totals))
        indices = {index for index, _total in shard_records}
        if len(shard_records) != total or indices != set(range(total)):
            raise ReleaseIntakeError(
                "SavedModel variables data shards are incomplete under %s" % root
            )


@dataclass(frozen=True)
class CandidateBundleReceipt(CanonicalJsonMixin):
    status: str
    source: OfficialReleaseSource
    required_sentinels: Tuple[str, ...]
    saved_model_roots: Tuple[str, ...]
    files: Tuple[BundleFile, ...]
    schema_version: str = RELEASE_INTAKE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status != CANDIDATE_STATUS:
            raise ReleaseIntakeError("candidate status must be candidate_unreviewed")
        if not isinstance(self.source, OfficialReleaseSource):
            raise ReleaseIntakeError("candidate source must be OfficialReleaseSource")
        if self.schema_version != RELEASE_INTAKE_SCHEMA_VERSION:
            raise ReleaseIntakeError("unsupported candidate schema_version")
        sentinels = _unique_paths(self.required_sentinels, "required_sentinels")
        if sentinels != tuple(sorted(REQUIRED_SENTINELS)):
            raise ReleaseIntakeError("candidate required_sentinels must equal the official sentinel set")
        roots = _unique_paths(self.saved_model_roots, "saved_model_roots")
        records = _validate_file_inventory(self.files)
        _validate_bundle_shape(records, roots)
        object.__setattr__(self, "required_sentinels", sentinels)
        object.__setattr__(self, "saved_model_roots", roots)
        object.__setattr__(self, "files", records)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateBundleReceipt":
        data = _mapping(data, "candidate receipt")
        _require_keys(
            data,
            required=("status", "source", "required_sentinels", "saved_model_roots", "files"),
            optional=("schema_version",),
            label="candidate receipt",
        )
        return cls(
            status=data["status"],
            source=OfficialReleaseSource.from_dict(data["source"]),
            required_sentinels=tuple(data["required_sentinels"]),
            saved_model_roots=tuple(data["saved_model_roots"]),
            files=tuple(BundleFile.from_dict(item) for item in data["files"]),
            schema_version=data.get("schema_version", RELEASE_INTAKE_SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, text: str) -> "CandidateBundleReceipt":
        return cls.from_dict(_load_json_text(text, "candidate receipt"))


@dataclass(frozen=True)
class InferenceMetadata(CanonicalJsonMixin):
    policy_format: str
    wpg_path: str
    reference_dataset_path: str
    policy_saved_model_roots: Tuple[str, ...]
    normalization_mode: NormalizationMode
    normalization_paths: Tuple[str, ...]
    normalization_description: str
    inference_artifact_path: str
    inference_entrypoint_uri: str
    inference_code_sha256: str
    observation_schema_uri: str
    observation_schema_sha256: str
    action_schema_uri: str
    action_schema_sha256: str
    controller_clock_s: float

    def __post_init__(self) -> None:
        if self.policy_format != "tensorflow_saved_model":
            raise ReleaseIntakeError("policy_format must be tensorflow_saved_model")
        if self.wpg_path != WPG_SENTINEL:
            raise ReleaseIntakeError("wpg_path must equal the official WPG sentinel")
        if self.reference_dataset_path != REFERENCE_SENTINEL:
            raise ReleaseIntakeError("reference_dataset_path must equal the official reference sentinel")
        roots = _unique_paths(self.policy_saved_model_roots, "policy_saved_model_roots")
        mode = _enum(NormalizationMode, self.normalization_mode, "normalization mode")
        paths = _unique_paths(self.normalization_paths, "normalization_paths", allow_empty=True)
        if mode is NormalizationMode.IDENTITY and paths:
            raise ReleaseIntakeError("identity normalization cannot declare normalization_paths")
        if mode is not NormalizationMode.IDENTITY and not paths:
            raise ReleaseIntakeError("non-identity normalization requires normalization_paths")
        _nonempty(self.normalization_description, "normalization_description")
        object.__setattr__(
            self,
            "inference_artifact_path",
            _relative_path(self.inference_artifact_path, "inference_artifact_path"),
        )
        _uri(self.inference_entrypoint_uri, "inference_entrypoint_uri")
        _sha256(self.inference_code_sha256, "inference_code_sha256")
        _uri(self.observation_schema_uri, "observation_schema_uri")
        _sha256(self.observation_schema_sha256, "observation_schema_sha256")
        _uri(self.action_schema_uri, "action_schema_uri")
        _sha256(self.action_schema_sha256, "action_schema_sha256")
        object.__setattr__(self, "controller_clock_s", _finite_positive(self.controller_clock_s, "controller_clock_s"))
        object.__setattr__(self, "policy_saved_model_roots", roots)
        object.__setattr__(self, "normalization_mode", mode)
        object.__setattr__(self, "normalization_paths", paths)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InferenceMetadata":
        data = _mapping(data, "inference metadata")
        names = tuple(field.name for field in fields(cls))
        _require_keys(data, required=names, label="inference metadata")
        values = dict(data)
        values["policy_saved_model_roots"] = tuple(values["policy_saved_model_roots"])
        values["normalization_paths"] = tuple(values["normalization_paths"])
        return cls(**values)


@dataclass(frozen=True)
class RuntimeMetadata(CanonicalJsonMixin):
    upstream_code_commit: str
    python_version: str
    flygym_version: str
    mujoco_version: str
    tensorflow_version: str
    platform_tag: str
    dependency_lock_path: str
    dependency_lock_uri: str
    dependency_lock_sha256: str
    container_image_digest: str
    physics_timestep_s: float
    determinism_notes: str

    def __post_init__(self) -> None:
        if self.upstream_code_commit != UPSTREAM_CODE_COMMIT:
            raise ReleaseIntakeError("runtime upstream_code_commit must equal the reviewed source commit")
        for value, label in (
            (self.python_version, "python_version"),
            (self.flygym_version, "flygym_version"),
            (self.mujoco_version, "mujoco_version"),
            (self.tensorflow_version, "tensorflow_version"),
            (self.platform_tag, "platform_tag"),
            (self.determinism_notes, "determinism_notes"),
        ):
            _nonempty(value, label)
        object.__setattr__(
            self,
            "dependency_lock_path",
            _relative_path(self.dependency_lock_path, "dependency_lock_path"),
        )
        _uri(self.dependency_lock_uri, "dependency_lock_uri")
        _sha256(self.dependency_lock_sha256, "dependency_lock_sha256")
        _image_digest(self.container_image_digest, "container_image_digest")
        object.__setattr__(self, "physics_timestep_s", _finite_positive(self.physics_timestep_s, "physics_timestep_s"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeMetadata":
        data = _mapping(data, "runtime metadata")
        names = tuple(field.name for field in fields(cls))
        _require_keys(data, required=names, label="runtime metadata")
        return cls(**dict(data))


@dataclass(frozen=True)
class ReviewAttestation(CanonicalJsonMixin):
    review_id: str
    reviewer: str
    reviewed_at_utc: str
    decision_uri: str
    decision_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.review_id, "review_id")
        _nonempty(self.reviewer, "reviewer")
        if not isinstance(self.reviewed_at_utc, str) or not _UTC_PATTERN.fullmatch(self.reviewed_at_utc):
            raise ReleaseIntakeError("reviewed_at_utc must be an RFC 3339 UTC timestamp")
        try:
            datetime.fromisoformat(self.reviewed_at_utc[:-1] + "+00:00")
        except ValueError as exc:
            raise ReleaseIntakeError(
                "reviewed_at_utc must be a valid RFC 3339 UTC timestamp"
            ) from exc
        _uri(self.decision_uri, "decision_uri")
        _sha256(self.decision_sha256, "decision_sha256")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewAttestation":
        data = _mapping(data, "review attestation")
        names = tuple(field.name for field in fields(cls))
        _require_keys(data, required=names, label="review attestation")
        return cls(**dict(data))


@dataclass(frozen=True)
class ReviewedExpectedReceipt(CanonicalJsonMixin):
    expected_id: str
    expected_version: str
    status: str
    source: OfficialReleaseSource
    required_sentinels: Tuple[str, ...]
    saved_model_roots: Tuple[str, ...]
    files: Tuple[BundleFile, ...]
    inference: InferenceMetadata
    runtime: RuntimeMetadata
    review: ReviewAttestation
    schema_version: str = RELEASE_INTAKE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.expected_id, "expected_id")
        _version(self.expected_version, "expected_version")
        if self.status != REVIEWED_EXPECTED_STATUS:
            raise ReleaseIntakeError("expected receipt status must be reviewed_expected")
        if not isinstance(self.source, OfficialReleaseSource):
            raise ReleaseIntakeError("expected source must be OfficialReleaseSource")
        if self.schema_version != RELEASE_INTAKE_SCHEMA_VERSION:
            raise ReleaseIntakeError("unsupported expected receipt schema_version")
        if not isinstance(self.inference, InferenceMetadata):
            raise ReleaseIntakeError("inference must be InferenceMetadata")
        if not isinstance(self.runtime, RuntimeMetadata):
            raise ReleaseIntakeError("runtime must be RuntimeMetadata")
        if not isinstance(self.review, ReviewAttestation):
            raise ReleaseIntakeError("review must be ReviewAttestation")
        sentinels = _unique_paths(self.required_sentinels, "required_sentinels")
        if sentinels != tuple(sorted(REQUIRED_SENTINELS)):
            raise ReleaseIntakeError("expected required_sentinels must equal the official sentinel set")
        roots = _unique_paths(self.saved_model_roots, "saved_model_roots")
        records = _validate_file_inventory(self.files)
        _validate_bundle_shape(records, roots)
        if roots != self.inference.policy_saved_model_roots:
            raise ReleaseIntakeError("inference policy roots must equal the expected SavedModel inventory")
        by_path = {item.path: item for item in records}
        for path in self.inference.normalization_paths:
            if path not in by_path:
                raise ReleaseIntakeError("normalization path is absent from expected inventory: %s" % path)
            allowed_roles = (
                (BundleRole.NORMALIZATION,)
                if self.inference.normalization_mode is NormalizationMode.ARTIFACT_FILES
                else (BundleRole.POLICY, BundleRole.NORMALIZATION)
            )
            if by_path[path].role not in allowed_roles:
                raise ReleaseIntakeError("normalization path has an incompatible file role: %s" % path)
        inference_path = self.inference.inference_artifact_path
        if inference_path not in by_path:
            raise ReleaseIntakeError("inference artifact is absent from expected inventory")
        if by_path[inference_path].role is not BundleRole.INFERENCE:
            raise ReleaseIntakeError("inference artifact must have inference role")
        if by_path[inference_path].sha256 != self.inference.inference_code_sha256:
            raise ReleaseIntakeError("inference artifact hash does not match inference metadata")
        runtime_path = self.runtime.dependency_lock_path
        if runtime_path not in by_path:
            raise ReleaseIntakeError("runtime dependency lock is absent from expected inventory")
        if by_path[runtime_path].role is not BundleRole.RUNTIME:
            raise ReleaseIntakeError("runtime dependency lock must have runtime role")
        if by_path[runtime_path].sha256 != self.runtime.dependency_lock_sha256:
            raise ReleaseIntakeError("runtime dependency lock hash does not match runtime metadata")
        object.__setattr__(self, "required_sentinels", sentinels)
        object.__setattr__(self, "saved_model_roots", roots)
        object.__setattr__(self, "files", records)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewedExpectedReceipt":
        data = _mapping(data, "reviewed expected receipt")
        _require_keys(
            data,
            required=(
                "expected_id", "expected_version", "status", "source",
                "required_sentinels", "saved_model_roots", "files",
                "inference", "runtime", "review",
            ),
            optional=("schema_version",),
            label="reviewed expected receipt",
        )
        return cls(
            expected_id=data["expected_id"],
            expected_version=data["expected_version"],
            status=data["status"],
            source=OfficialReleaseSource.from_dict(data["source"]),
            required_sentinels=tuple(data["required_sentinels"]),
            saved_model_roots=tuple(data["saved_model_roots"]),
            files=tuple(BundleFile.from_dict(item) for item in data["files"]),
            inference=InferenceMetadata.from_dict(data["inference"]),
            runtime=RuntimeMetadata.from_dict(data["runtime"]),
            review=ReviewAttestation.from_dict(data["review"]),
            schema_version=data.get("schema_version", RELEASE_INTAKE_SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, text: str) -> "ReviewedExpectedReceipt":
        return cls.from_dict(_load_json_text(text, "reviewed expected receipt"))


@dataclass(frozen=True)
class ReleaseReadinessReport(CanonicalJsonMixin):
    ready: bool
    status: str
    candidate_sha256: str
    expected_sha256: Optional[str]
    reason_codes: Tuple[str, ...]
    physics_timestep_convergence_unblocked: bool
    note: str
    schema_version: str = RELEASE_INTAKE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.ready, bool):
            raise ReleaseIntakeError("ready must be boolean")
        expected_status = "ready_for_offline_reproduction" if self.ready else "not_ready"
        if self.status != expected_status:
            raise ReleaseIntakeError("readiness status is inconsistent with ready")
        _sha256(self.candidate_sha256, "candidate_sha256")
        if self.expected_sha256 is not None:
            _sha256(self.expected_sha256, "expected_sha256")
        reasons = tuple(_identifier(item, "reason_code") for item in _tuple(self.reason_codes, "reason_codes"))
        if len(reasons) != len(set(reasons)):
            raise ReleaseIntakeError("reason_codes must be unique")
        if self.ready and reasons:
            raise ReleaseIntakeError("a ready report cannot contain reason codes")
        if not self.ready and not reasons:
            raise ReleaseIntakeError("a not-ready report requires reason codes")
        if self.physics_timestep_convergence_unblocked is not False:
            raise ReleaseIntakeError("this intake cannot unblock physics.timestep_convergence")
        _nonempty(self.note, "readiness note")
        if self.schema_version != RELEASE_INTAKE_SCHEMA_VERSION:
            raise ReleaseIntakeError("unsupported readiness schema_version")
        object.__setattr__(self, "reason_codes", tuple(sorted(reasons)))


def _default_role(path: str) -> BundleRole:
    if path == WPG_SENTINEL:
        return BundleRole.WPG
    if path == REFERENCE_SENTINEL:
        return BundleRole.REFERENCE
    if path.startswith(POLICY_ROOT + "/"):
        return BundleRole.POLICY
    return BundleRole.SUPPORTING


def _hash_open_file(path: Path) -> Tuple[int, str, Tuple[int, int, int, int]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise ReleaseIntakeError("cannot open bundle file %s: %s" % (path, exc)) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReleaseIntakeError("bundle member is not a regular file: %s" % path)
        if before.st_size <= 0:
            raise ReleaseIntakeError("bundle file is empty: %s" % path)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity != after_identity:
            raise ReleaseIntakeError("bundle file changed while hashing: %s" % path)
        return before.st_size, digest.hexdigest(), identity
    finally:
        os.close(descriptor)


def _scan_regular_files(root: Path) -> Tuple[Tuple[str, int, str, Tuple[int, int]], ...]:
    if root.is_symlink():
        raise ReleaseIntakeError("bundle root must not be a symlink")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ReleaseIntakeError("bundle root does not exist or cannot be resolved") from exc
    if not resolved_root.is_dir():
        raise ReleaseIntakeError("bundle root must be a directory")
    observed = []
    inode_paths: Dict[Tuple[int, int], str] = {}
    casefold_paths: Dict[str, str] = {}

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(str(directory)), key=lambda entry: entry.name)
        except OSError as exc:
            raise ReleaseIntakeError("cannot read bundle directory %s: %s" % (directory, exc)) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = _relative_path(path.relative_to(resolved_root).as_posix(), "observed path")
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ReleaseIntakeError("cannot stat bundle member %s: %s" % (relative, exc)) from exc
            if stat.S_ISLNK(info.st_mode):
                raise ReleaseIntakeError("symlinks are forbidden in the bundle: %s" % relative)
            if stat.S_ISDIR(info.st_mode):
                visit(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ReleaseIntakeError("non-regular bundle member is forbidden: %s" % relative)
            size, digest, identity = _hash_open_file(path)
            inode = (identity[0], identity[1])
            if inode in inode_paths:
                raise ReleaseIntakeError(
                    "duplicate hard-linked file %s also appears as %s" % (relative, inode_paths[inode])
                )
            folded = relative.casefold()
            if folded in casefold_paths:
                raise ReleaseIntakeError(
                    "case-colliding bundle paths %s and %s" % (relative, casefold_paths[folded])
                )
            inode_paths[inode] = relative
            casefold_paths[folded] = relative
            observed.append((relative, size, digest, inode))

    visit(resolved_root)
    if not observed:
        raise ReleaseIntakeError("bundle contains no regular files")
    return tuple(sorted(observed, key=lambda item: item[0]))


def load_role_overrides(path: Path) -> Dict[str, BundleRole]:
    try:
        text = Path(path).read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseIntakeError("cannot read role override JSON: %s" % exc) from exc
    data = _load_json_text(text, "role overrides")
    result: Dict[str, BundleRole] = {}
    for raw_path, raw_role in data.items():
        normalized = _relative_path(raw_path, "role override path")
        if normalized in result:
            raise ReleaseIntakeError("duplicate normalized role override path")
        result[normalized] = _enum(BundleRole, raw_role, "role override")
    return result


def inventory_release_candidate(
    bundle_root: Path,
    *,
    role_overrides: Optional[Mapping[str, BundleRole]] = None,
) -> CandidateBundleReceipt:
    """Hash an already extracted bundle without making any network request."""

    normalized_overrides: Dict[str, BundleRole] = {}
    for raw_path, raw_role in (role_overrides or {}).items():
        path = _relative_path(raw_path, "role override path")
        if path in normalized_overrides:
            raise ReleaseIntakeError("duplicate normalized role override path")
        normalized_overrides[path] = _enum(BundleRole, raw_role, "role override")

    observed = _scan_regular_files(Path(bundle_root))
    observed_paths = {item[0] for item in observed}
    unknown_overrides = sorted(set(normalized_overrides) - observed_paths)
    if unknown_overrides:
        raise ReleaseIntakeError("role overrides reference unknown files: %s" % ", ".join(unknown_overrides))
    files = tuple(
        BundleFile(
            path=path,
            bytes=size,
            sha256=digest,
            role=normalized_overrides.get(path, _default_role(path)),
        )
        for path, size, digest, _inode in observed
    )
    roots = _discover_saved_model_roots(files)
    return CandidateBundleReceipt(
        status=CANDIDATE_STATUS,
        source=OfficialReleaseSource(),
        required_sentinels=tuple(sorted(REQUIRED_SENTINELS)),
        saved_model_roots=roots,
        files=files,
    )


_READINESS_NOTE = (
    "This attests local bundle readiness for offline reproduction only. It does not "
    "modify, pass, or unblock physics.timestep_convergence."
)


def verify_release_readiness(
    candidate: CandidateBundleReceipt,
    bundle_root: Path,
    *,
    expected: Optional[ReviewedExpectedReceipt] = None,
    role_overrides: Optional[Mapping[str, BundleRole]] = None,
) -> ReleaseReadinessReport:
    """Reinventory local files and compare them with a separate reviewed receipt."""

    if not isinstance(candidate, CandidateBundleReceipt):
        raise ReleaseIntakeError("candidate must be CandidateBundleReceipt")
    reasons = []
    expected_digest = None if expected is None else expected.content_sha256
    if expected is None:
        reasons.append("reviewed_expected_receipt_missing")
    elif not isinstance(expected, ReviewedExpectedReceipt):
        raise ReleaseIntakeError("expected must be ReviewedExpectedReceipt")

    try:
        observed = inventory_release_candidate(bundle_root, role_overrides=role_overrides)
    except ReleaseIntakeError:
        observed = None
        reasons.append("local_inventory_invalid")
    if observed is not None and observed.to_dict() != candidate.to_dict():
        reasons.append("local_candidate_mismatch")
    if expected is not None:
        if candidate.source != expected.source:
            reasons.append("source_mismatch")
        if candidate.required_sentinels != expected.required_sentinels:
            reasons.append("required_sentinels_mismatch")
        if candidate.saved_model_roots != expected.saved_model_roots:
            reasons.append("saved_model_inventory_mismatch")
        if candidate.files != expected.files:
            reasons.append("full_file_inventory_mismatch")
    reasons = sorted(set(reasons))
    ready = not reasons
    return ReleaseReadinessReport(
        ready=ready,
        status="ready_for_offline_reproduction" if ready else "not_ready",
        candidate_sha256=candidate.content_sha256,
        expected_sha256=expected_digest,
        reason_codes=tuple(reasons),
        physics_timestep_convergence_unblocked=False,
        note=_READINESS_NOTE,
    )


def _write_exclusive_json(path: Path, value: CanonicalJsonMixin) -> None:
    output = Path(path)
    if not output.parent.is_dir():
        raise ReleaseIntakeError("output parent directory does not exist")
    payload = (value.to_json() + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % output.name,
        suffix=".tmp",
        dir=str(output.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(output))
        except FileExistsError as exc:
            raise ReleaseIntakeError("output already exists: %s" % output) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_candidate(path: Path) -> CandidateBundleReceipt:
    try:
        text = Path(path).read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseIntakeError("cannot read candidate receipt: %s" % exc) from exc
    return CandidateBundleReceipt.from_json(text)


def _read_expected(path: Path) -> ReviewedExpectedReceipt:
    try:
        text = Path(path).read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseIntakeError("cannot read reviewed expected receipt: %s" % exc) from exc
    return ReviewedExpectedReceipt.from_json(text)


def _output_outside_root(output: Path, root: Path) -> None:
    resolved_root = root.resolve(strict=True)
    resolved_output_parent = output.parent.resolve(strict=True)
    candidate = resolved_output_parent / output.name
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return
    raise ReleaseIntakeError("receipt output must be outside the inventoried bundle root")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    intake = subparsers.add_parser("intake", help="inventory an already extracted local bundle")
    intake.add_argument("--root", required=True, type=Path)
    intake.add_argument("--output", required=True, type=Path)
    intake.add_argument("--roles", type=Path, help="optional exact path-to-role JSON mapping")
    verify = subparsers.add_parser("verify", help="compare a candidate with a reviewed expected receipt")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--candidate", required=True, type=Path)
    verify.add_argument("--expected", required=True, type=Path)
    verify.add_argument("--roles", type=Path)
    verify.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        overrides = None if args.roles is None else load_role_overrides(args.roles)
        if args.command == "intake":
            _output_outside_root(args.output, args.root)
            candidate = inventory_release_candidate(args.root, role_overrides=overrides)
            _write_exclusive_json(args.output, candidate)
            print(json.dumps({"status": candidate.status, "sha256": candidate.content_sha256, "files": len(candidate.files)}, sort_keys=True))
            return 0
        candidate = _read_candidate(args.candidate)
        expected = _read_expected(args.expected)
        if args.output is not None:
            _output_outside_root(args.output, args.root)
        report = verify_release_readiness(
            candidate,
            args.root,
            expected=expected,
            role_overrides=overrides,
        )
        if args.output is None:
            print(report.to_json())
        else:
            _write_exclusive_json(args.output, report)
        return 0 if report.ready else 2
    except (OSError, ReleaseIntakeError) as exc:
        print("flybody release intake error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_STATUS",
    "FIGSHARE_DOI",
    "FIGSHARE_SOURCE_URI",
    "FIGSHARE_VERSION",
    "FLIGHT_IMITATION_ARCHIVE_FILE_ID",
    "POLICY_ROOT",
    "REFERENCE_SENTINEL",
    "RELEASE_LICENSE",
    "REVIEWED_EXPECTED_STATUS",
    "TRAINED_POLICIES_ARCHIVE_FILE_ID",
    "UPSTREAM_CODE_COMMIT",
    "WPG_SENTINEL",
    "BundleFile",
    "BundleRole",
    "CandidateBundleReceipt",
    "InferenceMetadata",
    "NormalizationMode",
    "OfficialReleaseSource",
    "ReleaseIntakeError",
    "ReleaseReadinessReport",
    "ReviewAttestation",
    "ReviewedExpectedReceipt",
    "RuntimeMetadata",
    "inventory_release_candidate",
    "load_role_overrides",
    "verify_release_readiness",
]
