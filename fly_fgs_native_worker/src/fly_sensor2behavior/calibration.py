"""Immutable, license-aware contracts for calibration and held-out evidence.

This module records what a completed calibration used without deciding whether
any scientific benchmark passes.  It is deliberately independent of the
benchmark registry: creating a valid bundle cannot unblock a registered gate.

All file receipts use exact byte counts and lowercase SHA-256 digests.  Trial
splits are permanent, exhaustive, and grouped by both biological individual
and recording session so calibration data cannot leak into validation or
held-out evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit


CALIBRATION_CONTRACT_SCHEMA_VERSION = "1.0.0"


class CalibrationContractError(ValueError):
    """Raised when calibration evidence is incomplete, ambiguous, or tampered."""


class RedistributionStatus(str, Enum):
    """Whether an artifact may be redistributed with the simulator."""

    REDISTRIBUTABLE = "redistributable"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class CalibrationSplit(str, Enum):
    """Permanent trial assignment; the three partitions are non-fungible."""

    CALIBRATION = "calibration"
    VALIDATION = "validation"
    HELD_OUT = "held_out"


_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationContractError("%s must be a non-empty string" % label)
    if value != value.strip():
        raise CalibrationContractError("%s must not contain surrounding whitespace" % label)
    return value


def _identifier(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if not _ID_PATTERN.fullmatch(text):
        raise CalibrationContractError(
            "%s must contain lowercase letters, numbers, dots, underscores, or hyphens"
            % label
        )
    return text


def _version(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if not _VERSION_PATTERN.fullmatch(text):
        raise CalibrationContractError("%s must be a semantic version" % label)
    return text


def _sha256(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise CalibrationContractError(
            "%s must be a lowercase 64-character SHA-256 digest" % label
        )
    return text


def _uri(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if any(character.isspace() for character in text):
        raise CalibrationContractError("%s must not contain whitespace" % label)
    scheme = urlsplit(text).scheme
    if not scheme or not _URI_SCHEME_PATTERN.fullmatch(scheme):
        raise CalibrationContractError("%s must be an absolute URI" % label)
    return text


def _enum(enum_type: Any, value: Any, label: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationContractError("invalid %s: %r" % (label, value)) from exc


def _tuple(values: Iterable[Any], label: str) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise CalibrationContractError("%s must be a sequence, not a string" % label)
    try:
        return tuple(values)
    except TypeError as exc:
        raise CalibrationContractError("%s must be a sequence" % label) from exc


def _unique_ids(values: Iterable[Any], label: str, *, nonempty: bool = True) -> Tuple[str, ...]:
    result = tuple(
        _identifier(value, "%s[%d]" % (label, index))
        for index, value in enumerate(_tuple(values, label))
    )
    if nonempty and not result:
        raise CalibrationContractError("%s must not be empty" % label)
    if len(result) != len(set(result)):
        raise CalibrationContractError("%s must be unique" % label)
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationContractError("%s must be an integer" % label)
    if value < 0:
        raise CalibrationContractError("%s must be non-negative" % label)
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CalibrationContractError("%s must be numeric" % label)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationContractError("%s must be numeric" % label) from exc
    if not math.isfinite(result):
        raise CalibrationContractError("%s must be finite" % label)
    return result


def _relative_file_path(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if "\\" in text:
        raise CalibrationContractError("%s must use POSIX separators" % label)
    parts = text.split("/")
    if text.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise CalibrationContractError(
            "%s must be a normalized relative file path without traversal" % label
        )
    path = PurePosixPath(text)
    if path.is_absolute():
        raise CalibrationContractError("%s must be relative" % label)
    return path.as_posix()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationContractError("%s must be a JSON object" % label)
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
        raise CalibrationContractError(
            "%s is missing required fields: %s" % (label, ", ".join(missing))
        )
    if unknown:
        raise CalibrationContractError(
            "%s contains unknown fields: %s" % (label, ", ".join(unknown))
        )


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
    return json.dumps(
        _primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationContractError("duplicate JSON key %r" % key)
        result[key] = value
    return result


class CanonicalJsonMixin:
    """Canonical JSON and a content digest for immutable contract records."""

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
class ArtifactFile(CanonicalJsonMixin):
    """A local verification target with an independently sourced URI receipt."""

    artifact_id: str
    uri: str
    relative_path: str
    sha256: str
    bytes: int
    media_type: str

    def __post_init__(self) -> None:
        _identifier(self.artifact_id, "artifact_id")
        _uri(self.uri, "artifact uri")
        object.__setattr__(
            self,
            "relative_path",
            _relative_file_path(self.relative_path, "artifact relative_path"),
        )
        _sha256(self.sha256, "artifact sha256")
        object.__setattr__(self, "bytes", _nonnegative_integer(self.bytes, "artifact bytes"))
        _nonempty(self.media_type, "artifact media_type")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactFile":
        data = _mapping(data, "artifact")
        _require_keys(
            data,
            required=("artifact_id", "uri", "relative_path", "sha256", "bytes", "media_type"),
            label="artifact",
        )
        return cls(**dict(data))


@dataclass(frozen=True)
class LicenseRecord(CanonicalJsonMixin):
    license_name: str
    license_uri: str
    redistribution: RedistributionStatus
    access_notes: Optional[str] = None

    def __post_init__(self) -> None:
        _nonempty(self.license_name, "license_name")
        _uri(self.license_uri, "license_uri")
        object.__setattr__(
            self,
            "redistribution",
            _enum(RedistributionStatus, self.redistribution, "redistribution status"),
        )
        if self.access_notes is not None:
            _nonempty(self.access_notes, "access_notes")
        if self.redistribution is not RedistributionStatus.REDISTRIBUTABLE and not self.access_notes:
            raise CalibrationContractError(
                "restricted or unknown licenses require access_notes"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LicenseRecord":
        data = _mapping(data, "license")
        _require_keys(
            data,
            required=("license_name", "license_uri", "redistribution"),
            optional=("access_notes",),
            label="license",
        )
        return cls(
            license_name=data["license_name"],
            license_uri=data["license_uri"],
            redistribution=data["redistribution"],
            access_notes=data.get("access_notes"),
        )


@dataclass(frozen=True)
class LicensedSource(CanonicalJsonMixin):
    source_id: str
    source_uri: str
    citation: str
    license: LicenseRecord
    artifact_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.source_id, "source_id")
        _uri(self.source_uri, "source_uri")
        _nonempty(self.citation, "citation")
        if not isinstance(self.license, LicenseRecord):
            raise CalibrationContractError("license must be a LicenseRecord")
        object.__setattr__(
            self, "artifact_ids", _unique_ids(self.artifact_ids, "source artifact_ids")
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LicensedSource":
        data = _mapping(data, "source")
        _require_keys(
            data,
            required=("source_id", "source_uri", "citation", "license", "artifact_ids"),
            label="source",
        )
        return cls(
            source_id=data["source_id"],
            source_uri=data["source_uri"],
            citation=data["citation"],
            license=LicenseRecord.from_dict(data["license"]),
            artifact_ids=tuple(data["artifact_ids"]),
        )


@dataclass(frozen=True)
class TrialRecord(CanonicalJsonMixin):
    trial_id: str
    individual_id: str
    session_id: str
    source_id: str
    artifact_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.trial_id, "trial_id")
        _identifier(self.individual_id, "individual_id")
        _identifier(self.session_id, "session_id")
        _identifier(self.source_id, "trial source_id")
        object.__setattr__(
            self, "artifact_ids", _unique_ids(self.artifact_ids, "trial artifact_ids")
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrialRecord":
        data = _mapping(data, "trial")
        _require_keys(
            data,
            required=("trial_id", "individual_id", "session_id", "source_id", "artifact_ids"),
            label="trial",
        )
        return cls(
            trial_id=data["trial_id"],
            individual_id=data["individual_id"],
            session_id=data["session_id"],
            source_id=data["source_id"],
            artifact_ids=tuple(data["artifact_ids"]),
        )


@dataclass(frozen=True)
class TrialAssignment(CanonicalJsonMixin):
    trial_id: str
    split: CalibrationSplit

    def __post_init__(self) -> None:
        _identifier(self.trial_id, "assigned trial_id")
        object.__setattr__(self, "split", _enum(CalibrationSplit, self.split, "trial split"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrialAssignment":
        data = _mapping(data, "trial assignment")
        _require_keys(data, required=("trial_id", "split"), label="trial assignment")
        return cls(trial_id=data["trial_id"], split=data["split"])


@dataclass(frozen=True)
class PermanentSplitPlan(CanonicalJsonMixin):
    split_id: str
    split_version: str
    assignments: Tuple[TrialAssignment, ...]

    def __post_init__(self) -> None:
        _identifier(self.split_id, "split_id")
        _version(self.split_version, "split_version")
        assignments = _tuple(self.assignments, "assignments")
        if not assignments:
            raise CalibrationContractError("assignments must not be empty")
        if not all(isinstance(item, TrialAssignment) for item in assignments):
            raise CalibrationContractError("assignments must contain TrialAssignment records")
        trial_ids = tuple(item.trial_id for item in assignments)
        if len(trial_ids) != len(set(trial_ids)):
            raise CalibrationContractError("split assignments contain duplicate trials")
        present = {item.split for item in assignments}
        missing = set(CalibrationSplit) - present
        if missing:
            raise CalibrationContractError(
                "split assignments must include calibration, validation, and held_out"
            )
        object.__setattr__(self, "assignments", assignments)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PermanentSplitPlan":
        data = _mapping(data, "split plan")
        _require_keys(
            data,
            required=("split_id", "split_version", "assignments"),
            label="split plan",
        )
        return cls(
            split_id=data["split_id"],
            split_version=data["split_version"],
            assignments=tuple(TrialAssignment.from_dict(item) for item in data["assignments"]),
        )


@dataclass(frozen=True)
class ProtocolReference(CanonicalJsonMixin):
    protocol_id: str
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        _identifier(self.protocol_id, "protocol_id")
        _uri(self.uri, "protocol uri")
        _sha256(self.sha256, "protocol sha256")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProtocolReference":
        data = _mapping(data, "protocol")
        _require_keys(data, required=("protocol_id", "uri", "sha256"), label="protocol")
        return cls(**dict(data))


@dataclass(frozen=True)
class MetricProtocol(CanonicalJsonMixin):
    metric_id: str
    protocol_uri: str
    protocol_sha256: str
    formula: str
    aggregation: str
    uncertainty: str
    unit: str

    def __post_init__(self) -> None:
        _identifier(self.metric_id, "metric_id")
        _uri(self.protocol_uri, "metric protocol_uri")
        _sha256(self.protocol_sha256, "metric protocol_sha256")
        _nonempty(self.formula, "metric formula")
        _nonempty(self.aggregation, "metric aggregation")
        _nonempty(self.uncertainty, "metric uncertainty")
        _nonempty(self.unit, "metric unit")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricProtocol":
        data = _mapping(data, "metric protocol")
        _require_keys(
            data,
            required=(
                "metric_id",
                "protocol_uri",
                "protocol_sha256",
                "formula",
                "aggregation",
                "uncertainty",
                "unit",
            ),
            label="metric protocol",
        )
        return cls(**dict(data))


@dataclass(frozen=True)
class FitDiagnostic(CanonicalJsonMixin):
    diagnostic_id: str
    value: float
    unit: str
    interpretation: str

    def __post_init__(self) -> None:
        _identifier(self.diagnostic_id, "diagnostic_id")
        object.__setattr__(self, "value", _finite(self.value, "diagnostic value"))
        _nonempty(self.unit, "diagnostic unit")
        _nonempty(self.interpretation, "diagnostic interpretation")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FitDiagnostic":
        data = _mapping(data, "fit diagnostic")
        _require_keys(
            data,
            required=("diagnostic_id", "value", "unit", "interpretation"),
            label="fit diagnostic",
        )
        return cls(**dict(data))


@dataclass(frozen=True)
class FitProvenance(CanonicalJsonMixin):
    fit_id: str
    code_uri: str
    code_sha256: str
    objective: str
    likelihood: str
    metric_ids: Tuple[str, ...]
    calibration_trial_ids: Tuple[str, ...]
    parameter_posterior_artifact_id: str
    diagnostics_artifact_id: str
    diagnostics: Tuple[FitDiagnostic, ...]

    def __post_init__(self) -> None:
        _identifier(self.fit_id, "fit_id")
        _uri(self.code_uri, "fit code_uri")
        _sha256(self.code_sha256, "fit code_sha256")
        _nonempty(self.objective, "fit objective")
        _nonempty(self.likelihood, "fit likelihood")
        object.__setattr__(self, "metric_ids", _unique_ids(self.metric_ids, "fit metric_ids"))
        object.__setattr__(
            self,
            "calibration_trial_ids",
            _unique_ids(self.calibration_trial_ids, "fit calibration_trial_ids"),
        )
        _identifier(
            self.parameter_posterior_artifact_id,
            "parameter_posterior_artifact_id",
        )
        _identifier(self.diagnostics_artifact_id, "diagnostics_artifact_id")
        if self.parameter_posterior_artifact_id == self.diagnostics_artifact_id:
            raise CalibrationContractError(
                "parameter posterior and diagnostics must use distinct artifacts"
            )
        diagnostics = _tuple(self.diagnostics, "fit diagnostics")
        if not diagnostics or not all(isinstance(item, FitDiagnostic) for item in diagnostics):
            raise CalibrationContractError(
                "fit diagnostics must contain at least one FitDiagnostic"
            )
        diagnostic_ids = tuple(item.diagnostic_id for item in diagnostics)
        if len(diagnostic_ids) != len(set(diagnostic_ids)):
            raise CalibrationContractError("fit diagnostic IDs must be unique")
        object.__setattr__(self, "diagnostics", diagnostics)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FitProvenance":
        data = _mapping(data, "fit provenance")
        _require_keys(
            data,
            required=(
                "fit_id",
                "code_uri",
                "code_sha256",
                "objective",
                "likelihood",
                "metric_ids",
                "calibration_trial_ids",
                "parameter_posterior_artifact_id",
                "diagnostics_artifact_id",
                "diagnostics",
            ),
            label="fit provenance",
        )
        return cls(
            fit_id=data["fit_id"],
            code_uri=data["code_uri"],
            code_sha256=data["code_sha256"],
            objective=data["objective"],
            likelihood=data["likelihood"],
            metric_ids=tuple(data["metric_ids"]),
            calibration_trial_ids=tuple(data["calibration_trial_ids"]),
            parameter_posterior_artifact_id=data["parameter_posterior_artifact_id"],
            diagnostics_artifact_id=data["diagnostics_artifact_id"],
            diagnostics=tuple(FitDiagnostic.from_dict(item) for item in data["diagnostics"]),
        )


@dataclass(frozen=True)
class CalibrationEvidenceContract(CanonicalJsonMixin):
    """A complete immutable receipt for a calibration and its reserved trials."""

    contract_id: str
    contract_version: str
    sources: Tuple[LicensedSource, ...]
    artifacts: Tuple[ArtifactFile, ...]
    trials: Tuple[TrialRecord, ...]
    split_plan: PermanentSplitPlan
    preprocessing_protocols: Tuple[ProtocolReference, ...]
    metric_protocols: Tuple[MetricProtocol, ...]
    fit: FitProvenance
    schema_version: str = CALIBRATION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.contract_id, "contract_id")
        _version(self.contract_version, "contract_version")
        if self.schema_version != CALIBRATION_CONTRACT_SCHEMA_VERSION:
            raise CalibrationContractError(
                "unsupported calibration schema_version %r; expected %s"
                % (self.schema_version, CALIBRATION_CONTRACT_SCHEMA_VERSION)
            )

        sources = _tuple(self.sources, "sources")
        artifacts = _tuple(self.artifacts, "artifacts")
        trials = _tuple(self.trials, "trials")
        preprocessing = _tuple(self.preprocessing_protocols, "preprocessing_protocols")
        metrics = _tuple(self.metric_protocols, "metric_protocols")
        for label, values, expected_type in (
            ("sources", sources, LicensedSource),
            ("artifacts", artifacts, ArtifactFile),
            ("trials", trials, TrialRecord),
            ("preprocessing_protocols", preprocessing, ProtocolReference),
            ("metric_protocols", metrics, MetricProtocol),
        ):
            if not values:
                raise CalibrationContractError("%s must not be empty" % label)
            if not all(isinstance(item, expected_type) for item in values):
                raise CalibrationContractError(
                    "%s must contain %s records" % (label, expected_type.__name__)
                )
        if not isinstance(self.split_plan, PermanentSplitPlan):
            raise CalibrationContractError("split_plan must be a PermanentSplitPlan")
        if not isinstance(self.fit, FitProvenance):
            raise CalibrationContractError("fit must be FitProvenance")

        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "preprocessing_protocols", preprocessing)
        object.__setattr__(self, "metric_protocols", metrics)

        source_by_id = _index_unique(sources, "source_id", "sources")
        artifact_by_id = _index_unique(artifacts, "artifact_id", "artifacts")
        _index_unique(artifacts, "relative_path", "artifacts")
        trial_by_id = _index_unique(trials, "trial_id", "trials")
        _index_unique(preprocessing, "protocol_id", "preprocessing_protocols")
        metric_by_id = _index_unique(metrics, "metric_id", "metric_protocols")

        for source in sources:
            _require_known_ids(
                source.artifact_ids,
                artifact_by_id,
                "source %s artifact_ids" % source.source_id,
            )

        session_owner: Dict[str, str] = {}
        for trial in trials:
            if trial.source_id not in source_by_id:
                raise CalibrationContractError(
                    "trial %s references unknown source %s"
                    % (trial.trial_id, trial.source_id)
                )
            _require_known_ids(
                trial.artifact_ids,
                artifact_by_id,
                "trial %s artifact_ids" % trial.trial_id,
            )
            source_artifacts = set(source_by_id[trial.source_id].artifact_ids)
            if not set(trial.artifact_ids).issubset(source_artifacts):
                raise CalibrationContractError(
                    "trial %s references artifacts outside source %s"
                    % (trial.trial_id, trial.source_id)
                )
            previous_owner = session_owner.setdefault(trial.session_id, trial.individual_id)
            if previous_owner != trial.individual_id:
                raise CalibrationContractError(
                    "session %s leaks across multiple individual identities"
                    % trial.session_id
                )

        assignment_by_trial = {
            assignment.trial_id: assignment.split
            for assignment in self.split_plan.assignments
        }
        unknown_assignments = sorted(set(assignment_by_trial) - set(trial_by_id))
        missing_assignments = sorted(set(trial_by_id) - set(assignment_by_trial))
        if unknown_assignments:
            raise CalibrationContractError(
                "split plan references unknown trials: %s"
                % ", ".join(unknown_assignments)
            )
        if missing_assignments:
            raise CalibrationContractError(
                "split plan omits trials: %s" % ", ".join(missing_assignments)
            )

        _validate_group_disjointness(trials, assignment_by_trial)

        _require_known_ids(self.fit.metric_ids, metric_by_id, "fit metric_ids")
        expected_calibration = {
            trial_id
            for trial_id, split in assignment_by_trial.items()
            if split is CalibrationSplit.CALIBRATION
        }
        supplied_calibration = set(self.fit.calibration_trial_ids)
        if supplied_calibration != expected_calibration:
            missing = sorted(expected_calibration - supplied_calibration)
            extra = sorted(supplied_calibration - expected_calibration)
            details = []
            if missing:
                details.append("missing %s" % ", ".join(missing))
            if extra:
                details.append("non-calibration %s" % ", ".join(extra))
            raise CalibrationContractError(
                "fit calibration_trial_ids must exactly equal the calibration split (%s)"
                % "; ".join(details)
            )
        _require_known_ids(
            (
                self.fit.parameter_posterior_artifact_id,
                self.fit.diagnostics_artifact_id,
            ),
            artifact_by_id,
            "fit artifact IDs",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationEvidenceContract":
        data = _mapping(data, "calibration contract")
        _require_keys(
            data,
            required=(
                "contract_id",
                "contract_version",
                "sources",
                "artifacts",
                "trials",
                "split_plan",
                "preprocessing_protocols",
                "metric_protocols",
                "fit",
            ),
            optional=("schema_version",),
            label="calibration contract",
        )
        return cls(
            contract_id=data["contract_id"],
            contract_version=data["contract_version"],
            sources=tuple(LicensedSource.from_dict(item) for item in data["sources"]),
            artifacts=tuple(ArtifactFile.from_dict(item) for item in data["artifacts"]),
            trials=tuple(TrialRecord.from_dict(item) for item in data["trials"]),
            split_plan=PermanentSplitPlan.from_dict(data["split_plan"]),
            preprocessing_protocols=tuple(
                ProtocolReference.from_dict(item)
                for item in data["preprocessing_protocols"]
            ),
            metric_protocols=tuple(
                MetricProtocol.from_dict(item) for item in data["metric_protocols"]
            ),
            fit=FitProvenance.from_dict(data["fit"]),
            schema_version=data.get(
                "schema_version", CALIBRATION_CONTRACT_SCHEMA_VERSION
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> "CalibrationEvidenceContract":
        try:
            data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise CalibrationContractError("invalid JSON: %s" % exc) from exc
        return cls.from_dict(_mapping(data, "calibration contract"))

    def verify_files(self, base_directory: Path) -> Tuple[ArtifactFile, ...]:
        """Verify every declared file under ``base_directory`` and fail closed."""

        return verify_artifact_files(self.artifacts, base_directory)


def _index_unique(values: Iterable[Any], attribute: str, label: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for value in values:
        key = getattr(value, attribute)
        if key in result:
            raise CalibrationContractError("%s contain duplicate %s %s" % (label, attribute, key))
        result[key] = value
    return result


def _require_known_ids(
    values: Iterable[str], known: Mapping[str, Any], label: str
) -> None:
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise CalibrationContractError(
            "%s reference unknown IDs: %s" % (label, ", ".join(unknown))
        )


def _validate_group_disjointness(
    trials: Iterable[TrialRecord], assignments: Mapping[str, CalibrationSplit]
) -> None:
    individual_splits: Dict[str, CalibrationSplit] = {}
    session_splits: Dict[str, CalibrationSplit] = {}
    for trial in trials:
        split = assignments[trial.trial_id]
        prior_individual = individual_splits.setdefault(trial.individual_id, split)
        if prior_individual is not split:
            raise CalibrationContractError(
                "individual %s leaks across %s and %s splits"
                % (trial.individual_id, prior_individual.value, split.value)
            )
        prior_session = session_splits.setdefault(trial.session_id, split)
        if prior_session is not split:
            raise CalibrationContractError(
                "session %s leaks across %s and %s splits"
                % (trial.session_id, prior_session.value, split.value)
            )


def _sha256_handle(handle: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_files(
    artifacts: Iterable[ArtifactFile], base_directory: Path
) -> Tuple[ArtifactFile, ...]:
    """Verify path containment, byte count, and SHA-256 for every artifact."""

    records = _tuple(artifacts, "artifacts")
    if not records or not all(isinstance(item, ArtifactFile) for item in records):
        raise CalibrationContractError(
            "artifacts must contain at least one ArtifactFile"
        )
    base = Path(base_directory)
    try:
        resolved_base = base.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CalibrationContractError(
            "artifact base directory does not exist: %s" % base
        ) from exc
    if not resolved_base.is_dir():
        raise CalibrationContractError(
            "artifact base path is not a directory: %s" % resolved_base
        )

    opened_identities: Dict[Tuple[int, int], str] = {}
    for artifact in records:
        candidate = resolved_base.joinpath(*PurePosixPath(artifact.relative_path).parts)
        current = resolved_base
        try:
            for part in PurePosixPath(artifact.relative_path).parts:
                current = current / part
                if stat.S_ISLNK(os.lstat(str(current)).st_mode):
                    raise CalibrationContractError(
                        "artifact %s contains a symlink path component"
                        % artifact.artifact_id
                    )
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(str(candidate), flags)
            with os.fdopen(descriptor, "rb") as handle:
                before = os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise CalibrationContractError(
                        "artifact %s is not a regular file" % artifact.artifact_id
                    )
                actual_bytes = before.st_size
                if actual_bytes != artifact.bytes:
                    raise CalibrationContractError(
                        "artifact %s byte count mismatch: expected %d, observed %d"
                        % (artifact.artifact_id, artifact.bytes, actual_bytes)
                    )
                actual_sha256 = _sha256_handle(handle)
                after = os.fstat(handle.fileno())
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if before_identity != after_identity:
                    raise CalibrationContractError(
                        "artifact %s changed while hashing" % artifact.artifact_id
                    )
        except OSError as exc:
            if isinstance(exc, FileNotFoundError):
                raise CalibrationContractError(
                    "artifact %s is missing: %s" % (artifact.artifact_id, candidate)
                ) from exc
            raise CalibrationContractError(
                "artifact %s could not be read: %s" % (artifact.artifact_id, exc)
            ) from exc
        inode = (before.st_dev, before.st_ino)
        if inode in opened_identities:
            raise CalibrationContractError(
                "artifact %s aliases the same file as artifact %s"
                % (artifact.artifact_id, opened_identities[inode])
            )
        opened_identities[inode] = artifact.artifact_id
        if actual_sha256 != artifact.sha256:
            raise CalibrationContractError(
                "artifact %s SHA-256 mismatch: expected %s, observed %s"
                % (artifact.artifact_id, artifact.sha256, actual_sha256)
            )
    return records


__all__ = [
    "CALIBRATION_CONTRACT_SCHEMA_VERSION",
    "ArtifactFile",
    "CalibrationContractError",
    "CalibrationEvidenceContract",
    "CalibrationSplit",
    "FitDiagnostic",
    "FitProvenance",
    "LicenseRecord",
    "LicensedSource",
    "MetricProtocol",
    "PermanentSplitPlan",
    "ProtocolReference",
    "RedistributionStatus",
    "TrialAssignment",
    "TrialRecord",
    "verify_artifact_files",
]
