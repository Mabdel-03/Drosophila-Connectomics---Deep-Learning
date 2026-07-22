"""Deterministic, evidence-aware scientific benchmark evaluation.

The simulator has several fundamentally different kinds of oracles.  A
manufactured solution can prove an implementation is correct, but it cannot
prove that a biological model is accurate.  This module preserves that
distinction in immutable benchmark definitions and validation reports.

The implementation is dependency-light and intentionally independent of the
simulation and artifact modules.  Evaluators supply scalar metric values and
evidence receipts; the runner, rather than the evaluator, decides pass/fail
against the registered source-backed tolerances.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import sysconfig
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


VALIDATION_SCHEMA_VERSION = "1.0.0"


class ValidationContractError(ValueError):
    """Raised when a benchmark definition or report violates its contract."""


class OracleClass(str, Enum):
    """Authority class for a benchmark claim."""

    EXACT_MANUFACTURED = "exact_manufactured"
    INVARIANT_METAMORPHIC = "invariant_metamorphic"
    DIFFERENTIAL_CONVERGENCE = "differential_convergence"
    FROZEN_REGRESSION = "frozen_regression"
    HELD_OUT_EMPIRICAL = "held_out_empirical"


class GateStatus(str, Enum):
    """Four-state outcome; blocked evidence is never silently treated as pass."""

    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class PromotionGate(str, Enum):
    """Ordered, non-fungible scientific promotion gates."""

    SOFTWARE_CORRECT = "software_correct"
    NUMERICALLY_CONVERGED = "numerically_converged"
    STRUCTURALLY_SUPPORTED = "structurally_supported"
    CALIBRATED = "calibrated"
    EMPIRICALLY_VALIDATED = "empirically_validated"


PROMOTION_GATE_ORDER: Tuple[PromotionGate, ...] = (
    PromotionGate.SOFTWARE_CORRECT,
    PromotionGate.NUMERICALLY_CONVERGED,
    PromotionGate.STRUCTURALLY_SUPPORTED,
    PromotionGate.CALIBRATED,
    PromotionGate.EMPIRICALLY_VALIDATED,
)


class MetricComparator(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    BETWEEN_INCLUSIVE = "between_inclusive"


class ToleranceAuthority(str, Enum):
    """Where the acceptance threshold came from."""

    MANUFACTURED_DERIVATION = "manufactured_derivation"
    ENGINEERING_REQUIREMENT = "engineering_requirement"
    REGISTERED_FIXTURE = "registered_fixture"
    RELEASED_BASELINE = "released_baseline"
    PUBLIC_DATASET = "public_dataset"
    PEER_REVIEWED_PUBLICATION = "peer_reviewed_publication"
    PREREGISTERED_PROTOCOL = "preregistered_protocol"


class DataSplit(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    HELD_OUT = "held_out"


class RuntimeTier(str, Enum):
    FAST = "fast"
    PULL_REQUEST = "pull_request"
    SCHEDULED = "scheduled"
    PROMOTION = "promotion"


_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationContractError("%s must be a non-empty string" % label)
    return value


def _identifier(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if not _ID_PATTERN.fullmatch(text):
        raise ValidationContractError(
            "%s must contain lowercase letters, numbers, dots, underscores, or hyphens" % label
        )
    return text


def _version(value: Any, label: str = "version") -> str:
    text = _nonempty(value, label)
    if not _VERSION_PATTERN.fullmatch(text):
        raise ValidationContractError("%s must be a semantic version" % label)
    return text


def _sha256(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ValidationContractError("%s must be a lowercase SHA-256 digest" % label)
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValidationContractError("%s must be numeric" % label)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationContractError("%s must be numeric" % label) from exc
    if not math.isfinite(number):
        raise ValidationContractError("%s must be finite" % label)
    return number


def _enum(enum_type: Any, value: Any, label: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValidationContractError("invalid %s: %r" % (label, value)) from exc


def _unique_strings(values: Iterable[Any], label: str) -> Tuple[str, ...]:
    result = tuple(_identifier(value, "%s[%d]" % (label, index)) for index, value in enumerate(values))
    if len(set(result)) != len(result):
        raise ValidationContractError("%s must be unique" % label)
    return result


def _check_schema_version(value: str) -> None:
    if value != VALIDATION_SCHEMA_VERSION:
        raise ValidationContractError(
            "unsupported validation schema_version %r; expected %s"
            % (value, VALIDATION_SCHEMA_VERSION)
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
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationContractError("duplicate JSON key %r" % key)
        result[key] = value
    return result


def _load_json(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValidationContractError("invalid JSON: %s" % exc) from exc


class DeterministicJsonMixin:
    """Canonical JSON serialization for content addressing and reproducibility."""

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
class ToleranceSource(DeterministicJsonMixin):
    authority: ToleranceAuthority
    source_uri: str
    rationale: str
    citation: Optional[str] = None
    source_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "authority", _enum(ToleranceAuthority, self.authority, "tolerance authority")
        )
        _nonempty(self.source_uri, "source_uri")
        _nonempty(self.rationale, "rationale")
        if self.citation is not None:
            _nonempty(self.citation, "citation")
        if self.source_sha256 is not None:
            _sha256(self.source_sha256, "source_sha256")
        if (
            self.authority is ToleranceAuthority.PREREGISTERED_PROTOCOL
            and self.source_sha256 is None
        ):
            raise ValidationContractError(
                "preregistered protocol tolerances require an immutable source_sha256"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToleranceSource":
        return cls(
            authority=data["authority"],
            source_uri=data["source_uri"],
            rationale=data["rationale"],
            citation=data.get("citation"),
            source_sha256=data.get("source_sha256"),
        )


@dataclass(frozen=True)
class MetricObservation(DeterministicJsonMixin):
    metric_id: str
    value: float
    unit: str
    passed: bool
    reference_value: Optional[float]
    delta: Optional[float]
    acceptance: str

    def __post_init__(self) -> None:
        _identifier(self.metric_id, "metric_id")
        object.__setattr__(self, "value", _finite(self.value, "value"))
        _nonempty(self.unit, "unit")
        if not isinstance(self.passed, bool):
            raise ValidationContractError("passed must be boolean")
        if self.reference_value is not None:
            object.__setattr__(
                self, "reference_value", _finite(self.reference_value, "reference_value")
            )
        if self.delta is not None:
            object.__setattr__(self, "delta", _finite(self.delta, "delta"))
        _nonempty(self.acceptance, "acceptance")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricObservation":
        return cls(
            metric_id=data["metric_id"],
            value=data["value"],
            unit=data["unit"],
            passed=data["passed"],
            reference_value=data.get("reference_value"),
            delta=data.get("delta"),
            acceptance=data["acceptance"],
        )


@dataclass(frozen=True)
class MetricSpec(DeterministicJsonMixin):
    metric_id: str
    unit: str
    comparator: MetricComparator
    tolerance_source: ToleranceSource
    target_value: Optional[float] = None
    absolute_tolerance: Optional[float] = None
    relative_tolerance: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def __post_init__(self) -> None:
        _identifier(self.metric_id, "metric_id")
        _nonempty(self.unit, "unit")
        comparator = _enum(MetricComparator, self.comparator, "comparator")
        object.__setattr__(self, "comparator", comparator)
        if not isinstance(self.tolerance_source, ToleranceSource):
            raise ValidationContractError("tolerance_source must be a ToleranceSource")

        numeric_names = (
            "target_value",
            "absolute_tolerance",
            "relative_tolerance",
            "lower_bound",
            "upper_bound",
        )
        for name in numeric_names:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        if self.absolute_tolerance is not None and self.absolute_tolerance < 0.0:
            raise ValidationContractError("absolute_tolerance must be non-negative")
        if self.relative_tolerance is not None and self.relative_tolerance < 0.0:
            raise ValidationContractError("relative_tolerance must be non-negative")

        target_comparators = {
            MetricComparator.EXACT,
            MetricComparator.APPROXIMATE,
            MetricComparator.LESS_THAN,
            MetricComparator.LESS_THAN_OR_EQUAL,
            MetricComparator.GREATER_THAN,
            MetricComparator.GREATER_THAN_OR_EQUAL,
        }
        if comparator in target_comparators and self.target_value is None:
            raise ValidationContractError("%s requires target_value" % comparator.value)
        if comparator is MetricComparator.APPROXIMATE:
            if self.absolute_tolerance is None and self.relative_tolerance is None:
                raise ValidationContractError("approximate comparisons require a tolerance")
            if (
                self.absolute_tolerance is None
                and self.relative_tolerance is not None
                and self.target_value == 0.0
            ):
                raise ValidationContractError(
                    "relative-only approximate comparison cannot target zero"
                )
        elif self.absolute_tolerance is not None or self.relative_tolerance is not None:
            raise ValidationContractError("tolerances are only valid for approximate comparisons")

        if comparator is MetricComparator.BETWEEN_INCLUSIVE:
            if self.lower_bound is None or self.upper_bound is None:
                raise ValidationContractError("between comparison requires both bounds")
            if self.lower_bound > self.upper_bound:
                raise ValidationContractError("lower_bound must not exceed upper_bound")
            if self.target_value is not None:
                raise ValidationContractError("between comparison does not use target_value")
        elif self.lower_bound is not None or self.upper_bound is not None:
            raise ValidationContractError("bounds are only valid for between comparisons")

    def observe(self, value: float) -> MetricObservation:
        observed = _finite(value, "observed metric %s" % self.metric_id)
        comparator = self.comparator
        target = self.target_value
        delta: Optional[float]
        reference: Optional[float]

        if comparator is MetricComparator.EXACT:
            passed = observed == target
            reference = target
            delta = observed - target  # type: ignore[operator]
            acceptance = "value == %s" % target
        elif comparator is MetricComparator.APPROXIMATE:
            assert target is not None
            allowed = (self.absolute_tolerance or 0.0) + (
                (self.relative_tolerance or 0.0) * abs(target)
            )
            delta = observed - target
            passed = abs(delta) <= allowed
            reference = target
            acceptance = "abs(value - %s) <= %s" % (target, allowed)
        elif comparator is MetricComparator.LESS_THAN:
            assert target is not None
            passed = observed < target
            reference = target
            delta = observed - target
            acceptance = "value < %s" % target
        elif comparator is MetricComparator.LESS_THAN_OR_EQUAL:
            assert target is not None
            passed = observed <= target
            reference = target
            delta = observed - target
            acceptance = "value <= %s" % target
        elif comparator is MetricComparator.GREATER_THAN:
            assert target is not None
            passed = observed > target
            reference = target
            delta = observed - target
            acceptance = "value > %s" % target
        elif comparator is MetricComparator.GREATER_THAN_OR_EQUAL:
            assert target is not None
            passed = observed >= target
            reference = target
            delta = observed - target
            acceptance = "value >= %s" % target
        else:
            assert self.lower_bound is not None and self.upper_bound is not None
            passed = self.lower_bound <= observed <= self.upper_bound
            reference = None
            if observed < self.lower_bound:
                delta = observed - self.lower_bound
            elif observed > self.upper_bound:
                delta = observed - self.upper_bound
            else:
                delta = 0.0
            acceptance = "%s <= value <= %s" % (self.lower_bound, self.upper_bound)

        return MetricObservation(
            metric_id=self.metric_id,
            value=observed,
            unit=self.unit,
            passed=passed,
            reference_value=reference,
            delta=delta,
            acceptance=acceptance,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricSpec":
        return cls(
            metric_id=data["metric_id"],
            unit=data["unit"],
            comparator=data["comparator"],
            tolerance_source=ToleranceSource.from_dict(data["tolerance_source"]),
            target_value=data.get("target_value"),
            absolute_tolerance=data.get("absolute_tolerance"),
            relative_tolerance=data.get("relative_tolerance"),
            lower_bound=data.get("lower_bound"),
            upper_bound=data.get("upper_bound"),
        )


@dataclass(frozen=True)
class EvidenceRequirement(DeterministicJsonMixin):
    requirement_id: str
    description: str
    source_uri: str
    expected_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier(self.requirement_id, "requirement_id")
        _nonempty(self.description, "description")
        _nonempty(self.source_uri, "source_uri")
        if self.expected_sha256 is not None:
            _sha256(self.expected_sha256, "expected_sha256")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceRequirement":
        return cls(
            requirement_id=data["requirement_id"],
            description=data["description"],
            source_uri=data["source_uri"],
            expected_sha256=data.get("expected_sha256"),
        )


_EMPIRICAL_AUTHORITIES = frozenset(
    {
        ToleranceAuthority.PUBLIC_DATASET,
        ToleranceAuthority.PEER_REVIEWED_PUBLICATION,
        ToleranceAuthority.PREREGISTERED_PROTOCOL,
    }
)


@dataclass(frozen=True)
class BenchmarkCase(DeterministicJsonMixin):
    case_id: str
    version: str
    title: str
    claim: str
    oracle_class: OracleClass
    promotion_gate: PromotionGate
    metrics: Tuple[MetricSpec, ...]
    dependency_keys: Tuple[str, ...]
    prerequisite_case_ids: Tuple[str, ...] = ()
    required_evidence: Tuple[EvidenceRequirement, ...] = ()
    data_split: DataSplit = DataSplit.NOT_APPLICABLE
    runtime_tier: RuntimeTier = RuntimeTier.FAST
    hard_gate: bool = True
    fixture_uri: Optional[str] = None
    input_sha256: Optional[str] = None
    tags: Tuple[str, ...] = ()
    schema_version: str = VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case_id")
        _version(self.version)
        _nonempty(self.title, "title")
        _nonempty(self.claim, "claim")
        object.__setattr__(
            self, "oracle_class", _enum(OracleClass, self.oracle_class, "oracle_class")
        )
        object.__setattr__(
            self, "promotion_gate", _enum(PromotionGate, self.promotion_gate, "promotion_gate")
        )
        object.__setattr__(self, "data_split", _enum(DataSplit, self.data_split, "data_split"))
        object.__setattr__(
            self, "runtime_tier", _enum(RuntimeTier, self.runtime_tier, "runtime_tier")
        )
        _check_schema_version(self.schema_version)
        if not isinstance(self.hard_gate, bool):
            raise ValidationContractError("hard_gate must be boolean")

        metrics = tuple(self.metrics)
        if not metrics or not all(isinstance(metric, MetricSpec) for metric in metrics):
            raise ValidationContractError("metrics must contain at least one MetricSpec")
        metric_ids = [metric.metric_id for metric in metrics]
        if len(set(metric_ids)) != len(metric_ids):
            raise ValidationContractError("metric IDs must be unique within a benchmark")
        object.__setattr__(self, "metrics", tuple(sorted(metrics, key=lambda item: item.metric_id)))

        dependencies = _unique_strings(self.dependency_keys, "dependency_keys")
        if not dependencies:
            raise ValidationContractError("dependency_keys must not be empty")
        object.__setattr__(self, "dependency_keys", tuple(sorted(dependencies)))
        prerequisites = _unique_strings(self.prerequisite_case_ids, "prerequisite_case_ids")
        if self.case_id in prerequisites:
            raise ValidationContractError("a benchmark cannot depend on itself")
        object.__setattr__(self, "prerequisite_case_ids", tuple(sorted(prerequisites)))

        evidence = tuple(self.required_evidence)
        if not all(isinstance(item, EvidenceRequirement) for item in evidence):
            raise ValidationContractError("required_evidence must contain EvidenceRequirement records")
        evidence_ids = [item.requirement_id for item in evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValidationContractError("evidence requirement IDs must be unique")
        object.__setattr__(
            self, "required_evidence", tuple(sorted(evidence, key=lambda item: item.requirement_id))
        )

        if self.fixture_uri is not None:
            _nonempty(self.fixture_uri, "fixture_uri")
        if self.input_sha256 is not None:
            _sha256(self.input_sha256, "input_sha256")
        object.__setattr__(self, "tags", tuple(sorted(_unique_strings(self.tags, "tags"))))

        if self.oracle_class is OracleClass.HELD_OUT_EMPIRICAL:
            if self.data_split is not DataSplit.HELD_OUT:
                raise ValidationContractError("held-out empirical cases require data_split=held_out")
            if not evidence:
                raise ValidationContractError("held-out empirical cases require public evidence")
            if any(metric.tolerance_source.authority not in _EMPIRICAL_AUTHORITIES for metric in metrics):
                raise ValidationContractError(
                    "held-out empirical tolerances require a public dataset, peer-reviewed "
                    "publication, or immutable preregistered protocol"
                )
        elif self.data_split is DataSplit.HELD_OUT:
            raise ValidationContractError("held_out data_split is reserved for empirical cases")

        if self.oracle_class is OracleClass.FROZEN_REGRESSION:
            allowed = {
                ToleranceAuthority.REGISTERED_FIXTURE,
                ToleranceAuthority.RELEASED_BASELINE,
            }
            if any(metric.tolerance_source.authority not in allowed for metric in metrics):
                raise ValidationContractError(
                    "frozen regression tolerances require a registered fixture or released baseline"
                )
            if self.fixture_uri is None or self.input_sha256 is None:
                raise ValidationContractError(
                    "frozen regression cases require a fixture_uri and input_sha256"
                )
            for metric_spec in metrics:
                tolerance = metric_spec.tolerance_source
                if tolerance.source_sha256 != self.input_sha256:
                    raise ValidationContractError(
                        "frozen regression tolerance digests must match input_sha256"
                    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=data["case_id"],
            version=data["version"],
            title=data["title"],
            claim=data["claim"],
            oracle_class=data["oracle_class"],
            promotion_gate=data["promotion_gate"],
            metrics=tuple(MetricSpec.from_dict(item) for item in data["metrics"]),
            dependency_keys=tuple(data["dependency_keys"]),
            prerequisite_case_ids=tuple(data.get("prerequisite_case_ids", ())),
            required_evidence=tuple(
                EvidenceRequirement.from_dict(item) for item in data.get("required_evidence", ())
            ),
            data_split=data.get("data_split", DataSplit.NOT_APPLICABLE.value),
            runtime_tier=data.get("runtime_tier", RuntimeTier.FAST.value),
            hard_gate=data.get("hard_gate", True),
            fixture_uri=data.get("fixture_uri"),
            input_sha256=data.get("input_sha256"),
            tags=tuple(data.get("tags", ())),
            schema_version=data.get("schema_version", VALIDATION_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class BenchmarkRegistry(DeterministicJsonMixin):
    registry_id: str
    version: str
    description: str
    cases: Tuple[BenchmarkCase, ...]
    schema_version: str = VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.registry_id, "registry_id")
        _version(self.version)
        _nonempty(self.description, "description")
        _check_schema_version(self.schema_version)
        cases = tuple(self.cases)
        if not cases or not all(isinstance(case, BenchmarkCase) for case in cases):
            raise ValidationContractError("registry must contain BenchmarkCase records")
        case_ids = [case.case_id for case in cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValidationContractError("benchmark case IDs must be unique")
        cases = tuple(sorted(cases, key=lambda item: item.case_id))
        object.__setattr__(self, "cases", cases)
        known = set(case_ids)
        for case in cases:
            missing = set(case.prerequisite_case_ids) - known
            if missing:
                raise ValidationContractError(
                    "benchmark %s has unknown prerequisites: %s"
                    % (case.case_id, ", ".join(sorted(missing)))
                )
        self._topological_order(cases)

    def case(self, case_id: str) -> BenchmarkCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    def _topological_order(
        self, cases: Iterable[BenchmarkCase]
    ) -> Tuple[BenchmarkCase, ...]:
        selected = {case.case_id: case for case in cases}
        emitted = []
        remaining = dict(selected)
        while remaining:
            ready = sorted(
                case_id
                for case_id, case in remaining.items()
                if all(
                    prerequisite not in selected
                    or prerequisite in {item.case_id for item in emitted}
                    for prerequisite in case.prerequisite_case_ids
                )
            )
            if not ready:
                raise ValidationContractError("benchmark prerequisite graph contains a cycle")
            for case_id in ready:
                emitted.append(remaining.pop(case_id))
        return tuple(emitted)

    @staticmethod
    def _dependency_matches(changed: str, registered: str) -> bool:
        return (
            changed == registered
            or changed.startswith(registered + ".")
            or registered.startswith(changed + ".")
        )

    def select_cases(
        self,
        *,
        changed_dependencies: Optional[Iterable[str]] = None,
        case_ids: Optional[Iterable[str]] = None,
        include_dependents: bool = True,
        include_prerequisites: bool = True,
    ) -> Tuple[BenchmarkCase, ...]:
        """Select impacted cases plus their downstream and prerequisite closure."""

        changed = None
        if changed_dependencies is not None:
            changed = _unique_strings(changed_dependencies, "changed_dependencies")
        requested = None
        if case_ids is not None:
            requested = _unique_strings(case_ids, "case_ids")
            unknown = set(requested) - {case.case_id for case in self.cases}
            if unknown:
                raise ValidationContractError("unknown requested cases: %s" % ", ".join(sorted(unknown)))

        if changed is None and requested is None:
            return self._topological_order(self.cases)

        selected = set(requested or ())
        if changed is not None:
            for case in self.cases:
                if any(
                    self._dependency_matches(item, dependency)
                    for item in changed
                    for dependency in case.dependency_keys
                ):
                    selected.add(case.case_id)

        if include_dependents:
            modified = True
            while modified:
                modified = False
                for case in self.cases:
                    if case.case_id not in selected and any(
                        prerequisite in selected for prerequisite in case.prerequisite_case_ids
                    ):
                        selected.add(case.case_id)
                        modified = True
        if include_prerequisites:
            modified = True
            while modified:
                modified = False
                for case_id in tuple(selected):
                    for prerequisite in self.case(case_id).prerequisite_case_ids:
                        if prerequisite not in selected:
                            selected.add(prerequisite)
                            modified = True

        return self._topological_order(case for case in self.cases if case.case_id in selected)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkRegistry":
        return cls(
            registry_id=data["registry_id"],
            version=data["version"],
            description=data["description"],
            cases=tuple(BenchmarkCase.from_dict(item) for item in data["cases"]),
            schema_version=data.get("schema_version", VALIDATION_SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkRegistry":
        data = _load_json(text)
        if not isinstance(data, Mapping):
            raise ValidationContractError("benchmark registry JSON must contain an object")
        return cls.from_dict(data)


@dataclass(frozen=True)
class MetricValue:
    metric_id: str
    value: float

    def __post_init__(self) -> None:
        _identifier(self.metric_id, "metric_id")
        object.__setattr__(self, "value", _finite(self.value, "value"))


@dataclass(frozen=True)
class EvidenceReceipt(DeterministicJsonMixin):
    requirement_id: str
    source_uri: str
    artifact_sha256: str
    verified_by: str

    def __post_init__(self) -> None:
        _identifier(self.requirement_id, "requirement_id")
        _nonempty(self.source_uri, "source_uri")
        _sha256(self.artifact_sha256, "artifact_sha256")
        _nonempty(self.verified_by, "verified_by")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceReceipt":
        return cls(
            requirement_id=data["requirement_id"],
            source_uri=data["source_uri"],
            artifact_sha256=data["artifact_sha256"],
            verified_by=data["verified_by"],
        )


@dataclass(frozen=True)
class EvaluatorResult:
    """Raw evaluator output; it deliberately has no pass status."""

    values: Tuple[MetricValue, ...] = ()
    evidence: Tuple[EvidenceReceipt, ...] = ()
    blocked_reason: Optional[str] = None
    not_applicable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        values = tuple(self.values)
        evidence = tuple(self.evidence)
        if not all(isinstance(item, MetricValue) for item in values):
            raise ValidationContractError("values must contain MetricValue records")
        if not all(isinstance(item, EvidenceReceipt) for item in evidence):
            raise ValidationContractError("evidence must contain EvidenceReceipt records")
        if len({item.metric_id for item in values}) != len(values):
            raise ValidationContractError("evaluator metric IDs must be unique")
        if len({item.requirement_id for item in evidence}) != len(evidence):
            raise ValidationContractError("evidence receipt IDs must be unique")
        object.__setattr__(self, "values", tuple(sorted(values, key=lambda item: item.metric_id)))
        object.__setattr__(
            self, "evidence", tuple(sorted(evidence, key=lambda item: item.requirement_id))
        )
        if self.blocked_reason is not None:
            _nonempty(self.blocked_reason, "blocked_reason")
        if self.not_applicable_reason is not None:
            _nonempty(self.not_applicable_reason, "not_applicable_reason")
        if self.blocked_reason is not None and self.not_applicable_reason is not None:
            raise ValidationContractError("an evaluator result cannot be blocked and not applicable")

    @classmethod
    def blocked(cls, reason: str) -> "EvaluatorResult":
        return cls(blocked_reason=reason)

    @classmethod
    def not_applicable(cls, reason: str) -> "EvaluatorResult":
        return cls(not_applicable_reason=reason)


@dataclass(frozen=True)
class SourceDigest(DeterministicJsonMixin):
    kind: str
    name: str
    sha256: str

    def __post_init__(self) -> None:
        _identifier(self.kind, "kind")
        _nonempty(self.name, "name")
        _sha256(self.sha256, "sha256")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceDigest":
        return cls(kind=data["kind"], name=data["name"], sha256=data["sha256"])


@dataclass(frozen=True)
class BenchmarkResult(DeterministicJsonMixin):
    case_id: str
    case_version: str
    oracle_class: OracleClass
    promotion_gate: PromotionGate
    hard_gate: bool
    status: GateStatus
    observations: Tuple[MetricObservation, ...] = ()
    evidence: Tuple[EvidenceReceipt, ...] = ()
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case_id")
        _version(self.case_version, "case_version")
        object.__setattr__(
            self, "oracle_class", _enum(OracleClass, self.oracle_class, "oracle_class")
        )
        object.__setattr__(
            self, "promotion_gate", _enum(PromotionGate, self.promotion_gate, "promotion_gate")
        )
        object.__setattr__(self, "status", _enum(GateStatus, self.status, "status"))
        if not isinstance(self.hard_gate, bool):
            raise ValidationContractError("hard_gate must be boolean")
        observations = tuple(self.observations)
        evidence = tuple(self.evidence)
        if len({item.metric_id for item in observations}) != len(observations):
            raise ValidationContractError("observation metric IDs must be unique")
        if len({item.requirement_id for item in evidence}) != len(evidence):
            raise ValidationContractError("evidence receipt IDs must be unique")
        object.__setattr__(
            self, "observations", tuple(sorted(observations, key=lambda item: item.metric_id))
        )
        object.__setattr__(
            self, "evidence", tuple(sorted(evidence, key=lambda item: item.requirement_id))
        )
        if self.status is GateStatus.PASS:
            if not observations:
                raise ValidationContractError(
                    "passing result must contain its registered metric observations"
                )
            if any(not item.passed for item in observations):
                raise ValidationContractError(
                    "passing result cannot contain failed observations"
                )
        if self.status is not GateStatus.PASS and self.reason is None:
            raise ValidationContractError("non-passing results require a reason")
        if self.reason is not None:
            _nonempty(self.reason, "reason")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkResult":
        return cls(
            case_id=data["case_id"],
            case_version=data["case_version"],
            oracle_class=data["oracle_class"],
            promotion_gate=data["promotion_gate"],
            hard_gate=data["hard_gate"],
            status=data["status"],
            observations=tuple(
                MetricObservation.from_dict(item) for item in data.get("observations", ())
            ),
            evidence=tuple(EvidenceReceipt.from_dict(item) for item in data.get("evidence", ())),
            reason=data.get("reason"),
        )


@dataclass(frozen=True)
class GateOutcome(DeterministicJsonMixin):
    gate: PromotionGate
    status: GateStatus
    passed_count: int
    failed_count: int
    blocked_count: int
    not_applicable_count: int
    case_ids: Tuple[str, ...]
    omitted_case_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate", _enum(PromotionGate, self.gate, "gate"))
        object.__setattr__(self, "status", _enum(GateStatus, self.status, "status"))
        for name in (
            "passed_count",
            "failed_count",
            "blocked_count",
            "not_applicable_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValidationContractError("%s must be a non-negative integer" % name)
        case_ids = _unique_strings(self.case_ids, "case_ids")
        object.__setattr__(self, "case_ids", tuple(sorted(case_ids)))
        omitted_case_ids = _unique_strings(
            self.omitted_case_ids, "omitted_case_ids"
        )
        if set(case_ids) & set(omitted_case_ids):
            raise ValidationContractError(
                "gate case_ids and omitted_case_ids must be disjoint"
            )
        object.__setattr__(
            self, "omitted_case_ids", tuple(sorted(omitted_case_ids))
        )
        total = self.passed_count + self.failed_count + self.blocked_count + self.not_applicable_count
        if total != len(case_ids):
            raise ValidationContractError("gate counts must match case_ids")

        if self.failed_count:
            expected_status = GateStatus.FAIL
        elif self.blocked_count:
            expected_status = GateStatus.BLOCKED
        elif self.not_applicable_count or omitted_case_ids:
            expected_status = GateStatus.NOT_APPLICABLE
        elif self.passed_count:
            expected_status = GateStatus.PASS
        else:
            expected_status = GateStatus.NOT_APPLICABLE
        if self.status is not expected_status:
            raise ValidationContractError(
                "gate status does not match result counts and omitted hard cases"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GateOutcome":
        return cls(
            gate=data["gate"],
            status=data["status"],
            passed_count=data["passed_count"],
            failed_count=data["failed_count"],
            blocked_count=data["blocked_count"],
            not_applicable_count=data["not_applicable_count"],
            case_ids=tuple(data["case_ids"]),
            omitted_case_ids=tuple(data.get("omitted_case_ids", ())),
        )


def _build_gate_vector(
    results: Sequence[BenchmarkResult],
    omitted_hard_case_ids: Optional[
        Mapping[PromotionGate, Iterable[str]]
    ] = None,
) -> Tuple[GateOutcome, ...]:
    omitted_by_gate = omitted_hard_case_ids or {}
    vector = []
    for gate in PROMOTION_GATE_ORDER:
        relevant = [result for result in results if result.hard_gate and result.promotion_gate is gate]
        counts = {status: sum(result.status is status for result in relevant) for status in GateStatus}
        omitted = tuple(omitted_by_gate.get(gate, ()))
        if counts[GateStatus.FAIL]:
            status = GateStatus.FAIL
        elif counts[GateStatus.BLOCKED]:
            status = GateStatus.BLOCKED
        elif counts[GateStatus.NOT_APPLICABLE] or omitted:
            status = GateStatus.NOT_APPLICABLE
        elif counts[GateStatus.PASS]:
            status = GateStatus.PASS
        else:
            status = GateStatus.NOT_APPLICABLE
        vector.append(
            GateOutcome(
                gate=gate,
                status=status,
                passed_count=counts[GateStatus.PASS],
                failed_count=counts[GateStatus.FAIL],
                blocked_count=counts[GateStatus.BLOCKED],
                not_applicable_count=counts[GateStatus.NOT_APPLICABLE],
                case_ids=tuple(result.case_id for result in relevant),
                omitted_case_ids=omitted,
            )
        )
    return tuple(vector)


@dataclass(frozen=True)
class ValidationReport(DeterministicJsonMixin):
    evaluation_id: str
    registry_id: str
    registry_version: str
    registry_sha256: str
    selected_case_ids: Tuple[str, ...]
    source_digests: Tuple[SourceDigest, ...]
    results: Tuple[BenchmarkResult, ...]
    gate_vector: Tuple[GateOutcome, ...]
    suite_complete: bool = False
    omitted_case_ids: Tuple[str, ...] = ()
    schema_version: str = VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.evaluation_id, "evaluation_id")
        _identifier(self.registry_id, "registry_id")
        _version(self.registry_version, "registry_version")
        _sha256(self.registry_sha256, "registry_sha256")
        _check_schema_version(self.schema_version)
        selected = _unique_strings(self.selected_case_ids, "selected_case_ids")
        results = tuple(self.results)
        if len({result.case_id for result in results}) != len(results):
            raise ValidationContractError("report result case IDs must be unique")
        if set(selected) != {result.case_id for result in results}:
            raise ValidationContractError("selected_case_ids must match report results")
        object.__setattr__(self, "selected_case_ids", tuple(selected))
        object.__setattr__(self, "results", tuple(results))
        if not isinstance(self.suite_complete, bool):
            raise ValidationContractError("suite_complete must be boolean")
        omitted = _unique_strings(self.omitted_case_ids, "omitted_case_ids")
        if set(selected) & set(omitted):
            raise ValidationContractError(
                "selected_case_ids and omitted_case_ids must be disjoint"
            )
        if self.suite_complete and omitted:
            raise ValidationContractError(
                "a complete suite cannot contain omitted_case_ids"
            )
        object.__setattr__(self, "omitted_case_ids", tuple(sorted(omitted)))
        digests = tuple(self.source_digests)
        if len({(item.kind, item.name) for item in digests}) != len(digests):
            raise ValidationContractError("source digest kind/name pairs must be unique")
        object.__setattr__(
            self, "source_digests", tuple(sorted(digests, key=lambda item: (item.kind, item.name)))
        )
        vector = tuple(self.gate_vector)
        if tuple(item.gate for item in vector) != PROMOTION_GATE_ORDER:
            raise ValidationContractError("gate_vector must contain every promotion gate in order")
        gate_omissions = {
            outcome.gate: outcome.omitted_case_ids for outcome in vector
        }
        omitted_hard_ids = {
            case_id
            for outcome in vector
            for case_id in outcome.omitted_case_ids
        }
        if not omitted_hard_ids.issubset(set(omitted)):
            raise ValidationContractError(
                "gate omitted_case_ids must be present in report omitted_case_ids"
            )
        expected = _build_gate_vector(results, gate_omissions)
        if vector != expected:
            raise ValidationContractError("gate_vector does not match hard benchmark results")
        object.__setattr__(self, "gate_vector", vector)

    @property
    def overall_status(self) -> GateStatus:
        statuses = tuple(item.status for item in self.gate_vector)
        if GateStatus.FAIL in statuses:
            return GateStatus.FAIL
        if GateStatus.BLOCKED in statuses:
            return GateStatus.BLOCKED
        if not self.suite_complete:
            return GateStatus.NOT_APPLICABLE
        if all(status is GateStatus.PASS for status in statuses):
            return GateStatus.PASS
        return GateStatus.NOT_APPLICABLE

    @property
    def promotion_ceiling(self) -> Optional[PromotionGate]:
        if not self.suite_complete:
            return None
        ceiling = None
        for outcome in self.gate_vector:
            if outcome.status is not GateStatus.PASS:
                break
            ceiling = outcome.gate
        return ceiling

    def passes_through(self, gate: PromotionGate) -> bool:
        if not self.suite_complete:
            return False
        requested = _enum(PromotionGate, gate, "gate")
        for outcome in self.gate_vector:
            if outcome.status is not GateStatus.PASS:
                return False
            if outcome.gate is requested:
                return True
        return False

    def validate_against(self, registry: BenchmarkRegistry) -> "ValidationReport":
        """Bind every report claim to its exact declarative registry contract."""

        if not isinstance(registry, BenchmarkRegistry):
            raise ValidationContractError(
                "report registry binding requires a BenchmarkRegistry"
            )
        if self.registry_id != registry.registry_id:
            raise ValidationContractError("report registry_id does not match registry")
        if self.registry_version != registry.version:
            raise ValidationContractError(
                "report registry_version does not match registry"
            )
        if self.registry_sha256 != registry.content_sha256:
            raise ValidationContractError(
                "report registry_sha256 does not match canonical registry digest"
            )

        registry_ids = {case.case_id for case in registry.cases}
        selected_ids = set(self.selected_case_ids)
        omitted_ids = set(self.omitted_case_ids)
        expected_omitted = registry_ids - selected_ids
        if selected_ids - registry_ids:
            raise ValidationContractError(
                "report contains case IDs absent from the bound registry"
            )
        if omitted_ids != expected_omitted:
            raise ValidationContractError(
                "report omitted_case_ids do not complete the bound registry partition"
            )
        if self.suite_complete != (not expected_omitted):
            raise ValidationContractError(
                "report suite_complete does not match registry case coverage"
            )

        results_by_id = {result.case_id: result for result in self.results}
        for result in self.results:
            benchmark = registry.case(result.case_id)
            expected_metadata = (
                benchmark.version,
                benchmark.oracle_class,
                benchmark.promotion_gate,
                benchmark.hard_gate,
            )
            observed_metadata = (
                result.case_version,
                result.oracle_class,
                result.promotion_gate,
                result.hard_gate,
            )
            if observed_metadata != expected_metadata:
                raise ValidationContractError(
                    "report metadata does not match registry case %s"
                    % result.case_id
                )

            metric_specs = {
                metric_spec.metric_id: metric_spec
                for metric_spec in benchmark.metrics
            }
            observations = {
                observation.metric_id: observation
                for observation in result.observations
            }
            unknown_metrics = set(observations) - set(metric_specs)
            if unknown_metrics:
                raise ValidationContractError(
                    "report case %s contains unregistered metric observations: %s"
                    % (result.case_id, ", ".join(sorted(unknown_metrics)))
                )
            if result.status is GateStatus.PASS and set(observations) != set(
                metric_specs
            ):
                raise ValidationContractError(
                    "passing report case %s does not contain every registered metric"
                    % result.case_id
                )
            for metric_id, observation in observations.items():
                expected_observation = metric_specs[metric_id].observe(
                    observation.value
                )
                if observation != expected_observation:
                    raise ValidationContractError(
                        "report observation does not match registry metric %s/%s"
                        % (result.case_id, metric_id)
                    )

            requirements = {
                requirement.requirement_id: requirement
                for requirement in benchmark.required_evidence
            }
            receipts = {
                receipt.requirement_id: receipt for receipt in result.evidence
            }
            if set(receipts) - set(requirements):
                raise ValidationContractError(
                    "report case %s contains unregistered evidence receipts"
                    % result.case_id
                )
            if result.status is GateStatus.PASS and set(receipts) != set(
                requirements
            ):
                raise ValidationContractError(
                    "passing report case %s lacks registered evidence receipts"
                    % result.case_id
                )
            for requirement_id, receipt in receipts.items():
                requirement = requirements[requirement_id]
                if receipt.source_uri != requirement.source_uri or (
                    requirement.expected_sha256 is not None
                    and receipt.artifact_sha256
                    != requirement.expected_sha256
                ):
                    raise ValidationContractError(
                        "report evidence does not match registry requirement %s/%s"
                        % (result.case_id, requirement_id)
                    )

            selected_prerequisites = tuple(
                results_by_id[prerequisite]
                for prerequisite in benchmark.prerequisite_case_ids
                if prerequisite in results_by_id
            )
            if result.status is GateStatus.PASS and any(
                prerequisite.status is not GateStatus.PASS
                for prerequisite in selected_prerequisites
            ):
                raise ValidationContractError(
                    "passing report case %s has a non-passing selected prerequisite"
                    % result.case_id
                )

        expected_gate_omissions = {
            gate: tuple(
                case.case_id
                for case in registry.cases
                if case.hard_gate
                and case.promotion_gate is gate
                and case.case_id in expected_omitted
            )
            for gate in PROMOTION_GATE_ORDER
        }
        expected_vector = _build_gate_vector(
            self.results, expected_gate_omissions
        )
        if self.gate_vector != expected_vector:
            raise ValidationContractError(
                "report gate_vector does not match bound registry coverage"
            )
        return self

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        registry: Optional[BenchmarkRegistry] = None,
    ) -> "ValidationReport":
        report = cls(
            evaluation_id=data["evaluation_id"],
            registry_id=data["registry_id"],
            registry_version=data["registry_version"],
            registry_sha256=data["registry_sha256"],
            selected_case_ids=tuple(data["selected_case_ids"]),
            source_digests=tuple(SourceDigest.from_dict(item) for item in data["source_digests"]),
            results=tuple(BenchmarkResult.from_dict(item) for item in data["results"]),
            gate_vector=tuple(GateOutcome.from_dict(item) for item in data["gate_vector"]),
            suite_complete=data.get("suite_complete", False),
            omitted_case_ids=tuple(data.get("omitted_case_ids", ())),
            schema_version=data.get("schema_version", VALIDATION_SCHEMA_VERSION),
        )
        if registry is not None:
            report.validate_against(registry)
        return report

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        registry: Optional[BenchmarkRegistry] = None,
    ) -> "ValidationReport":
        data = _load_json(text)
        if not isinstance(data, Mapping):
            raise ValidationContractError("validation report JSON must contain an object")
        return cls.from_dict(data, registry=registry)


BenchmarkEvaluator = Callable[[BenchmarkCase], EvaluatorResult]


class ValidationRunner:
    """Execute registered evaluators and enforce registry-owned decisions."""

    def __init__(
        self,
        registry: BenchmarkRegistry,
        evaluators: Mapping[str, BenchmarkEvaluator],
    ) -> None:
        if not isinstance(registry, BenchmarkRegistry):
            raise ValidationContractError("registry must be a BenchmarkRegistry")
        unknown = set(evaluators) - {case.case_id for case in registry.cases}
        if unknown:
            raise ValidationContractError("evaluators registered for unknown cases: %s" % ", ".join(sorted(unknown)))
        self.registry = registry
        self._evaluators = dict(evaluators)

    @staticmethod
    def _result(
        case: BenchmarkCase,
        status: GateStatus,
        *,
        observations: Tuple[MetricObservation, ...] = (),
        evidence: Tuple[EvidenceReceipt, ...] = (),
        reason: Optional[str] = None,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            case_id=case.case_id,
            case_version=case.version,
            oracle_class=case.oracle_class,
            promotion_gate=case.promotion_gate,
            hard_gate=case.hard_gate,
            status=status,
            observations=observations,
            evidence=evidence,
            reason=reason,
        )

    def _evaluate(self, case: BenchmarkCase) -> BenchmarkResult:
        evaluator = self._evaluators.get(case.case_id)
        if evaluator is None:
            return self._result(
                case,
                GateStatus.BLOCKED,
                reason="no evaluator is registered for this benchmark",
            )
        try:
            raw = evaluator(case)
        except Exception as exc:  # Evaluator failures are reportable scientific failures.
            return self._result(
                case,
                GateStatus.FAIL,
                reason="evaluator raised %s: %s" % (type(exc).__name__, exc),
            )
        if not isinstance(raw, EvaluatorResult):
            return self._result(
                case,
                GateStatus.FAIL,
                reason="evaluator must return EvaluatorResult",
            )
        if raw.not_applicable_reason is not None:
            return self._result(
                case,
                GateStatus.NOT_APPLICABLE,
                evidence=raw.evidence,
                reason=raw.not_applicable_reason,
            )
        if raw.blocked_reason is not None:
            return self._result(
                case,
                GateStatus.BLOCKED,
                evidence=raw.evidence,
                reason=raw.blocked_reason,
            )

        requirements = {item.requirement_id: item for item in case.required_evidence}
        receipts = {item.requirement_id: item for item in raw.evidence}
        unknown_receipts = set(receipts) - set(requirements)
        if unknown_receipts:
            return self._result(
                case,
                GateStatus.FAIL,
                evidence=raw.evidence,
                reason="evaluator returned unregistered evidence receipts: %s"
                % ", ".join(sorted(unknown_receipts)),
            )
        missing = set(requirements) - set(receipts)
        if missing:
            return self._result(
                case,
                GateStatus.BLOCKED,
                evidence=raw.evidence,
                reason="required evidence is unavailable: %s" % ", ".join(sorted(missing)),
            )
        if case.oracle_class is OracleClass.HELD_OUT_EMPIRICAL:
            unpinned = sorted(
                requirement_id
                for requirement_id, requirement in requirements.items()
                if requirement.expected_sha256 is None
            )
            if unpinned:
                return self._result(
                    case,
                    GateStatus.BLOCKED,
                    evidence=raw.evidence,
                    reason="held-out evidence digest is not registered: %s"
                    % ", ".join(unpinned),
                )
        for requirement_id, requirement in requirements.items():
            receipt = receipts[requirement_id]
            if receipt.source_uri != requirement.source_uri:
                return self._result(
                    case,
                    GateStatus.BLOCKED,
                    evidence=raw.evidence,
                    reason="evidence source mismatch for %s" % requirement_id,
                )
            if (
                requirement.expected_sha256 is not None
                and receipt.artifact_sha256 != requirement.expected_sha256
            ):
                return self._result(
                    case,
                    GateStatus.BLOCKED,
                    evidence=raw.evidence,
                    reason="evidence digest mismatch for %s" % requirement_id,
                )

        metric_specs = {metric.metric_id: metric for metric in case.metrics}
        values = {item.metric_id: item.value for item in raw.values}
        unknown_metrics = set(values) - set(metric_specs)
        missing_metrics = set(metric_specs) - set(values)
        if unknown_metrics or missing_metrics:
            parts = []
            if missing_metrics:
                parts.append("missing metrics: %s" % ", ".join(sorted(missing_metrics)))
            if unknown_metrics:
                parts.append("unregistered metrics: %s" % ", ".join(sorted(unknown_metrics)))
            return self._result(
                case,
                GateStatus.FAIL,
                evidence=raw.evidence,
                reason="; ".join(parts),
            )

        observations = tuple(metric.observe(values[metric.metric_id]) for metric in case.metrics)
        failed = tuple(item.metric_id for item in observations if not item.passed)
        if failed:
            return self._result(
                case,
                GateStatus.FAIL,
                observations=observations,
                evidence=raw.evidence,
                reason="registered tolerances failed: %s" % ", ".join(failed),
            )
        return self._result(
            case,
            GateStatus.PASS,
            observations=observations,
            evidence=raw.evidence,
        )

    def run(
        self,
        *,
        evaluation_id: str,
        source_digests: Iterable[SourceDigest] = (),
        changed_dependencies: Optional[Iterable[str]] = None,
        case_ids: Optional[Iterable[str]] = None,
        include_dependents: bool = True,
        include_prerequisites: bool = True,
    ) -> ValidationReport:
        selected = self.registry.select_cases(
            changed_dependencies=changed_dependencies,
            case_ids=case_ids,
            include_dependents=include_dependents,
            include_prerequisites=include_prerequisites,
        )
        results_list = []
        results_by_id: Dict[str, BenchmarkResult] = {}
        for case in selected:
            result = self._evaluate(case)
            nonpassing_prerequisites = tuple(
                results_by_id[prerequisite]
                for prerequisite in case.prerequisite_case_ids
                if prerequisite in results_by_id
                and results_by_id[prerequisite].status is not GateStatus.PASS
            )
            if result.status is GateStatus.PASS and nonpassing_prerequisites:
                prerequisite_statuses = {
                    prerequisite.status
                    for prerequisite in nonpassing_prerequisites
                }
                if GateStatus.FAIL in prerequisite_statuses:
                    propagated_status = GateStatus.FAIL
                elif GateStatus.BLOCKED in prerequisite_statuses:
                    propagated_status = GateStatus.BLOCKED
                else:
                    propagated_status = GateStatus.NOT_APPLICABLE
                result = self._result(
                    case,
                    propagated_status,
                    observations=result.observations,
                    evidence=result.evidence,
                    reason=(
                        "selected prerequisites did not pass: %s; evaluator "
                        "observations are retained but cannot accept this claim"
                        % ", ".join(
                            "%s=%s"
                            % (prerequisite.case_id, prerequisite.status.value)
                            for prerequisite in nonpassing_prerequisites
                        )
                    ),
                )
            results_list.append(result)
            results_by_id[case.case_id] = result
        results = tuple(results_list)

        selected_ids = {case.case_id for case in selected}
        omitted_cases = tuple(
            case for case in self.registry.cases if case.case_id not in selected_ids
        )
        omitted_hard_case_ids = {
            gate: tuple(
                case.case_id
                for case in omitted_cases
                if case.hard_gate and case.promotion_gate is gate
            )
            for gate in PROMOTION_GATE_ORDER
        }
        return ValidationReport(
            evaluation_id=evaluation_id,
            registry_id=self.registry.registry_id,
            registry_version=self.registry.version,
            registry_sha256=self.registry.content_sha256,
            selected_case_ids=tuple(case.case_id for case in selected),
            source_digests=tuple(source_digests),
            results=results,
            gate_vector=_build_gate_vector(results, omitted_hard_case_ids),
            suite_complete=not omitted_cases,
            omitted_case_ids=tuple(case.case_id for case in omitted_cases),
        )


def default_benchmark_registry_path() -> Path:
    """Locate the source-tree or installed default registry without network I/O."""

    source_tree = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "registry.v1.json"
    installed = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / "benchmarks"
        / "registry.v1.json"
    )
    for candidate in (source_tree, installed):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("could not locate data/benchmarks/registry.v1.json")


def load_benchmark_registry(path: Optional[Path] = None) -> BenchmarkRegistry:
    target = default_benchmark_registry_path() if path is None else Path(path)
    registry = BenchmarkRegistry.from_json(target.read_text(encoding="utf-8"))
    verified_preregistered_protocol_files(registry, target)
    return registry


def verified_preregistered_protocol_files(
    registry: BenchmarkRegistry,
    registry_path: Path,
) -> Tuple[Tuple[str, Path, str, bytes], ...]:
    """Resolve and hash every local protocol that supplies an empirical tolerance.

    The digest is embedded in the registry, but validation also requires the
    corresponding regular file to be present and byte-identical.  This makes a
    protocol an auditable release input rather than an unresolvable prose URN.
    """

    target = Path(registry_path)
    expected_by_uri: Dict[str, str] = {}
    for benchmark in registry.cases:
        for metric_spec in benchmark.metrics:
            tolerance = metric_spec.tolerance_source
            if tolerance.authority is not ToleranceAuthority.PREREGISTERED_PROTOCOL:
                continue
            assert tolerance.source_sha256 is not None
            previous = expected_by_uri.setdefault(
                tolerance.source_uri, tolerance.source_sha256
            )
            if previous != tolerance.source_sha256:
                raise ValidationContractError(
                    "one preregistered protocol URI declares conflicting digests"
                )
        for requirement in benchmark.required_evidence:
            if not requirement.source_uri.startswith("data/benchmarks/protocols/"):
                continue
            if requirement.expected_sha256 is None:
                raise ValidationContractError(
                    "local protocol evidence requires an expected_sha256"
                )
            previous = expected_by_uri.setdefault(
                requirement.source_uri, requirement.expected_sha256
            )
            if previous != requirement.expected_sha256:
                raise ValidationContractError(
                    "one preregistered protocol URI declares conflicting digests"
                )

    receipts = []
    for source_uri, expected_sha256 in sorted(expected_by_uri.items()):
        relative = PurePosixPath(source_uri)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
            or source_uri != relative.as_posix()
            or relative.parts[:3] != ("data", "benchmarks", "protocols")
            or len(relative.parts) != 4
            or relative.suffix != ".json"
        ):
            raise ValidationContractError(
                "preregistered protocol source_uri must be a normalized "
                "data/benchmarks/protocols file"
            )
        candidates = []
        if len(target.parents) >= 3:
            candidates.append(target.parents[2].joinpath(*relative.parts))
        candidates.append(target.parent / "protocols" / relative.name)
        source_path = next((item for item in candidates if item.exists()), None)
        if source_path is None:
            raise ValidationContractError(
                "preregistered protocol source is unavailable: %s" % source_uri
            )
        before = source_path.lstat()
        if source_path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise ValidationContractError(
                "preregistered protocol source must be a non-symlink regular file"
            )
        payload = source_path.read_bytes()
        after = source_path.lstat()
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValidationContractError(
                "preregistered protocol source changed while it was read"
            )
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValidationContractError(
                "preregistered protocol source digest mismatch for %s" % source_uri
            )
        receipts.append((source_uri, source_path, observed_sha256, payload))
    return tuple(receipts)


def _stable_regular_file_bytes(path: Path, label: str) -> bytes:
    target = Path(path)
    before = target.lstat()
    if target.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValidationContractError("%s must be a non-symlink regular file" % label)
    payload = target.read_bytes()
    after = target.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise ValidationContractError("%s changed while it was read" % label)
    return payload


def python_source_tree_sha256(package_root: Path) -> str:
    """Hash relative paths and stable bytes for every runtime Python source."""

    root = Path(package_root)
    sources = sorted(root.rglob("*.py"), key=lambda path: path.as_posix())
    if not sources:
        raise ValidationContractError("Python package source tree is empty")
    digest = hashlib.sha256()
    for source in sources:
        relative = source.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(_stable_regular_file_bytes(source, "Python source"))
        digest.update(b"\0")
    return digest.hexdigest()


def default_validation_source_digests(
    registry: BenchmarkRegistry,
    registry_path: Path,
) -> Tuple[SourceDigest, ...]:
    """Build the exact non-image receipts required for a publishable report."""

    target = Path(registry_path)
    registry_bytes = _stable_regular_file_bytes(target, "benchmark registry")
    on_disk = BenchmarkRegistry.from_json(registry_bytes.decode("utf-8", errors="strict"))
    if on_disk != registry:
        raise ValidationContractError(
            "benchmark registry object does not match the supplied registry file"
        )
    protocol_files = verified_preregistered_protocol_files(registry, target)

    # Imported lazily to keep the contract module independent during module
    # initialization; these paths are part of the execution receipt itself.
    from .evaluators import default_evaluators
    from .flybody_adapter import worker_dependency_lock_sha256
    from .fly_fgs_runtime import default_fly_fgs_runtime_script_path
    from .flybody_ordinary_flight_release import default_expected_manifest_path

    evaluator_source = Path(default_evaluators.__code__.co_filename).resolve()
    fly_fgs_runtime_source = default_fly_fgs_runtime_script_path().resolve()
    ordinary_flight_contract = default_expected_manifest_path().resolve()
    package_root = Path(__file__).resolve().parent
    receipts = [
        SourceDigest(
            kind="registry",
            name=target.name,
            sha256=hashlib.sha256(registry_bytes).hexdigest(),
        ),
        SourceDigest(
            kind="code",
            name="built-in-evaluators.py",
            sha256=hashlib.sha256(
                _stable_regular_file_bytes(evaluator_source, "built-in evaluator source")
            ).hexdigest(),
        ),
        SourceDigest(
            kind="code",
            name="fly_sensor2behavior-python-tree",
            sha256=python_source_tree_sha256(package_root),
        ),
        SourceDigest(
            kind="code",
            name="fly_fgs_runtime_rpc.mjs",
            sha256=hashlib.sha256(
                _stable_regular_file_bytes(
                    fly_fgs_runtime_source, "fly-FGS runtime sidecar source"
                )
            ).hexdigest(),
        ),
        SourceDigest(
            kind="dependency_lock",
            name="requirements.lock",
            sha256=worker_dependency_lock_sha256().removeprefix("sha256:"),
        ),
        SourceDigest(
            kind="reference_contract",
            name=ordinary_flight_contract.name,
            sha256=hashlib.sha256(
                _stable_regular_file_bytes(
                    ordinary_flight_contract,
                    "ordinary-flight expected-artifact contract",
                )
            ).hexdigest(),
        ),
    ]
    receipts.extend(
        SourceDigest(kind="protocol", name=source_uri, sha256=sha256)
        for source_uri, _source_path, sha256, _payload in protocol_files
    )
    return tuple(receipts)


def validate_required_source_digests(
    report: ValidationReport,
    expected: Sequence[SourceDigest],
    *,
    exact_kinds: Sequence[str] = (),
) -> None:
    """Require exactly one matching receipt for every release-critical source."""

    reported: Dict[Tuple[str, str], str] = {}
    for item in report.source_digests:
        key = (item.kind, item.name)
        if key in reported:
            raise ValidationContractError(
                "validation report contains a duplicate source receipt"
            )
        reported[key] = item.sha256
    for item in expected:
        key = (item.kind, item.name)
        if reported.get(key) != item.sha256:
            raise ValidationContractError(
                "validation report source receipt %s:%s is stale or missing" % key
            )
    exact_kind_set = set(exact_kinds)
    expected_keys = {(item.kind, item.name) for item in expected}
    unexpected = sorted(
        key for key in reported if key[0] in exact_kind_set and key not in expected_keys
    )
    if unexpected:
        raise ValidationContractError(
            "validation report contains unexpected release-critical source receipts: %s"
            % ", ".join("%s:%s" % key for key in unexpected)
        )


__all__ = [
    "VALIDATION_SCHEMA_VERSION",
    "ValidationContractError",
    "OracleClass",
    "GateStatus",
    "PromotionGate",
    "PROMOTION_GATE_ORDER",
    "MetricComparator",
    "ToleranceAuthority",
    "DataSplit",
    "RuntimeTier",
    "ToleranceSource",
    "MetricSpec",
    "MetricValue",
    "MetricObservation",
    "EvidenceRequirement",
    "EvidenceReceipt",
    "BenchmarkCase",
    "BenchmarkRegistry",
    "EvaluatorResult",
    "SourceDigest",
    "BenchmarkResult",
    "GateOutcome",
    "ValidationReport",
    "BenchmarkEvaluator",
    "ValidationRunner",
    "default_benchmark_registry_path",
    "load_benchmark_registry",
    "verified_preregistered_protocol_files",
    "python_source_tree_sha256",
    "default_validation_source_digests",
    "validate_required_source_digests",
]
