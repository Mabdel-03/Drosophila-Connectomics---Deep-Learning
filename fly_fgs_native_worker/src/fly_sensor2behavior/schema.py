"""Versioned, dependency-light scientific interchange schemas.

This module is intentionally usable on the Python 3.9 web/analysis host.  The
FlyBody worker can consume the same JSON contracts from its Python 3.12
environment without making the evidence layer depend on MuJoCo or FlyGym.

Field names carry SI units.  Connectome identifiers are scoped by an explicit
``DatasetRef`` and FlyWire root IDs are always strings so that they remain
lossless in JavaScript.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0.0"
FAFB_V783_MATERIALIZATION = 783
FAFB_MIDLINE_X_NM = 533_000.0

# Exact FlyGym 2.1.0 / FlyBody yaw-roll-pitch skeleton order reviewed by the
# native worker. The torque mapper and measured-wing artifacts share this
# contract so an upstream reorder cannot silently swap generalized forces.
REVIEWED_FLYBODY_WING_AXIS_ORDER: Tuple[str, ...] = (
    "c_thorax-l_wing-yaw",
    "c_thorax-l_wing-roll",
    "c_thorax-l_wing-pitch",
    "c_thorax-r_wing-yaw",
    "c_thorax-r_wing-roll",
    "c_thorax-r_wing-pitch",
)


class SchemaValidationError(ValueError):
    """Raised when a scientific interchange object violates its contract."""


class CrossAtlasJoinError(SchemaValidationError):
    """Raised when identifiers from different atlas spaces are directly joined."""


class DatasetNamespace(str, Enum):
    FLYWIRE_FAFB = "flywire_fafb"
    BANC = "banc"
    FANC = "fanc"
    MANC = "manc"
    LITERATURE = "literature"
    SIMULATION = "simulation"
    EXPERIMENT = "experiment"


class EntityKind(str, Enum):
    NEURON = "neuron"
    MOTOR_NEURON = "motor_neuron"
    CELL_TYPE = "cell_type"
    CIRCUIT_POPULATION = "circuit_population"
    MUSCLE = "muscle"
    MODEL = "model"


class AnatomicalSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDLINE = "midline"
    BILATERAL = "bilateral"
    UNKNOWN = "unknown"


class EyeSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BINOCULAR = "binocular"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class SideMappingMethod(str, Enum):
    """Method used to translate a dataset label into simulation side frames."""

    FAFB_SOMA_X = "fafb_soma_x"
    CURATED_ATLAS_LABEL = "curated_atlas_label"
    CELL_TYPE_CROSSWALK = "cell_type_crosswalk"
    EXPERIMENTAL_MAPPING = "experimental_mapping"
    SIMULATION_CONVENTION = "simulation_convention"
    NOT_APPLICABLE = "not_applicable"


class EvidenceTier(str, Enum):
    DIRECT_OBSERVATION = "direct_observation"
    CURATED_CONNECTOME = "curated_connectome"
    TYPE_CROSSWALK = "type_crosswalk"
    MODEL_INFERENCE = "model_inference"
    HYPOTHESIS = "hypothesis"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CircuitSignalKind(str, Enum):
    VOLTAGE = "voltage"
    FIRING_RATE = "firing_rate"
    SPIKE_EVENTS = "spike_events"
    ACTIVATION = "activation"


class SignalOrigin(str, Enum):
    """Epistemic origin of a signal, independent of its numerical timebase."""

    OBSERVED = "observed"
    SIMULATED = "simulated"
    INFERRED = "inferred"
    SYNTHETIC = "synthetic"
    UNSPECIFIED = "unspecified"


class RetinalSignalKind(str, Enum):
    NORMALIZED_LUMINANCE = "normalized_luminance"
    LUMINANCE = "luminance"
    IRRADIANCE = "irradiance"


class AtlasMappingKind(str, Enum):
    SAME_IDENTIFIER_SPACE = "same_identifier_space"
    CELL_TYPE_CROSSWALK = "cell_type_crosswalk"
    MODEL_BRIDGE = "model_bridge"


class MotorSignalKind(str, Enum):
    EXACT_SPIKES = "exact_spikes"
    INFERRED_RATE = "inferred_rate"


class MuscleClass(str, Enum):
    ASYNCHRONOUS_POWER = "asynchronous_power"
    STEERING = "steering"
    TENSION = "tension"


class ValidationStatus(str, Enum):
    EXPLORATORY = "exploratory"
    CALIBRATED = "calibrated"
    VALIDATED = "validated"


class EpisodeStatus(str, Enum):
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial"


class MechanicsBackend(str, Enum):
    REDUCED_ANALYTIC = "reduced_analytic"
    FLYBODY_MUJOCO = "flybody_mujoco"
    OFFLINE_CFD = "offline_cfd"


class AerodynamicsOwner(str, Enum):
    NONE = "none"
    REDUCED_ANALYTIC = "reduced_analytic"
    FLYBODY = "flybody"
    CFD = "cfd"


class ActuationOwner(str, Enum):
    NEUROMUSCULAR_ADAPTER = "neuromuscular_adapter"
    FLYBODY_NATIVE = "flybody_native"
    PRESCRIBED_KINEMATICS = "prescribed_kinematics"


_SI_UNITS = frozenset(
    {
        "1",
        "s",
        "Hz",
        "V",
        "A",
        "m",
        "nm",
        "m s^-1",
        "m s^-2",
        "rad",
        "rad s^-1",
        "rad s^-2",
        "N",
        "N m",
        "J",
        "W",
        "kg",
        "kg m^2",
        "mol m^-3",
        "Pa",
        "W m^-2",
        "cd m^-2",
    }
)


def _enum(enum_type: Any, value: Any, label: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("invalid %s: %r" % (label, value)) from exc


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError("%s must be a non-empty string" % label)
    return value


def _finite(value: Any, label: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise SchemaValidationError("%s must be numeric" % label)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("%s must be numeric" % label) from exc
    if not math.isfinite(number):
        raise SchemaValidationError("%s must be finite" % label)
    if minimum is not None and number < minimum:
        raise SchemaValidationError("%s must be >= %s" % (label, minimum))
    return number


def _float_tuple(values: Iterable[Any], label: str) -> Tuple[float, ...]:
    return tuple(_finite(value, "%s[%d]" % (label, index)) for index, value in enumerate(values))


def _increasing_times(values: Iterable[Any], label: str, *, allow_empty: bool = False) -> Tuple[float, ...]:
    result = _float_tuple(values, label)
    if not allow_empty and not result:
        raise SchemaValidationError("%s must not be empty" % label)
    if result and result[0] < 0.0:
        raise SchemaValidationError("%s must be non-negative" % label)
    if any(current <= previous for previous, current in zip(result, result[1:])):
        raise SchemaValidationError("%s must be strictly increasing" % label)
    return result


def _event_times(values: Iterable[Any], label: str) -> Tuple[float, ...]:
    result = _float_tuple(values, label)
    if any(value < 0.0 for value in result):
        raise SchemaValidationError("%s must be non-negative" % label)
    if tuple(sorted(result)) != result:
        raise SchemaValidationError("%s must be sorted" % label)
    return result


def _vector(values: Sequence[Any], length: int, label: str) -> Tuple[float, ...]:
    result = _float_tuple(values, label)
    if len(result) != length:
        raise SchemaValidationError("%s must contain %d values" % (label, length))
    return result


def _vectors(values: Iterable[Sequence[Any]], length: int, label: str) -> Tuple[Tuple[float, ...], ...]:
    return tuple(_vector(value, length, "%s[%d]" % (label, index)) for index, value in enumerate(values))


def _require_unit(unit: str, label: str = "unit") -> str:
    if unit not in _SI_UNITS:
        raise SchemaValidationError("%s must be a supported SI unit, got %r" % (label, unit))
    return unit


def _check_schema_version(value: str) -> None:
    if value != SCHEMA_VERSION:
        raise SchemaValidationError(
            "unsupported schema_version %r; expected %s" % (value, SCHEMA_VERSION)
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class JsonSchemaMixin:
    """Stable JSON serialization shared by all public contracts."""

    schema_version: ClassVar[str]

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class DatasetRef(JsonSchemaMixin):
    namespace: DatasetNamespace
    release: str
    materialization: Optional[int] = None
    source_uri: Optional[str] = None
    neuron_universe: Optional[str] = None
    coordinate_units: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _enum(DatasetNamespace, self.namespace, "namespace"))
        _nonempty(self.release, "release")
        _check_schema_version(self.schema_version)
        if self.namespace is DatasetNamespace.FLYWIRE_FAFB:
            if not isinstance(self.materialization, int) or isinstance(self.materialization, bool) or self.materialization <= 0:
                raise SchemaValidationError("FlyWire FAFB datasets require a positive integer materialization")
        elif self.materialization is not None:
            raise SchemaValidationError("materialization is only valid for FlyWire FAFB")
        if self.source_uri is not None:
            _nonempty(self.source_uri, "source_uri")
        if self.coordinate_units is not None:
            _require_unit(self.coordinate_units, "coordinate_units")

    @property
    def identity_space(self) -> str:
        materialization = "" if self.materialization is None else "@%d" % self.materialization
        return "%s:%s%s" % (self.namespace.value, self.release, materialization)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetRef":
        return cls(
            namespace=data["namespace"],
            release=data["release"],
            materialization=data.get("materialization"),
            source_uri=data.get("source_uri"),
            neuron_universe=data.get("neuron_universe"),
            coordinate_units=data.get("coordinate_units"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class EntityRef(JsonSchemaMixin):
    dataset: DatasetRef
    entity_id: str
    kind: EntityKind
    cell_type: Optional[str] = None
    anatomical_side: AnatomicalSide = AnatomicalSide.UNKNOWN
    schema_version: str = SCHEMA_VERSION
    side_context: Optional["SideContext"] = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetRef):
            raise SchemaValidationError("dataset must be a DatasetRef")
        if not isinstance(self.entity_id, str):
            raise SchemaValidationError("entity_id must be a string")
        _nonempty(self.entity_id, "entity_id")
        object.__setattr__(self, "kind", _enum(EntityKind, self.kind, "kind"))
        object.__setattr__(
            self, "anatomical_side", _enum(AnatomicalSide, self.anatomical_side, "anatomical_side")
        )
        _check_schema_version(self.schema_version)
        if self.cell_type is not None:
            _nonempty(self.cell_type, "cell_type")
        if self.dataset.namespace is DatasetNamespace.FLYWIRE_FAFB and self.kind is EntityKind.NEURON:
            if not self.entity_id.isdigit():
                raise SchemaValidationError("FlyWire neuron root IDs must be decimal strings")
        if self.side_context is not None:
            if not isinstance(self.side_context, SideContext):
                raise SchemaValidationError("side_context must be a SideContext")
            if self.side_context.dataset.identity_space != self.dataset.identity_space:
                raise CrossAtlasJoinError(
                    "entity side context cannot come from a different atlas identity space"
                )
            context_side = self.side_context.anatomical_side
            if self.anatomical_side is AnatomicalSide.UNKNOWN:
                object.__setattr__(self, "anatomical_side", context_side)
            elif (
                context_side is not AnatomicalSide.UNKNOWN
                and context_side is not self.anatomical_side
            ):
                raise SchemaValidationError(
                    "entity anatomical_side conflicts with side_context anatomical_side"
                )

    @property
    def scoped_id(self) -> str:
        return "%s/%s/%s" % (self.dataset.identity_space, self.kind.value, self.entity_id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntityRef":
        return cls(
            dataset=DatasetRef.from_dict(data["dataset"]),
            entity_id=data["entity_id"],
            kind=data["kind"],
            cell_type=data.get("cell_type"),
            anatomical_side=data.get("anatomical_side", AnatomicalSide.UNKNOWN.value),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            side_context=(
                None
                if data.get("side_context") is None
                else SideContext.from_dict(data["side_context"])
            ),
        )


def require_same_identifier_space(source: EntityRef, target: EntityRef) -> None:
    """Reject a direct identifier join across atlases, releases, or materializations."""

    if source.dataset.identity_space != target.dataset.identity_space:
        raise CrossAtlasJoinError(
            "direct identifier join is forbidden across %s and %s; use an explicit cell-type crosswalk"
            % (source.dataset.identity_space, target.dataset.identity_space)
        )


def fafb_anatomical_side_from_soma_x_nm(
    soma_x_nm: float, *, midline_x_nm: float = FAFB_MIDLINE_X_NM
) -> AnatomicalSide:
    """Return corrected fly anatomy: higher FAFB soma x is fly LEFT."""

    x_value = _finite(soma_x_nm, "soma_x_nm", minimum=0.0)
    midline = _finite(midline_x_nm, "midline_x_nm", minimum=0.0)
    if x_value > midline:
        return AnatomicalSide.LEFT
    if x_value < midline:
        return AnatomicalSide.RIGHT
    return AnatomicalSide.MIDLINE


@dataclass(frozen=True)
class Provenance(JsonSchemaMixin):
    source_uri: str
    method: str
    citation: Optional[str] = None
    accessed_at_utc: Optional[str] = None
    dataset_identity: Optional[str] = None
    filters: Mapping[str, Any] = None  # type: ignore[assignment]
    notes: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    artifact_hash: Optional[str] = None
    source_run_id: Optional[str] = None
    derived_from: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.source_uri, "source_uri")
        _nonempty(self.method, "method")
        _check_schema_version(self.schema_version)
        if self.citation is not None:
            _nonempty(self.citation, "citation")
        if self.dataset_identity is not None:
            _nonempty(self.dataset_identity, "dataset_identity")
        if self.artifact_hash is not None:
            _nonempty(self.artifact_hash, "artifact_hash")
        if self.source_run_id is not None:
            _nonempty(self.source_run_id, "source_run_id")
        derived_from = tuple(self.derived_from)
        for index, source_id in enumerate(derived_from):
            _nonempty(source_id, "derived_from[%d]" % index)
        if len(set(derived_from)) != len(derived_from):
            raise SchemaValidationError("derived_from entries must be unique")
        object.__setattr__(self, "derived_from", derived_from)
        if self.filters is None:
            object.__setattr__(self, "filters", {})
        elif not isinstance(self.filters, Mapping):
            raise SchemaValidationError("filters must be a mapping")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        return cls(
            source_uri=data["source_uri"],
            method=data["method"],
            citation=data.get("citation"),
            accessed_at_utc=data.get("accessed_at_utc"),
            dataset_identity=data.get("dataset_identity"),
            filters=data.get("filters", {}),
            notes=data.get("notes"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            artifact_hash=data.get("artifact_hash"),
            source_run_id=data.get("source_run_id"),
            derived_from=tuple(data.get("derived_from", ())),
        )


@dataclass(frozen=True)
class Confidence(JsonSchemaMixin):
    tier: EvidenceTier
    level: ConfidenceLevel
    score: float
    basis: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", _enum(EvidenceTier, self.tier, "tier"))
        object.__setattr__(self, "level", _enum(ConfidenceLevel, self.level, "level"))
        score = _finite(self.score, "score", minimum=0.0)
        if score > 1.0:
            raise SchemaValidationError("score must be <= 1")
        object.__setattr__(self, "score", score)
        _nonempty(self.basis, "basis")
        _check_schema_version(self.schema_version)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Confidence":
        return cls(
            tier=data["tier"],
            level=data["level"],
            score=data["score"],
            basis=data["basis"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class SideContext(JsonSchemaMixin):
    """Explicit translations between dataset, visual, display, and effector sides.

    ``raw_dataset_side`` is preserved verbatim and never treated as anatomical
    truth.  In particular, FlyWire FAFB anatomical side is position-derived:
    higher soma x is fly-left and lower soma x is fly-right.
    """

    dataset: DatasetRef
    raw_dataset_side: Optional[str]
    anatomical_side: AnatomicalSide
    visual_field_side: AnatomicalSide
    app_rendering_side: AnatomicalSide
    eye_side: EyeSide
    effector_side: AnatomicalSide
    mapping_method: SideMappingMethod
    confidence: Confidence
    soma_x_nm: Optional[float] = None
    notes: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetRef):
            raise SchemaValidationError("side context dataset must be a DatasetRef")
        if self.raw_dataset_side is not None:
            _nonempty(self.raw_dataset_side, "raw_dataset_side")
        for name in (
            "anatomical_side",
            "visual_field_side",
            "app_rendering_side",
            "effector_side",
        ):
            object.__setattr__(
                self, name, _enum(AnatomicalSide, getattr(self, name), name)
            )
        object.__setattr__(self, "eye_side", _enum(EyeSide, self.eye_side, "eye_side"))
        method = _enum(SideMappingMethod, self.mapping_method, "mapping_method")
        object.__setattr__(self, "mapping_method", method)
        if not isinstance(self.confidence, Confidence):
            raise SchemaValidationError("side context confidence must be Confidence")
        if self.notes is not None:
            _nonempty(self.notes, "notes")
        if self.soma_x_nm is not None:
            object.__setattr__(
                self, "soma_x_nm", _finite(self.soma_x_nm, "soma_x_nm", minimum=0.0)
            )
        if method is SideMappingMethod.FAFB_SOMA_X:
            if self.dataset.namespace is not DatasetNamespace.FLYWIRE_FAFB:
                raise SchemaValidationError(
                    "fafb_soma_x mapping is only valid for FlyWire FAFB"
                )
            if self.soma_x_nm is None:
                raise SchemaValidationError("fafb_soma_x mapping requires soma_x_nm")
            expected = fafb_anatomical_side_from_soma_x_nm(self.soma_x_nm)
            if self.anatomical_side is not expected:
                raise SchemaValidationError(
                    "anatomical_side conflicts with the corrected FAFB soma-x convention"
                )
        elif (
            self.dataset.namespace is DatasetNamespace.FLYWIRE_FAFB
            and method is SideMappingMethod.CURATED_ATLAS_LABEL
            and self.anatomical_side
            in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT, AnatomicalSide.MIDLINE)
        ):
            raise SchemaValidationError(
                "FlyWire dataset side labels are not anatomical truth; use fafb_soma_x or an explicit crosswalk"
            )
        _check_schema_version(self.schema_version)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SideContext":
        return cls(
            dataset=DatasetRef.from_dict(data["dataset"]),
            raw_dataset_side=data.get("raw_dataset_side"),
            anatomical_side=data["anatomical_side"],
            visual_field_side=data["visual_field_side"],
            app_rendering_side=data["app_rendering_side"],
            eye_side=data["eye_side"],
            effector_side=data["effector_side"],
            mapping_method=data["mapping_method"],
            confidence=Confidence.from_dict(data["confidence"]),
            soma_x_nm=data.get("soma_x_nm"),
            notes=data.get("notes"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class CircuitSignal(JsonSchemaMixin):
    neuron: EntityRef
    signal_kind: CircuitSignalKind
    unit: str
    provenance: Provenance
    confidence: Confidence
    values: Tuple[float, ...] = ()
    spike_times_s: Tuple[float, ...] = ()
    uncertainty: Tuple[float, ...] = ()
    schema_version: str = SCHEMA_VERSION
    origin: SignalOrigin = SignalOrigin.UNSPECIFIED
    availability_times_s: Tuple[float, ...] = ()
    source_signal_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.neuron.kind not in (EntityKind.NEURON, EntityKind.CELL_TYPE, EntityKind.CIRCUIT_POPULATION):
            raise SchemaValidationError("circuit signals require a neuron, cell type, or circuit population")
        signal_kind = _enum(CircuitSignalKind, self.signal_kind, "signal_kind")
        object.__setattr__(self, "signal_kind", signal_kind)
        _require_unit(self.unit)
        values = _float_tuple(self.values, "values")
        spikes = _event_times(self.spike_times_s, "spike_times_s")
        uncertainty = _float_tuple(self.uncertainty, "uncertainty")
        availability = _event_times(self.availability_times_s, "availability_times_s")
        source_signal_ids = tuple(self.source_signal_ids)
        for index, source_id in enumerate(source_signal_ids):
            _nonempty(source_id, "source_signal_ids[%d]" % index)
        if len(set(source_signal_ids)) != len(source_signal_ids):
            raise SchemaValidationError("source_signal_ids must be unique")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "spike_times_s", spikes)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "availability_times_s", availability)
        object.__setattr__(self, "source_signal_ids", source_signal_ids)
        object.__setattr__(self, "origin", _enum(SignalOrigin, self.origin, "origin"))
        _check_schema_version(self.schema_version)
        expected_unit = {
            CircuitSignalKind.VOLTAGE: "V",
            CircuitSignalKind.FIRING_RATE: "Hz",
            CircuitSignalKind.SPIKE_EVENTS: "1",
            CircuitSignalKind.ACTIVATION: "1",
        }[signal_kind]
        if self.unit != expected_unit:
            raise SchemaValidationError("%s signals require unit %s" % (signal_kind.value, expected_unit))
        if signal_kind is CircuitSignalKind.SPIKE_EVENTS:
            if values:
                raise SchemaValidationError("spike-event signals use spike_times_s, not values")
        elif spikes:
            raise SchemaValidationError("non-spike signals cannot contain spike_times_s")
        if uncertainty and len(uncertainty) != len(values):
            raise SchemaValidationError("uncertainty must match values length")
        expected_availability_length = len(spikes) if signal_kind is CircuitSignalKind.SPIKE_EVENTS else len(values)
        if availability and len(availability) != expected_availability_length:
            raise SchemaValidationError(
                "availability_times_s must match the signal measurements"
            )


@dataclass(frozen=True)
class CircuitOutputTrace(JsonSchemaMixin):
    dataset: DatasetRef
    sample_times_s: Tuple[float, ...]
    signals: Tuple[CircuitSignal, ...]
    provenance: Provenance
    confidence: Confidence
    exact_timebase: bool
    schema_version: str = SCHEMA_VERSION
    sample_interval_end_s: Optional[float] = None

    def __post_init__(self) -> None:
        times = _increasing_times(self.sample_times_s, "sample_times_s")
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "signals", tuple(self.signals))
        interval_end = self.sample_interval_end_s
        if interval_end is not None:
            interval_end = _finite(
                interval_end,
                "sample_interval_end_s",
                minimum=0.0,
            )
            if interval_end <= times[-1]:
                raise SchemaValidationError(
                    "sample_interval_end_s must be greater than the final sample time"
                )
            object.__setattr__(self, "sample_interval_end_s", interval_end)
        _check_schema_version(self.schema_version)
        for signal in self.signals:
            require_same_identifier_space(
                EntityRef(self.dataset, "trace", EntityKind.CIRCUIT_POPULATION), signal.neuron
            )
            if signal.signal_kind is not CircuitSignalKind.SPIKE_EVENTS and len(signal.values) != len(times):
                raise SchemaValidationError("each sampled circuit signal must match sample_times_s")
            if signal.spike_times_s:
                if interval_end is None and signal.spike_times_s[-1] > times[-1]:
                    raise SchemaValidationError("circuit spike event is outside trace timebase")
                if (
                    interval_end is not None
                    and signal.spike_times_s[-1] >= interval_end
                ):
                    raise SchemaValidationError(
                        "circuit spike event is outside the half-open sample interval"
                    )
            measurements = (
                signal.spike_times_s
                if signal.signal_kind is CircuitSignalKind.SPIKE_EVENTS
                else times
            )
            if signal.availability_times_s and any(
                available < measured
                for measured, available in zip(measurements, signal.availability_times_s)
            ):
                raise SchemaValidationError(
                    "circuit signal cannot be available before it is measured"
                )


@dataclass(frozen=True)
class RetinalFrame(JsonSchemaMixin):
    """One causal retinal observation with explicit radiometric semantics."""

    measurement_time_s: float
    availability_time_s: float
    exposure_start_s: float
    exposure_end_s: float
    eye_side: EyeSide
    signal_kind: RetinalSignalKind
    unit: str
    samples: Tuple[float, ...]
    provenance: Provenance
    confidence: Confidence
    ommatidial_directions_body: Tuple[Tuple[float, float, float], ...] = ()
    uncertainty: Tuple[float, ...] = ()
    calibration_id: Optional[str] = None
    side_context: Optional[SideContext] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        measured = _finite(self.measurement_time_s, "measurement_time_s", minimum=0.0)
        available = _finite(self.availability_time_s, "availability_time_s", minimum=0.0)
        exposure_start = _finite(self.exposure_start_s, "exposure_start_s", minimum=0.0)
        exposure_end = _finite(self.exposure_end_s, "exposure_end_s", minimum=0.0)
        if exposure_end <= exposure_start:
            raise SchemaValidationError("exposure_end_s must be greater than exposure_start_s")
        if measured < exposure_start or measured > exposure_end:
            raise SchemaValidationError(
                "measurement_time_s must lie within the exposure interval"
            )
        if available < exposure_end:
            raise SchemaValidationError(
                "availability_time_s cannot precede the end of exposure"
            )
        object.__setattr__(self, "measurement_time_s", measured)
        object.__setattr__(self, "availability_time_s", available)
        object.__setattr__(self, "exposure_start_s", exposure_start)
        object.__setattr__(self, "exposure_end_s", exposure_end)
        eye = _enum(EyeSide, self.eye_side, "eye_side")
        if eye in (EyeSide.NOT_APPLICABLE, EyeSide.UNKNOWN):
            raise SchemaValidationError("retinal frames require a left, right, or binocular eye")
        object.__setattr__(self, "eye_side", eye)
        signal_kind = _enum(RetinalSignalKind, self.signal_kind, "signal_kind")
        object.__setattr__(self, "signal_kind", signal_kind)
        expected_unit = {
            RetinalSignalKind.NORMALIZED_LUMINANCE: "1",
            RetinalSignalKind.LUMINANCE: "cd m^-2",
            RetinalSignalKind.IRRADIANCE: "W m^-2",
        }[signal_kind]
        _require_unit(self.unit)
        if self.unit != expected_unit:
            raise SchemaValidationError(
                "%s retinal samples require unit %s"
                % (signal_kind.value, expected_unit)
            )
        samples = _float_tuple(self.samples, "samples")
        if not samples:
            raise SchemaValidationError("retinal samples must not be empty")
        if any(value < 0.0 for value in samples):
            raise SchemaValidationError("retinal samples must be non-negative")
        if (
            signal_kind is RetinalSignalKind.NORMALIZED_LUMINANCE
            and any(value > 1.0 for value in samples)
        ):
            raise SchemaValidationError("normalized luminance must lie in [0, 1]")
        object.__setattr__(self, "samples", samples)
        directions = _vectors(
            self.ommatidial_directions_body, 3, "ommatidial_directions_body"
        )
        if directions and len(directions) != len(samples):
            raise SchemaValidationError(
                "ommatidial directions must match the retinal sample count"
            )
        for direction in directions:
            norm = math.sqrt(sum(value * value for value in direction))
            if abs(norm - 1.0) > 1e-6:
                raise SchemaValidationError("ommatidial directions must have unit norm")
        object.__setattr__(self, "ommatidial_directions_body", directions)
        uncertainty = _float_tuple(self.uncertainty, "uncertainty")
        if uncertainty and len(uncertainty) != len(samples):
            raise SchemaValidationError("retinal uncertainty must match samples")
        if any(value < 0.0 for value in uncertainty):
            raise SchemaValidationError("retinal uncertainty must be non-negative")
        object.__setattr__(self, "uncertainty", uncertainty)
        if self.calibration_id is not None:
            _nonempty(self.calibration_id, "calibration_id")
        if self.side_context is not None:
            if not isinstance(self.side_context, SideContext):
                raise SchemaValidationError("side_context must be a SideContext")
            context_eye = self.side_context.eye_side
            if context_eye not in (EyeSide.UNKNOWN, eye):
                raise SchemaValidationError("retinal eye_side conflicts with side_context")
        if not isinstance(self.provenance, Provenance):
            raise SchemaValidationError("retinal provenance must be Provenance")
        if not isinstance(self.confidence, Confidence):
            raise SchemaValidationError("retinal confidence must be Confidence")
        _check_schema_version(self.schema_version)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetinalFrame":
        side_context = data.get("side_context")
        return cls(
            measurement_time_s=data["measurement_time_s"],
            availability_time_s=data["availability_time_s"],
            exposure_start_s=data["exposure_start_s"],
            exposure_end_s=data["exposure_end_s"],
            eye_side=data["eye_side"],
            signal_kind=data["signal_kind"],
            unit=data["unit"],
            samples=tuple(data["samples"]),
            provenance=Provenance.from_dict(data["provenance"]),
            confidence=Confidence.from_dict(data["confidence"]),
            ommatidial_directions_body=tuple(
                tuple(item) for item in data.get("ommatidial_directions_body", ())
            ),
            uncertainty=tuple(data.get("uncertainty", ())),
            calibration_id=data.get("calibration_id"),
            side_context=(
                None if side_context is None else SideContext.from_dict(side_context)
            ),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class EncoderProvenance(JsonSchemaMixin):
    """Provenance for a circuit-to-DN encoder and any atlas bridge it uses."""

    name: str
    version: str
    artifact_hash: str
    input_dataset: DatasetRef
    output_dataset: DatasetRef
    mapping_kind: AtlasMappingKind
    method: str
    provenance: Provenance
    confidence: Confidence
    cross_atlas_mapping: Optional[Provenance] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("name", "version", "artifact_hash", "method"):
            _nonempty(getattr(self, name), name)
        if not isinstance(self.input_dataset, DatasetRef) or not isinstance(
            self.output_dataset, DatasetRef
        ):
            raise SchemaValidationError("encoder datasets must be DatasetRef objects")
        mapping_kind = _enum(AtlasMappingKind, self.mapping_kind, "mapping_kind")
        object.__setattr__(self, "mapping_kind", mapping_kind)
        same_space = (
            self.input_dataset.identity_space == self.output_dataset.identity_space
        )
        if same_space and mapping_kind is not AtlasMappingKind.SAME_IDENTIFIER_SPACE:
            raise SchemaValidationError(
                "same-atlas encoders must declare same_identifier_space"
            )
        if not same_space:
            if mapping_kind is AtlasMappingKind.SAME_IDENTIFIER_SPACE:
                raise CrossAtlasJoinError(
                    "cross-atlas encoders cannot claim a shared identifier space"
                )
            if self.cross_atlas_mapping is None:
                raise CrossAtlasJoinError(
                    "cross-atlas encoders require explicit mapping provenance"
                )
        if self.cross_atlas_mapping is not None and not isinstance(
            self.cross_atlas_mapping, Provenance
        ):
            raise SchemaValidationError("cross_atlas_mapping must be Provenance")
        if not isinstance(self.provenance, Provenance):
            raise SchemaValidationError("encoder provenance must be Provenance")
        if not isinstance(self.confidence, Confidence):
            raise SchemaValidationError("encoder confidence must be Confidence")
        _check_schema_version(self.schema_version)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EncoderProvenance":
        mapping = data.get("cross_atlas_mapping")
        return cls(
            name=data["name"],
            version=data["version"],
            artifact_hash=data["artifact_hash"],
            input_dataset=DatasetRef.from_dict(data["input_dataset"]),
            output_dataset=DatasetRef.from_dict(data["output_dataset"]),
            mapping_kind=data["mapping_kind"],
            method=data["method"],
            provenance=Provenance.from_dict(data["provenance"]),
            confidence=Confidence.from_dict(data["confidence"]),
            cross_atlas_mapping=(
                None if mapping is None else Provenance.from_dict(mapping)
            ),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class DescendingSignal(JsonSchemaMixin):
    descending_neuron: EntityRef
    signal_kind: CircuitSignalKind
    unit: str
    origin: SignalOrigin
    encoder_delay_s: float
    availability_times_s: Tuple[float, ...]
    provenance: Provenance
    confidence: Confidence
    values: Tuple[float, ...] = ()
    spike_times_s: Tuple[float, ...] = ()
    uncertainty: Tuple[float, ...] = ()
    source_signal_ids: Tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.descending_neuron.kind not in (EntityKind.NEURON, EntityKind.CELL_TYPE):
            raise SchemaValidationError(
                "descending signals require an individual neuron or DN cell type"
            )
        signal_kind = _enum(CircuitSignalKind, self.signal_kind, "signal_kind")
        if signal_kind is CircuitSignalKind.ACTIVATION:
            raise SchemaValidationError(
                "descending signals must be voltage, firing rate, or spike events"
            )
        object.__setattr__(self, "signal_kind", signal_kind)
        origin = _enum(SignalOrigin, self.origin, "origin")
        if origin is SignalOrigin.UNSPECIFIED:
            raise SchemaValidationError("descending signal origin must be explicit")
        object.__setattr__(self, "origin", origin)
        expected_unit = {
            CircuitSignalKind.VOLTAGE: "V",
            CircuitSignalKind.FIRING_RATE: "Hz",
            CircuitSignalKind.SPIKE_EVENTS: "1",
        }[signal_kind]
        _require_unit(self.unit)
        if self.unit != expected_unit:
            raise SchemaValidationError(
                "%s descending signals require unit %s"
                % (signal_kind.value, expected_unit)
            )
        delay = _finite(self.encoder_delay_s, "encoder_delay_s", minimum=0.0)
        object.__setattr__(self, "encoder_delay_s", delay)
        values = _float_tuple(self.values, "values")
        spikes = _event_times(self.spike_times_s, "spike_times_s")
        availability = _event_times(self.availability_times_s, "availability_times_s")
        uncertainty = _float_tuple(self.uncertainty, "uncertainty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "spike_times_s", spikes)
        object.__setattr__(self, "availability_times_s", availability)
        object.__setattr__(self, "uncertainty", uncertainty)
        if signal_kind is CircuitSignalKind.SPIKE_EVENTS:
            if values:
                raise SchemaValidationError("spike-event signals use spike_times_s, not values")
            measurements = spikes
        else:
            if spikes:
                raise SchemaValidationError("sampled descending signals cannot contain spikes")
            measurements = values
            if signal_kind is CircuitSignalKind.FIRING_RATE and any(
                value < 0.0 for value in values
            ):
                raise SchemaValidationError("descending firing rates must be non-negative")
        if len(availability) != len(measurements):
            raise SchemaValidationError(
                "availability_times_s must match descending measurements"
            )
        if signal_kind is CircuitSignalKind.SPIKE_EVENTS and any(
            available + 1e-15 < measured + delay
            for measured, available in zip(spikes, availability)
        ):
            raise SchemaValidationError(
                "descending signal availability violates encoder_delay_s"
            )
        if uncertainty and len(uncertainty) != len(values):
            raise SchemaValidationError("uncertainty must match values length")
        if any(value < 0.0 for value in uncertainty):
            raise SchemaValidationError("uncertainty must be non-negative")
        source_signal_ids = tuple(self.source_signal_ids)
        for index, source_id in enumerate(source_signal_ids):
            _nonempty(source_id, "source_signal_ids[%d]" % index)
        if len(set(source_signal_ids)) != len(source_signal_ids):
            raise SchemaValidationError("source_signal_ids must be unique")
        if origin in (SignalOrigin.INFERRED, SignalOrigin.SYNTHETIC) and not source_signal_ids:
            raise SchemaValidationError(
                "inferred or synthetic descending signals require source_signal_ids"
            )
        object.__setattr__(self, "source_signal_ids", source_signal_ids)
        if not isinstance(self.provenance, Provenance):
            raise SchemaValidationError("descending provenance must be Provenance")
        if not isinstance(self.confidence, Confidence):
            raise SchemaValidationError("descending confidence must be Confidence")
        _check_schema_version(self.schema_version)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DescendingSignal":
        return cls(
            descending_neuron=EntityRef.from_dict(data["descending_neuron"]),
            signal_kind=data["signal_kind"],
            unit=data["unit"],
            origin=data["origin"],
            encoder_delay_s=data["encoder_delay_s"],
            availability_times_s=tuple(data["availability_times_s"]),
            provenance=Provenance.from_dict(data["provenance"]),
            confidence=Confidence.from_dict(data["confidence"]),
            values=tuple(data.get("values", ())),
            spike_times_s=tuple(data.get("spike_times_s", ())),
            uncertainty=tuple(data.get("uncertainty", ())),
            source_signal_ids=tuple(data.get("source_signal_ids", ())),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class DescendingTrace(JsonSchemaMixin):
    dataset: DatasetRef
    duration_s: float
    sample_times_s: Tuple[float, ...]
    signals: Tuple[DescendingSignal, ...]
    encoder: EncoderProvenance
    input_trace_hash: str
    provenance: Provenance
    confidence: Confidence
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetRef):
            raise SchemaValidationError("descending trace dataset must be DatasetRef")
        duration = _finite(self.duration_s, "duration_s", minimum=0.0)
        if duration <= 0.0:
            raise SchemaValidationError("duration_s must be > 0")
        object.__setattr__(self, "duration_s", duration)
        times = _increasing_times(
            self.sample_times_s, "sample_times_s", allow_empty=True
        )
        if times and times[-1] > duration:
            raise SchemaValidationError("descending sample time exceeds duration_s")
        object.__setattr__(self, "sample_times_s", times)
        signals = tuple(self.signals)
        if not signals:
            raise SchemaValidationError("descending trace must contain signals")
        object.__setattr__(self, "signals", signals)
        if not isinstance(self.encoder, EncoderProvenance):
            raise SchemaValidationError("encoder must be EncoderProvenance")
        if self.encoder.output_dataset.identity_space != self.dataset.identity_space:
            raise CrossAtlasJoinError(
                "encoder output dataset does not match descending trace dataset"
            )
        _nonempty(self.input_trace_hash, "input_trace_hash")
        trace_ref = EntityRef(self.dataset, "trace", EntityKind.CIRCUIT_POPULATION)
        seen = set()
        for signal in signals:
            if not isinstance(signal, DescendingSignal):
                raise SchemaValidationError("signals must contain DescendingSignal objects")
            require_same_identifier_space(trace_ref, signal.descending_neuron)
            key = (signal.descending_neuron.scoped_id, signal.signal_kind.value)
            if key in seen:
                raise SchemaValidationError("descending neuron/signal-kind pairs must be unique")
            seen.add(key)
            if signal.signal_kind is CircuitSignalKind.SPIKE_EVENTS:
                measurements = signal.spike_times_s
                if measurements and measurements[-1] > duration:
                    raise SchemaValidationError("descending spike exceeds duration_s")
            else:
                measurements = times
                if len(signal.values) != len(times):
                    raise SchemaValidationError(
                        "sampled descending signal must match sample_times_s"
                    )
            if any(
                available + 1e-15 < measured + signal.encoder_delay_s
                for measured, available in zip(
                    measurements, signal.availability_times_s
                )
            ):
                raise SchemaValidationError(
                    "descending signal availability violates encoder_delay_s"
                )
        if not isinstance(self.provenance, Provenance):
            raise SchemaValidationError("descending trace provenance must be Provenance")
        if not isinstance(self.confidence, Confidence):
            raise SchemaValidationError("descending trace confidence must be Confidence")
        _check_schema_version(self.schema_version)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DescendingTrace":
        return cls(
            dataset=DatasetRef.from_dict(data["dataset"]),
            duration_s=data["duration_s"],
            sample_times_s=tuple(data.get("sample_times_s", ())),
            signals=tuple(
                DescendingSignal.from_dict(item) for item in data["signals"]
            ),
            encoder=EncoderProvenance.from_dict(data["encoder"]),
            input_trace_hash=data["input_trace_hash"],
            provenance=Provenance.from_dict(data["provenance"]),
            confidence=Confidence.from_dict(data["confidence"]),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class MotorSignal(JsonSchemaMixin):
    motor_neuron: EntityRef
    muscle: EntityRef
    side: AnatomicalSide
    signal_kind: MotorSignalKind
    provenance: Provenance
    confidence: Confidence
    spike_times_s: Tuple[float, ...] = ()
    wingbeat_phase_rad: Tuple[float, ...] = ()
    sample_times_s: Tuple[float, ...] = ()
    rate_hz: Tuple[float, ...] = ()
    schema_version: str = SCHEMA_VERSION
    origin: SignalOrigin = SignalOrigin.UNSPECIFIED
    availability_times_s: Tuple[float, ...] = ()
    source_signal_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.motor_neuron.kind is not EntityKind.MOTOR_NEURON:
            raise SchemaValidationError("motor_neuron must have kind motor_neuron")
        if self.muscle.kind is not EntityKind.MUSCLE:
            raise SchemaValidationError("muscle must have kind muscle")
        require_same_identifier_space(self.motor_neuron, self.muscle)
        side = _enum(AnatomicalSide, self.side, "side")
        if side not in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT, AnatomicalSide.BILATERAL):
            raise SchemaValidationError("motor signal side must be left, right, or bilateral")
        object.__setattr__(self, "side", side)
        for label, entity in (
            ("motor_neuron", self.motor_neuron),
            ("muscle", self.muscle),
        ):
            if entity.anatomical_side is AnatomicalSide.UNKNOWN:
                raise SchemaValidationError(
                    "%s anatomical_side must be declared for a motor signal" % label
                )
            if entity.anatomical_side is not side:
                raise SchemaValidationError(
                    "%s anatomical_side must match motor signal side" % label
                )
        signal_kind = _enum(MotorSignalKind, self.signal_kind, "signal_kind")
        object.__setattr__(self, "signal_kind", signal_kind)
        spikes = _event_times(self.spike_times_s, "spike_times_s")
        phases = _float_tuple(self.wingbeat_phase_rad, "wingbeat_phase_rad")
        sample_times = _increasing_times(self.sample_times_s, "sample_times_s", allow_empty=True)
        rates = _float_tuple(self.rate_hz, "rate_hz")
        availability = _event_times(self.availability_times_s, "availability_times_s")
        source_signal_ids = tuple(self.source_signal_ids)
        for index, source_id in enumerate(source_signal_ids):
            _nonempty(source_id, "source_signal_ids[%d]" % index)
        if len(set(source_signal_ids)) != len(source_signal_ids):
            raise SchemaValidationError("source_signal_ids must be unique")
        object.__setattr__(self, "spike_times_s", spikes)
        object.__setattr__(self, "wingbeat_phase_rad", phases)
        object.__setattr__(self, "sample_times_s", sample_times)
        object.__setattr__(self, "rate_hz", rates)
        object.__setattr__(self, "availability_times_s", availability)
        object.__setattr__(self, "source_signal_ids", source_signal_ids)
        object.__setattr__(self, "origin", _enum(SignalOrigin, self.origin, "origin"))
        _check_schema_version(self.schema_version)
        if any(rate < 0.0 for rate in rates):
            raise SchemaValidationError("rate_hz must be non-negative")
        if phases and len(phases) != len(spikes):
            raise SchemaValidationError("wingbeat_phase_rad must match spike_times_s")
        if any(phase < 0.0 or phase >= 2.0 * math.pi for phase in phases):
            raise SchemaValidationError("wingbeat phases must be in [0, 2*pi)")
        if signal_kind is MotorSignalKind.EXACT_SPIKES:
            if rates or sample_times:
                raise SchemaValidationError("exact spike signals cannot also contain inferred rates")
            measurements = spikes
        else:
            if spikes or phases:
                raise SchemaValidationError("inferred-rate signals cannot claim exact spike events")
            if not rates or len(rates) != len(sample_times):
                raise SchemaValidationError("inferred-rate signals require matching rate_hz and sample_times_s")
            measurements = sample_times
        if availability:
            if len(availability) != len(measurements):
                raise SchemaValidationError(
                    "availability_times_s must match the motor measurements"
                )
            if any(available < measured for measured, available in zip(measurements, availability)):
                raise SchemaValidationError(
                    "motor signal cannot be available before it is measured"
                )


@dataclass(frozen=True)
class WingMotorTrace(JsonSchemaMixin):
    dataset: DatasetRef
    duration_s: float
    signals: Tuple[MotorSignal, ...]
    provenance: Provenance
    confidence: Confidence
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        duration = _finite(self.duration_s, "duration_s", minimum=0.0)
        if duration <= 0.0:
            raise SchemaValidationError("duration_s must be > 0")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "signals", tuple(self.signals))
        _check_schema_version(self.schema_version)
        trace_ref = EntityRef(self.dataset, "trace", EntityKind.CIRCUIT_POPULATION)
        for signal in self.signals:
            require_same_identifier_space(trace_ref, signal.motor_neuron)
            if signal.spike_times_s and signal.spike_times_s[-1] >= duration:
                raise SchemaValidationError("motor spike event is outside trace duration")
            if signal.sample_times_s and signal.sample_times_s[-1] >= duration:
                raise SchemaValidationError("motor rate sample is outside trace duration")


@dataclass(frozen=True)
class MuscleSeries(JsonSchemaMixin):
    muscle: EntityRef
    muscle_class: MuscleClass
    provenance: Provenance
    confidence: Confidence
    activation: Tuple[float, ...]
    calcium_mol_m3: Tuple[float, ...]
    length_m: Tuple[float, ...]
    velocity_m_s: Tuple[float, ...]
    force_n: Tuple[float, ...]
    moment_arm_m: Tuple[float, ...] = ()
    attachment_or_virtual_mapping: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.muscle.kind is not EntityKind.MUSCLE:
            raise SchemaValidationError("muscle must have kind muscle")
        object.__setattr__(self, "muscle_class", _enum(MuscleClass, self.muscle_class, "muscle_class"))
        names = ("activation", "calcium_mol_m3", "length_m", "velocity_m_s", "force_n", "moment_arm_m")
        converted = {name: _float_tuple(getattr(self, name), name) for name in names}
        for name, value in converted.items():
            object.__setattr__(self, name, value)
        lengths = {len(converted[name]) for name in names[:-1]}
        if len(lengths) != 1 or not lengths or 0 in lengths:
            raise SchemaValidationError("muscle state arrays must be non-empty and have equal length")
        if converted["moment_arm_m"] and len(converted["moment_arm_m"]) not in (1, len(converted["activation"])):
            raise SchemaValidationError("moment_arm_m must be scalar-per-series or match the timebase")
        # The reduced asynchronous/steering models retain a dimensionless
        # stretch- or event-amplified activation state.  It is not a
        # probability and is explicitly bounded at 1.5 by the runtime.
        if any(value < 0.0 or value > 1.5 for value in converted["activation"]):
            raise SchemaValidationError("activation must lie in [0, 1.5]")
        for name in ("calcium_mol_m3", "length_m", "force_n"):
            if any(value < 0.0 for value in converted[name]):
                raise SchemaValidationError("%s must be non-negative" % name)
        if not converted["moment_arm_m"] and not self.attachment_or_virtual_mapping:
            raise SchemaValidationError("muscle state requires a moment arm or an attachment/virtual mapping")
        _check_schema_version(self.schema_version)


@dataclass(frozen=True)
class MuscleState(JsonSchemaMixin):
    dataset: DatasetRef
    sample_times_s: Tuple[float, ...]
    muscles: Tuple[MuscleSeries, ...]
    provenance: Provenance
    confidence: Confidence
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        times = _increasing_times(self.sample_times_s, "sample_times_s")
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "muscles", tuple(self.muscles))
        _check_schema_version(self.schema_version)
        trace_ref = EntityRef(self.dataset, "state", EntityKind.CIRCUIT_POPULATION)
        for muscle in self.muscles:
            require_same_identifier_space(trace_ref, muscle.muscle)
            if len(muscle.activation) != len(times):
                raise SchemaValidationError("each muscle series must match sample_times_s")


@dataclass(frozen=True)
class MechanicsFrame(JsonSchemaMixin):
    """One authoritative mechanics sample with exclusive backend ownership."""

    measurement_time_s: float
    availability_time_s: float
    backend: MechanicsBackend
    backend_version: str
    aerodynamics_owner: AerodynamicsOwner
    actuation_owner: ActuationOwner
    body_position_m: Tuple[float, float, float]
    body_orientation_quaternion_wxyz: Tuple[float, float, float, float]
    body_linear_velocity_m_s: Tuple[float, float, float]
    body_angular_velocity_rad_s: Tuple[float, float, float]
    wing_angles_rad: Mapping[str, Tuple[float, ...]]
    wing_angular_velocity_rad_s: Mapping[str, Tuple[float, ...]]
    aerodynamic_force_n: Tuple[float, float, float]
    aerodynamic_moment_n_m: Tuple[float, float, float]
    net_force_n: Tuple[float, float, float]
    net_moment_n_m: Tuple[float, float, float]
    kinetic_energy_j: float
    potential_energy_j: float
    actuator_work_j: float
    aerodynamic_work_j: float
    provenance: Provenance
    confidence: Confidence
    numerical_diagnostics: Mapping[str, Any] = None  # type: ignore[assignment]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        measured = _finite(self.measurement_time_s, "measurement_time_s", minimum=0.0)
        available = _finite(self.availability_time_s, "availability_time_s", minimum=0.0)
        if available < measured:
            raise SchemaValidationError(
                "mechanics frame cannot be available before it is measured"
            )
        object.__setattr__(self, "measurement_time_s", measured)
        object.__setattr__(self, "availability_time_s", available)
        backend = _enum(MechanicsBackend, self.backend, "backend")
        aero_owner = _enum(
            AerodynamicsOwner, self.aerodynamics_owner, "aerodynamics_owner"
        )
        actuation_owner = _enum(
            ActuationOwner, self.actuation_owner, "actuation_owner"
        )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "aerodynamics_owner", aero_owner)
        object.__setattr__(self, "actuation_owner", actuation_owner)
        _nonempty(self.backend_version, "backend_version")
        allowed_aerodynamics = {
            MechanicsBackend.REDUCED_ANALYTIC: {
                AerodynamicsOwner.NONE,
                AerodynamicsOwner.REDUCED_ANALYTIC,
            },
            MechanicsBackend.FLYBODY_MUJOCO: {
                AerodynamicsOwner.NONE,
                AerodynamicsOwner.FLYBODY,
            },
            MechanicsBackend.OFFLINE_CFD: {AerodynamicsOwner.CFD},
        }[backend]
        if aero_owner not in allowed_aerodynamics:
            raise SchemaValidationError(
                "aerodynamics_owner is incompatible with mechanics backend"
            )
        if (
            actuation_owner is ActuationOwner.FLYBODY_NATIVE
            and backend is not MechanicsBackend.FLYBODY_MUJOCO
        ):
            raise SchemaValidationError(
                "flybody_native actuation requires the FlyBody MuJoCo backend"
            )
        if (
            backend is MechanicsBackend.OFFLINE_CFD
            and actuation_owner is not ActuationOwner.PRESCRIBED_KINEMATICS
        ):
            raise SchemaValidationError(
                "offline CFD verification requires prescribed kinematics"
            )
        for name, length in (
            ("body_position_m", 3),
            ("body_orientation_quaternion_wxyz", 4),
            ("body_linear_velocity_m_s", 3),
            ("body_angular_velocity_rad_s", 3),
            ("aerodynamic_force_n", 3),
            ("aerodynamic_moment_n_m", 3),
            ("net_force_n", 3),
            ("net_moment_n_m", 3),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), length, name))
        norm = math.sqrt(
            sum(value * value for value in self.body_orientation_quaternion_wxyz)
        )
        if abs(norm - 1.0) > 1e-6:
            raise SchemaValidationError("body orientation quaternion must have unit norm")
        wing_angles: Dict[str, Tuple[float, ...]] = {}
        wing_velocities: Dict[str, Tuple[float, ...]] = {}
        expected_wing_keys = {
            AnatomicalSide.LEFT.value,
            AnatomicalSide.RIGHT.value,
        }
        if set(self.wing_angles_rad) != expected_wing_keys or set(
            self.wing_angular_velocity_rad_s
        ) != expected_wing_keys:
            raise SchemaValidationError(
                "wing angle and velocity mappings must contain exactly left and right"
            )
        for side in (AnatomicalSide.LEFT.value, AnatomicalSide.RIGHT.value):
            angles = _float_tuple(self.wing_angles_rad[side], "wing_angles_rad[%s]" % side)
            velocities = _float_tuple(
                self.wing_angular_velocity_rad_s[side],
                "wing_angular_velocity_rad_s[%s]" % side,
            )
            if not angles or len(angles) != len(velocities):
                raise SchemaValidationError(
                    "each wing requires matching non-empty angle and velocity vectors"
                )
            wing_angles[side] = angles
            wing_velocities[side] = velocities
        object.__setattr__(self, "wing_angles_rad", wing_angles)
        object.__setattr__(self, "wing_angular_velocity_rad_s", wing_velocities)
        if aero_owner is AerodynamicsOwner.NONE and any(
            value != 0.0
            for value in self.aerodynamic_force_n + self.aerodynamic_moment_n_m
        ):
            raise SchemaValidationError(
                "aerodynamic wrench must be zero when aerodynamics_owner is none"
            )
        for name in (
            "kinetic_energy_j",
            "potential_energy_j",
            "actuator_work_j",
            "aerodynamic_work_j",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.kinetic_energy_j < 0.0:
            raise SchemaValidationError("kinetic_energy_j must be non-negative")
        if not isinstance(self.provenance, Provenance):
            raise SchemaValidationError("mechanics provenance must be Provenance")
        if not isinstance(self.confidence, Confidence):
            raise SchemaValidationError("mechanics confidence must be Confidence")
        if self.numerical_diagnostics is None:
            object.__setattr__(self, "numerical_diagnostics", {})
        elif not isinstance(self.numerical_diagnostics, Mapping):
            raise SchemaValidationError("numerical_diagnostics must be a mapping")
        _check_schema_version(self.schema_version)

    @property
    def mechanical_energy_j(self) -> float:
        return self.kinetic_energy_j + self.potential_energy_j

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MechanicsFrame":
        return cls(
            measurement_time_s=data["measurement_time_s"],
            availability_time_s=data["availability_time_s"],
            backend=data["backend"],
            backend_version=data["backend_version"],
            aerodynamics_owner=data["aerodynamics_owner"],
            actuation_owner=data["actuation_owner"],
            body_position_m=tuple(data["body_position_m"]),
            body_orientation_quaternion_wxyz=tuple(
                data["body_orientation_quaternion_wxyz"]
            ),
            body_linear_velocity_m_s=tuple(data["body_linear_velocity_m_s"]),
            body_angular_velocity_rad_s=tuple(data["body_angular_velocity_rad_s"]),
            wing_angles_rad={
                str(side): tuple(values)
                for side, values in data["wing_angles_rad"].items()
            },
            wing_angular_velocity_rad_s={
                str(side): tuple(values)
                for side, values in data["wing_angular_velocity_rad_s"].items()
            },
            aerodynamic_force_n=tuple(data["aerodynamic_force_n"]),
            aerodynamic_moment_n_m=tuple(data["aerodynamic_moment_n_m"]),
            net_force_n=tuple(data["net_force_n"]),
            net_moment_n_m=tuple(data["net_moment_n_m"]),
            kinetic_energy_j=data["kinetic_energy_j"],
            potential_energy_j=data["potential_energy_j"],
            actuator_work_j=data["actuator_work_j"],
            aerodynamic_work_j=data["aerodynamic_work_j"],
            provenance=Provenance.from_dict(data["provenance"]),
            confidence=Confidence.from_dict(data["confidence"]),
            numerical_diagnostics=data.get("numerical_diagnostics", {}),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class FeedbackFrame(JsonSchemaMixin):
    timestamp_s: float
    retinal_irradiance_w_m2: Tuple[float, ...]
    body_position_m: Tuple[float, float, float]
    body_orientation_quaternion_wxyz: Tuple[float, float, float, float]
    body_linear_velocity_m_s: Tuple[float, float, float]
    body_angular_velocity_rad_s: Tuple[float, float, float]
    wing_phase_rad: Tuple[float, float]
    wing_strain: Tuple[float, float]
    haltere_angular_velocity_rad_s: Tuple[float, float, float]
    latency_s: float
    provenance: Provenance
    confidence: Confidence
    schema_version: str = SCHEMA_VERSION
    availability_time_s: Optional[float] = None
    source_frame_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s", minimum=0.0))
        retinal = _float_tuple(self.retinal_irradiance_w_m2, "retinal_irradiance_w_m2")
        if not retinal or any(value < 0.0 for value in retinal):
            raise SchemaValidationError("retinal irradiance must be non-empty and non-negative")
        object.__setattr__(self, "retinal_irradiance_w_m2", retinal)
        for name, length in (
            ("body_position_m", 3),
            ("body_orientation_quaternion_wxyz", 4),
            ("body_linear_velocity_m_s", 3),
            ("body_angular_velocity_rad_s", 3),
            ("wing_phase_rad", 2),
            ("wing_strain", 2),
            ("haltere_angular_velocity_rad_s", 3),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), length, name))
        norm = math.sqrt(sum(value * value for value in self.body_orientation_quaternion_wxyz))
        if abs(norm - 1.0) > 1e-6:
            raise SchemaValidationError("body orientation quaternion must have unit norm")
        if any(phase < 0.0 or phase >= 2.0 * math.pi for phase in self.wing_phase_rad):
            raise SchemaValidationError("wing phases must be in [0, 2*pi)")
        object.__setattr__(self, "latency_s", _finite(self.latency_s, "latency_s", minimum=0.0))
        expected_availability = self.timestamp_s + self.latency_s
        if self.availability_time_s is None:
            object.__setattr__(self, "availability_time_s", expected_availability)
        else:
            available = _finite(
                self.availability_time_s, "availability_time_s", minimum=0.0
            )
            if abs(available - expected_availability) > 1e-12:
                raise SchemaValidationError(
                    "availability_time_s must equal timestamp_s + latency_s"
                )
            object.__setattr__(self, "availability_time_s", available)
        source_frame_ids = tuple(self.source_frame_ids)
        for index, source_id in enumerate(source_frame_ids):
            _nonempty(source_id, "source_frame_ids[%d]" % index)
        if len(set(source_frame_ids)) != len(source_frame_ids):
            raise SchemaValidationError("source_frame_ids must be unique")
        object.__setattr__(self, "source_frame_ids", source_frame_ids)
        _check_schema_version(self.schema_version)


@dataclass(frozen=True)
class PerturbationSpec(JsonSchemaMixin):
    target_kind: str
    target_id: str
    mode: str
    start_s: float
    end_s: float
    magnitude: float = 1.0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.target_kind, "target_kind")
        _nonempty(self.target_id, "target_id")
        _nonempty(self.mode, "mode")
        start = _finite(self.start_s, "start_s", minimum=0.0)
        end = _finite(self.end_s, "end_s", minimum=0.0)
        if end <= start:
            raise SchemaValidationError("perturbation end_s must be greater than start_s")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)
        object.__setattr__(self, "magnitude", _finite(self.magnitude, "magnitude"))
        _check_schema_version(self.schema_version)


@dataclass(frozen=True)
class FlightEpisodeConfig(JsonSchemaMixin):
    episode_id: str
    scenario: str
    duration_s: float
    physics_timestep_s: float
    neural_timestep_s: float
    muscle_timestep_s: float
    render_timestep_s: float
    seed: int
    circuit_snapshot: str
    model_hashes: Mapping[str, str]
    world: Mapping[str, Any]
    perturbations: Tuple[PerturbationSpec, ...] = ()
    starts_airborne: bool = True
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.episode_id, "episode_id")
        _nonempty(self.scenario, "scenario")
        _nonempty(self.circuit_snapshot, "circuit_snapshot")
        duration = _finite(self.duration_s, "duration_s", minimum=0.0)
        if duration <= 0.0:
            raise SchemaValidationError("duration_s must be > 0")
        object.__setattr__(self, "duration_s", duration)
        for name in ("physics_timestep_s", "neural_timestep_s", "muscle_timestep_s", "render_timestep_s"):
            value = _finite(getattr(self, name), name, minimum=0.0)
            if value <= 0.0:
                raise SchemaValidationError("%s must be > 0" % name)
            object.__setattr__(self, name, value)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise SchemaValidationError("seed must be a non-negative integer")
        if not isinstance(self.model_hashes, Mapping) or not self.model_hashes:
            raise SchemaValidationError("model_hashes must be a non-empty mapping")
        for name, digest in self.model_hashes.items():
            _nonempty(name, "model name")
            _nonempty(digest, "model hash")
        if not isinstance(self.world, Mapping):
            raise SchemaValidationError("world must be a mapping")
        object.__setattr__(self, "perturbations", tuple(self.perturbations))
        if any(item.end_s > duration for item in self.perturbations):
            raise SchemaValidationError("perturbation extends beyond episode duration")
        _check_schema_version(self.schema_version)


@dataclass(frozen=True)
class CoverageSummary(JsonSchemaMixin):
    branch: str
    total_structural_synapses: int
    resolved_structural_synapses: int
    unresolved_structural_synapses: int
    provenance: Provenance
    confidence: Confidence
    scope: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.branch, "branch")
        _nonempty(self.scope, "scope")
        counts = (
            self.total_structural_synapses,
            self.resolved_structural_synapses,
            self.unresolved_structural_synapses,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
            raise SchemaValidationError("coverage counts must be non-negative integers")
        if self.resolved_structural_synapses + self.unresolved_structural_synapses != self.total_structural_synapses:
            raise SchemaValidationError("resolved + unresolved must equal total structural synapses")
        _check_schema_version(self.schema_version)

    @property
    def resolved_fraction(self) -> float:
        if self.total_structural_synapses == 0:
            return 0.0
        return self.resolved_structural_synapses / self.total_structural_synapses


@dataclass(frozen=True)
class FlightEpisodeResult(JsonSchemaMixin):
    config: FlightEpisodeConfig
    status: EpisodeStatus
    validation_status: ValidationStatus
    sample_times_s: Tuple[float, ...]
    body_position_m: Tuple[Tuple[float, ...], ...]
    body_orientation_quaternion_wxyz: Tuple[Tuple[float, ...], ...]
    body_linear_velocity_m_s: Tuple[Tuple[float, ...], ...]
    body_angular_velocity_rad_s: Tuple[Tuple[float, ...], ...]
    wing_angles_rad: Mapping[str, Tuple[Tuple[float, ...], ...]]
    aerodynamic_force_n: Tuple[Tuple[float, ...], ...]
    aerodynamic_moment_n_m: Tuple[Tuple[float, ...], ...]
    energy_j: Tuple[float, ...]
    coverage: Tuple[CoverageSummary, ...]
    uncertainty: Mapping[str, Any]
    numerical_diagnostics: Mapping[str, Any]
    provenance: Provenance
    confidence: Confidence
    circuit_output: Optional[CircuitOutputTrace] = None
    wing_motor_trace: Optional[WingMotorTrace] = None
    muscle_state: Optional[MuscleState] = None
    feedback_frames: Tuple[FeedbackFrame, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(EpisodeStatus, self.status, "status"))
        object.__setattr__(
            self, "validation_status", _enum(ValidationStatus, self.validation_status, "validation_status")
        )
        times = _increasing_times(self.sample_times_s, "sample_times_s")
        object.__setattr__(self, "sample_times_s", times)
        series_spec = (
            ("body_position_m", 3),
            ("body_orientation_quaternion_wxyz", 4),
            ("body_linear_velocity_m_s", 3),
            ("body_angular_velocity_rad_s", 3),
            ("aerodynamic_force_n", 3),
            ("aerodynamic_moment_n_m", 3),
        )
        for name, length in series_spec:
            values = _vectors(getattr(self, name), length, name)
            if len(values) != len(times):
                raise SchemaValidationError("%s must match sample_times_s" % name)
            object.__setattr__(self, name, values)
        if any(
            abs(sum(component * component for component in quaternion) - 1.0)
            > 1.0e-6
            for quaternion in self.body_orientation_quaternion_wxyz
        ):
            raise SchemaValidationError(
                "body_orientation_quaternion_wxyz must contain unit quaternions"
            )
        energy = _float_tuple(self.energy_j, "energy_j")
        if len(energy) != len(times):
            raise SchemaValidationError("energy_j must match sample_times_s")
        object.__setattr__(self, "energy_j", energy)
        for side in ("left", "right"):
            if side not in self.wing_angles_rad:
                raise SchemaValidationError("wing_angles_rad requires left and right series")
            values = _vectors(self.wing_angles_rad[side], 3, "wing_angles_rad[%s]" % side)
            if len(values) != len(times):
                raise SchemaValidationError("wing angle series must match sample_times_s")
        object.__setattr__(self, "coverage", tuple(self.coverage))
        object.__setattr__(self, "feedback_frames", tuple(self.feedback_frames))
        if abs(times[0]) > 1.0e-12:
            raise SchemaValidationError("result timebase must begin at zero")
        if abs(times[-1] - self.config.duration_s) > 1.0e-12:
            raise SchemaValidationError(
                "result timebase endpoint must equal configured duration"
            )
        _check_schema_version(self.schema_version)


@dataclass(frozen=True)
class ModelParameter(JsonSchemaMixin):
    name: str
    mean: float
    unit: str
    standard_deviation: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.name, "name")
        _require_unit(self.unit)
        object.__setattr__(self, "mean", _finite(self.mean, "mean"))
        if self.standard_deviation is not None:
            object.__setattr__(
                self,
                "standard_deviation",
                _finite(self.standard_deviation, "standard_deviation", minimum=0.0),
            )
        if self.lower_bound is not None:
            object.__setattr__(self, "lower_bound", _finite(self.lower_bound, "lower_bound"))
        if self.upper_bound is not None:
            object.__setattr__(self, "upper_bound", _finite(self.upper_bound, "upper_bound"))
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise SchemaValidationError("parameter lower_bound cannot exceed upper_bound")
        if self.lower_bound is not None and self.mean < self.lower_bound:
            raise SchemaValidationError("parameter mean is below lower_bound")
        if self.upper_bound is not None and self.mean > self.upper_bound:
            raise SchemaValidationError("parameter mean is above upper_bound")
        _check_schema_version(self.schema_version)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelParameter":
        return cls(
            name=data["name"],
            mean=data["mean"],
            unit=data["unit"],
            standard_deviation=data.get("standard_deviation"),
            lower_bound=data.get("lower_bound"),
            upper_bound=data.get("upper_bound"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class ModelRecord(JsonSchemaMixin):
    name: str
    version: str
    role: str
    artifact_hash: str
    artifact_uri: str
    parameters: Tuple[ModelParameter, ...]
    calibration_dataset: Optional[DatasetRef]
    provenance: Provenance
    confidence: Confidence
    validation_status: ValidationStatus
    license: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("name", "version", "role", "artifact_hash", "artifact_uri", "license"):
            _nonempty(getattr(self, name), name)
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(
            self, "validation_status", _enum(ValidationStatus, self.validation_status, "validation_status")
        )
        if len({parameter.name for parameter in self.parameters}) != len(self.parameters):
            raise SchemaValidationError("model parameter names must be unique")
        _check_schema_version(self.schema_version)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelRecord":
        calibration = data.get("calibration_dataset")
        return cls(
            name=data["name"],
            version=data["version"],
            role=data["role"],
            artifact_hash=data["artifact_hash"],
            artifact_uri=data["artifact_uri"],
            parameters=tuple(
                ModelParameter.from_dict(item) for item in data.get("parameters", ())
            ),
            calibration_dataset=(
                None if calibration is None else DatasetRef.from_dict(calibration)
            ),
            provenance=Provenance.from_dict(data["provenance"]),
            confidence=Confidence.from_dict(data["confidence"]),
            validation_status=data["validation_status"],
            license=data["license"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class ModelRegistry(JsonSchemaMixin):
    models: Tuple[ModelRecord, ...]
    generated_at_utc: str
    provenance: Provenance
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(self.models))
        _nonempty(self.generated_at_utc, "generated_at_utc")
        keys = [(model.name, model.version) for model in self.models]
        if len(set(keys)) != len(keys):
            raise SchemaValidationError("model name/version pairs must be unique")
        _check_schema_version(self.schema_version)

    def get(self, name: str, version: Optional[str] = None) -> ModelRecord:
        matches = [model for model in self.models if model.name == name and (version is None or model.version == version)]
        if len(matches) != 1:
            raise KeyError("expected one model for %s/%s, found %d" % (name, version or "*", len(matches)))
        return matches[0]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelRegistry":
        return cls(
            models=tuple(ModelRecord.from_dict(item) for item in data["models"]),
            generated_at_utc=data["generated_at_utc"],
            provenance=Provenance.from_dict(data["provenance"]),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


__all__ = [
    "SCHEMA_VERSION",
    "FAFB_V783_MATERIALIZATION",
    "FAFB_MIDLINE_X_NM",
    "SchemaValidationError",
    "CrossAtlasJoinError",
    "DatasetNamespace",
    "DatasetRef",
    "EntityKind",
    "EntityRef",
    "AnatomicalSide",
    "EyeSide",
    "SideMappingMethod",
    "SideContext",
    "EvidenceTier",
    "ConfidenceLevel",
    "Confidence",
    "Provenance",
    "CircuitSignalKind",
    "SignalOrigin",
    "CircuitSignal",
    "CircuitOutputTrace",
    "RetinalSignalKind",
    "RetinalFrame",
    "AtlasMappingKind",
    "EncoderProvenance",
    "DescendingSignal",
    "DescendingTrace",
    "MotorSignalKind",
    "MotorSignal",
    "WingMotorTrace",
    "MuscleClass",
    "MuscleSeries",
    "MuscleState",
    "MechanicsBackend",
    "AerodynamicsOwner",
    "ActuationOwner",
    "MechanicsFrame",
    "FeedbackFrame",
    "PerturbationSpec",
    "FlightEpisodeConfig",
    "CoverageSummary",
    "EpisodeStatus",
    "ValidationStatus",
    "FlightEpisodeResult",
    "ModelParameter",
    "ModelRecord",
    "ModelRegistry",
    "require_same_identifier_space",
    "fafb_anatomical_side_from_soma_x_nm",
]
