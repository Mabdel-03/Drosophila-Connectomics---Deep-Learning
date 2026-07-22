"""Stateful online bridge from the canonical fly-FGS NOD1 output to wing MNs.

This module is the causal, re-entrant counterpart to :mod:`.bridge`.  It
accepts only the four individual NOD1 voltages exposed by
``FlyFGSCircuitSample`` and advances on an exact 0.5 ms grid.  The source
application's ``L``/``R`` labels are deliberately retained as *raw display
labels*: every public channel explicitly carries ``AnatomicalSide.UNKNOWN``.

The model is exploratory.  Functional gains and preferred phases are explicit
parameters, never connectome synapse counts.  Seeded synthetic steering events
are emitted only when the caller-supplied, unwrapped wing phase crosses an
individual muscle's preferred phase.  Encoder, VNC, and NMJ delays are
separate causal queues, and a hash-protected checkpoint contains every queue,
hold, RNG, phase-history, event, and intervention datum needed for exact
fresh-runtime continuation.

``StreamingBridgeStage.MUSCLE`` is a historical command-generation label: it
modulates the inferred MN rate before event generation and cannot silence
already pending NMJ events or pre-existing muscle force.  Experiments that
claim a mechanics-level muscle perturbation must instead use the immutable
force-stage contracts in :mod:`.streaming_mechanics`.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..fly_fgs import FLY_FGS_NOD1_RAW_APP_SIDES, FLY_FGS_NOD1_ROOT_IDS
from ..fly_fgs_runtime import FlyFGSCircuitSample
from ..schema import AnatomicalSide


STREAMING_BRIDGE_SCHEMA_VERSION = "1.0.0"
STREAMING_BRIDGE_RUNTIME_VERSION = "1.2.0"
STREAMING_BRIDGE_DT_S = 0.0005
_CANONICAL_CIRCUIT_DT_S = 0.005
_TWO_PI = 2.0 * math.pi
_TIME_TOLERANCE_S = 1e-12
_EXPECTED_PATHWAY_IDENTITIES = (
    ("MN-iv2", "iv2"),
    ("MN-i1", "i1"),
    ("MN-iv1", "iv1"),
    ("MN-b3", "b3"),
)


class RawAppSide(str, Enum):
    """Uninterpreted side label used by the captured fly-FGS application."""

    L = "L"
    R = "R"

    @property
    def mirror(self) -> "RawAppSide":
        return RawAppSide.R if self is RawAppSide.L else RawAppSide.L


class StreamingBridgeStage(str, Enum):
    NOD1 = "nod1"
    DN = "dn"
    MN = "mn"
    MUSCLE = "muscle"


class StreamingInterventionMode(str, Enum):
    SILENCE = "silence"
    ACTIVATE = "activate"
    SCALE = "scale"


class StreamingSignalSemantics(str, Enum):
    INFERRED_RATE = "inferred_rate"
    SEEDED_SYNTHETIC = "seeded_synthetic"


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError("%s must be a finite number" % label)
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite number" % label) from exc
    if not math.isfinite(converted):
        raise ValueError("%s must be a finite number" % label)
    return converted


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % label)
    return value


def _on_grid(time_s: float, dt_s: float = STREAMING_BRIDGE_DT_S) -> bool:
    index = round(time_s / dt_s)
    return abs(index * dt_s - time_s) <= _TIME_TOLERANCE_S


def _require_grid(time_s: float, label: str) -> float:
    value = _finite(time_s, label)
    if value < 0.0 or not _on_grid(value):
        raise ValueError("%s must be non-negative and on the 0.5 ms grid" % label)
    # Persist the integer-grid reconstruction, not the caller's potentially
    # drifted binary representation (for example 0.0045000000000000005).
    return round(value / STREAMING_BRIDGE_DT_S) * STREAMING_BRIDGE_DT_S


def _channel_key(
    side: RawAppSide, motor_neuron: str, muscle: str
) -> str:
    return "%s|%s|%s" % (side.value, motor_neuron, muscle)


def _split_channel_key(value: str) -> Tuple[RawAppSide, str, str]:
    if not isinstance(value, str):
        raise ValueError("motor channel key must be a string")
    parts = value.split("|")
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise ValueError("invalid motor channel key")
    return RawAppSide(parts[0]), parts[1], parts[2]


def _channel_seed(seed: int, side: RawAppSide, motor: str, muscle: str) -> int:
    payload = "%d|%s|%s|%s" % (seed, side.value, motor, muscle)
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(child) for child in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _deep_thaw(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(payload_without_digest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(payload_without_digest).encode("utf-8")
    ).hexdigest()


def _exact_mapping(
    value: Any, expected: Sequence[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("%s fields do not match the checkpoint schema" % label)
    return value


@dataclass(frozen=True)
class StreamingMotorPathway:
    """One explicit exploratory DNp26-to-wing-MN functional route."""

    motor_neuron: str
    muscle: str
    functional_rate_gain: float
    preferred_phase_rad: float
    structural_supported: bool = True
    structural_synapse_count: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.motor_neuron, str) or not self.motor_neuron:
            raise ValueError("motor_neuron must be a non-empty string")
        if not isinstance(self.muscle, str) or not self.muscle:
            raise ValueError("muscle must be a non-empty string")
        gain = _finite(self.functional_rate_gain, "functional_rate_gain")
        if gain < 0.0:
            raise ValueError("functional_rate_gain must be non-negative")
        phase = _finite(self.preferred_phase_rad, "preferred_phase_rad") % _TWO_PI
        if not isinstance(self.structural_supported, bool):
            raise TypeError("structural_supported must be boolean")
        if self.structural_synapse_count is not None:
            if (
                isinstance(self.structural_synapse_count, bool)
                or not isinstance(self.structural_synapse_count, int)
                or self.structural_synapse_count <= 0
            ):
                raise ValueError(
                    "structural_synapse_count must be a positive integer"
                )
        object.__setattr__(self, "functional_rate_gain", gain)
        object.__setattr__(self, "preferred_phase_rad", phase)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "motor_neuron": self.motor_neuron,
            "muscle": self.muscle,
            "functional_rate_gain": self.functional_rate_gain,
            "preferred_phase_rad": self.preferred_phase_rad,
            "structural_supported": self.structural_supported,
            "structural_synapse_count": self.structural_synapse_count,
        }


def default_streaming_motor_pathways() -> Tuple[StreamingMotorPathway, ...]:
    """Return the same explicit provisional gains as the frozen batch bridge."""

    return (
        StreamingMotorPathway("MN-iv2", "iv2", 0.60, 0.20 * _TWO_PI, True, 53),
        StreamingMotorPathway("MN-i1", "i1", 0.50, 0.35 * _TWO_PI, True, 29),
        StreamingMotorPathway("MN-iv1", "iv1", 0.40, 0.55 * _TWO_PI, True, 21),
        StreamingMotorPathway("MN-b3", "b3", 0.35, 0.75 * _TWO_PI, True, 14),
    )


@dataclass(frozen=True)
class StreamingBridgeConfig:
    """Numerical and functional configuration for the online bridge."""

    base_dt_s: float = STREAMING_BRIDGE_DT_S
    encoder_delay_s: float = 0.003
    vnc_delay_s: float = 0.002
    nmj_delay_s: float = 0.0005
    resting_voltage_v: float = -0.060
    voltage_scale_v: float = 0.005
    dn_baseline_rate_hz: float = 0.0
    dn_functional_gain_hz: float = 40.0
    maximum_dn_rate_hz: float = 200.0
    maximum_motor_rate_hz: float = 200.0
    initial_wingbeat_frequency_hz: float = 200.0
    source_cell_type: str = "NOD1"
    target_dn_type: str = "DNp26"
    contralateral_raw_app_lanes: bool = True
    pathways: Tuple[StreamingMotorPathway, ...] = field(
        default_factory=default_streaming_motor_pathways
    )

    def __post_init__(self) -> None:
        base_dt = _finite(self.base_dt_s, "base_dt_s")
        if abs(base_dt - STREAMING_BRIDGE_DT_S) > 1e-15:
            raise ValueError("the online bridge requires an exact 0.5 ms base grid")
        for name in ("encoder_delay_s", "vnc_delay_s", "nmj_delay_s"):
            object.__setattr__(self, name, _require_grid(getattr(self, name), name))
        if self.nmj_delay_s + _TIME_TOLERANCE_S < base_dt:
            raise ValueError(
                "nmj_delay_s must be at least one 0.5 ms bridge interval"
            )
        voltage_scale = _finite(self.voltage_scale_v, "voltage_scale_v")
        if voltage_scale <= 0.0:
            raise ValueError("voltage_scale_v must be positive")
        for name in (
            "dn_baseline_rate_hz",
            "dn_functional_gain_hz",
            "maximum_dn_rate_hz",
            "maximum_motor_rate_hz",
            "initial_wingbeat_frequency_hz",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0 or (
                name
                in (
                    "maximum_dn_rate_hz",
                    "maximum_motor_rate_hz",
                    "initial_wingbeat_frequency_hz",
                )
                and value <= 0.0
            ):
                raise ValueError("%s has an invalid rate" % name)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "resting_voltage_v", _finite(self.resting_voltage_v, "resting_voltage_v")
        )
        object.__setattr__(self, "voltage_scale_v", voltage_scale)
        if not self.source_cell_type or not self.target_dn_type:
            raise ValueError("source_cell_type and target_dn_type must be non-empty")
        if not isinstance(self.contralateral_raw_app_lanes, bool):
            raise TypeError("contralateral_raw_app_lanes must be boolean")
        pathways = tuple(self.pathways)
        identities = tuple((path.motor_neuron, path.muscle) for path in pathways)
        if identities != _EXPECTED_PATHWAY_IDENTITIES:
            raise ValueError(
                "streaming pathways must be the ordered iv2/i1/iv1/b3 inventory"
            )
        object.__setattr__(self, "pathways", pathways)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "base_dt_s": self.base_dt_s,
            "encoder_delay_s": self.encoder_delay_s,
            "vnc_delay_s": self.vnc_delay_s,
            "nmj_delay_s": self.nmj_delay_s,
            "resting_voltage_v": self.resting_voltage_v,
            "voltage_scale_v": self.voltage_scale_v,
            "dn_baseline_rate_hz": self.dn_baseline_rate_hz,
            "dn_functional_gain_hz": self.dn_functional_gain_hz,
            "maximum_dn_rate_hz": self.maximum_dn_rate_hz,
            "maximum_motor_rate_hz": self.maximum_motor_rate_hz,
            "initial_wingbeat_frequency_hz": self.initial_wingbeat_frequency_hz,
            "source_cell_type": self.source_cell_type,
            "target_dn_type": self.target_dn_type,
            "contralateral_raw_app_lanes": self.contralateral_raw_app_lanes,
            "pathways": [path.to_dict() for path in self.pathways],
        }


@dataclass(frozen=True)
class StreamingBridgeIntervention:
    """Grid-aligned command intervention on raw app lanes, not anatomy.

    The ``MUSCLE`` target stage acts on pre-event inferred motor command rate;
    it is not a force-stage muscle intervention.
    """

    intervention_id: str
    target_stage: StreamingBridgeStage
    target_name: str
    raw_app_side: Optional[RawAppSide]
    mode: StreamingInterventionMode
    start_s: float
    end_s: float
    magnitude: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.intervention_id, str) or not self.intervention_id:
            raise ValueError("intervention_id must be a non-empty string")
        if not isinstance(self.target_stage, StreamingBridgeStage):
            raise TypeError("target_stage must be a StreamingBridgeStage")
        if not isinstance(self.target_name, str) or not self.target_name:
            raise ValueError("target_name must be a non-empty string")
        if self.raw_app_side is not None and not isinstance(
            self.raw_app_side, RawAppSide
        ):
            raise TypeError("raw_app_side must be RawAppSide or None")
        if not isinstance(self.mode, StreamingInterventionMode):
            raise TypeError("mode must be a StreamingInterventionMode")
        start = _require_grid(self.start_s, "intervention start_s")
        end = _require_grid(self.end_s, "intervention end_s")
        if end <= start:
            raise ValueError("intervention end_s must be greater than start_s")
        magnitude = _finite(self.magnitude, "intervention magnitude")
        if self.mode in (
            StreamingInterventionMode.SCALE,
            StreamingInterventionMode.ACTIVATE,
        ) and magnitude < 0.0:
            raise ValueError("scale and activation magnitudes must be non-negative")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)
        object.__setattr__(self, "magnitude", magnitude)

    def active(self, time_s: float) -> bool:
        return self.start_s <= time_s < self.end_s

    def matches(
        self,
        stage: StreamingBridgeStage,
        target_name: str,
        side: RawAppSide,
        time_s: float,
    ) -> bool:
        return (
            self.target_stage is stage
            and self.target_name == target_name
            and (self.raw_app_side is None or self.raw_app_side is side)
            and self.active(time_s)
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "target_stage": self.target_stage.value,
            "target_name": self.target_name,
            "raw_app_side": (
                None if self.raw_app_side is None else self.raw_app_side.value
            ),
            "mode": self.mode.value,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "magnitude": self.magnitude,
        }


@dataclass(frozen=True)
class StreamingNOD1Hold:
    root_id: str
    raw_app_side: RawAppSide
    anatomical_side: AnatomicalSide
    sample_index: int
    measurement_time_s: float
    availability_time_s: float
    voltage_v: float

    def __post_init__(self) -> None:
        if self.root_id not in FLY_FGS_NOD1_ROOT_IDS:
            raise ValueError("NOD1 hold root ID is outside the four-channel inventory")
        if not isinstance(self.raw_app_side, RawAppSide):
            raise TypeError("raw_app_side must be RawAppSide")
        if FLY_FGS_NOD1_RAW_APP_SIDES[self.root_id] != self.raw_app_side.value:
            raise ValueError("NOD1 hold raw app side does not match the source receipt")
        if self.anatomical_side is not AnatomicalSide.UNKNOWN:
            raise ValueError("fly-FGS raw application labels are not anatomical sides")
        _nonnegative_int(self.sample_index, "sample_index")
        measurement = _require_grid(self.measurement_time_s, "measurement_time_s")
        availability = _require_grid(self.availability_time_s, "availability_time_s")
        if availability + _TIME_TOLERANCE_S < measurement:
            raise ValueError("NOD1 availability cannot precede measurement")
        object.__setattr__(self, "measurement_time_s", measurement)
        object.__setattr__(self, "availability_time_s", availability)
        object.__setattr__(self, "voltage_v", _finite(self.voltage_v, "voltage_v"))


@dataclass(frozen=True)
class StreamingRateSample:
    stage: StreamingBridgeStage
    target_name: str
    raw_app_side: RawAppSide
    anatomical_side: AnatomicalSide
    measurement_time_s: float
    availability_time_s: float
    rate_hz: float
    source_measurement_time_s: Optional[float]
    semantics: StreamingSignalSemantics = StreamingSignalSemantics.INFERRED_RATE

    def __post_init__(self) -> None:
        if self.stage not in (StreamingBridgeStage.DN, StreamingBridgeStage.MN):
            raise ValueError("rate samples are defined only for DN and MN stages")
        if not self.target_name:
            raise ValueError("rate target_name must be non-empty")
        if not isinstance(self.raw_app_side, RawAppSide):
            raise TypeError("raw_app_side must be RawAppSide")
        if self.anatomical_side is not AnatomicalSide.UNKNOWN:
            raise ValueError("raw application lanes must retain unknown anatomy")
        measurement = _require_grid(self.measurement_time_s, "rate measurement_time_s")
        availability = _require_grid(self.availability_time_s, "rate availability_time_s")
        if availability + _TIME_TOLERANCE_S < measurement:
            raise ValueError("rate availability cannot precede measurement")
        rate = _finite(self.rate_hz, "rate_hz")
        if rate < 0.0:
            raise ValueError("rate_hz must be non-negative")
        if self.source_measurement_time_s is not None:
            source = _require_grid(
                self.source_measurement_time_s, "source_measurement_time_s"
            )
            if source > measurement + _TIME_TOLERANCE_S:
                raise ValueError("source measurement cannot lie in the future")
            object.__setattr__(self, "source_measurement_time_s", source)
        if self.semantics is not StreamingSignalSemantics.INFERRED_RATE:
            raise ValueError("streaming rates must retain inferred-rate semantics")
        object.__setattr__(self, "measurement_time_s", measurement)
        object.__setattr__(self, "availability_time_s", availability)
        object.__setattr__(self, "rate_hz", rate)


@dataclass(frozen=True)
class StreamingMotorEvent:
    event_id: str
    motor_neuron: str
    muscle: str
    raw_app_side: RawAppSide
    anatomical_side: AnatomicalSide
    event_time_s: float
    availability_time_s: float
    wingbeat_phase_rad: float
    rate_hz: float
    emission_probability: float
    source_measurement_time_s: Optional[float]
    generator_seed: int
    phase_crossing_index: int
    semantics: StreamingSignalSemantics = StreamingSignalSemantics.SEEDED_SYNTHETIC

    def __post_init__(self) -> None:
        if not self.event_id or not self.motor_neuron or not self.muscle:
            raise ValueError("motor event identities must be non-empty")
        if not isinstance(self.raw_app_side, RawAppSide):
            raise TypeError("raw_app_side must be RawAppSide")
        if self.anatomical_side is not AnatomicalSide.UNKNOWN:
            raise ValueError("raw application lanes must retain unknown anatomy")
        event_time = _finite(self.event_time_s, "event_time_s")
        available = _finite(self.availability_time_s, "availability_time_s")
        if event_time < 0.0 or available + _TIME_TOLERANCE_S < event_time:
            raise ValueError("motor event timing is invalid")
        phase = _finite(self.wingbeat_phase_rad, "wingbeat_phase_rad") % _TWO_PI
        rate = _finite(self.rate_hz, "rate_hz")
        probability = _finite(self.emission_probability, "emission_probability")
        if rate < 0.0 or not 0.0 <= probability <= 1.0:
            raise ValueError("motor event rate/probability is invalid")
        if self.source_measurement_time_s is not None:
            source = _finite(
                self.source_measurement_time_s, "source_measurement_time_s"
            )
            if source < 0.0 or source > event_time + _TIME_TOLERANCE_S:
                raise ValueError("event source measurement cannot lie in the future")
            object.__setattr__(self, "source_measurement_time_s", source)
        _nonnegative_int(self.generator_seed, "generator_seed")
        _nonnegative_int(self.phase_crossing_index, "phase_crossing_index")
        if self.semantics is not StreamingSignalSemantics.SEEDED_SYNTHETIC:
            raise ValueError("online inferred events must remain seeded-synthetic")
        object.__setattr__(self, "event_time_s", event_time)
        object.__setattr__(self, "availability_time_s", available)
        object.__setattr__(self, "wingbeat_phase_rad", phase)
        object.__setattr__(self, "rate_hz", rate)
        object.__setattr__(self, "emission_probability", probability)


@dataclass(frozen=True)
class StreamingBridgeFrame:
    """One causal half-open 0.5 ms bridge interval."""

    tick_index: int
    interval_start_s: float
    interval_end_s: float
    wing_phase_start_unwrapped_rad: float
    wing_phase_end_unwrapped_rad: float
    wing_phase_path_unwrapped_rad: Tuple[float, ...]
    source_hold: Tuple[StreamingNOD1Hold, ...]
    generated_dn_rates: Tuple[StreamingRateSample, ...]
    held_dn_rates: Tuple[StreamingRateSample, ...]
    generated_motor_rates: Tuple[StreamingRateSample, ...]
    held_motor_rates: Tuple[StreamingRateSample, ...]
    generated_events: Tuple[StreamingMotorEvent, ...]
    delivered_events: Tuple[StreamingMotorEvent, ...]
    active_intervention_ids: Tuple[str, ...]
    pending_event_count: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.tick_index, "tick_index")
        start = _require_grid(self.interval_start_s, "interval_start_s")
        end = _require_grid(self.interval_end_s, "interval_end_s")
        if abs(end - start - STREAMING_BRIDGE_DT_S) > _TIME_TOLERANCE_S:
            raise ValueError("streaming frame must span exactly 0.5 ms")
        phase_start = _finite(
            self.wing_phase_start_unwrapped_rad,
            "wing_phase_start_unwrapped_rad",
        )
        phase_end = _finite(
            self.wing_phase_end_unwrapped_rad,
            "wing_phase_end_unwrapped_rad",
        )
        if phase_end + _TIME_TOLERANCE_S < phase_start:
            raise ValueError("unwrapped wing phase cannot decrease")
        if phase_end - phase_start > _TWO_PI + _TIME_TOLERANCE_S:
            raise ValueError("wing phase cannot advance by more than one cycle per tick")
        phase_path = tuple(
            _finite(value, "wing_phase_path_unwrapped_rad")
            for value in self.wing_phase_path_unwrapped_rad
        )
        if len(phase_path) not in (2, 6):
            raise ValueError("wing phase path must contain two or six endpoints")
        if (
            abs(phase_path[0] - phase_start) > _TIME_TOLERANCE_S
            or abs(phase_path[-1] - phase_end) > _TIME_TOLERANCE_S
        ):
            raise ValueError("wing phase path endpoints do not match the frame")
        if any(
            right + _TIME_TOLERANCE_S < left
            for left, right in zip(phase_path, phase_path[1:])
        ):
            raise ValueError("wing phase path cannot decrease")
        if tuple(hold.root_id for hold in self.source_hold) not in (
            (),
            FLY_FGS_NOD1_ROOT_IDS,
        ):
            raise ValueError("source hold must be empty or contain exactly four NOD1 roots")
        if len(self.generated_dn_rates) != 2:
            raise ValueError("each frame must generate two raw-app DN lane rates")
        if len(self.generated_motor_rates) != 8:
            raise ValueError("each frame must generate eight individual wing-MN rates")
        if len(set(self.active_intervention_ids)) != len(
            self.active_intervention_ids
        ):
            raise ValueError("active intervention IDs must be unique")
        _nonnegative_int(self.pending_event_count, "pending_event_count")
        object.__setattr__(self, "interval_start_s", start)
        object.__setattr__(self, "interval_end_s", end)
        object.__setattr__(self, "wing_phase_start_unwrapped_rad", phase_start)
        object.__setattr__(self, "wing_phase_end_unwrapped_rad", phase_end)
        object.__setattr__(self, "wing_phase_path_unwrapped_rad", phase_path)


@dataclass(frozen=True)
class StreamingBridgeIntervalStart:
    """Causally available state at the start of one bridge interval.

    ``delivered_events`` contains only events generated in earlier intervals.
    The mechanics layer can therefore schedule them by NMJ availability before
    reporting the interval's actual end phase to :meth:`end_interval`.
    """

    tick_index: int
    interval_start_s: float
    interval_end_s: float
    wing_phase_start_unwrapped_rad: float
    source_hold: Tuple[StreamingNOD1Hold, ...]
    generated_dn_rates: Tuple[StreamingRateSample, ...]
    held_dn_rates: Tuple[StreamingRateSample, ...]
    generated_motor_rates: Tuple[StreamingRateSample, ...]
    held_motor_rates: Tuple[StreamingRateSample, ...]
    delivered_events: Tuple[StreamingMotorEvent, ...]
    active_intervention_ids: Tuple[str, ...]
    pending_event_count: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.tick_index, "tick_index")
        start = _require_grid(self.interval_start_s, "interval_start_s")
        end = _require_grid(self.interval_end_s, "interval_end_s")
        if abs(end - start - STREAMING_BRIDGE_DT_S) > _TIME_TOLERANCE_S:
            raise ValueError("streaming interval start must span exactly 0.5 ms")
        phase = _finite(
            self.wing_phase_start_unwrapped_rad,
            "wing_phase_start_unwrapped_rad",
        )
        if tuple(hold.root_id for hold in self.source_hold) not in (
            (),
            FLY_FGS_NOD1_ROOT_IDS,
        ):
            raise ValueError("source hold must be empty or contain exactly four NOD1 roots")
        if len(self.generated_dn_rates) != 2:
            raise ValueError("each interval must generate two raw-app DN lane rates")
        if len(self.generated_motor_rates) != 8:
            raise ValueError("each interval must generate eight individual wing-MN rates")
        if len(set(self.active_intervention_ids)) != len(
            self.active_intervention_ids
        ):
            raise ValueError("active intervention IDs must be unique")
        if len({event.event_id for event in self.delivered_events}) != len(
            self.delivered_events
        ):
            raise ValueError("delivered interval event IDs must be unique")
        _nonnegative_int(self.pending_event_count, "pending_event_count")
        object.__setattr__(self, "interval_start_s", start)
        object.__setattr__(self, "interval_end_s", end)
        object.__setattr__(self, "wing_phase_start_unwrapped_rad", phase)


@dataclass(frozen=True)
class StreamingBridgeCheckpoint:
    """Immutable, canonical-JSON and SHA-256 protected bridge state."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = json.loads(_canonical_json(self.payload))
        expected = {
            "schema_version",
            "runtime_version",
            "config",
            "seed",
            "interventions",
            "state",
            "payload_sha256",
        }
        mapping = _exact_mapping(copied, expected, "streaming checkpoint")
        if mapping["schema_version"] != STREAMING_BRIDGE_SCHEMA_VERSION:
            raise ValueError("streaming checkpoint schema version mismatch")
        if mapping["runtime_version"] != STREAMING_BRIDGE_RUNTIME_VERSION:
            raise ValueError("streaming checkpoint runtime version mismatch")
        digest = mapping["payload_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("streaming checkpoint digest is invalid")
        unsigned = {key: value for key, value in mapping.items() if key != "payload_sha256"}
        if _payload_sha256(unsigned) != digest:
            raise ValueError("streaming checkpoint payload SHA-256 mismatch")
        object.__setattr__(self, "payload", _deep_freeze(copied))

    def to_dict(self) -> Mapping[str, Any]:
        return _deep_thaw(self.payload)


@dataclass(frozen=True)
class _CircuitInput:
    sample_index: int
    measurement_time_s: float
    availability_time_s: float
    voltages_v: Tuple[float, ...]

    @classmethod
    def from_sample(cls, sample: FlyFGSCircuitSample) -> "_CircuitInput":
        return cls(
            sample_index=sample.sample_index,
            measurement_time_s=sample.measurement_time_s,
            availability_time_s=sample.availability_time_s,
            voltages_v=tuple(
                float(sample.nod1_voltage_v[root_id])
                for root_id in FLY_FGS_NOD1_ROOT_IDS
            ),
        )

    def __post_init__(self) -> None:
        index = _nonnegative_int(self.sample_index, "sample_index")
        measurement = _require_grid(self.measurement_time_s, "measurement_time_s")
        availability = _require_grid(self.availability_time_s, "availability_time_s")
        expected_time = index * _CANONICAL_CIRCUIT_DT_S
        if abs(measurement - expected_time) > _TIME_TOLERANCE_S:
            raise ValueError(
                "canonical fly-FGS measurement_time_s must equal sample_index * 5 ms"
            )
        if abs(availability - expected_time) > _TIME_TOLERANCE_S:
            raise ValueError(
                "canonical fly-FGS availability_time_s must equal sample_index * 5 ms"
            )
        if len(self.voltages_v) != len(FLY_FGS_NOD1_ROOT_IDS):
            raise ValueError("circuit input must contain exactly four NOD1 voltages")
        voltages = tuple(_finite(value, "NOD1 voltage") for value in self.voltages_v)
        object.__setattr__(self, "measurement_time_s", measurement)
        object.__setattr__(self, "availability_time_s", availability)
        object.__setattr__(self, "voltages_v", voltages)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "sample_index": self.sample_index,
            "measurement_time_s": self.measurement_time_s,
            "availability_time_s": self.availability_time_s,
            "voltages_v": list(self.voltages_v),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "_CircuitInput":
        mapping = _exact_mapping(
            value,
            (
                "sample_index",
                "measurement_time_s",
                "availability_time_s",
                "voltages_v",
            ),
            "circuit queue item",
        )
        voltages = mapping["voltages_v"]
        if not isinstance(voltages, (list, tuple)):
            raise ValueError("circuit queue voltages must be an array")
        return cls(
            sample_index=mapping["sample_index"],
            measurement_time_s=mapping["measurement_time_s"],
            availability_time_s=mapping["availability_time_s"],
            voltages_v=tuple(voltages),
        )


def _rate_to_dict(sample: StreamingRateSample) -> Mapping[str, Any]:
    return {
        "stage": sample.stage.value,
        "target_name": sample.target_name,
        "raw_app_side": sample.raw_app_side.value,
        "anatomical_side": sample.anatomical_side.value,
        "measurement_time_s": sample.measurement_time_s,
        "availability_time_s": sample.availability_time_s,
        "rate_hz": sample.rate_hz,
        "source_measurement_time_s": sample.source_measurement_time_s,
        "semantics": sample.semantics.value,
    }


def _rate_from_dict(value: Any) -> StreamingRateSample:
    mapping = _exact_mapping(
        value,
        (
            "stage",
            "target_name",
            "raw_app_side",
            "anatomical_side",
            "measurement_time_s",
            "availability_time_s",
            "rate_hz",
            "source_measurement_time_s",
            "semantics",
        ),
        "rate queue item",
    )
    return StreamingRateSample(
        stage=StreamingBridgeStage(mapping["stage"]),
        target_name=mapping["target_name"],
        raw_app_side=RawAppSide(mapping["raw_app_side"]),
        anatomical_side=AnatomicalSide(mapping["anatomical_side"]),
        measurement_time_s=mapping["measurement_time_s"],
        availability_time_s=mapping["availability_time_s"],
        rate_hz=mapping["rate_hz"],
        source_measurement_time_s=mapping["source_measurement_time_s"],
        semantics=StreamingSignalSemantics(mapping["semantics"]),
    )


def _event_to_dict(event: StreamingMotorEvent) -> Mapping[str, Any]:
    return {
        "event_id": event.event_id,
        "motor_neuron": event.motor_neuron,
        "muscle": event.muscle,
        "raw_app_side": event.raw_app_side.value,
        "anatomical_side": event.anatomical_side.value,
        "event_time_s": event.event_time_s,
        "availability_time_s": event.availability_time_s,
        "wingbeat_phase_rad": event.wingbeat_phase_rad,
        "rate_hz": event.rate_hz,
        "emission_probability": event.emission_probability,
        "source_measurement_time_s": event.source_measurement_time_s,
        "generator_seed": event.generator_seed,
        "phase_crossing_index": event.phase_crossing_index,
        "semantics": event.semantics.value,
    }


def _event_from_dict(value: Any) -> StreamingMotorEvent:
    mapping = _exact_mapping(
        value,
        (
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
        ),
        "motor event",
    )
    return StreamingMotorEvent(
        event_id=mapping["event_id"],
        motor_neuron=mapping["motor_neuron"],
        muscle=mapping["muscle"],
        raw_app_side=RawAppSide(mapping["raw_app_side"]),
        anatomical_side=AnatomicalSide(mapping["anatomical_side"]),
        event_time_s=mapping["event_time_s"],
        availability_time_s=mapping["availability_time_s"],
        wingbeat_phase_rad=mapping["wingbeat_phase_rad"],
        rate_hz=mapping["rate_hz"],
        emission_probability=mapping["emission_probability"],
        source_measurement_time_s=mapping["source_measurement_time_s"],
        generator_seed=mapping["generator_seed"],
        phase_crossing_index=mapping["phase_crossing_index"],
        semantics=StreamingSignalSemantics(mapping["semantics"]),
    )


def _apply_interventions(
    value: float,
    interventions: Sequence[StreamingBridgeIntervention],
    stage: StreamingBridgeStage,
    target_name: str,
    side: RawAppSide,
    time_s: float,
) -> float:
    result = float(value)
    for intervention in interventions:
        if not intervention.matches(stage, target_name, side, time_s):
            continue
        if intervention.mode is StreamingInterventionMode.SILENCE:
            result = 0.0
        elif intervention.mode is StreamingInterventionMode.SCALE:
            result *= intervention.magnitude
        else:
            result += intervention.magnitude
    return max(0.0, result)


class StreamingNOD1MotorBridge:
    # Frozen legacy fly-FGS fixture contract. Paper assay mode subclasses this
    # bridge with a protocol-owned duration instead of mutating this boundary.
    _MAXIMUM_SOURCE_SAMPLE_INDEX: Optional[int] = 99
    """Deterministic state machine for one online neural-to-wing-MN stream."""

    def __init__(
        self,
        config: Optional[StreamingBridgeConfig] = None,
        *,
        seed: int = 0,
        interventions: Sequence[StreamingBridgeIntervention] = (),
        initial_wing_phase_unwrapped_rad: float = 0.0,
    ) -> None:
        self.config = config if config is not None else StreamingBridgeConfig()
        if not isinstance(self.config, StreamingBridgeConfig):
            raise TypeError("config must be StreamingBridgeConfig")
        self.seed = _nonnegative_int(seed, "seed")
        self.interventions = tuple(interventions)
        if any(
            not isinstance(item, StreamingBridgeIntervention)
            for item in self.interventions
        ):
            raise TypeError("interventions must be StreamingBridgeIntervention values")
        if len({item.intervention_id for item in self.interventions}) != len(
            self.interventions
        ):
            raise ValueError("intervention IDs must be unique")
        self._validate_intervention_targets()

        initial_phase = _finite(
            initial_wing_phase_unwrapped_rad,
            "initial_wing_phase_unwrapped_rad",
        )
        self._tick_index = 0
        self._initial_phase_unwrapped_rad = initial_phase
        self._phase_unwrapped_rad = initial_phase
        self._phase_path_history: List[Tuple[float, ...]] = []
        self._open_interval: Optional[StreamingBridgeIntervalStart] = None
        self._queue_sequence = 0
        self._source_pending: List[Tuple[float, int, _CircuitInput]] = []
        self._source_hold: Optional[_CircuitInput] = None
        self._dn_pending: Dict[
            RawAppSide, List[Tuple[float, int, StreamingRateSample]]
        ] = {RawAppSide.L: [], RawAppSide.R: []}
        self._dn_hold: Dict[RawAppSide, StreamingRateSample] = {}
        self._motor_pending: Dict[
            str, List[Tuple[float, int, StreamingRateSample]]
        ] = {key: [] for key in self._channel_keys()}
        self._motor_hold: Dict[str, StreamingRateSample] = {}
        self._event_pending: List[Tuple[float, int, StreamingMotorEvent]] = []
        self._delivered_events: List[StreamingMotorEvent] = []
        self._last_pushed_sample_index: Optional[int] = None
        self._last_pushed_measurement_time_s: Optional[float] = None
        self._last_pushed_availability_time_s: Optional[float] = None
        self._rng: Dict[str, random.Random] = {}
        self._generator_seeds: Dict[str, int] = {}
        self._last_crossing_time_s: Dict[str, Optional[float]] = {}
        self._phase_crossing_count: Dict[str, int] = {}
        for key in self._channel_keys():
            side, motor, muscle = _split_channel_key(key)
            channel_seed = _channel_seed(self.seed, side, motor, muscle)
            self._generator_seeds[key] = channel_seed
            self._rng[key] = random.Random(channel_seed)
            self._last_crossing_time_s[key] = None
            self._phase_crossing_count[key] = 0

    @property
    def tick_index(self) -> int:
        return self._tick_index

    @property
    def current_time_s(self) -> float:
        return self._tick_index * self.config.base_dt_s

    @property
    def delivered_events(self) -> Tuple[StreamingMotorEvent, ...]:
        return tuple(self._delivered_events)

    @property
    def interval_is_open(self) -> bool:
        return self._open_interval is not None

    def _channel_keys(self) -> Tuple[str, ...]:
        return tuple(
            _channel_key(side, path.motor_neuron, path.muscle)
            for path in self.config.pathways
            for side in (RawAppSide.L, RawAppSide.R)
        )

    def _validate_intervention_targets(self) -> None:
        motor_names = {path.motor_neuron for path in self.config.pathways}
        muscle_names = {path.muscle for path in self.config.pathways}
        for item in self.interventions:
            if (
                item.target_stage is StreamingBridgeStage.NOD1
                and item.target_name != self.config.source_cell_type
            ):
                raise ValueError("intervention targets an unknown NOD1 stage")
            if (
                item.target_stage is StreamingBridgeStage.DN
                and item.target_name != self.config.target_dn_type
            ):
                raise ValueError("intervention targets an unknown DN stage")
            if (
                item.target_stage is StreamingBridgeStage.MN
                and item.target_name not in motor_names
            ):
                raise ValueError("intervention targets an unknown wing motor neuron")
            if (
                item.target_stage is StreamingBridgeStage.MUSCLE
                and item.target_name not in muscle_names
            ):
                raise ValueError("intervention targets an unknown wing muscle")

    def _next_sequence(self) -> int:
        result = self._queue_sequence
        self._queue_sequence += 1
        return result

    def push_circuit_sample(self, sample: FlyFGSCircuitSample) -> None:
        """Queue one validated sample without reading it before availability."""

        if self._open_interval is not None:
            raise RuntimeError("circuit samples may be pushed only between intervals")
        if not isinstance(sample, FlyFGSCircuitSample):
            raise TypeError("sample must be FlyFGSCircuitSample")
        # Access exactly the motor-eligible boundary. Pooled/full/retinal fields
        # are intentionally neither inspected nor serialized.
        item = _CircuitInput.from_sample(sample)
        maximum_index = self._MAXIMUM_SOURCE_SAMPLE_INDEX
        if maximum_index is not None and item.sample_index > maximum_index:
            raise ValueError(
                "canonical circuit sample_index must be in [0, {}]".format(
                    maximum_index
                )
            )
        if item.availability_time_s + _TIME_TOLERANCE_S < self.current_time_s:
            raise ValueError("cannot insert a circuit sample retroactively")
        if self._last_pushed_sample_index is None:
            if item.sample_index != 0:
                raise ValueError("the canonical fly-FGS stream must begin at sample 0")
        else:
            if item.sample_index != self._last_pushed_sample_index + 1:
                raise ValueError(
                    "canonical fly-FGS circuit sample indices must be contiguous"
                )
            assert self._last_pushed_measurement_time_s is not None
            assert self._last_pushed_availability_time_s is not None
            if abs(
                item.measurement_time_s
                - self._last_pushed_measurement_time_s
                - _CANONICAL_CIRCUIT_DT_S
            ) > _TIME_TOLERANCE_S or abs(
                item.availability_time_s
                - self._last_pushed_availability_time_s
                - _CANONICAL_CIRCUIT_DT_S
            ) > _TIME_TOLERANCE_S:
                raise ValueError("canonical fly-FGS samples must have exact 5 ms cadence")
        heapq.heappush(
            self._source_pending,
            (item.availability_time_s, self._next_sequence(), item),
        )
        self._last_pushed_sample_index = item.sample_index
        self._last_pushed_measurement_time_s = item.measurement_time_s
        self._last_pushed_availability_time_s = item.availability_time_s

    def _release_source(self, time_s: float) -> None:
        while (
            self._source_pending
            and self._source_pending[0][0] <= time_s + _TIME_TOLERANCE_S
        ):
            _, _, self._source_hold = heapq.heappop(self._source_pending)

    @staticmethod
    def _release_rate_queue(
        queue: List[Tuple[float, int, StreamingRateSample]],
        time_s: float,
        current: Optional[StreamingRateSample],
    ) -> Optional[StreamingRateSample]:
        while queue and queue[0][0] <= time_s + _TIME_TOLERANCE_S:
            _, _, current = heapq.heappop(queue)
        return current

    def _source_drive(
        self, side: RawAppSide, time_s: float
    ) -> Tuple[float, Optional[float]]:
        if self._source_hold is None:
            return 0.0, None
        values = tuple(
            voltage
            for root_id, voltage in zip(
                FLY_FGS_NOD1_ROOT_IDS, self._source_hold.voltages_v
            )
            if FLY_FGS_NOD1_RAW_APP_SIDES[root_id] == side.value
        )
        if len(values) != 2:
            raise RuntimeError("fly-FGS NOD1 source inventory is not two cells per app lane")
        voltage = sum(values) / len(values)
        drive = max(
            0.0,
            (voltage - self.config.resting_voltage_v)
            / self.config.voltage_scale_v,
        )
        drive = _apply_interventions(
            drive,
            self.interventions,
            StreamingBridgeStage.NOD1,
            self.config.source_cell_type,
            side,
            time_s,
        )
        return drive, self._source_hold.measurement_time_s

    def _generate_dn_rates(self, time_s: float) -> Tuple[StreamingRateSample, ...]:
        source = {
            side: self._source_drive(side, time_s)
            for side in (RawAppSide.L, RawAppSide.R)
        }
        output: List[StreamingRateSample] = []
        for target_side in (RawAppSide.L, RawAppSide.R):
            source_side = (
                target_side.mirror
                if self.config.contralateral_raw_app_lanes
                else target_side
            )
            drive, source_measurement = source[source_side]
            rate_hz = min(
                self.config.maximum_dn_rate_hz,
                self.config.dn_baseline_rate_hz
                + self.config.dn_functional_gain_hz * drive,
            )
            rate_hz = min(
                self.config.maximum_dn_rate_hz,
                _apply_interventions(
                    rate_hz,
                    self.interventions,
                    StreamingBridgeStage.DN,
                    self.config.target_dn_type,
                    target_side,
                    time_s,
                ),
            )
            sample = StreamingRateSample(
                stage=StreamingBridgeStage.DN,
                target_name=self.config.target_dn_type,
                raw_app_side=target_side,
                anatomical_side=AnatomicalSide.UNKNOWN,
                measurement_time_s=time_s,
                availability_time_s=time_s + self.config.encoder_delay_s,
                rate_hz=rate_hz,
                source_measurement_time_s=source_measurement,
            )
            heapq.heappush(
                self._dn_pending[target_side],
                (sample.availability_time_s, self._next_sequence(), sample),
            )
            output.append(sample)
        return tuple(output)

    def _release_dn(self, time_s: float) -> None:
        for side in (RawAppSide.L, RawAppSide.R):
            latest = self._release_rate_queue(
                self._dn_pending[side], time_s, self._dn_hold.get(side)
            )
            if latest is not None:
                self._dn_hold[side] = latest

    def _generate_motor_rates(
        self, time_s: float
    ) -> Tuple[StreamingRateSample, ...]:
        output: List[StreamingRateSample] = []
        for path in self.config.pathways:
            for side in (RawAppSide.L, RawAppSide.R):
                dn = self._dn_hold.get(side)
                dn_rate_hz = 0.0 if dn is None else dn.rate_hz
                rate_hz = (
                    path.functional_rate_gain * dn_rate_hz
                    if path.structural_supported
                    else 0.0
                )
                rate_hz = _apply_interventions(
                    rate_hz,
                    self.interventions,
                    StreamingBridgeStage.MN,
                    path.motor_neuron,
                    side,
                    time_s,
                )
                rate_hz = _apply_interventions(
                    rate_hz,
                    self.interventions,
                    StreamingBridgeStage.MUSCLE,
                    path.muscle,
                    side,
                    time_s,
                )
                rate_hz = min(rate_hz, self.config.maximum_motor_rate_hz)
                sample = StreamingRateSample(
                    stage=StreamingBridgeStage.MN,
                    target_name=path.motor_neuron,
                    raw_app_side=side,
                    anatomical_side=AnatomicalSide.UNKNOWN,
                    measurement_time_s=time_s,
                    availability_time_s=time_s + self.config.vnc_delay_s,
                    rate_hz=rate_hz,
                    source_measurement_time_s=(
                        None if dn is None else dn.source_measurement_time_s
                    ),
                )
                key = _channel_key(side, path.motor_neuron, path.muscle)
                heapq.heappush(
                    self._motor_pending[key],
                    (sample.availability_time_s, self._next_sequence(), sample),
                )
                output.append(sample)
        return tuple(output)

    def _release_motor(self, time_s: float) -> None:
        for key in self._channel_keys():
            latest = self._release_rate_queue(
                self._motor_pending[key], time_s, self._motor_hold.get(key)
            )
            if latest is not None:
                self._motor_hold[key] = latest

    @staticmethod
    def _phase_crossings(
        preferred_phase_rad: float,
        phase_start: float,
        phase_end: float,
        time_start_s: float,
        dt_s: float,
    ) -> Tuple[float, ...]:
        """Return crossing times in the half-open runtime interval [start, end)."""

        phase_delta = phase_end - phase_start
        if phase_delta <= _TIME_TOLERANCE_S:
            return ()
        first_cycle = math.ceil(
            (phase_start - preferred_phase_rad) / _TWO_PI
            - _TIME_TOLERANCE_S
        )
        crossings: List[float] = []
        cycle = first_cycle
        while True:
            target = preferred_phase_rad + cycle * _TWO_PI
            if target < phase_start - _TIME_TOLERANCE_S:
                cycle += 1
                continue
            if target >= phase_end - _TIME_TOLERANCE_S:
                break
            fraction = min(1.0, max(0.0, (target - phase_start) / phase_delta))
            crossings.append(time_start_s + fraction * dt_s)
            cycle += 1
        return tuple(crossings)

    def _generate_events(
        self,
        time_s: float,
        phase_path: Sequence[float],
    ) -> Tuple[StreamingMotorEvent, ...]:
        generated: List[StreamingMotorEvent] = []
        nominal_period_s = 1.0 / self.config.initial_wingbeat_frequency_hz
        segment_dt_s = self.config.base_dt_s / (len(phase_path) - 1)
        for path in self.config.pathways:
            crossing_times = tuple(
                crossing_time
                for segment_index, (phase_start, phase_end) in enumerate(
                    zip(phase_path, phase_path[1:])
                )
                for crossing_time in self._phase_crossings(
                    path.preferred_phase_rad,
                    phase_start,
                    phase_end,
                    time_s + segment_index * segment_dt_s,
                    segment_dt_s,
                )
            )
            for crossing_time_s in crossing_times:
                for side in (RawAppSide.L, RawAppSide.R):
                    key = _channel_key(side, path.motor_neuron, path.muscle)
                    rate_sample = self._motor_hold.get(key)
                    rate_hz = 0.0 if rate_sample is None else rate_sample.rate_hz
                    last_crossing = self._last_crossing_time_s[key]
                    period_s = (
                        nominal_period_s
                        if last_crossing is None
                        else crossing_time_s - last_crossing
                    )
                    if period_s <= 0.0 or not math.isfinite(period_s):
                        raise RuntimeError("wing phase crossing history is not monotone")
                    probability = min(1.0, max(0.0, rate_hz * period_s))
                    random_draw = self._rng[key].random()
                    crossing_index = self._phase_crossing_count[key]
                    self._phase_crossing_count[key] = crossing_index + 1
                    self._last_crossing_time_s[key] = crossing_time_s
                    if random_draw >= probability:
                        continue
                    event_id = "%s|crossing-%d" % (key, crossing_index)
                    event = StreamingMotorEvent(
                        event_id=event_id,
                        motor_neuron=path.motor_neuron,
                        muscle=path.muscle,
                        raw_app_side=side,
                        anatomical_side=AnatomicalSide.UNKNOWN,
                        event_time_s=crossing_time_s,
                        availability_time_s=crossing_time_s
                        + self.config.nmj_delay_s,
                        wingbeat_phase_rad=path.preferred_phase_rad,
                        rate_hz=rate_hz,
                        emission_probability=probability,
                        source_measurement_time_s=(
                            None
                            if rate_sample is None
                            else rate_sample.source_measurement_time_s
                        ),
                        generator_seed=self._generator_seeds[key],
                        phase_crossing_index=crossing_index,
                    )
                    heapq.heappush(
                        self._event_pending,
                        (event.availability_time_s, self._next_sequence(), event),
                    )
                    generated.append(event)
        generated.sort(key=lambda event: (event.event_time_s, event.event_id))
        return tuple(generated)

    def _deliver_events(self, interval_end_s: float) -> Tuple[StreamingMotorEvent, ...]:
        delivered: List[StreamingMotorEvent] = []
        while (
            self._event_pending
            and self._event_pending[0][0] < interval_end_s - _TIME_TOLERANCE_S
        ):
            _, _, event = heapq.heappop(self._event_pending)
            delivered.append(event)
            self._delivered_events.append(event)
        delivered.sort(key=lambda event: (event.availability_time_s, event.event_id))
        return tuple(delivered)

    def _source_hold_public(self) -> Tuple[StreamingNOD1Hold, ...]:
        if self._source_hold is None:
            return ()
        return tuple(
            StreamingNOD1Hold(
                root_id=root_id,
                raw_app_side=RawAppSide(FLY_FGS_NOD1_RAW_APP_SIDES[root_id]),
                anatomical_side=AnatomicalSide.UNKNOWN,
                sample_index=self._source_hold.sample_index,
                measurement_time_s=self._source_hold.measurement_time_s,
                availability_time_s=self._source_hold.availability_time_s,
                voltage_v=voltage,
            )
            for root_id, voltage in zip(
                FLY_FGS_NOD1_ROOT_IDS, self._source_hold.voltages_v
            )
        )

    def _active_intervention_ids(self, time_s: float) -> Tuple[str, ...]:
        return tuple(
            item.intervention_id for item in self.interventions if item.active(time_s)
        )

    def _validate_phase_end(self, value: float) -> float:
        phase_end = _finite(value, "wing_phase_end_unwrapped_rad")
        phase_start = self._phase_unwrapped_rad
        if phase_end + _TIME_TOLERANCE_S < phase_start:
            raise ValueError("unwrapped wing phase cannot decrease")
        if phase_end - phase_start > _TWO_PI + _TIME_TOLERANCE_S:
            raise ValueError("wing phase cannot advance by more than one cycle per tick")
        return phase_end

    def _normalize_phase_path(self, value: Any) -> Tuple[float, ...]:
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            path = tuple(
                _finite(item, "wing_phase_path_unwrapped_rad") for item in value
            )
        else:
            path = (
                self._phase_unwrapped_rad,
                _finite(value, "wing_phase_end_unwrapped_rad"),
            )
        if len(path) not in (2, 6):
            raise ValueError("wing phase path must contain two or six endpoints")
        if abs(path[0] - self._phase_unwrapped_rad) > _TIME_TOLERANCE_S:
            raise ValueError("wing phase path must begin at the open interval phase")
        if any(
            right + _TIME_TOLERANCE_S < left
            for left, right in zip(path, path[1:])
        ):
            raise ValueError("unwrapped wing phase path cannot decrease")
        self._validate_phase_end(path[-1])
        return path

    def begin_interval(
        self,
        *,
        circuit_sample: Optional[FlyFGSCircuitSample] = None,
    ) -> StreamingBridgeIntervalStart:
        """Open one half-open interval and expose only causal start state.

        Events returned here were generated in an earlier interval and are
        already scheduled at the NMJ.  The configured NMJ delay is at least one
        complete bridge interval, so events inferred from the phase path that
        is about to occur cannot feed back into that same path.
        """

        if self._open_interval is not None:
            raise RuntimeError("a streaming bridge interval is already open")
        if circuit_sample is not None:
            self.push_circuit_sample(circuit_sample)

        time_s = self.current_time_s
        interval_end_s = time_s + self.config.base_dt_s
        self._release_source(time_s)
        generated_dn = self._generate_dn_rates(time_s)
        self._release_dn(time_s)
        generated_motor = self._generate_motor_rates(time_s)
        self._release_motor(time_s)
        delivered_events = self._deliver_events(interval_end_s)

        interval = StreamingBridgeIntervalStart(
            tick_index=self._tick_index,
            interval_start_s=time_s,
            interval_end_s=interval_end_s,
            wing_phase_start_unwrapped_rad=self._phase_unwrapped_rad,
            source_hold=self._source_hold_public(),
            generated_dn_rates=generated_dn,
            held_dn_rates=tuple(
                self._dn_hold[side]
                for side in (RawAppSide.L, RawAppSide.R)
                if side in self._dn_hold
            ),
            generated_motor_rates=generated_motor,
            held_motor_rates=tuple(
                self._motor_hold[key]
                for key in self._channel_keys()
                if key in self._motor_hold
            ),
            delivered_events=delivered_events,
            active_intervention_ids=self._active_intervention_ids(time_s),
            pending_event_count=len(self._event_pending),
        )
        self._open_interval = interval
        return interval

    def end_interval(
        self,
        wing_phase_path_unwrapped_rad: Any,
    ) -> StreamingBridgeFrame:
        """Close using an observed two- or six-point unwrapped phase path.

        Six points are the bridge-left endpoint plus the five 0.1 ms mechanics
        endpoints.  Crossing times are then located piecewise rather than by a
        single 0.5 ms linear interpolation.  A scalar remains a compatibility
        shorthand for an independently known linear path.
        """

        interval = self._open_interval
        if interval is None:
            raise RuntimeError("no streaming bridge interval is open")
        phase_path = self._normalize_phase_path(wing_phase_path_unwrapped_rad)
        phase_end = phase_path[-1]
        generated_events = self._generate_events(
            interval.interval_start_s,
            phase_path,
        )
        # The minimum one-interval NMJ delay guarantees strict causality: every
        # newly generated event belongs to a later begin_interval() call.
        if any(
            event.availability_time_s
            < interval.interval_end_s - _TIME_TOLERANCE_S
            for event in generated_events
        ):
            raise RuntimeError("a newly generated event became available circularly")

        frame = StreamingBridgeFrame(
            tick_index=interval.tick_index,
            interval_start_s=interval.interval_start_s,
            interval_end_s=interval.interval_end_s,
            wing_phase_start_unwrapped_rad=interval.wing_phase_start_unwrapped_rad,
            wing_phase_end_unwrapped_rad=phase_end,
            wing_phase_path_unwrapped_rad=phase_path,
            source_hold=interval.source_hold,
            generated_dn_rates=interval.generated_dn_rates,
            held_dn_rates=interval.held_dn_rates,
            generated_motor_rates=interval.generated_motor_rates,
            held_motor_rates=interval.held_motor_rates,
            generated_events=generated_events,
            delivered_events=interval.delivered_events,
            active_intervention_ids=interval.active_intervention_ids,
            pending_event_count=len(self._event_pending),
        )
        self._phase_unwrapped_rad = phase_end
        self._phase_path_history.append(phase_path)
        self._tick_index += 1
        self._open_interval = None
        return frame

    def step(
        self,
        wing_phase_end_unwrapped_rad: float,
        *,
        circuit_sample: Optional[FlyFGSCircuitSample] = None,
    ) -> StreamingBridgeFrame:
        """Compatibility wrapper around :meth:`begin_interval`/`end_interval`.

        Closed-loop callers should use the two methods directly, advance their
        mechanics with ``begin_interval().delivered_events``, and pass the
        observed mechanics phase to ``end_interval``.  This wrapper is safe
        only when its phase end is independently known before the interval.
        """

        if self._open_interval is not None:
            raise RuntimeError("a streaming bridge interval is already open")
        # Validate before begin_interval mutates queue/hold state.
        phase_end = self._validate_phase_end(wing_phase_end_unwrapped_rad)
        self.begin_interval(circuit_sample=circuit_sample)
        return self.end_interval(phase_end)

    @staticmethod
    def _queue_to_list(
        queue: Sequence[Tuple[float, int, Any]], serializer: Any
    ) -> List[Mapping[str, Any]]:
        return [
            {
                "availability_time_s": availability,
                "sequence": sequence,
                "value": serializer(value),
            }
            for availability, sequence, value in sorted(queue)
        ]

    @staticmethod
    def _queue_from_list(
        value: Any, parser: Any, label: str
    ) -> List[Tuple[float, int, Any]]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("%s must be an array" % label)
        queue: List[Tuple[float, int, Any]] = []
        seen_sequences = set()
        for raw in value:
            mapping = _exact_mapping(
                raw,
                ("availability_time_s", "sequence", "value"),
                label,
            )
            availability = _finite(
                mapping["availability_time_s"], "%s availability" % label
            )
            sequence = _nonnegative_int(mapping["sequence"], "%s sequence" % label)
            if sequence in seen_sequences:
                raise ValueError("%s contains a duplicate sequence" % label)
            item = parser(mapping["value"])
            if abs(item.availability_time_s - availability) > _TIME_TOLERANCE_S:
                raise ValueError("%s availability does not match its value" % label)
            seen_sequences.add(sequence)
            queue.append((availability, sequence, item))
        heapq.heapify(queue)
        return queue

    def checkpoint(self) -> StreamingBridgeCheckpoint:
        """Capture all causal state needed for exactly-once continuation."""

        if self._open_interval is not None:
            raise RuntimeError("checkpointing is allowed only between intervals")
        current_time = self.current_time_s
        state: Mapping[str, Any] = {
            "tick_index": self._tick_index,
            "initial_wing_phase_unwrapped_rad": self._initial_phase_unwrapped_rad,
            "wing_phase_unwrapped_rad": self._phase_unwrapped_rad,
            "wing_phase_path_history": [
                list(path) for path in self._phase_path_history
            ],
            "queue_sequence": self._queue_sequence,
            "last_pushed": {
                "sample_index": self._last_pushed_sample_index,
                "measurement_time_s": self._last_pushed_measurement_time_s,
                "availability_time_s": self._last_pushed_availability_time_s,
            },
            "source_pending": self._queue_to_list(
                self._source_pending, lambda item: item.to_dict()
            ),
            "source_hold": (
                None if self._source_hold is None else self._source_hold.to_dict()
            ),
            "dn_pending": {
                side.value: self._queue_to_list(
                    self._dn_pending[side], _rate_to_dict
                )
                for side in (RawAppSide.L, RawAppSide.R)
            },
            "dn_hold": {
                side.value: _rate_to_dict(sample)
                for side, sample in self._dn_hold.items()
            },
            "motor_pending": {
                key: self._queue_to_list(self._motor_pending[key], _rate_to_dict)
                for key in self._channel_keys()
            },
            "motor_hold": {
                key: _rate_to_dict(sample) for key, sample in self._motor_hold.items()
            },
            "rng_state": {
                key: self._rng[key].getstate() for key in self._channel_keys()
            },
            "generator_seeds": dict(self._generator_seeds),
            "last_crossing_time_s": dict(self._last_crossing_time_s),
            "phase_crossing_count": dict(self._phase_crossing_count),
            "event_pending": self._queue_to_list(
                self._event_pending, _event_to_dict
            ),
            "delivered_events": [
                _event_to_dict(event) for event in self._delivered_events
            ],
            "active_intervention_ids": list(
                self._active_intervention_ids(current_time)
            ),
        }
        unsigned: Dict[str, Any] = {
            "schema_version": STREAMING_BRIDGE_SCHEMA_VERSION,
            "runtime_version": STREAMING_BRIDGE_RUNTIME_VERSION,
            "config": self.config.to_dict(),
            "seed": self.seed,
            "interventions": [item.to_dict() for item in self.interventions],
            "state": state,
        }
        payload = dict(unsigned)
        payload["payload_sha256"] = _payload_sha256(unsigned)
        return StreamingBridgeCheckpoint(payload)

    @staticmethod
    def _random_state_from_json(value: Any) -> Tuple[Any, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(
                StreamingNOD1MotorBridge._random_state_from_json(child)
                if isinstance(child, (list, tuple))
                else child
                for child in value
            )
        raise ValueError("RNG state must be an array")

    def restore(self, checkpoint: StreamingBridgeCheckpoint) -> None:
        """Transactionally restore a compatible hash-protected checkpoint."""

        if self._open_interval is not None:
            raise RuntimeError("restore is allowed only between intervals")
        if not isinstance(checkpoint, StreamingBridgeCheckpoint):
            raise TypeError("checkpoint must be StreamingBridgeCheckpoint")
        # Reconstruct to defend against a checkpoint object's nested payload
        # being replaced through non-standard reflection after construction.
        payload = StreamingBridgeCheckpoint(checkpoint.to_dict()).to_dict()
        if payload["config"] != self.config.to_dict():
            raise ValueError("streaming checkpoint configuration mismatch")
        if payload["seed"] != self.seed:
            raise ValueError("streaming checkpoint seed mismatch")
        expected_interventions = [item.to_dict() for item in self.interventions]
        if payload["interventions"] != expected_interventions:
            raise ValueError("streaming checkpoint intervention state mismatch")

        expected_state_fields = {
            "tick_index",
            "initial_wing_phase_unwrapped_rad",
            "wing_phase_unwrapped_rad",
            "wing_phase_path_history",
            "queue_sequence",
            "last_pushed",
            "source_pending",
            "source_hold",
            "dn_pending",
            "dn_hold",
            "motor_pending",
            "motor_hold",
            "rng_state",
            "generator_seeds",
            "last_crossing_time_s",
            "phase_crossing_count",
            "event_pending",
            "delivered_events",
            "active_intervention_ids",
        }
        state = _exact_mapping(payload["state"], expected_state_fields, "bridge state")
        tick_index = _nonnegative_int(state["tick_index"], "tick_index")
        current_time = tick_index * self.config.base_dt_s
        initial_phase = _finite(
            state["initial_wing_phase_unwrapped_rad"],
            "checkpoint initial wing phase",
        )
        phase = _finite(state["wing_phase_unwrapped_rad"], "checkpoint wing phase")
        if phase + _TIME_TOLERANCE_S < initial_phase:
            raise ValueError("checkpoint wing phase precedes its initial phase")
        if tick_index == 0 and abs(phase - initial_phase) > _TIME_TOLERANCE_S:
            raise ValueError("zero-tick checkpoint wing phase is inconsistent")
        raw_phase_history = state["wing_phase_path_history"]
        if not isinstance(raw_phase_history, (list, tuple)):
            raise ValueError("wing phase path history must be an array")
        if len(raw_phase_history) != tick_index:
            raise ValueError("wing phase path history length does not match tick_index")
        phase_history: List[Tuple[float, ...]] = []
        phase_cursor = initial_phase
        for raw_path in raw_phase_history:
            if not isinstance(raw_path, (list, tuple)) or len(raw_path) not in (2, 6):
                raise ValueError("checkpoint wing phase paths must have two or six points")
            parsed_path = tuple(
                _finite(value, "checkpoint wing phase path") for value in raw_path
            )
            if abs(parsed_path[0] - phase_cursor) > _TIME_TOLERANCE_S:
                raise ValueError("checkpoint wing phase paths are not contiguous")
            if any(
                right + _TIME_TOLERANCE_S < left
                for left, right in zip(parsed_path, parsed_path[1:])
            ):
                raise ValueError("checkpoint wing phase path decreases")
            if parsed_path[-1] - parsed_path[0] > _TWO_PI + _TIME_TOLERANCE_S:
                raise ValueError("checkpoint wing phase path advances by over one cycle")
            phase_history.append(parsed_path)
            phase_cursor = parsed_path[-1]
        if abs(phase_cursor - phase) > _TIME_TOLERANCE_S:
            raise ValueError("checkpoint phase endpoint does not match its path history")
        queue_sequence = _nonnegative_int(state["queue_sequence"], "queue_sequence")

        last_pushed = _exact_mapping(
            state["last_pushed"],
            ("sample_index", "measurement_time_s", "availability_time_s"),
            "last_pushed",
        )
        last_index = last_pushed["sample_index"]
        last_measurement = last_pushed["measurement_time_s"]
        last_availability = last_pushed["availability_time_s"]
        if last_index is None:
            if last_measurement is not None or last_availability is not None:
                raise ValueError("empty last-pushed state must contain only nulls")
        else:
            last_index = _nonnegative_int(last_index, "last pushed sample index")
            last_measurement = _require_grid(
                last_measurement, "last pushed measurement"
            )
            last_availability = _require_grid(
                last_availability, "last pushed availability"
            )

        source_pending = self._queue_from_list(
            state["source_pending"], _CircuitInput.from_dict, "source_pending"
        )
        source_hold = (
            None
            if state["source_hold"] is None
            else _CircuitInput.from_dict(state["source_hold"])
        )
        if any(item[0] + _TIME_TOLERANCE_S < current_time for item in source_pending):
            raise ValueError("checkpoint contains an overdue source sample")
        if source_hold is not None and source_hold.availability_time_s > current_time + _TIME_TOLERANCE_S:
            raise ValueError("checkpoint source hold is not yet available")
        retained_source = [item[2] for item in source_pending]
        if source_hold is not None:
            retained_source.append(source_hold)
        if last_index is None:
            if retained_source:
                raise ValueError("checkpoint source state is inconsistent with last_pushed")
        else:
            matches = [item for item in retained_source if item.sample_index == last_index]
            if len(matches) != 1:
                raise ValueError("checkpoint last-pushed sample is not retained exactly once")
            if (
                matches[0].measurement_time_s != last_measurement
                or matches[0].availability_time_s != last_availability
            ):
                raise ValueError("checkpoint last-pushed timing is inconsistent")
            if any(item.sample_index > last_index for item in retained_source):
                raise ValueError("checkpoint contains a sample after last_pushed")
            retained_indices = sorted(item.sample_index for item in retained_source)
            first_retained = 0 if source_hold is None else source_hold.sample_index
            if retained_indices != list(range(first_retained, last_index + 1)):
                raise ValueError(
                    "checkpoint canonical source samples are not contiguous"
                )

        raw_dn_pending = _exact_mapping(
            state["dn_pending"], (RawAppSide.L.value, RawAppSide.R.value), "dn_pending"
        )
        dn_pending = {
            side: self._queue_from_list(
                raw_dn_pending[side.value], _rate_from_dict, "dn_pending.%s" % side.value
            )
            for side in (RawAppSide.L, RawAppSide.R)
        }
        raw_dn_hold = state["dn_hold"]
        if not isinstance(raw_dn_hold, Mapping) or not set(raw_dn_hold).issubset(
            {RawAppSide.L.value, RawAppSide.R.value}
        ):
            raise ValueError("dn_hold contains invalid raw app lanes")
        dn_hold = {
            RawAppSide(key): _rate_from_dict(value)
            for key, value in raw_dn_hold.items()
        }
        for side, queue in dn_pending.items():
            for availability, _, sample in queue:
                if (
                    sample.stage is not StreamingBridgeStage.DN
                    or sample.target_name != self.config.target_dn_type
                    or sample.raw_app_side is not side
                    or sample.rate_hz
                    > self.config.maximum_dn_rate_hz + _TIME_TOLERANCE_S
                    or abs(
                        sample.availability_time_s
                        - sample.measurement_time_s
                        - self.config.encoder_delay_s
                    )
                    > _TIME_TOLERANCE_S
                ):
                    raise ValueError("dn_pending channel identity mismatch")
                if availability + _TIME_TOLERANCE_S < current_time:
                    raise ValueError("checkpoint contains an overdue DN rate")
        for side, sample in dn_hold.items():
            if (
                sample.stage is not StreamingBridgeStage.DN
                or sample.raw_app_side is not side
                or sample.rate_hz
                > self.config.maximum_dn_rate_hz + _TIME_TOLERANCE_S
                or abs(
                    sample.availability_time_s
                    - sample.measurement_time_s
                    - self.config.encoder_delay_s
                )
                > _TIME_TOLERANCE_S
            ):
                raise ValueError("dn_hold channel identity mismatch")
            if sample.target_name != self.config.target_dn_type:
                raise ValueError("dn_hold target identity mismatch")
            if sample.availability_time_s > current_time + _TIME_TOLERANCE_S:
                raise ValueError("checkpoint DN hold is not yet available")

        channel_keys = self._channel_keys()
        raw_motor_pending = _exact_mapping(
            state["motor_pending"], channel_keys, "motor_pending"
        )
        motor_pending = {
            key: self._queue_from_list(
                raw_motor_pending[key], _rate_from_dict, "motor_pending.%s" % key
            )
            for key in channel_keys
        }
        for key, queue in motor_pending.items():
            side, motor, _ = _split_channel_key(key)
            for availability, _, sample in queue:
                if (
                    sample.stage is not StreamingBridgeStage.MN
                    or sample.target_name != motor
                    or sample.raw_app_side is not side
                    or sample.rate_hz
                    > self.config.maximum_motor_rate_hz + _TIME_TOLERANCE_S
                    or abs(
                        sample.availability_time_s
                        - sample.measurement_time_s
                        - self.config.vnc_delay_s
                    )
                    > _TIME_TOLERANCE_S
                ):
                    raise ValueError("motor_pending channel identity mismatch")
                if availability + _TIME_TOLERANCE_S < current_time:
                    raise ValueError("checkpoint contains an overdue motor rate")
        raw_motor_hold = state["motor_hold"]
        if not isinstance(raw_motor_hold, Mapping) or not set(raw_motor_hold).issubset(
            set(channel_keys)
        ):
            raise ValueError("motor_hold contains invalid channels")
        motor_hold = {key: _rate_from_dict(value) for key, value in raw_motor_hold.items()}
        for key, sample in motor_hold.items():
            side, motor, _ = _split_channel_key(key)
            if (
                sample.stage is not StreamingBridgeStage.MN
                or sample.raw_app_side is not side
                or sample.target_name != motor
                or sample.rate_hz
                > self.config.maximum_motor_rate_hz + _TIME_TOLERANCE_S
                or abs(
                    sample.availability_time_s
                    - sample.measurement_time_s
                    - self.config.vnc_delay_s
                )
                > _TIME_TOLERANCE_S
            ):
                raise ValueError("motor_hold channel identity mismatch")
            if sample.availability_time_s > current_time + _TIME_TOLERANCE_S:
                raise ValueError("checkpoint motor hold is not yet available")

        raw_rng = _exact_mapping(state["rng_state"], channel_keys, "rng_state")
        rng: Dict[str, random.Random] = {}
        for key in channel_keys:
            generator = random.Random()
            try:
                generator.setstate(self._random_state_from_json(raw_rng[key]))
            except (TypeError, ValueError) as exc:
                raise ValueError("checkpoint RNG state is invalid") from exc
            rng[key] = generator
        generator_seeds = _exact_mapping(
            state["generator_seeds"], channel_keys, "generator_seeds"
        )
        parsed_seeds = {
            key: _nonnegative_int(value, "generator seed")
            for key, value in generator_seeds.items()
        }
        if parsed_seeds != self._generator_seeds:
            raise ValueError("checkpoint per-channel generator seeds mismatch")

        last_crossing_raw = _exact_mapping(
            state["last_crossing_time_s"], channel_keys, "last_crossing_time_s"
        )
        last_crossing: Dict[str, Optional[float]] = {}
        for key, value in last_crossing_raw.items():
            if value is None:
                last_crossing[key] = None
            else:
                parsed = _finite(value, "last crossing time")
                if parsed < 0.0 or parsed >= current_time + _TIME_TOLERANCE_S:
                    raise ValueError("last crossing time lies outside elapsed runtime")
                last_crossing[key] = parsed
        crossing_count_raw = _exact_mapping(
            state["phase_crossing_count"], channel_keys, "phase_crossing_count"
        )
        crossing_count = {
            key: _nonnegative_int(value, "phase crossing count")
            for key, value in crossing_count_raw.items()
        }
        for key in channel_keys:
            if (crossing_count[key] == 0) != (last_crossing[key] is None):
                raise ValueError("checkpoint phase crossing history is inconsistent")
        path_by_identity = {
            (path.motor_neuron, path.muscle): path for path in self.config.pathways
        }
        expected_crossing_times: Dict[str, Tuple[float, ...]] = {}
        for key in channel_keys:
            _side, motor, muscle = _split_channel_key(key)
            preferred = path_by_identity[(motor, muscle)].preferred_phase_rad
            crossings = tuple(
                crossing_time
                for tick, phase_path in enumerate(phase_history)
                for segment, (phase_start, phase_end) in enumerate(
                    zip(phase_path, phase_path[1:])
                )
                for crossing_time in self._phase_crossings(
                    preferred,
                    phase_start,
                    phase_end,
                    tick * self.config.base_dt_s
                    + segment
                    * self.config.base_dt_s
                    / (len(phase_path) - 1),
                    self.config.base_dt_s / (len(phase_path) - 1),
                )
            )
            expected_crossing_times[key] = crossings
            if crossing_count[key] != len(crossings):
                raise ValueError("checkpoint phase crossing count is inconsistent")
            expected_last = None if not crossings else crossings[-1]
            if expected_last is None:
                if last_crossing[key] is not None:
                    raise ValueError("checkpoint last crossing time is inconsistent")
            elif (
                last_crossing[key] is None
                or abs(last_crossing[key] - expected_last) > _TIME_TOLERANCE_S
            ):
                raise ValueError("checkpoint last crossing time is inconsistent")

        event_pending = self._queue_from_list(
            state["event_pending"], _event_from_dict, "event_pending"
        )
        delivered_raw = state["delivered_events"]
        if not isinstance(delivered_raw, (list, tuple)):
            raise ValueError("delivered_events must be an array")
        delivered = [_event_from_dict(value) for value in delivered_raw]
        pending_ids = {item[2].event_id for item in event_pending}
        delivered_ids = {event.event_id for event in delivered}
        if len(pending_ids) != len(event_pending) or len(delivered_ids) != len(delivered):
            raise ValueError("checkpoint event IDs must be unique")
        if pending_ids & delivered_ids:
            raise ValueError("an event cannot be both pending and delivered")
        if any(item[0] < current_time - _TIME_TOLERANCE_S for item in event_pending):
            raise ValueError("checkpoint contains an overdue NMJ event")
        if any(event.availability_time_s >= current_time - _TIME_TOLERANCE_S for event in delivered):
            raise ValueError("checkpoint marks an unavailable NMJ event as delivered")
        if delivered != sorted(
            delivered, key=lambda event: (event.availability_time_s, event.event_id)
        ):
            raise ValueError("checkpoint delivered events are not chronologically ordered")
        for event in [item[2] for item in event_pending] + delivered:
            identity = (event.motor_neuron, event.muscle)
            if identity not in path_by_identity:
                raise ValueError("checkpoint event targets an unknown motor pathway")
            key = _channel_key(event.raw_app_side, *identity)
            expected_event_id = "%s|crossing-%d" % (
                key,
                event.phase_crossing_index,
            )
            if (
                event.event_id != expected_event_id
                or event.generator_seed != parsed_seeds[key]
                or event.phase_crossing_index >= crossing_count[key]
                or abs(
                    event.event_time_s
                    - expected_crossing_times[key][event.phase_crossing_index]
                )
                > _TIME_TOLERANCE_S
                or abs(
                    event.availability_time_s
                    - event.event_time_s
                    - self.config.nmj_delay_s
                )
                > _TIME_TOLERANCE_S
                or abs(
                    event.wingbeat_phase_rad
                    - path_by_identity[identity].preferred_phase_rad
                )
                > _TIME_TOLERANCE_S
                or event.event_time_s >= current_time + _TIME_TOLERANCE_S
            ):
                raise ValueError("checkpoint event provenance is inconsistent")

        active_raw = state["active_intervention_ids"]
        if not isinstance(active_raw, (list, tuple)):
            raise ValueError("active_intervention_ids must be an array")
        expected_active = self._active_intervention_ids(current_time)
        if tuple(active_raw) != expected_active:
            raise ValueError("checkpoint active intervention state mismatch")

        all_sequences = [item[1] for item in source_pending]
        for queue in dn_pending.values():
            all_sequences.extend(item[1] for item in queue)
        for queue in motor_pending.values():
            all_sequences.extend(item[1] for item in queue)
        all_sequences.extend(item[1] for item in event_pending)
        if len(set(all_sequences)) != len(all_sequences):
            raise ValueError("checkpoint queue sequence values must be globally unique")
        if all_sequences and queue_sequence <= max(all_sequences):
            raise ValueError("checkpoint queue sequence counter is stale")

        # Commit only after every field has passed validation.
        self._tick_index = tick_index
        self._initial_phase_unwrapped_rad = initial_phase
        self._phase_unwrapped_rad = phase
        self._phase_path_history = phase_history
        self._open_interval = None
        self._queue_sequence = queue_sequence
        self._last_pushed_sample_index = last_index
        self._last_pushed_measurement_time_s = last_measurement
        self._last_pushed_availability_time_s = last_availability
        self._source_pending = source_pending
        self._source_hold = source_hold
        self._dn_pending = dn_pending
        self._dn_hold = dn_hold
        self._motor_pending = motor_pending
        self._motor_hold = motor_hold
        self._rng = rng
        self._last_crossing_time_s = last_crossing
        self._phase_crossing_count = crossing_count
        self._event_pending = event_pending
        self._delivered_events = delivered


__all__ = [
    "RawAppSide",
    "STREAMING_BRIDGE_DT_S",
    "STREAMING_BRIDGE_RUNTIME_VERSION",
    "STREAMING_BRIDGE_SCHEMA_VERSION",
    "StreamingBridgeCheckpoint",
    "StreamingBridgeConfig",
    "StreamingBridgeFrame",
    "StreamingBridgeIntervalStart",
    "StreamingBridgeIntervention",
    "StreamingBridgeStage",
    "StreamingInterventionMode",
    "StreamingMotorEvent",
    "StreamingMotorPathway",
    "StreamingNOD1Hold",
    "StreamingNOD1MotorBridge",
    "StreamingRateSample",
    "StreamingSignalSemantics",
    "default_streaming_motor_pathways",
]
