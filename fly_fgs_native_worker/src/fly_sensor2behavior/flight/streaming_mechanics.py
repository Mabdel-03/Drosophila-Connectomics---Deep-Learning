"""Checkpointable online wing-muscle and virtual-hinge mechanics.

The stepper in this module is the causal downstream consumer of
``StreamingBridgeFrame.delivered_events``.  It deliberately consumes *only*
that tuple of already NMJ-available events; rates, NOD1 voltages, generated
events, and the bridge's display-side holds never cross this boundary.

Four synchronous steering-muscle channels (iv2, i1, iv1, and b3) are retained
individually on each raw fly-FGS application lane.  The application labels
``L`` and ``R`` do not establish anatomical laterality.  For this exploratory
mechanics model only, raw ``L`` drives the first/left virtual-wing array entry
and raw ``R`` drives the second/right entry.  This is a visible simulation
coordinate convention, not an anatomical claim.

Mechanics-level interventions are immutable, digest-bound contracts applied at
the force/hinge boundary.  ``SILENCE`` suppresses NMJ excitation while active
and gates the named muscle's effective output to zero; ``SCALE`` gates output
by a dimensionless factor without changing the muscle's hidden state.  Both
operate on exact 0.1 ms-aligned half-open intervals.  Natural muscle state is
therefore retained separately from the effective state sent to the hinge.

The bilateral DLM/DVM/tp1 state is an explicit, uncalibrated airborne carrier
baseline.  It provides a wingbeat carrier on which steering events can act;
it must never be described as biological stable flight.  The component reuses
the reduced asynchronous-power, phase-coded steering, tension, thorax, and
virtual-hinge models.  Their parameters remain exploratory.

Clocks and interval semantics
-----------------------------

* mechanics advances on exact 0.1 ms intervals ``[t, t + dt)``;
* a bridge interval is exactly five mechanics intervals;
* an event is injected at ``availability_time_s`` (the NMJ boundary), never at
  its earlier generation time;
* an event exactly at an interval's right endpoint belongs to the next
  interval;
* sub-grid availability times split muscle integration at the causal time;
* every event ID is accepted and applied at most once.

The carrier is advanced exactly once per fixed physics interval.  Delivered
iv2/i1/iv1/b3 events update only steering-muscle state and cannot alter thorax
phase or frequency in this reduced model.  Consequently
``project_phase_end_unwrapped_rad`` supplies an exact, deterministic,
non-mutating phase projection for a bridge interval; it is not merely a
constant-frequency extrapolation.

There is no RNG, seed, wall clock, or implicit spike generator here.  Event
generation remains wholly owned by :mod:`.streaming_bridge`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np

from ..schema import AnatomicalSide
from .hinge import VirtualWingHinge, VirtualWingHingeParameters
from .muscles import (
    AsynchronousPowerMuscle,
    AsynchronousPowerParameters,
    PhaseCodedSteeringMuscle,
    SteeringParameters,
    TensionMuscle,
    TensionParameters,
    ThoraxOscillator,
    ThoraxOscillatorParameters,
)
from .streaming_bridge import (
    RawAppSide,
    StreamingBridgeIntervalStart,
    StreamingMotorEvent,
)
from .types import (
    IndividualMuscleState,
    MuscleClass,
    MuscleSnapshot,
    Side,
    WingKinematics,
)


STREAMING_MECHANICS_SCHEMA_VERSION = "1.1.0"
STREAMING_MECHANICS_RUNTIME_VERSION = "1.3.0"
STREAMING_MECHANICS_DT_S = 0.0001
STREAMING_MECHANICS_BRIDGE_DT_S = 0.0005
STREAMING_MECHANICS_STEPS_PER_BRIDGE = 5
_TWO_PI = 2.0 * math.pi
_TIME_TOLERANCE_S = 1.0e-12
_STEERING_IDENTITIES = (
    ("MN-iv2", "iv2"),
    ("MN-i1", "i1"),
    ("MN-iv1", "iv1"),
    ("MN-b3", "b3"),
)
_RAW_LANE_TO_VIRTUAL_WING = (
    "exploratory coordinate convention only: raw fly-FGS app L -> first/left "
    "virtual-wing entry; raw app R -> second/right virtual-wing entry; "
    "anatomical laterality remains unknown"
)
_CARRIER_BASELINE_STATUS = (
    "exploratory uncalibrated bilateral DLM/DVM/tp1 airborne carrier; "
    "not biological stable flight"
)


class WingPhaseSource(str, Enum):
    """Authority for the kinematic wingbeat phase."""

    MODEL_OWNED_OSCILLATOR = "model_owned_oscillator"
    MEASURED_UNWRAPPED = "measured_unwrapped"


class MechanicsInterventionMode(str, Enum):
    """Mechanics-stage rules with precise hidden-state semantics."""

    SILENCE = "silence"
    SCALE = "scale"


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("%s must be a finite number" % label)
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite number" % label) from exc
    if not math.isfinite(converted):
        raise ValueError("%s must be a finite number" % label)
    return converted


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % label)
    return value


def _exact_mapping(
    value: Any, expected: Iterable[str], label: str
) -> Mapping[str, Any]:
    expected_set = set(expected)
    if not isinstance(value, Mapping) or set(value) != expected_set:
        raise ValueError("%s fields do not match the schema" % label)
    return value


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


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(_deep_thaw(value)),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(payload_without_digest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(payload_without_digest).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class StreamingMuscleIntervention:
    """Immutable steering-muscle intervention at the mechanics boundary.

    ``raw_app_side=None`` targets both raw application lanes.  Those lanes are
    deliberately not called anatomical sides.  Times are canonicalized to the
    exact 0.1 ms mechanics grid and use half-open ``[start_s, end_s)``
    semantics.  ``SILENCE`` has a fixed output scale of zero and also blocks
    NMJ excitation; ``SCALE`` changes effective output only.
    """

    intervention_id: str
    muscle: str
    start_s: float
    end_s: float
    mode: MechanicsInterventionMode
    raw_app_side: Optional[RawAppSide] = None
    output_scale: float = 0.0
    contract_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.intervention_id, str) or not self.intervention_id:
            raise ValueError("intervention_id must be a non-empty string")
        if self.muscle not in tuple(muscle for _motor, muscle in _STEERING_IDENTITIES):
            raise ValueError("intervention muscle must be one of iv2/i1/iv1/b3")
        if not isinstance(self.mode, MechanicsInterventionMode):
            raise TypeError("intervention mode must be MechanicsInterventionMode")
        if self.raw_app_side is not None and not isinstance(
            self.raw_app_side, RawAppSide
        ):
            raise TypeError("raw_app_side must be RawAppSide or None")
        start = _finite(self.start_s, "intervention start_s")
        end = _finite(self.end_s, "intervention end_s")
        if start < 0.0 or end <= start:
            raise ValueError("intervention interval must have end > start >= 0")
        start_tick = round(start / STREAMING_MECHANICS_DT_S)
        end_tick = round(end / STREAMING_MECHANICS_DT_S)
        canonical_start = start_tick * STREAMING_MECHANICS_DT_S
        canonical_end = end_tick * STREAMING_MECHANICS_DT_S
        if (
            abs(start - canonical_start) > _TIME_TOLERANCE_S
            or abs(end - canonical_end) > _TIME_TOLERANCE_S
        ):
            raise ValueError("intervention bounds must be on the exact 0.1 ms grid")
        scale = _finite(self.output_scale, "intervention output_scale")
        if not 0.0 <= scale <= 1.0:
            raise ValueError("intervention output_scale must lie in [0, 1]")
        if self.mode is MechanicsInterventionMode.SILENCE and scale != 0.0:
            raise ValueError("SILENCE output_scale must be exactly zero")
        unsigned = {
            "intervention_id": self.intervention_id,
            "muscle": self.muscle,
            "start_s": canonical_start,
            "end_s": canonical_end,
            "mode": self.mode.value,
            "raw_app_side": (
                None if self.raw_app_side is None else self.raw_app_side.value
            ),
            "output_scale": scale,
        }
        object.__setattr__(self, "start_s", canonical_start)
        object.__setattr__(self, "end_s", canonical_end)
        object.__setattr__(self, "output_scale", scale)
        object.__setattr__(self, "contract_sha256", _payload_sha256(unsigned))

    def active(self, time_s: float) -> bool:
        parsed = _finite(time_s, "intervention query time")
        nearest_grid_time = (
            round(parsed / STREAMING_MECHANICS_DT_S) * STREAMING_MECHANICS_DT_S
        )
        if abs(parsed - nearest_grid_time) <= _TIME_TOLERANCE_S:
            parsed = nearest_grid_time
        return self.start_s <= parsed < self.end_s

    def targets(self, lane: RawAppSide, muscle: str) -> bool:
        return self.muscle == muscle and (
            self.raw_app_side is None or self.raw_app_side is lane
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "muscle": self.muscle,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "mode": self.mode.value,
            "raw_app_side": (
                None if self.raw_app_side is None else self.raw_app_side.value
            ),
            "output_scale": self.output_scale,
            "contract_sha256": self.contract_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "StreamingMuscleIntervention":
        mapping = _exact_mapping(
            value,
            (
                "intervention_id",
                "muscle",
                "start_s",
                "end_s",
                "mode",
                "raw_app_side",
                "output_scale",
                "contract_sha256",
            ),
            "streaming muscle intervention",
        )
        try:
            lane = (
                None
                if mapping["raw_app_side"] is None
                else RawAppSide(mapping["raw_app_side"])
            )
            instance = cls(
                intervention_id=mapping["intervention_id"],
                muscle=mapping["muscle"],
                start_s=mapping["start_s"],
                end_s=mapping["end_s"],
                mode=MechanicsInterventionMode(mapping["mode"]),
                raw_app_side=lane,
                output_scale=mapping["output_scale"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid streaming muscle intervention") from exc
        if mapping["contract_sha256"] != instance.contract_sha256:
            raise ValueError("intervention contract SHA-256 mismatch")
        return instance


def _intervention_schedule_sha256(
    interventions: Sequence[StreamingMuscleIntervention],
) -> str:
    return hashlib.sha256(
        _canonical_json([item.to_dict() for item in interventions]).encode("utf-8")
    ).hexdigest()


def _parameter_from_dict(cls: Type[Any], value: Any, label: str) -> Any:
    expected = tuple(item.name for item in dataclasses.fields(cls))
    mapping = _exact_mapping(value, expected, label)
    kwargs = dict(mapping)
    if cls is VirtualWingHingeParameters:
        axis = kwargs["steering_axis_weights"]
        if not isinstance(axis, (tuple, list)):
            raise ValueError("steering_axis_weights must be an array")
        converted_axis = []
        for item in axis:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("invalid steering axis weight entry")
            name, weights = item
            if not isinstance(name, str) or not isinstance(weights, (tuple, list)):
                raise ValueError("invalid steering axis weight entry")
            converted_axis.append((name, tuple(weights)))
        kwargs["steering_axis_weights"] = tuple(converted_axis)
        default = kwargs["default_steering_axis_weights"]
        if not isinstance(default, (tuple, list)):
            raise ValueError("default steering axis weights must be an array")
        kwargs["default_steering_axis_weights"] = tuple(default)
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid %s" % label) from exc


@dataclass(frozen=True)
class StreamingMechanicsConfig:
    """All parameters that affect deterministic online mechanics.

    Baseline biases are dimensionless reduced-model inputs.  They are exposed
    so a calibrated model registry can replace them; they are not physiological
    firing rates and no carrier spikes are invented here.
    """

    physics_dt_s: float = STREAMING_MECHANICS_DT_S
    bridge_dt_s: float = STREAMING_MECHANICS_BRIDGE_DT_S
    phase_source: WingPhaseSource = WingPhaseSource.MODEL_OWNED_OSCILLATOR
    initial_phase_unwrapped_rad: float = 0.0
    baseline_power_initial_calcium: float = 0.48
    baseline_power_activation_bias: float = 0.18
    baseline_tension_activation_bias: float = 0.0
    steering_preferred_phases_rad: Tuple[Tuple[str, float], ...] = (
        ("iv2", 0.20 * _TWO_PI),
        ("i1", 0.35 * _TWO_PI),
        ("iv1", 0.55 * _TWO_PI),
        ("b3", 0.75 * _TWO_PI),
    )
    muscle_interventions: Tuple[StreamingMuscleIntervention, ...] = ()
    power_parameters: AsynchronousPowerParameters = field(
        default_factory=AsynchronousPowerParameters
    )
    steering_parameters: SteeringParameters = field(default_factory=SteeringParameters)
    tension_parameters: TensionParameters = field(default_factory=TensionParameters)
    oscillator_parameters: ThoraxOscillatorParameters = field(
        default_factory=ThoraxOscillatorParameters
    )
    hinge_parameters: VirtualWingHingeParameters = field(
        default_factory=VirtualWingHingeParameters
    )

    def __post_init__(self) -> None:
        physics_dt = _finite(self.physics_dt_s, "physics_dt_s")
        bridge_dt = _finite(self.bridge_dt_s, "bridge_dt_s")
        if abs(physics_dt - STREAMING_MECHANICS_DT_S) > 1.0e-15:
            raise ValueError("online mechanics requires an exact 0.1 ms physics grid")
        if abs(bridge_dt - STREAMING_MECHANICS_BRIDGE_DT_S) > 1.0e-15:
            raise ValueError("online mechanics requires an exact 0.5 ms bridge grid")
        if round(bridge_dt / physics_dt) != STREAMING_MECHANICS_STEPS_PER_BRIDGE:
            raise ValueError("bridge/physics clock ratio must be exactly 5:1")
        if not isinstance(self.phase_source, WingPhaseSource):
            raise TypeError("phase_source must be a WingPhaseSource")
        initial_phase = _finite(
            self.initial_phase_unwrapped_rad, "initial_phase_unwrapped_rad"
        )
        initial_calcium = _finite(
            self.baseline_power_initial_calcium,
            "baseline_power_initial_calcium",
        )
        power_bias = _finite(
            self.baseline_power_activation_bias,
            "baseline_power_activation_bias",
        )
        tension_bias = _finite(
            self.baseline_tension_activation_bias,
            "baseline_tension_activation_bias",
        )
        if initial_calcium < 0.0:
            raise ValueError("baseline power calcium must be non-negative")
        if not 0.0 <= power_bias <= 1.5 or not 0.0 <= tension_bias <= 1.5:
            raise ValueError("baseline activation biases must lie in [0, 1.5]")
        nested = (
            (self.power_parameters, AsynchronousPowerParameters, "power_parameters"),
            (self.steering_parameters, SteeringParameters, "steering_parameters"),
            (self.tension_parameters, TensionParameters, "tension_parameters"),
            (
                self.oscillator_parameters,
                ThoraxOscillatorParameters,
                "oscillator_parameters",
            ),
            (self.hinge_parameters, VirtualWingHingeParameters, "hinge_parameters"),
        )
        for value, expected, label in nested:
            if not isinstance(value, expected):
                raise TypeError("%s must be %s" % (label, expected.__name__))
        phases = tuple(self.steering_preferred_phases_rad)
        if tuple(name for name, _ in phases) != tuple(
            muscle for _, muscle in _STEERING_IDENTITIES
        ):
            raise ValueError("steering phases must be the ordered iv2/i1/iv1/b3 inventory")
        normalized_phases = tuple(
            (name, _finite(phase, "%s preferred phase" % name) % _TWO_PI)
            for name, phase in phases
        )
        interventions = tuple(self.muscle_interventions)
        if not all(
            isinstance(item, StreamingMuscleIntervention)
            for item in interventions
        ):
            raise TypeError(
                "muscle_interventions must contain StreamingMuscleIntervention values"
            )
        intervention_ids = tuple(item.intervention_id for item in interventions)
        if len(set(intervention_ids)) != len(intervention_ids):
            raise ValueError("muscle intervention IDs must be unique")
        interventions = tuple(
            sorted(interventions, key=lambda item: (item.start_s, item.intervention_id))
        )
        object.__setattr__(self, "physics_dt_s", physics_dt)
        object.__setattr__(self, "bridge_dt_s", bridge_dt)
        object.__setattr__(self, "initial_phase_unwrapped_rad", initial_phase)
        object.__setattr__(self, "baseline_power_initial_calcium", initial_calcium)
        object.__setattr__(self, "baseline_power_activation_bias", power_bias)
        object.__setattr__(self, "baseline_tension_activation_bias", tension_bias)
        object.__setattr__(self, "steering_preferred_phases_rad", normalized_phases)
        object.__setattr__(self, "muscle_interventions", interventions)

    def to_dict(self) -> Mapping[str, Any]:
        value = _jsonable(self)
        value["muscle_interventions"] = [
            intervention.to_dict() for intervention in self.muscle_interventions
        ]
        return value

    @classmethod
    def from_dict(cls, value: Any) -> "StreamingMechanicsConfig":
        mapping = _exact_mapping(
            value,
            tuple(item.name for item in dataclasses.fields(cls)),
            "streaming mechanics config",
        )
        phase_source = mapping["phase_source"]
        try:
            phase_source_enum = WingPhaseSource(phase_source)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid mechanics phase source") from exc
        phases = mapping["steering_preferred_phases_rad"]
        if not isinstance(phases, (tuple, list)):
            raise ValueError("steering preferred phases must be an array")
        parsed_phases = []
        for item in phases:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("invalid steering preferred phase entry")
            parsed_phases.append((item[0], item[1]))
        interventions = mapping["muscle_interventions"]
        if not isinstance(interventions, (tuple, list)):
            raise ValueError("muscle interventions must be an array")
        parsed_interventions = tuple(
            StreamingMuscleIntervention.from_dict(item) for item in interventions
        )
        return cls(
            physics_dt_s=mapping["physics_dt_s"],
            bridge_dt_s=mapping["bridge_dt_s"],
            phase_source=phase_source_enum,
            initial_phase_unwrapped_rad=mapping["initial_phase_unwrapped_rad"],
            baseline_power_initial_calcium=mapping[
                "baseline_power_initial_calcium"
            ],
            baseline_power_activation_bias=mapping[
                "baseline_power_activation_bias"
            ],
            baseline_tension_activation_bias=mapping[
                "baseline_tension_activation_bias"
            ],
            steering_preferred_phases_rad=tuple(parsed_phases),
            muscle_interventions=parsed_interventions,
            power_parameters=_parameter_from_dict(
                AsynchronousPowerParameters,
                mapping["power_parameters"],
                "power parameters",
            ),
            steering_parameters=_parameter_from_dict(
                SteeringParameters,
                mapping["steering_parameters"],
                "steering parameters",
            ),
            tension_parameters=_parameter_from_dict(
                TensionParameters,
                mapping["tension_parameters"],
                "tension parameters",
            ),
            oscillator_parameters=_parameter_from_dict(
                ThoraxOscillatorParameters,
                mapping["oscillator_parameters"],
                "oscillator parameters",
            ),
            hinge_parameters=_parameter_from_dict(
                VirtualWingHingeParameters,
                mapping["hinge_parameters"],
                "hinge parameters",
            ),
        )


@dataclass(frozen=True)
class StreamingMechanicsCheckpoint:
    """Canonical-JSON/SHA-256 protected complete mutable mechanics state."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = json.loads(_canonical_json(self.payload))
        mapping = _exact_mapping(
            copied,
            (
                "schema_version",
                "runtime_version",
                "config",
                "state",
                "payload_sha256",
            ),
            "streaming mechanics checkpoint",
        )
        if mapping["schema_version"] != STREAMING_MECHANICS_SCHEMA_VERSION:
            raise ValueError("streaming mechanics checkpoint schema version mismatch")
        if mapping["runtime_version"] != STREAMING_MECHANICS_RUNTIME_VERSION:
            raise ValueError("streaming mechanics checkpoint runtime version mismatch")
        digest = mapping["payload_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("streaming mechanics checkpoint digest is invalid")
        unsigned = {key: child for key, child in mapping.items() if key != "payload_sha256"}
        if _payload_sha256(unsigned) != digest:
            raise ValueError("streaming mechanics checkpoint payload SHA-256 mismatch")
        object.__setattr__(self, "payload", _deep_freeze(copied))

    def to_dict(self) -> Mapping[str, Any]:
        return _deep_thaw(self.payload)


@dataclass(frozen=True)
class StreamingMechanicsFrame:
    """One completed 0.1 ms mechanics interval and endpoint kinematics."""

    tick_index: int
    interval_start_s: float
    interval_end_s: float
    phase_source: WingPhaseSource
    phase_start_unwrapped_rad: float
    phase_end_unwrapped_rad: float
    model_owned_frequency_hz: float
    kinematic_frequency_hz: float
    applied_event_ids: Tuple[str, ...]
    suppressed_event_ids: Tuple[str, ...]
    active_intervention_ids: Tuple[str, ...]
    intervention_schedule_sha256: str
    pending_event_count: int
    actuation_muscle_snapshot: MuscleSnapshot
    actuation_wing_kinematics: WingKinematics
    natural_muscle_snapshot: MuscleSnapshot
    muscle_snapshot: MuscleSnapshot
    wing_kinematics: WingKinematics
    physics_command_interval_semantics: str = (
        "left-boundary zero-order-hold command over [interval_start, interval_end); "
        "NMJ events inside the interval affect the next physics command"
    )
    raw_lane_to_virtual_wing_convention: str = _RAW_LANE_TO_VIRTUAL_WING
    carrier_baseline_status: str = _CARRIER_BASELINE_STATUS

    def __post_init__(self) -> None:
        _nonnegative_int(self.tick_index, "tick_index")
        start = _finite(self.interval_start_s, "interval_start_s")
        end = _finite(self.interval_end_s, "interval_end_s")
        if abs(end - start - STREAMING_MECHANICS_DT_S) > _TIME_TOLERANCE_S:
            raise ValueError("mechanics frame must span exactly 0.1 ms")
        if not isinstance(self.phase_source, WingPhaseSource):
            raise TypeError("phase_source must be WingPhaseSource")
        phase_start = _finite(
            self.phase_start_unwrapped_rad, "phase_start_unwrapped_rad"
        )
        phase_end = _finite(self.phase_end_unwrapped_rad, "phase_end_unwrapped_rad")
        if phase_end + _TIME_TOLERANCE_S < phase_start:
            raise ValueError("unwrapped kinematic phase cannot decrease")
        model_frequency = _finite(
            self.model_owned_frequency_hz, "model_owned_frequency_hz"
        )
        kinematic_frequency = _finite(
            self.kinematic_frequency_hz, "kinematic_frequency_hz"
        )
        if model_frequency <= 0.0 or kinematic_frequency <= 0.0:
            raise ValueError("wing frequencies must be positive")
        if len(set(self.applied_event_ids)) != len(self.applied_event_ids):
            raise ValueError("frame event IDs must be unique")
        if len(set(self.suppressed_event_ids)) != len(self.suppressed_event_ids):
            raise ValueError("frame suppressed event IDs must be unique")
        if set(self.applied_event_ids).intersection(self.suppressed_event_ids):
            raise ValueError("an event cannot be both applied and suppressed")
        if len(set(self.active_intervention_ids)) != len(
            self.active_intervention_ids
        ):
            raise ValueError("frame active intervention IDs must be unique")
        if (
            not isinstance(self.intervention_schedule_sha256, str)
            or len(self.intervention_schedule_sha256) != 64
        ):
            raise ValueError("frame intervention schedule digest is invalid")
        _nonnegative_int(self.pending_event_count, "pending_event_count")
        if not isinstance(self.actuation_muscle_snapshot, MuscleSnapshot):
            raise TypeError("actuation_muscle_snapshot must be MuscleSnapshot")
        if not isinstance(self.natural_muscle_snapshot, MuscleSnapshot):
            raise TypeError("natural_muscle_snapshot must be MuscleSnapshot")
        if not isinstance(self.muscle_snapshot, MuscleSnapshot):
            raise TypeError("muscle_snapshot must be MuscleSnapshot")
        if not isinstance(self.actuation_wing_kinematics, WingKinematics):
            raise TypeError("actuation_wing_kinematics must be WingKinematics")
        if not isinstance(self.wing_kinematics, WingKinematics):
            raise TypeError("wing_kinematics must be WingKinematics")
        object.__setattr__(self, "interval_start_s", start)
        object.__setattr__(self, "interval_end_s", end)
        object.__setattr__(self, "phase_start_unwrapped_rad", phase_start)
        object.__setattr__(self, "phase_end_unwrapped_rad", phase_end)
        object.__setattr__(self, "model_owned_frequency_hz", model_frequency)
        object.__setattr__(self, "kinematic_frequency_hz", kinematic_frequency)


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
        "streaming motor event",
    )
    # StreamingMotorEvent itself verifies the only allowed signal semantics.
    from .streaming_bridge import StreamingSignalSemantics

    try:
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
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid checkpoint motor event") from exc


class StreamingMuscleWingStepper:
    """Deterministic event-to-muscle-to-``WingKinematics`` online stepper."""

    validation_status = "exploratory"
    raw_lane_to_virtual_wing_convention = _RAW_LANE_TO_VIRTUAL_WING
    carrier_baseline_status = _CARRIER_BASELINE_STATUS

    def __init__(
        self, config: StreamingMechanicsConfig = StreamingMechanicsConfig()
    ) -> None:
        if not isinstance(config, StreamingMechanicsConfig):
            raise TypeError("config must be StreamingMechanicsConfig")
        self.config = config
        phase_by_muscle = dict(config.steering_preferred_phases_rad)
        self._power: Dict[Tuple[RawAppSide, str], AsynchronousPowerMuscle] = {}
        self._steering: Dict[
            Tuple[RawAppSide, str], PhaseCodedSteeringMuscle
        ] = {}
        self._tension: Dict[Tuple[RawAppSide, str], TensionMuscle] = {}
        for lane in (RawAppSide.L, RawAppSide.R):
            for muscle in ("DLM", "DVM"):
                self._power[(lane, muscle)] = AsynchronousPowerMuscle(
                    parameters=config.power_parameters,
                    initial_calcium=config.baseline_power_initial_calcium,
                )
            for _motor, muscle in _STEERING_IDENTITIES:
                self._steering[(lane, muscle)] = PhaseCodedSteeringMuscle(
                    preferred_phase_rad=phase_by_muscle[muscle],
                    parameters=config.steering_parameters,
                )
            self._tension[(lane, "tp1")] = TensionMuscle(
                parameters=config.tension_parameters
            )
        self._oscillator = ThoraxOscillator(
            parameters=config.oscillator_parameters,
            initial_phase_rad=config.initial_phase_unwrapped_rad,
        )
        self._hinge = VirtualWingHinge(config.hinge_parameters)
        self._tick_index = 0
        self._phase_unwrapped_rad = config.initial_phase_unwrapped_rad
        self._queue_sequence = 0
        self._pending_events: List[Tuple[float, int, StreamingMotorEvent]] = []
        self._seen_event_ids: set[str] = set()
        self._applied_event_ids: List[str] = []
        self._applied_events: List[StreamingMotorEvent] = []
        self._suppressed_event_ids: List[str] = []
        self._suppressed_events: List[StreamingMotorEvent] = []
        self._intervention_schedule_sha256 = _intervention_schedule_sha256(
            config.muscle_interventions
        )

    @property
    def tick_index(self) -> int:
        return self._tick_index

    @property
    def current_time_s(self) -> float:
        return self._tick_index * self.config.physics_dt_s

    @property
    def phase_unwrapped_rad(self) -> float:
        return self._phase_unwrapped_rad

    @property
    def pending_event_count(self) -> int:
        return len(self._pending_events)

    @property
    def applied_event_ids(self) -> Tuple[str, ...]:
        return tuple(self._applied_event_ids)

    @property
    def suppressed_event_ids(self) -> Tuple[str, ...]:
        return tuple(self._suppressed_event_ids)

    @property
    def intervention_schedule_sha256(self) -> str:
        return self._intervention_schedule_sha256

    @property
    def active_intervention_ids(self) -> Tuple[str, ...]:
        return tuple(
            item.intervention_id
            for item in self._active_interventions(self.current_time_s)
        )

    @staticmethod
    def _virtual_side(lane: RawAppSide) -> Side:
        # This mapping is deliberately a display-coordinate convention only.
        return Side.LEFT if lane is RawAppSide.L else Side.RIGHT

    @staticmethod
    def _validate_event_identity(event: StreamingMotorEvent) -> None:
        if not isinstance(event, StreamingMotorEvent):
            raise TypeError("delivered events must be StreamingMotorEvent values")
        if event.anatomical_side is not AnatomicalSide.UNKNOWN:
            raise ValueError("streaming mechanics cannot infer anatomical laterality")
        if (event.motor_neuron, event.muscle) not in _STEERING_IDENTITIES:
            raise ValueError("event is outside the iv2/i1/iv1/b3 steering inventory")

    def push_delivered_events(
        self, delivered_events: Sequence[StreamingMotorEvent]
    ) -> None:
        """Queue one bridge frame's ``delivered_events`` transactionally.

        No other field of ``StreamingBridgeFrame`` is accepted or inspected.
        The batch is fully validated before the queue or event-ID ledger is
        mutated.  Events older than the mechanics clock are rejected rather
        than applied retroactively.
        """

        try:
            events = tuple(delivered_events)
        except TypeError as exc:
            raise TypeError("delivered_events must be a finite sequence") from exc
        batch_ids: set[str] = set()
        for event in events:
            self._validate_event_identity(event)
            if event.event_id in batch_ids or event.event_id in self._seen_event_ids:
                raise ValueError("duplicate delivered event ID: %s" % event.event_id)
            if event.availability_time_s < self.current_time_s - _TIME_TOLERANCE_S:
                raise ValueError("cannot apply an NMJ event retroactively")
            batch_ids.add(event.event_id)
        # Commit only after all validation succeeds.
        for event in events:
            heapq.heappush(
                self._pending_events,
                (event.availability_time_s, self._queue_sequence, event),
            )
            self._queue_sequence += 1
            self._seen_event_ids.add(event.event_id)

    def _active_interventions(
        self, time_s: float
    ) -> Tuple[StreamingMuscleIntervention, ...]:
        return tuple(
            item for item in self.config.muscle_interventions if item.active(time_s)
        )

    def _effective_output_scale(
        self, lane: RawAppSide, muscle: str, time_s: float
    ) -> float:
        scale = 1.0
        for intervention in self._active_interventions(time_s):
            if not intervention.targets(lane, muscle):
                continue
            if intervention.mode is MechanicsInterventionMode.SILENCE:
                return 0.0
            scale *= intervention.output_scale
        return scale

    def _event_excitation_is_suppressed(
        self, event: StreamingMotorEvent, time_s: float
    ) -> bool:
        return any(
            intervention.mode is MechanicsInterventionMode.SILENCE
            and intervention.targets(event.raw_app_side, event.muscle)
            for intervention in self._active_interventions(time_s)
        )

    def _snapshot(
        self, *, time_s: Optional[float] = None, effective: bool = False
    ) -> MuscleSnapshot:
        if effective and time_s is None:
            raise ValueError("effective snapshots require an explicit causal time")
        individual: Dict[str, IndividualMuscleState] = {}
        power_values: Dict[RawAppSide, List[float]] = {
            RawAppSide.L: [],
            RawAppSide.R: [],
        }
        steering_values: Dict[RawAppSide, List[float]] = {
            RawAppSide.L: [],
            RawAppSide.R: [],
        }
        tension_values: Dict[RawAppSide, List[float]] = {
            RawAppSide.L: [],
            RawAppSide.R: [],
        }
        for (lane, muscle), model in self._power.items():
            virtual_side = self._virtual_side(lane)
            power_values[lane].append(model.activation)
            key = "%s:%s" % (virtual_side.value, muscle)
            individual[key] = IndividualMuscleState(
                muscle=muscle,
                side=virtual_side,
                muscle_class=MuscleClass.ASYNCHRONOUS_POWER,
                activation=model.activation,
                force_n=model.force_n,
                virtual_moment_arm_m=self.config.hinge_parameters.power_virtual_moment_arm_m,
            )
        for (lane, muscle), model in self._steering.items():
            virtual_side = self._virtual_side(lane)
            output_scale = (
                self._effective_output_scale(lane, muscle, time_s)
                if effective
                else 1.0
            )
            steering_values[lane].append(
                model.activation * model.phase_effect * output_scale
            )
            key = "%s:%s" % (virtual_side.value, muscle)
            individual[key] = IndividualMuscleState(
                muscle=muscle,
                side=virtual_side,
                muscle_class=MuscleClass.STEERING,
                activation=model.activation * output_scale,
                force_n=model.force_n * output_scale,
                phase_effect=model.phase_effect,
                virtual_moment_arm_m=self.config.hinge_parameters.steering_virtual_moment_arm_m,
            )
        for (lane, muscle), model in self._tension.items():
            virtual_side = self._virtual_side(lane)
            p = model.parameters
            resonance = 1.0 + p.resonance_gain * (
                model.activation - p.baseline_activation
            )
            tension_values[lane].append(resonance)
            key = "%s:%s" % (virtual_side.value, muscle)
            individual[key] = IndividualMuscleState(
                muscle=muscle,
                side=virtual_side,
                muscle_class=MuscleClass.TENSION,
                activation=model.activation,
                force_n=0.0,
                resonance_scale=resonance,
            )

        def lane_mean(values: Mapping[RawAppSide, Sequence[float]], lane: RawAppSide) -> float:
            selected = values[lane]
            return float(np.mean(selected))

        return MuscleSnapshot(
            power_left=lane_mean(power_values, RawAppSide.L),
            power_right=lane_mean(power_values, RawAppSide.R),
            steering_left=lane_mean(steering_values, RawAppSide.L),
            steering_right=lane_mean(steering_values, RawAppSide.R),
            tension_left=lane_mean(tension_values, RawAppSide.L),
            tension_right=lane_mean(tension_values, RawAppSide.R),
            individual=individual,
        )

    def _advance_carrier(
        self,
        dt_s: float,
        kinematic_phase_start_rad: float,
    ) -> None:
        """Advance event-independent DLM/DVM/tp1/thorax state one fixed tick."""

        omega = 2.0 * math.pi * self._oscillator.frequency_hz
        for (_lane, muscle), model in self._power.items():
            sign = 1.0 if muscle == "DLM" else -1.0
            stretch = sign * math.sin(kinematic_phase_start_rad)
            shortening = sign * omega * math.cos(kinematic_phase_start_rad) / max(
                omega, 1.0
            )
            model.step(
                spike_count=0,
                stretch=stretch,
                shortening_velocity_s=shortening,
                dt_s=dt_s,
                activation_bias=self.config.baseline_power_activation_bias,
            )
        for model in self._tension.values():
            model.step(
                spike_count=0,
                dt_s=dt_s,
                activation_bias=self.config.baseline_tension_activation_bias,
            )

        snapshot = self._snapshot()
        tension_scale = 0.5 * (snapshot.tension_left + snapshot.tension_right)
        self._oscillator.step(
            power_activation=0.5 * (snapshot.power_left + snapshot.power_right),
            tension_resonance_scale=tension_scale,
            dt_s=dt_s,
        )

    def _advance_steering_segment(
        self,
        dt_s: float,
        events_at_start: Sequence[StreamingMotorEvent],
    ) -> None:
        """Advance steering state, injecting events at this segment's left edge."""

        if dt_s <= 0.0:
            raise ValueError("steering subinterval must be positive")
        event_phases: Dict[Tuple[RawAppSide, str], List[float]] = {}
        for event in events_at_start:
            event_phases.setdefault((event.raw_app_side, event.muscle), []).append(
                event.wingbeat_phase_rad
            )
        for key, model in self._steering.items():
            model.step(event_phases.get(key, ()), dt_s)

    def _validate_phase_input(
        self, measured_phase_end_unwrapped_rad: Optional[float]
    ) -> Optional[float]:
        if self.config.phase_source is WingPhaseSource.MODEL_OWNED_OSCILLATOR:
            if measured_phase_end_unwrapped_rad is not None:
                raise ValueError(
                    "measured phase is forbidden when the model-owned oscillator is authoritative"
                )
            return None
        if measured_phase_end_unwrapped_rad is None:
            raise ValueError("measured-unwrapped phase mode requires every endpoint")
        endpoint = _finite(
            measured_phase_end_unwrapped_rad,
            "measured_phase_end_unwrapped_rad",
        )
        phase_delta = endpoint - self._phase_unwrapped_rad
        if phase_delta <= _TIME_TOLERANCE_S:
            raise ValueError("measured unwrapped phase must increase strictly")
        if phase_delta > _TWO_PI + _TIME_TOLERANCE_S:
            raise ValueError("measured phase cannot advance by more than one cycle per step")
        return endpoint

    def step(
        self, *, measured_phase_end_unwrapped_rad: Optional[float] = None
    ) -> StreamingMechanicsFrame:
        """Advance one causal half-open 0.1 ms mechanics interval."""

        measured_endpoint = self._validate_phase_input(
            measured_phase_end_unwrapped_rad
        )
        start_s = self.current_time_s
        end_s = start_s + self.config.physics_dt_s
        phase_start = self._phase_unwrapped_rad
        active_intervention_ids = tuple(
            item.intervention_id for item in self._active_interventions(start_s)
        )
        # This is the only command causal at the interval's left boundary. The
        # endpoint state below is logged for state continuation and becomes the
        # basis of the following interval's command; it is never applied
        # retroactively to the interval that produced it.
        actuation_snapshot = self._snapshot(time_s=start_s, effective=True)
        actuation_wings = self._hinge.evaluate(
            phase_rad=phase_start,
            frequency_hz=self._oscillator.frequency_hz,
            muscles=actuation_snapshot,
        )

        due: List[StreamingMotorEvent] = []
        while (
            self._pending_events
            and self._pending_events[0][0] < end_s - _TIME_TOLERANCE_S
        ):
            availability, _sequence, event = heapq.heappop(self._pending_events)
            if availability < start_s - _TIME_TOLERANCE_S:
                raise RuntimeError("pending NMJ event missed its causal interval")
            due.append(event)
        due.sort(key=lambda item: (item.availability_time_s, item.event_id))

        grouped: List[Tuple[float, List[StreamingMotorEvent]]] = []
        for event in due:
            effective_availability = (
                start_s
                if abs(event.availability_time_s - start_s) <= _TIME_TOLERANCE_S
                else event.availability_time_s
            )
            if grouped and abs(effective_availability - grouped[-1][0]) <= _TIME_TOLERANCE_S:
                grouped[-1][1].append(event)
            else:
                grouped.append((effective_availability, [event]))

        def phase_at(time_s: float) -> float:
            if measured_endpoint is None:
                return self._oscillator.phase_rad
            fraction = (time_s - start_s) / self.config.physics_dt_s
            return phase_start + fraction * (measured_endpoint - phase_start)

        cursor_s = start_s
        applied_this_frame: List[StreamingMotorEvent] = []
        suppressed_this_frame: List[StreamingMotorEvent] = []
        for event_time_s, events in grouped:
            if event_time_s > cursor_s:
                self._advance_steering_segment(
                    event_time_s - cursor_s,
                    (),
                )
            cursor_s = event_time_s
            next_times = [
                candidate_time
                for candidate_time, _candidate_events in grouped
                if candidate_time > event_time_s
            ]
            segment_end_s = min(next_times) if next_times else end_s
            applied_events = [
                event
                for event in events
                if not self._event_excitation_is_suppressed(event, event_time_s)
            ]
            suppressed_events = [
                event for event in events if event not in applied_events
            ]
            self._advance_steering_segment(
                segment_end_s - cursor_s,
                applied_events,
            )
            self._applied_event_ids.extend(
                event.event_id for event in applied_events
            )
            self._applied_events.extend(applied_events)
            self._suppressed_event_ids.extend(
                event.event_id for event in suppressed_events
            )
            self._suppressed_events.extend(suppressed_events)
            applied_this_frame.extend(applied_events)
            suppressed_this_frame.extend(suppressed_events)
            cursor_s = segment_end_s
        if not grouped:
            self._advance_steering_segment(
                self.config.physics_dt_s,
                (),
            )
        elif cursor_s < end_s - _TIME_TOLERANCE_S:
            # Normally the final event segment reaches ``end_s``.  Retain this
            # guard for round-off without allowing a zero-duration muscle step.
            self._advance_steering_segment(end_s - cursor_s, ())

        # The carrier has no steering-state input.  Advancing it on the fixed
        # 0.1 ms grid makes its phase schedule exactly independent of event
        # count and sub-grid event times.
        self._advance_carrier(
            self.config.physics_dt_s,
            phase_at(start_s),
        )

        if measured_endpoint is None:
            phase_end = self._oscillator.phase_rad
            kinematic_frequency_hz = self._oscillator.frequency_hz
        else:
            phase_end = measured_endpoint
            kinematic_frequency_hz = (
                (phase_end - phase_start)
                / (_TWO_PI * self.config.physics_dt_s)
            )
        self._phase_unwrapped_rad = phase_end
        natural_snapshot = self._snapshot()
        snapshot = self._snapshot(time_s=end_s, effective=True)
        wings = self._hinge.evaluate(
            phase_rad=phase_end,
            frequency_hz=kinematic_frequency_hz,
            muscles=snapshot,
        )
        frame = StreamingMechanicsFrame(
            tick_index=self._tick_index,
            interval_start_s=start_s,
            interval_end_s=end_s,
            phase_source=self.config.phase_source,
            phase_start_unwrapped_rad=phase_start,
            phase_end_unwrapped_rad=phase_end,
            model_owned_frequency_hz=self._oscillator.frequency_hz,
            kinematic_frequency_hz=kinematic_frequency_hz,
            applied_event_ids=tuple(
                event.event_id for event in applied_this_frame
            ),
            suppressed_event_ids=tuple(
                event.event_id for event in suppressed_this_frame
            ),
            active_intervention_ids=active_intervention_ids,
            intervention_schedule_sha256=self._intervention_schedule_sha256,
            pending_event_count=len(self._pending_events),
            actuation_muscle_snapshot=actuation_snapshot,
            actuation_wing_kinematics=actuation_wings,
            natural_muscle_snapshot=natural_snapshot,
            muscle_snapshot=snapshot,
            wing_kinematics=wings,
        )
        self._tick_index += 1
        return frame

    def project_phase_end_unwrapped_rad(
        self, duration_s: float = STREAMING_MECHANICS_BRIDGE_DT_S
    ) -> float:
        """Return an exact non-mutating model-owned carrier phase projection.

        ``duration_s`` must be a positive integer number of 0.1 ms physics
        ticks.  The default projects the five ticks of one 0.5 ms bridge
        interval.  It clones the complete deterministic state and executes the
        same fixed-grid carrier updates used by :meth:`step`; it does not use a
        constant-frequency approximation.  Steering events may be present in
        the clone, but by model construction they have no path to DLM/DVM/tp1
        or the thorax oscillator, so the projection is event-independent.
        """

        if self.config.phase_source is not WingPhaseSource.MODEL_OWNED_OSCILLATOR:
            raise ValueError("phase projection is defined only for model-owned phase")
        duration = _finite(duration_s, "duration_s")
        if duration <= 0.0:
            raise ValueError("projection duration must be positive")
        tick_count = round(duration / self.config.physics_dt_s)
        if (
            tick_count < 1
            or abs(tick_count * self.config.physics_dt_s - duration)
            > _TIME_TOLERANCE_S
        ):
            raise ValueError("projection duration must be on the 0.1 ms grid")
        clone = type(self).from_checkpoint(self.checkpoint())
        for _ in range(tick_count):
            clone.step()
        return clone.phase_unwrapped_rad

    def advance_bridge_interval(
        self,
        interval: StreamingBridgeIntervalStart,
        *,
        measured_phase_ends_unwrapped_rad: Optional[Sequence[float]] = None,
    ) -> Tuple[StreamingMechanicsFrame, ...]:
        """Consume one exact bridge receipt and emit five physics frames.

        The bridge tick and half-open bounds must exactly match the mechanics
        clock, preventing skipped, duplicated, or reordered handoffs. In measured
        phase mode the caller must provide five unwrapped phase endpoints, one
        for each 0.1 ms physics interval.  In model-owned mode that argument is
        omitted.
        """

        if not isinstance(interval, StreamingBridgeIntervalStart):
            raise TypeError("interval must be StreamingBridgeIntervalStart")
        if self._tick_index % STREAMING_MECHANICS_STEPS_PER_BRIDGE != 0:
            raise RuntimeError("mechanics is not at a bridge interval boundary")
        expected_bridge_tick = (
            self._tick_index // STREAMING_MECHANICS_STEPS_PER_BRIDGE
        )
        expected_start_s = self.current_time_s
        expected_end_s = expected_start_s + self.config.bridge_dt_s
        if (
            interval.tick_index != expected_bridge_tick
            or abs(interval.interval_start_s - expected_start_s)
            > _TIME_TOLERANCE_S
            or abs(interval.interval_end_s - expected_end_s)
            > _TIME_TOLERANCE_S
        ):
            raise ValueError("bridge interval receipt does not match mechanics clock")
        if any(
            event.availability_time_s
            < interval.interval_start_s - _TIME_TOLERANCE_S
            or event.availability_time_s
            >= interval.interval_end_s - _TIME_TOLERANCE_S
            for event in interval.delivered_events
        ):
            raise ValueError("bridge receipt contains an event outside its interval")

        if self.config.phase_source is WingPhaseSource.MEASURED_UNWRAPPED:
            if measured_phase_ends_unwrapped_rad is None:
                raise ValueError("measured phase mode requires five endpoints")
            endpoints = tuple(measured_phase_ends_unwrapped_rad)
            if len(endpoints) != STREAMING_MECHANICS_STEPS_PER_BRIDGE:
                raise ValueError("one bridge interval requires exactly five phase endpoints")
            # Validate the complete measured schedule before accepting events,
            # so a bad endpoint cannot partially mutate the queue.
            previous = self._phase_unwrapped_rad
            validated_endpoints = []
            for endpoint in endpoints:
                parsed = _finite(endpoint, "measured phase endpoint")
                delta = parsed - previous
                if delta <= _TIME_TOLERANCE_S:
                    raise ValueError("measured unwrapped phase must increase strictly")
                if delta > _TWO_PI + _TIME_TOLERANCE_S:
                    raise ValueError(
                        "measured phase cannot advance by more than one cycle per step"
                    )
                validated_endpoints.append(parsed)
                previous = parsed
            endpoints = tuple(validated_endpoints)
        else:
            if measured_phase_ends_unwrapped_rad is not None:
                raise ValueError("model-owned phase mode cannot accept measured endpoints")
            endpoints = (None,) * STREAMING_MECHANICS_STEPS_PER_BRIDGE
        self.push_delivered_events(interval.delivered_events)
        return tuple(
            self.step(measured_phase_end_unwrapped_rad=endpoint)
            for endpoint in endpoints
        )

    @staticmethod
    def _muscle_state(model: Any, kind: str) -> Mapping[str, float]:
        if kind == "power":
            return {
                "calcium": model.calcium,
                "activation": model.activation,
                "force_n": model.force_n,
            }
        if kind == "steering":
            return {
                "activation": model.activation,
                "phase_effect": model.phase_effect,
                "force_n": model.force_n,
            }
        if kind == "tension":
            return {
                "activation": model.activation,
                "excitation": model.excitation,
            }
        raise RuntimeError("unknown muscle state kind")

    @staticmethod
    def _ordered_model_state(models: Mapping[Tuple[RawAppSide, str], Any], kind: str) -> List[Mapping[str, Any]]:
        return [
            {
                "raw_app_side": lane.value,
                "muscle": muscle,
                "state": StreamingMuscleWingStepper._muscle_state(model, kind),
            }
            for (lane, muscle), model in sorted(
                models.items(), key=lambda item: (item[0][0].value, item[0][1])
            )
        ]

    def checkpoint(self) -> StreamingMechanicsCheckpoint:
        pending = [
            {
                "availability_time_s": availability,
                "sequence": sequence,
                "event": _event_to_dict(event),
            }
            for availability, sequence, event in sorted(self._pending_events)
        ]
        unsigned: Dict[str, Any] = {
            "schema_version": STREAMING_MECHANICS_SCHEMA_VERSION,
            "runtime_version": STREAMING_MECHANICS_RUNTIME_VERSION,
            "config": self.config.to_dict(),
            "state": {
                "tick_index": self._tick_index,
                "phase_unwrapped_rad": self._phase_unwrapped_rad,
                "intervention_schedule_sha256": self._intervention_schedule_sha256,
                "active_intervention_ids": list(self.active_intervention_ids),
                "queue_sequence": self._queue_sequence,
                "pending_events": pending,
                "seen_event_ids": sorted(self._seen_event_ids),
                "applied_event_ids": list(self._applied_event_ids),
                "applied_events": [
                    _event_to_dict(event) for event in self._applied_events
                ],
                "suppressed_event_ids": list(self._suppressed_event_ids),
                "suppressed_events": [
                    _event_to_dict(event) for event in self._suppressed_events
                ],
                "power": self._ordered_model_state(self._power, "power"),
                "steering": self._ordered_model_state(self._steering, "steering"),
                "tension": self._ordered_model_state(self._tension, "tension"),
                "oscillator": {
                    "phase_rad": self._oscillator.phase_rad,
                    "frequency_hz": self._oscillator.frequency_hz,
                },
            },
        }
        return StreamingMechanicsCheckpoint(
            {**unsigned, "payload_sha256": _payload_sha256(unsigned)}
        )

    @staticmethod
    def _validate_state_value(value: Any, low: float, high: Optional[float], label: str) -> float:
        parsed = _finite(value, label)
        if parsed < low or (high is not None and parsed > high):
            raise ValueError("%s is outside its valid range" % label)
        return parsed

    def _restore_model_array(
        self,
        raw: Any,
        models: Mapping[Tuple[RawAppSide, str], Any],
        kind: str,
    ) -> None:
        if not isinstance(raw, (list, tuple)) or len(raw) != len(models):
            raise ValueError("checkpoint %s inventory is invalid" % kind)
        expected_keys = set(models)
        observed_keys: set[Tuple[RawAppSide, str]] = set()
        for item in raw:
            mapping = _exact_mapping(
                item, ("raw_app_side", "muscle", "state"), "%s model" % kind
            )
            try:
                key = (RawAppSide(mapping["raw_app_side"]), mapping["muscle"])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid %s model identity" % kind) from exc
            if key not in expected_keys or key in observed_keys:
                raise ValueError("checkpoint %s model identity is invalid" % kind)
            observed_keys.add(key)
            model = models[key]
            if kind == "power":
                state = _exact_mapping(
                    mapping["state"],
                    ("calcium", "activation", "force_n"),
                    "power state",
                )
                calcium = self._validate_state_value(
                    state["calcium"], 0.0, None, "power calcium"
                )
                activation = self._validate_state_value(
                    state["activation"], 0.0, 1.5, "power activation"
                )
                force = self._validate_state_value(
                    state["force_n"], 0.0, None, "power force"
                )
                if abs(force - model.parameters.max_isometric_force_n * activation) > 1.0e-15:
                    raise ValueError("power force is inconsistent with activation")
                model.calcium = calcium
                model.activation = activation
                model.force_n = force
            elif kind == "steering":
                state = _exact_mapping(
                    mapping["state"],
                    ("activation", "phase_effect", "force_n"),
                    "steering state",
                )
                activation = self._validate_state_value(
                    state["activation"], 0.0, 1.5, "steering activation"
                )
                phase_effect = self._validate_state_value(
                    state["phase_effect"], -1.5, 1.5, "steering phase effect"
                )
                force = self._validate_state_value(
                    state["force_n"], 0.0, None, "steering force"
                )
                if abs(force - model.parameters.max_force_n * activation) > 1.0e-15:
                    raise ValueError("steering force is inconsistent with activation")
                model.activation = activation
                model.phase_effect = phase_effect
                model.force_n = force
            elif kind == "tension":
                state = _exact_mapping(
                    mapping["state"],
                    ("activation", "excitation"),
                    "tension state",
                )
                model.activation = self._validate_state_value(
                    state["activation"], 0.0, 1.5, "tension activation"
                )
                model.excitation = self._validate_state_value(
                    state["excitation"], 0.0, None, "tension excitation"
                )
            else:
                raise RuntimeError("unknown checkpoint muscle state")
        if observed_keys != expected_keys:
            raise ValueError("checkpoint %s inventory is incomplete" % kind)

    def _load_validated_state(self, raw: Any) -> None:
        state = _exact_mapping(
            raw,
            (
                "tick_index",
                "phase_unwrapped_rad",
                "intervention_schedule_sha256",
                "active_intervention_ids",
                "queue_sequence",
                "pending_events",
                "seen_event_ids",
                "applied_event_ids",
                "applied_events",
                "suppressed_event_ids",
                "suppressed_events",
                "power",
                "steering",
                "tension",
                "oscillator",
            ),
            "streaming mechanics state",
        )
        tick_index = _nonnegative_int(state["tick_index"], "tick_index")
        phase = _finite(state["phase_unwrapped_rad"], "phase_unwrapped_rad")
        queue_sequence = _nonnegative_int(state["queue_sequence"], "queue_sequence")
        schedule_digest = state["intervention_schedule_sha256"]
        if schedule_digest != self._intervention_schedule_sha256:
            raise ValueError("checkpoint intervention schedule digest mismatch")
        active_raw = state["active_intervention_ids"]
        if not isinstance(active_raw, (list, tuple)) or not all(
            isinstance(item, str) and item for item in active_raw
        ):
            raise ValueError("active intervention IDs must be non-empty strings")
        if len(set(active_raw)) != len(active_raw):
            raise ValueError("active intervention IDs contain duplicates")
        checkpoint_time = tick_index * self.config.physics_dt_s
        expected_active = tuple(
            item.intervention_id
            for item in self._active_interventions(checkpoint_time)
        )
        if tuple(active_raw) != expected_active:
            raise ValueError("checkpoint active intervention IDs are inconsistent")

        seen_raw = state["seen_event_ids"]
        applied_raw = state["applied_event_ids"]
        suppressed_raw = state["suppressed_event_ids"]
        if not isinstance(seen_raw, (list, tuple)) or not all(
            isinstance(item, str) and item for item in seen_raw
        ):
            raise ValueError("seen event IDs must be non-empty strings")
        if not isinstance(applied_raw, (list, tuple)) or not all(
            isinstance(item, str) and item for item in applied_raw
        ):
            raise ValueError("applied event IDs must be non-empty strings")
        if not isinstance(suppressed_raw, (list, tuple)) or not all(
            isinstance(item, str) and item for item in suppressed_raw
        ):
            raise ValueError("suppressed event IDs must be non-empty strings")
        seen = set(seen_raw)
        applied = list(applied_raw)
        suppressed = list(suppressed_raw)
        if (
            len(seen) != len(seen_raw)
            or len(set(applied)) != len(applied)
            or len(set(suppressed)) != len(suppressed)
            or set(applied).intersection(suppressed)
        ):
            raise ValueError("checkpoint event ID ledgers contain duplicates")

        pending_raw = state["pending_events"]
        if not isinstance(pending_raw, (list, tuple)):
            raise ValueError("pending events must be an array")
        pending: List[Tuple[float, int, StreamingMotorEvent]] = []
        pending_ids: set[str] = set()
        sequences: set[int] = set()
        applied_events_raw = state["applied_events"]
        if not isinstance(applied_events_raw, (list, tuple)):
            raise ValueError("applied events must be an array")
        applied_events = [_event_from_dict(value) for value in applied_events_raw]
        for event in applied_events:
            self._validate_event_identity(event)
            if event.availability_time_s >= checkpoint_time - _TIME_TOLERANCE_S:
                raise ValueError("checkpoint applies an event before its causal interval")
            if self._event_excitation_is_suppressed(
                event, event.availability_time_s
            ):
                raise ValueError(
                    "checkpoint applies an event during an active SILENCE intervention"
                )
        if applied_events != sorted(
            applied_events,
            key=lambda event: (event.availability_time_s, event.event_id),
        ):
            raise ValueError("checkpoint applied events are not chronologically ordered")
        if [event.event_id for event in applied_events] != applied:
            raise ValueError("applied event receipts do not match their ID ledger")
        suppressed_events_raw = state["suppressed_events"]
        if not isinstance(suppressed_events_raw, (list, tuple)):
            raise ValueError("suppressed events must be an array")
        suppressed_events = [
            _event_from_dict(value) for value in suppressed_events_raw
        ]
        for event in suppressed_events:
            self._validate_event_identity(event)
            if event.availability_time_s >= checkpoint_time - _TIME_TOLERANCE_S:
                raise ValueError(
                    "checkpoint suppresses an event before its causal interval"
                )
            if not self._event_excitation_is_suppressed(
                event, event.availability_time_s
            ):
                raise ValueError(
                    "checkpoint suppresses an event without an active SILENCE intervention"
                )
        if suppressed_events != sorted(
            suppressed_events,
            key=lambda event: (event.availability_time_s, event.event_id),
        ):
            raise ValueError(
                "checkpoint suppressed events are not chronologically ordered"
            )
        if [event.event_id for event in suppressed_events] != suppressed:
            raise ValueError(
                "suppressed event receipts do not match their ID ledger"
            )
        for item in pending_raw:
            mapping = _exact_mapping(
                item,
                ("availability_time_s", "sequence", "event"),
                "pending event",
            )
            availability = _finite(
                mapping["availability_time_s"], "pending availability"
            )
            sequence = _nonnegative_int(mapping["sequence"], "pending sequence")
            event = _event_from_dict(mapping["event"])
            self._validate_event_identity(event)
            if availability != event.availability_time_s:
                raise ValueError("pending event availability fields disagree")
            if availability < checkpoint_time - _TIME_TOLERANCE_S:
                raise ValueError("checkpoint contains a retroactive pending event")
            if event.event_id in pending_ids or sequence in sequences:
                raise ValueError("checkpoint pending event identities are duplicated")
            pending_ids.add(event.event_id)
            sequences.add(sequence)
            pending.append((availability, sequence, event))
        if (
            not pending_ids.issubset(seen)
            or not set(applied).issubset(seen)
            or not set(suppressed).issubset(seen)
        ):
            raise ValueError("checkpoint event ledgers are inconsistent")
        consumed = set(applied).union(suppressed)
        if pending_ids.intersection(consumed):
            raise ValueError("an event cannot be pending and consumed")
        if seen != pending_ids.union(consumed):
            raise ValueError("seen event ledger contains an unaccounted event")
        if sequences and queue_sequence <= max(sequences):
            raise ValueError("queue sequence does not follow pending events")

        self._restore_model_array(state["power"], self._power, "power")
        self._restore_model_array(state["steering"], self._steering, "steering")
        self._restore_model_array(state["tension"], self._tension, "tension")
        oscillator = _exact_mapping(
            state["oscillator"], ("phase_rad", "frequency_hz"), "oscillator state"
        )
        oscillator_phase = _finite(oscillator["phase_rad"], "oscillator phase")
        oscillator_frequency = _finite(
            oscillator["frequency_hz"], "oscillator frequency"
        )
        p = self.config.oscillator_parameters
        if not p.minimum_frequency_hz <= oscillator_frequency <= p.maximum_frequency_hz:
            raise ValueError("oscillator frequency is outside configured bounds")
        if (
            self.config.phase_source is WingPhaseSource.MODEL_OWNED_OSCILLATOR
            and oscillator_phase != phase
        ):
            raise ValueError("model-owned oscillator and kinematic phase disagree")
        if self.config.phase_source is WingPhaseSource.MODEL_OWNED_OSCILLATOR:
            # DLM/DVM/tp1 and the thorax carrier are event-independent by model
            # construction. Replay them from the frozen initial state so a
            # checksum-recomputed checkpoint cannot forge carrier dynamics.
            expected_carrier = type(self)(self.config)
            for _ in range(tick_index):
                expected_carrier.step()
            if (
                expected_carrier._phase_unwrapped_rad != phase
                or expected_carrier._oscillator.phase_rad != oscillator_phase
                or expected_carrier._oscillator.frequency_hz
                != oscillator_frequency
                or _canonical_json(
                    expected_carrier._ordered_model_state(
                        expected_carrier._power, "power"
                    )
                )
                != _canonical_json(self._ordered_model_state(self._power, "power"))
                or _canonical_json(
                    expected_carrier._ordered_model_state(
                        expected_carrier._tension, "tension"
                    )
                )
                != _canonical_json(
                    self._ordered_model_state(self._tension, "tension")
                )
            ):
                raise ValueError("checkpoint model-owned carrier state is inconsistent")

        self._tick_index = tick_index
        self._phase_unwrapped_rad = phase
        self._queue_sequence = queue_sequence
        self._pending_events = pending
        heapq.heapify(self._pending_events)
        self._seen_event_ids = seen
        self._applied_event_ids = applied
        self._applied_events = applied_events
        self._suppressed_event_ids = suppressed
        self._suppressed_events = suppressed_events
        self._oscillator.phase_rad = oscillator_phase
        self._oscillator.frequency_hz = oscillator_frequency

    def restore(self, checkpoint: StreamingMechanicsCheckpoint) -> None:
        """Restore atomically; every validation failure precedes mutation."""

        if not isinstance(checkpoint, StreamingMechanicsCheckpoint):
            raise TypeError("checkpoint must be StreamingMechanicsCheckpoint")
        payload = checkpoint.to_dict()
        checkpoint_config = StreamingMechanicsConfig.from_dict(payload["config"])
        if _canonical_json(checkpoint_config.to_dict()) != _canonical_json(
            self.config.to_dict()
        ):
            raise ValueError("streaming mechanics checkpoint config mismatch")
        # Parse and mutate a fresh candidate.  ``self`` is untouched unless the
        # complete state passes structural and semantic validation.
        candidate = type(self)(self.config)
        candidate._load_validated_state(payload["state"])
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: StreamingMechanicsCheckpoint
    ) -> "StreamingMuscleWingStepper":
        if not isinstance(checkpoint, StreamingMechanicsCheckpoint):
            raise TypeError("checkpoint must be StreamingMechanicsCheckpoint")
        config = StreamingMechanicsConfig.from_dict(checkpoint.to_dict()["config"])
        instance = cls(config)
        instance.restore(checkpoint)
        return instance


__all__ = [
    "STREAMING_MECHANICS_BRIDGE_DT_S",
    "STREAMING_MECHANICS_DT_S",
    "STREAMING_MECHANICS_RUNTIME_VERSION",
    "STREAMING_MECHANICS_SCHEMA_VERSION",
    "STREAMING_MECHANICS_STEPS_PER_BRIDGE",
    "MechanicsInterventionMode",
    "StreamingMechanicsCheckpoint",
    "StreamingMechanicsConfig",
    "StreamingMechanicsFrame",
    "StreamingMuscleIntervention",
    "StreamingMuscleWingStepper",
    "WingPhaseSource",
]
