"""Causal exploratory bridge from NOD1 circuit output to wing motor events.

The bridge is deliberately narrower than a complete VNC model.  It implements
the first software-validation vertical slice::

    individual NOD1 signals -> contralateral DNp26 rate
      -> ipsilateral iv2/i1/iv1/b3 motor-neuron rates
      -> seeded, phase-tagged synthetic steering events

All functional gains in this module are explicitly model parameters.  A
structural synapse count can enable a route and is retained as evidence, but
its magnitude is never used as a physiological gain.  Input atlas identities
are not joined to downstream identities: schema conversion creates new
simulation-scoped motor and muscle entities and records the source atlas only
as provenance.

The public trace schemas cannot represent a spike whose time was synthesized
from a rate.  Consequently :class:`BridgeWingMotorTrace` exposes generated
events separately with ``seeded_synthetic`` semantics, while
``to_schema_trace`` exports the underlying inferred rates.  Only caller-
supplied exact motor events are exported as ``exact_spikes``.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np

from ..schema import (
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
    MotorSignal,
    MotorSignalKind,
    Provenance,
    SideMappingMethod,
    WingMotorTrace,
)


_T = TypeVar("_T")
_TWO_PI = 2.0 * math.pi
_TIME_TOLERANCE_S = 1e-12


class BridgeSignalSemantics(str, Enum):
    """Scientific status of a bridge signal or event."""

    EXACT_SPIKES = "exact_spikes"
    INFERRED_RATE = "inferred_rate"
    SEEDED_SYNTHETIC = "seeded_synthetic"


class BridgeTargetType(str, Enum):
    NOD1 = "nod1"
    DN = "dn"
    MN = "mn"
    MUSCLE = "muscle"


class BridgeInterventionMode(str, Enum):
    SILENCE = "silence"
    ACTIVATE = "activate"
    SCALE = "scale"


def _side(value: AnatomicalSide) -> AnatomicalSide:
    if not isinstance(value, AnatomicalSide):
        try:
            value = AnatomicalSide(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid anatomical side %r" % (value,)) from exc
    if value not in (
        AnatomicalSide.LEFT,
        AnatomicalSide.RIGHT,
        AnatomicalSide.BILATERAL,
    ):
        raise ValueError("bridge sides must be left, right, or bilateral")
    return value


def mirror_side(side: AnatomicalSide) -> AnatomicalSide:
    """Return the anatomical mirror without applying a display convention."""

    side = _side(side)
    if side is AnatomicalSide.LEFT:
        return AnatomicalSide.RIGHT
    if side is AnatomicalSide.RIGHT:
        return AnatomicalSide.LEFT
    return side


def _circuit_bridge_input_side(
    signal: CircuitSignal,
) -> Tuple[AnatomicalSide, str]:
    """Resolve a bridge grouping side without promoting an app label.

    Anatomically established input sides remain eligible.  A source whose
    anatomical side is unknown may instead opt in through an explicit,
    low-confidence ``SIMULATION_CONVENTION`` app-rendering side.  The returned
    side is local to this exploratory bridge and never mutates the source
    FlyWire entity or its anatomical side.
    """

    anatomical_side = signal.neuron.anatomical_side
    if anatomical_side in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT):
        return anatomical_side, "anatomical_side"

    context = signal.neuron.side_context
    if (
        anatomical_side is AnatomicalSide.UNKNOWN
        and context is not None
        and context.anatomical_side is AnatomicalSide.UNKNOWN
        and context.mapping_method is SideMappingMethod.SIMULATION_CONVENTION
        and context.confidence.level is ConfidenceLevel.LOW
        and context.app_rendering_side
        in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT)
    ):
        return context.app_rendering_side, "low_confidence_app_simulation_convention"

    raise ValueError(
        "individual NOD1 bridge signals require an anatomical left/right side "
        "or an explicit low-confidence simulation-convention app side"
    )


@dataclass(frozen=True)
class AvailableValue(Generic[_T]):
    """A value measured at one time and usable no earlier than another."""

    measurement_time_s: float
    availability_time_s: float
    value: _T

    def __post_init__(self) -> None:
        if not math.isfinite(self.measurement_time_s) or self.measurement_time_s < 0.0:
            raise ValueError("measurement_time_s must be finite and non-negative")
        if (
            not math.isfinite(self.availability_time_s)
            or self.availability_time_s + _TIME_TOLERANCE_S < self.measurement_time_s
        ):
            raise ValueError("availability cannot precede measurement")


class AvailabilityQueue(Generic[_T]):
    """Timestamped causal queue with deterministic zero-order hold.

    Items may be inserted in any order before or during a run.  ``advance``
    releases only items whose availability time is at or before the requested
    clock time.  The last released item is held until another becomes
    available.  Insertion order breaks exact timestamp ties deterministically.
    """

    def __init__(self) -> None:
        self._pending: List[Tuple[float, int, AvailableValue[_T]]] = []
        self._sequence = 0
        self._latest: Optional[AvailableValue[_T]] = None
        self._time_s = -math.inf

    def push(self, item: AvailableValue[_T]) -> None:
        if item.availability_time_s + _TIME_TOLERANCE_S < self._time_s:
            raise ValueError("cannot insert an item retroactively after queue advancement")
        heapq.heappush(
            self._pending,
            (item.availability_time_s, self._sequence, item),
        )
        self._sequence += 1

    def advance(self, time_s: float) -> Optional[AvailableValue[_T]]:
        if not math.isfinite(time_s) or time_s + _TIME_TOLERANCE_S < self._time_s:
            raise ValueError("queue time must advance monotonically")
        self._time_s = float(time_s)
        while self._pending and self._pending[0][0] <= time_s + _TIME_TOLERANCE_S:
            _, _, self._latest = heapq.heappop(self._pending)
        return self._latest

    @property
    def latest(self) -> Optional[AvailableValue[_T]]:
        return self._latest


@dataclass(frozen=True)
class BridgeIntervention:
    """Typed, half-open and side-specific bridge intervention.

    ``ACTIVATE`` adds normalized drive at NOD1 and Hz at DN/MN/muscle stages.
    ``SCALE`` is dimensionless.  A bilateral side matches both anatomical
    sides.  Exact motor events support silencing only; modifying their count or
    timing would turn them into inferred events and is rejected.
    """

    target_type: BridgeTargetType
    target_name: str
    side: AnatomicalSide
    mode: BridgeInterventionMode
    start_s: float
    end_s: float
    magnitude: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.target_type, BridgeTargetType):
            object.__setattr__(self, "target_type", BridgeTargetType(self.target_type))
        if not isinstance(self.mode, BridgeInterventionMode):
            object.__setattr__(self, "mode", BridgeInterventionMode(self.mode))
        object.__setattr__(self, "side", _side(self.side))
        if not self.target_name:
            raise ValueError("target_name must be non-empty")
        if (
            not math.isfinite(self.start_s)
            or not math.isfinite(self.end_s)
            or self.start_s < 0.0
            or self.end_s <= self.start_s
        ):
            raise ValueError("intervention interval must satisfy end > start >= 0")
        if not math.isfinite(self.magnitude):
            raise ValueError("intervention magnitude must be finite")
        if self.mode is BridgeInterventionMode.SCALE and self.magnitude < 0.0:
            raise ValueError("scale magnitude must be non-negative")
        if self.mode is BridgeInterventionMode.ACTIVATE and self.magnitude < 0.0:
            raise ValueError("activation magnitude must be non-negative")

    def matches(
        self,
        target_type: BridgeTargetType,
        target_name: str,
        side: AnatomicalSide,
        time_s: float,
    ) -> bool:
        return (
            self.target_type is target_type
            and self.target_name == target_name
            and (self.side is AnatomicalSide.BILATERAL or self.side is side)
            and self.start_s <= time_s < self.end_s
        )


def _apply_interventions(
    value: float,
    interventions: Sequence[BridgeIntervention],
    target_type: BridgeTargetType,
    target_name: str,
    side: AnatomicalSide,
    time_s: float,
) -> float:
    result = float(value)
    for intervention in interventions:
        if not intervention.matches(target_type, target_name, side, time_s):
            continue
        if intervention.mode is BridgeInterventionMode.SILENCE:
            result = 0.0
        elif intervention.mode is BridgeInterventionMode.SCALE:
            result *= intervention.magnitude
        else:
            result += intervention.magnitude
    return max(0.0, result)


@dataclass(frozen=True)
class TimedRateSample:
    measurement_time_s: float
    availability_time_s: float
    rate_hz: float
    source_measurement_time_s: Optional[float]
    semantics: BridgeSignalSemantics = BridgeSignalSemantics.INFERRED_RATE

    def __post_init__(self) -> None:
        AvailableValue(self.measurement_time_s, self.availability_time_s, self.rate_hz)
        if not math.isfinite(self.rate_hz) or self.rate_hz < 0.0:
            raise ValueError("rate_hz must be finite and non-negative")
        if self.source_measurement_time_s is not None:
            if (
                not math.isfinite(self.source_measurement_time_s)
                or self.source_measurement_time_s > self.measurement_time_s + _TIME_TOLERANCE_S
            ):
                raise ValueError("source measurement must not be in the future")
        if self.semantics is not BridgeSignalSemantics.INFERRED_RATE:
            raise ValueError("sampled bridge rates must retain inferred_rate semantics")


@dataclass(frozen=True)
class DescendingChannelTrace:
    dn_type: str
    side: AnatomicalSide
    samples: Tuple[TimedRateSample, ...]
    source_scoped_ids: Tuple[str, ...]
    mapping_method: str
    structural_route_present: bool
    structural_synapse_count: Optional[int]
    functional_gain_hz: float


@dataclass(frozen=True)
class DescendingTrace:
    duration_s: float
    update_dt_s: float
    update_count: int
    channels: Tuple[DescendingChannelTrace, ...]
    source_dataset_identity: str
    status: str = "exploratory"

    def channel(self, dn_type: str, side: AnatomicalSide) -> DescendingChannelTrace:
        side = _side(side)
        matches = tuple(
            channel
            for channel in self.channels
            if channel.dn_type == dn_type and channel.side is side
        )
        if len(matches) != 1:
            raise KeyError("expected one %s:%s channel" % (dn_type, side.value))
        return matches[0]


@dataclass(frozen=True)
class CircuitToDNEncoderConfig:
    update_dt_s: float = 0.0005
    input_availability_delay_s: float = 0.0
    encoder_delay_s: float = 0.003
    resting_voltage_v: float = -0.060
    voltage_scale_v: float = 0.005
    input_rate_scale_hz: float = 100.0
    dn_baseline_rate_hz: float = 0.0
    functional_gain_hz: float = 40.0
    maximum_dn_rate_hz: float = 200.0
    source_cell_type: str = "NOD1"
    target_dn_type: str = "DNp26"
    contralateral: bool = True
    require_bilateral_input: bool = True
    structural_route_present: bool = True
    structural_synapse_count: Optional[int] = None

    def __post_init__(self) -> None:
        positive = (
            ("update_dt_s", self.update_dt_s),
            ("voltage_scale_v", self.voltage_scale_v),
            ("input_rate_scale_hz", self.input_rate_scale_hz),
            ("maximum_dn_rate_hz", self.maximum_dn_rate_hz),
        )
        if any(not math.isfinite(value) or value <= 0.0 for _, value in positive):
            raise ValueError("encoder time/scaling constants must be finite and positive")
        if (
            not math.isfinite(self.input_availability_delay_s)
            or self.input_availability_delay_s < 0.0
            or not math.isfinite(self.encoder_delay_s)
            or self.encoder_delay_s < 0.0
        ):
            raise ValueError("encoder delays must be finite and non-negative")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.dn_baseline_rate_hz, self.functional_gain_hz)
        ):
            raise ValueError("encoder rates/gain must be finite and non-negative")
        if not self.source_cell_type or not self.target_dn_type:
            raise ValueError("encoder source and target names must be non-empty")
        if self.structural_synapse_count is not None and self.structural_synapse_count <= 0:
            raise ValueError("structural_synapse_count must be positive when supplied")


class _SampledSignalRuntime:
    def __init__(
        self,
        signal: CircuitSignal,
        sample_times_s: Sequence[float],
        availability_delay_s: float,
    ) -> None:
        self.signal = signal
        self.queue: AvailabilityQueue[float] = AvailabilityQueue()
        self._spike_index = 0
        spike_availability_times_s = (
            signal.availability_times_s
            if signal.availability_times_s
            else signal.spike_times_s
        )
        self._spike_times_available_s = tuple(
            time_s + availability_delay_s
            for time_s in spike_availability_times_s
        )
        if signal.signal_kind is not CircuitSignalKind.SPIKE_EVENTS:
            sample_availability_times_s = (
                signal.availability_times_s
                if signal.availability_times_s
                else tuple(sample_times_s)
            )
            for time_s, availability_time_s, value in zip(
                sample_times_s,
                sample_availability_times_s,
                signal.values,
            ):
                self.queue.push(
                    AvailableValue(
                        time_s,
                        availability_time_s + availability_delay_s,
                        float(value),
                    )
                )

    def drive(
        self,
        time_s: float,
        previous_time_s: float,
        config: CircuitToDNEncoderConfig,
    ) -> Tuple[float, Optional[float]]:
        if self.signal.signal_kind is CircuitSignalKind.SPIKE_EVENTS:
            count = 0
            latest_measurement: Optional[float] = None
            while (
                self._spike_index < len(self._spike_times_available_s)
                and self._spike_times_available_s[self._spike_index]
                <= time_s + _TIME_TOLERANCE_S
            ):
                available = self._spike_times_available_s[self._spike_index]
                if available > previous_time_s + _TIME_TOLERANCE_S or time_s == 0.0:
                    count += 1
                    latest_measurement = self.signal.spike_times_s[self._spike_index]
                self._spike_index += 1
            inferred_rate = count / config.update_dt_s
            return inferred_rate / config.input_rate_scale_hz, latest_measurement

        latest = self.queue.advance(time_s)
        if latest is None:
            return 0.0, None
        if self.signal.signal_kind is CircuitSignalKind.VOLTAGE:
            drive = (latest.value - config.resting_voltage_v) / config.voltage_scale_v
        elif self.signal.signal_kind is CircuitSignalKind.FIRING_RATE:
            drive = latest.value / config.input_rate_scale_hz
        else:
            drive = latest.value
        return max(0.0, drive), latest.measurement_time_s


class CircuitToDNEncoder:
    """Causally encode individual NOD1 signals into bilateral DNp26 rates."""

    def __init__(self, config: Optional[CircuitToDNEncoderConfig] = None) -> None:
        self.config = config if config is not None else CircuitToDNEncoderConfig()

    def encode(
        self,
        trace: CircuitOutputTrace,
        interventions: Sequence[BridgeIntervention] = (),
    ) -> DescendingTrace:
        config = self.config
        if not trace.sample_times_s or abs(trace.sample_times_s[0]) > _TIME_TOLERANCE_S:
            raise ValueError("bridge circuit traces must begin at t=0")
        duration_s = float(
            trace.sample_times_s[-1]
            if trace.sample_interval_end_s is None
            else trace.sample_interval_end_s
        )
        if duration_s <= 0.0:
            raise ValueError("bridge circuit traces must span a positive duration")
        update_count = int(round(duration_s / config.update_dt_s))
        if abs(update_count * config.update_dt_s - duration_s) > 1e-9 * duration_s:
            raise ValueError("trace duration must be an integer multiple of encoder update_dt_s")

        selected: Dict[AnatomicalSide, List[CircuitSignal]] = {
            AnatomicalSide.LEFT: [],
            AnatomicalSide.RIGHT: [],
        }
        selected_side_semantics: Dict[AnatomicalSide, List[str]] = {
            AnatomicalSide.LEFT: [],
            AnatomicalSide.RIGHT: [],
        }
        for signal in trace.signals:
            if signal.neuron.cell_type != config.source_cell_type:
                continue
            side, side_semantics = _circuit_bridge_input_side(signal)
            selected[side].append(signal)
            selected_side_semantics[side].append(side_semantics)
        if not any(selected.values()):
            raise ValueError("trace contains no %s signals" % config.source_cell_type)
        if config.require_bilateral_input and any(not selected[side] for side in selected):
            raise ValueError("bridge requires NOD1 input on both bridge input sides")

        allowed_encoder_targets = {
            (BridgeTargetType.NOD1, config.source_cell_type),
            (BridgeTargetType.DN, config.target_dn_type),
        }
        for intervention in interventions:
            if intervention.target_type in (BridgeTargetType.NOD1, BridgeTargetType.DN):
                if (intervention.target_type, intervention.target_name) not in allowed_encoder_targets:
                    raise ValueError("unknown encoder intervention target %s" % intervention.target_name)

        runtimes: Dict[AnatomicalSide, Tuple[_SampledSignalRuntime, ...]] = {
            side: tuple(
                _SampledSignalRuntime(
                    signal,
                    trace.sample_times_s,
                    config.input_availability_delay_s,
                )
                for signal in signals
            )
            for side, signals in selected.items()
        }
        output: Dict[AnatomicalSide, List[TimedRateSample]] = {
            AnatomicalSide.LEFT: [],
            AnatomicalSide.RIGHT: [],
        }
        source_ids: Dict[AnatomicalSide, Tuple[str, ...]] = {
            side: tuple(signal.neuron.scoped_id for signal in selected[mirror_side(side)])
            if config.contralateral
            else tuple(signal.neuron.scoped_id for signal in selected[side])
            for side in output
        }

        for index in range(update_count):
            time_s = index * config.update_dt_s
            previous_time_s = (index - 1) * config.update_dt_s
            side_drives: Dict[AnatomicalSide, Tuple[float, Optional[float]]] = {}
            for source_side, source_runtimes in runtimes.items():
                values: List[float] = []
                measurement_times: List[float] = []
                for runtime in source_runtimes:
                    drive, measurement_time = runtime.drive(time_s, previous_time_s, config)
                    values.append(drive)
                    if measurement_time is not None:
                        measurement_times.append(measurement_time)
                drive = sum(values) / len(values) if values else 0.0
                drive = _apply_interventions(
                    drive,
                    interventions,
                    BridgeTargetType.NOD1,
                    config.source_cell_type,
                    source_side,
                    time_s,
                )
                side_drives[source_side] = (
                    drive,
                    max(measurement_times) if measurement_times else None,
                )

            for target_side in output:
                source_side = mirror_side(target_side) if config.contralateral else target_side
                drive, source_measurement = side_drives[source_side]
                if config.structural_route_present:
                    rate_hz = config.dn_baseline_rate_hz + config.functional_gain_hz * drive
                else:
                    rate_hz = 0.0
                rate_hz = min(rate_hz, config.maximum_dn_rate_hz)
                rate_hz = _apply_interventions(
                    rate_hz,
                    interventions,
                    BridgeTargetType.DN,
                    config.target_dn_type,
                    target_side,
                    time_s,
                )
                rate_hz = min(rate_hz, config.maximum_dn_rate_hz)
                output[target_side].append(
                    TimedRateSample(
                        measurement_time_s=time_s,
                        availability_time_s=time_s + config.encoder_delay_s,
                        rate_hz=rate_hz,
                        source_measurement_time_s=source_measurement,
                    )
                )

        channels = []
        for side in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT):
            source_side = mirror_side(side) if config.contralateral else side
            simulation_convention_used = any(
                semantics == "low_confidence_app_simulation_convention"
                for semantics in selected_side_semantics[source_side]
            )
            side_basis = (
                "input side derived from an explicit low-confidence app-rendering "
                "simulation convention; it is not anatomical laterality"
                if simulation_convention_used
                else "input side is anatomically labelled by the source trace"
            )
            channels.append(
                DescendingChannelTrace(
                    dn_type=config.target_dn_type,
                    side=side,
                    samples=tuple(output[side]),
                    source_scoped_ids=source_ids[side],
                    mapping_method=(
                        "contralateral cell-type/side model; %s; source atlas identity "
                        "retained as provenance; no cross-atlas identifier join"
                        % side_basis
                        if config.contralateral
                        else "ipsilateral cell-type/side model; %s; no cross-atlas "
                        "identifier join" % side_basis
                    ),
                    structural_route_present=config.structural_route_present,
                    structural_synapse_count=config.structural_synapse_count,
                    functional_gain_hz=config.functional_gain_hz,
                )
            )
        return DescendingTrace(
            duration_s=duration_s,
            update_dt_s=config.update_dt_s,
            update_count=update_count,
            channels=tuple(channels),
            source_dataset_identity=trace.dataset.identity_space,
        )


@dataclass(frozen=True)
class MotorPathway:
    motor_neuron: str
    muscle: str
    functional_rate_gain: float
    preferred_phase_rad: float
    structural_supported: bool = True
    structural_synapse_count: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.motor_neuron or not self.muscle:
            raise ValueError("motor_neuron and muscle must be non-empty")
        if not math.isfinite(self.functional_rate_gain) or self.functional_rate_gain < 0.0:
            raise ValueError("functional_rate_gain must be finite and non-negative")
        if not math.isfinite(self.preferred_phase_rad):
            raise ValueError("preferred_phase_rad must be finite")
        object.__setattr__(self, "preferred_phase_rad", self.preferred_phase_rad % _TWO_PI)
        if self.structural_synapse_count is not None and self.structural_synapse_count <= 0:
            raise ValueError("structural_synapse_count must be positive when supplied")


def default_steering_pathways() -> Tuple[MotorPathway, ...]:
    """Return the evidence-supported exploratory DNp26 steering mask.

    Counts are retained from the current audit solely as structural evidence.
    Functional gains and phases are explicit provisional parameters and are
    intentionally not functions of those counts.
    """

    return (
        MotorPathway("MN-iv2", "iv2", 0.60, 0.20 * _TWO_PI, True, 53),
        MotorPathway("MN-i1", "i1", 0.50, 0.35 * _TWO_PI, True, 29),
        MotorPathway("MN-iv1", "iv1", 0.40, 0.55 * _TWO_PI, True, 21),
        MotorPathway("MN-b3", "b3", 0.35, 0.75 * _TWO_PI, True, 14),
    )


@dataclass(frozen=True)
class ExactMotorEvent:
    motor_neuron: str
    muscle: str
    side: AnatomicalSide
    event_time_s: float
    availability_time_s: Optional[float] = None
    wingbeat_phase_rad: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _side(self.side))
        if self.side is AnatomicalSide.BILATERAL:
            raise ValueError("exact motor events must have an individual side")
        if not self.motor_neuron or not self.muscle:
            raise ValueError("exact event motor neuron and muscle must be non-empty")
        available = self.event_time_s if self.availability_time_s is None else self.availability_time_s
        AvailableValue(self.event_time_s, available, 1)
        object.__setattr__(self, "availability_time_s", float(available))
        if self.wingbeat_phase_rad is not None:
            if not math.isfinite(self.wingbeat_phase_rad):
                raise ValueError("wingbeat phase must be finite")
            object.__setattr__(self, "wingbeat_phase_rad", self.wingbeat_phase_rad % _TWO_PI)


@dataclass(frozen=True)
class MotorEvent:
    event_time_s: float
    availability_time_s: float
    wingbeat_phase_rad: float
    semantics: BridgeSignalSemantics
    generator_seed: Optional[int] = None

    def __post_init__(self) -> None:
        AvailableValue(self.event_time_s, self.availability_time_s, 1)
        if not math.isfinite(self.wingbeat_phase_rad):
            raise ValueError("wingbeat phase must be finite")
        object.__setattr__(self, "wingbeat_phase_rad", self.wingbeat_phase_rad % _TWO_PI)
        if self.semantics is BridgeSignalSemantics.EXACT_SPIKES:
            if self.generator_seed is not None:
                raise ValueError("exact events cannot carry a generator seed")
        elif self.semantics is BridgeSignalSemantics.SEEDED_SYNTHETIC:
            if self.generator_seed is None:
                raise ValueError("synthetic events require a generator seed")
        else:
            raise ValueError("motor events must be exact or seeded-synthetic")


@dataclass(frozen=True)
class MotorChannelTrace:
    motor_neuron: str
    muscle: str
    side: AnatomicalSide
    rate_samples: Tuple[TimedRateSample, ...]
    events: Tuple[MotorEvent, ...]
    preferred_phase_rad: float
    structural_synapse_count: Optional[int]
    functional_rate_gain: float
    signal_semantics: BridgeSignalSemantics

    def __post_init__(self) -> None:
        if self.signal_semantics is BridgeSignalSemantics.EXACT_SPIKES:
            if self.rate_samples:
                raise ValueError("exact motor channels cannot also contain inferred rates")
            if any(
                event.semantics is not BridgeSignalSemantics.EXACT_SPIKES
                for event in self.events
            ):
                raise ValueError("exact motor channels can contain only exact events")
        elif self.signal_semantics is BridgeSignalSemantics.INFERRED_RATE:
            if not self.rate_samples:
                raise ValueError("inferred motor channels require rate samples")
            if any(
                event.semantics is not BridgeSignalSemantics.SEEDED_SYNTHETIC
                for event in self.events
            ):
                raise ValueError("rate-derived motor events must remain seeded-synthetic")
        else:
            raise ValueError("motor channel signal semantics must be exact_spikes or inferred_rate")

    @property
    def event_semantics(self) -> Optional[BridgeSignalSemantics]:
        semantics = {event.semantics for event in self.events}
        if not semantics:
            return (
                BridgeSignalSemantics.EXACT_SPIKES
                if self.signal_semantics is BridgeSignalSemantics.EXACT_SPIKES
                else None
            )
        if len(semantics) != 1:
            raise RuntimeError("a motor channel cannot mix exact and synthetic events")
        return next(iter(semantics))


@dataclass(frozen=True)
class BridgeWingMotorTrace:
    duration_s: float
    update_dt_s: float
    update_count: int
    wingbeat_frequency_hz: float
    channels: Tuple[MotorChannelTrace, ...]
    source_dataset_identity: str
    seed: int
    status: str = "exploratory"

    def channel(
        self,
        muscle: str,
        side: AnatomicalSide,
    ) -> MotorChannelTrace:
        side = _side(side)
        matches = tuple(
            channel
            for channel in self.channels
            if channel.muscle == muscle and channel.side is side
        )
        if len(matches) != 1:
            raise KeyError("expected one %s:%s motor channel" % (muscle, side.value))
        return matches[0]

    def to_schema_trace(self) -> WingMotorTrace:
        """Export exact events or inferred rates without relabelling synthesis.

        Seeded synthetic events are deliberately omitted from the v1
        ``WingMotorTrace`` because its spike representation means exact spikes.
        Their causal event stream remains available on ``channels[].events``.
        """

        dataset = DatasetRef(
            namespace=DatasetNamespace.SIMULATION,
            release="exploratory-nod1-dnp26-bridge-v1",
            source_uri="urn:fly-sensor2behavior:flight:bridge:v1",
        )
        provenance = Provenance(
            source_uri="urn:fly-sensor2behavior:flight:bridge:v1",
            method="causal exploratory NOD1-to-DNp26-to-wing-MN bridge",
            dataset_identity=dataset.identity_space,
            notes=(
                "Source circuit atlas %s is provenance only; downstream entities are "
                "new simulation identities, not a cross-atlas identifier join."
                % self.source_dataset_identity
            ),
        )
        confidence = Confidence(
            tier=EvidenceTier.MODEL_INFERENCE,
            level=ConfidenceLevel.LOW,
            score=0.2,
            basis="exploratory functional gains and phase model; not empirically calibrated",
        )
        signals: List[MotorSignal] = []
        for channel in self.channels:
            motor = EntityRef(
                dataset,
                "%s-%s" % (channel.motor_neuron, channel.side.value),
                EntityKind.MOTOR_NEURON,
                cell_type=channel.motor_neuron,
                anatomical_side=channel.side,
            )
            muscle = EntityRef(
                dataset,
                "%s-%s" % (channel.muscle, channel.side.value),
                EntityKind.MUSCLE,
                cell_type=channel.muscle,
                anatomical_side=channel.side,
            )
            if channel.signal_semantics is BridgeSignalSemantics.EXACT_SPIKES:
                signals.append(
                    MotorSignal(
                        motor_neuron=motor,
                        muscle=muscle,
                        side=channel.side,
                        signal_kind=MotorSignalKind.EXACT_SPIKES,
                        spike_times_s=tuple(event.event_time_s for event in channel.events),
                        availability_times_s=tuple(
                            event.availability_time_s for event in channel.events
                        ),
                        wingbeat_phase_rad=tuple(
                            event.wingbeat_phase_rad for event in channel.events
                        ),
                        provenance=provenance,
                        confidence=confidence,
                    )
                )
            else:
                signals.append(
                    MotorSignal(
                        motor_neuron=motor,
                        muscle=muscle,
                        side=channel.side,
                        signal_kind=MotorSignalKind.INFERRED_RATE,
                        sample_times_s=tuple(
                            sample.measurement_time_s for sample in channel.rate_samples
                        ),
                        availability_times_s=tuple(
                            sample.availability_time_s
                            for sample in channel.rate_samples
                        ),
                        rate_hz=tuple(sample.rate_hz for sample in channel.rate_samples),
                        provenance=provenance,
                        confidence=confidence,
                    )
                )
        return WingMotorTrace(
            dataset=dataset,
            duration_s=self.duration_s,
            signals=tuple(signals),
            provenance=provenance,
            confidence=confidence,
        )

    def to_flight_motor_commands(self) -> Tuple[object, ...]:
        """Adapt bridge events to the reduced/FlyBody flight runtime.

        Synthetic events remain a distinct command kind and use availability
        time as their actuation time.  This prevents an event from affecting a
        muscle before the declared VNC/NMJ delay.  The richer bridge trace
        remains the source of record for measurement times and rate samples.
        """

        from .types import (
            MotorCommand as FlightMotorCommand,
            MotorSignalKind as FlightMotorSignalKind,
            MuscleClass as FlightMuscleClass,
            Side as FlightSide,
        )

        side_map = {
            AnatomicalSide.LEFT: FlightSide.LEFT,
            AnatomicalSide.RIGHT: FlightSide.RIGHT,
        }
        commands: List[object] = []
        for channel in self.channels:
            if channel.side not in side_map:
                raise ValueError("flight motor commands require individual left/right sides")
            if channel.signal_semantics is BridgeSignalSemantics.EXACT_SPIKES:
                signal_kind = FlightMotorSignalKind.EXACT_SPIKES
                seed = None
            else:
                signal_kind = FlightMotorSignalKind.SEEDED_SYNTHETIC_SPIKES
                seed = _channel_seed(
                    self.seed,
                    channel.motor_neuron,
                    channel.muscle,
                    channel.side,
                )
            actionable_events = tuple(
                event
                for event in channel.events
                if event.availability_time_s < self.duration_s
            )
            commands.append(
                FlightMotorCommand(
                    neuron_id="%s-%s" % (channel.motor_neuron, channel.side.value),
                    muscle=channel.muscle,
                    side=side_map[channel.side],
                    muscle_class=FlightMuscleClass.STEERING,
                    signal_kind=signal_kind,
                    spike_times_s=tuple(
                        event.availability_time_s for event in actionable_events
                    ),
                    measurement_spike_times_s=tuple(
                        event.event_time_s for event in actionable_events
                    ),
                    preferred_phase_rad=channel.preferred_phase_rad,
                    provenance=(
                        "causal exploratory NOD1-DNp26-VNC bridge; source events "
                        "retain measurement/availability times in BridgeWingMotorTrace"
                    ),
                    generator_seed=seed,
                )
            )
        return tuple(commands)


@dataclass(frozen=True)
class VNCMotorNetworkConfig:
    update_dt_s: float = 0.0005
    vnc_delay_s: float = 0.002
    wingbeat_frequency_hz: float = 200.0
    maximum_motor_rate_hz: float = 200.0
    input_dn_type: str = "DNp26"
    pathways: Tuple[MotorPathway, ...] = field(default_factory=default_steering_pathways)

    def __post_init__(self) -> None:
        for value in (
            self.update_dt_s,
            self.wingbeat_frequency_hz,
            self.maximum_motor_rate_hz,
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("VNC timing/rate parameters must be finite and positive")
        if not math.isfinite(self.vnc_delay_s) or self.vnc_delay_s < 0.0:
            raise ValueError("vnc_delay_s must be finite and non-negative")
        if not self.input_dn_type or not self.pathways:
            raise ValueError("VNC input type and pathways must be non-empty")
        keys = [(path.motor_neuron, path.muscle) for path in self.pathways]
        if len(set(keys)) != len(keys):
            raise ValueError("VNC pathway motor-neuron/muscle pairs must be unique")


def _channel_seed(seed: int, motor_neuron: str, muscle: str, side: AnatomicalSide) -> int:
    payload = "%d|%s|%s|%s" % (seed, motor_neuron, muscle, side.value)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


class VNCMotorNetwork:
    """Map inferred DNp26 rates to individual phase-coded steering MNs."""

    def __init__(self, config: Optional[VNCMotorNetworkConfig] = None) -> None:
        self.config = config if config is not None else VNCMotorNetworkConfig()

    def run(
        self,
        descending: DescendingTrace,
        *,
        seed: int = 0,
        interventions: Sequence[BridgeIntervention] = (),
        exact_motor_events: Sequence[ExactMotorEvent] = (),
    ) -> BridgeWingMotorTrace:
        config = self.config
        duration_s = descending.duration_s
        update_count = int(round(duration_s / config.update_dt_s))
        if abs(update_count * config.update_dt_s - duration_s) > 1e-9 * duration_s:
            raise ValueError("descending duration must be an integer multiple of VNC update_dt_s")

        known_mns = {path.motor_neuron for path in config.pathways}
        known_muscles = {path.muscle for path in config.pathways}
        for intervention in interventions:
            if intervention.target_type is BridgeTargetType.DN:
                if intervention.target_name != config.input_dn_type:
                    raise ValueError("unknown VNC DN intervention target %s" % intervention.target_name)
            elif intervention.target_type is BridgeTargetType.MN:
                if intervention.target_name not in known_mns:
                    raise ValueError("unknown motor-neuron intervention target %s" % intervention.target_name)
            elif intervention.target_type is BridgeTargetType.MUSCLE:
                if intervention.target_name not in known_muscles:
                    raise ValueError("unknown muscle intervention target %s" % intervention.target_name)

        exact_by_key: Dict[Tuple[str, str, AnatomicalSide], List[ExactMotorEvent]] = {}
        pathway_keys = {
            (path.motor_neuron, path.muscle, side)
            for path in config.pathways
            for side in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT)
        }
        for event in exact_motor_events:
            key = (event.motor_neuron, event.muscle, event.side)
            if key not in pathway_keys:
                raise ValueError("exact event targets an unknown motor pathway")
            if event.event_time_s >= duration_s:
                raise ValueError("exact motor event lies outside the episode")
            exact_by_key.setdefault(key, []).append(event)
        for events in exact_by_key.values():
            events.sort(key=lambda event: event.event_time_s)

        dn_queues: Dict[AnatomicalSide, AvailabilityQueue[TimedRateSample]] = {}
        for side in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT):
            queue: AvailabilityQueue[TimedRateSample] = AvailabilityQueue()
            for sample in descending.channel(config.input_dn_type, side).samples:
                queue.push(
                    AvailableValue(
                        sample.measurement_time_s,
                        sample.availability_time_s,
                        sample,
                    )
                )
            dn_queues[side] = queue

        rates: Dict[Tuple[str, str, AnatomicalSide], List[TimedRateSample]] = {
            key: [] for key in pathway_keys
        }
        for index in range(update_count):
            time_s = index * config.update_dt_s
            held_dn = {side: dn_queues[side].advance(time_s) for side in dn_queues}
            for path in config.pathways:
                for side in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT):
                    key = (path.motor_neuron, path.muscle, side)
                    if key in exact_by_key:
                        continue
                    available = held_dn[side]
                    dn_rate_hz = 0.0 if available is None else available.value.rate_hz
                    dn_rate_hz = _apply_interventions(
                        dn_rate_hz,
                        interventions,
                        BridgeTargetType.DN,
                        config.input_dn_type,
                        side,
                        time_s,
                    )
                    # The structural count is deliberately absent from this
                    # equation. Only its Boolean support can mask the route.
                    motor_rate_hz = (
                        path.functional_rate_gain * dn_rate_hz
                        if path.structural_supported
                        else 0.0
                    )
                    motor_rate_hz = _apply_interventions(
                        motor_rate_hz,
                        interventions,
                        BridgeTargetType.MN,
                        path.motor_neuron,
                        side,
                        time_s,
                    )
                    motor_rate_hz = _apply_interventions(
                        motor_rate_hz,
                        interventions,
                        BridgeTargetType.MUSCLE,
                        path.muscle,
                        side,
                        time_s,
                    )
                    motor_rate_hz = min(motor_rate_hz, config.maximum_motor_rate_hz)
                    rates[key].append(
                        TimedRateSample(
                            measurement_time_s=time_s,
                            availability_time_s=time_s + config.vnc_delay_s,
                            rate_hz=motor_rate_hz,
                            source_measurement_time_s=(
                                None
                                if available is None
                                else available.value.measurement_time_s
                            ),
                        )
                    )

        channels: List[MotorChannelTrace] = []
        for path in config.pathways:
            for side in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT):
                key = (path.motor_neuron, path.muscle, side)
                exact = exact_by_key.get(key)
                if exact is not None:
                    for intervention in interventions:
                        if (
                            intervention.target_type in (BridgeTargetType.MN, BridgeTargetType.MUSCLE)
                            and intervention.target_name
                            in (path.motor_neuron, path.muscle)
                            and (
                                intervention.side is AnatomicalSide.BILATERAL
                                or intervention.side is side
                            )
                            and intervention.mode is not BridgeInterventionMode.SILENCE
                        ):
                            raise ValueError(
                                "exact motor events support silencing only; modifying them would infer timing"
                            )
                    events = tuple(
                        MotorEvent(
                            event_time_s=event.event_time_s,
                            availability_time_s=float(event.availability_time_s),
                            wingbeat_phase_rad=(
                                event.wingbeat_phase_rad
                                if event.wingbeat_phase_rad is not None
                                else (_TWO_PI * config.wingbeat_frequency_hz * event.event_time_s)
                                % _TWO_PI
                            ),
                            semantics=BridgeSignalSemantics.EXACT_SPIKES,
                        )
                        for event in exact
                        if not any(
                            intervention.mode is BridgeInterventionMode.SILENCE
                            and (
                                intervention.matches(
                                    BridgeTargetType.MN,
                                    path.motor_neuron,
                                    side,
                                    event.event_time_s,
                                )
                                or intervention.matches(
                                    BridgeTargetType.MUSCLE,
                                    path.muscle,
                                    side,
                                    event.event_time_s,
                                )
                            )
                            for intervention in interventions
                        )
                    )
                    rate_samples: Tuple[TimedRateSample, ...] = ()
                    signal_semantics = BridgeSignalSemantics.EXACT_SPIKES
                else:
                    rate_samples = tuple(rates[key])
                    events = self._generate_synthetic_events(
                        path,
                        side,
                        rate_samples,
                        duration_s,
                        seed,
                    )
                    signal_semantics = BridgeSignalSemantics.INFERRED_RATE
                channels.append(
                    MotorChannelTrace(
                        motor_neuron=path.motor_neuron,
                        muscle=path.muscle,
                        side=side,
                        rate_samples=rate_samples,
                        events=events,
                        preferred_phase_rad=path.preferred_phase_rad,
                        structural_synapse_count=path.structural_synapse_count,
                        functional_rate_gain=path.functional_rate_gain,
                        signal_semantics=signal_semantics,
                    )
                )

        return BridgeWingMotorTrace(
            duration_s=duration_s,
            update_dt_s=config.update_dt_s,
            update_count=update_count,
            wingbeat_frequency_hz=config.wingbeat_frequency_hz,
            channels=tuple(channels),
            source_dataset_identity=descending.source_dataset_identity,
            seed=int(seed),
        )

    def _generate_synthetic_events(
        self,
        path: MotorPathway,
        side: AnatomicalSide,
        samples: Sequence[TimedRateSample],
        duration_s: float,
        seed: int,
    ) -> Tuple[MotorEvent, ...]:
        channel_seed = _channel_seed(seed, path.motor_neuron, path.muscle, side)
        generator = np.random.default_rng(channel_seed)
        queue: AvailabilityQueue[TimedRateSample] = AvailabilityQueue()
        for sample in samples:
            queue.push(
                AvailableValue(
                    sample.measurement_time_s,
                    sample.availability_time_s,
                    sample,
                )
            )
        period_s = 1.0 / self.config.wingbeat_frequency_hz
        first_time_s = path.preferred_phase_rad / _TWO_PI * period_s
        events: List[MotorEvent] = []
        event_time_s = first_time_s
        while event_time_s < duration_s - _TIME_TOLERANCE_S:
            available = queue.advance(event_time_s)
            rate_hz = 0.0 if available is None else available.value.rate_hz
            probability = min(1.0, rate_hz / self.config.wingbeat_frequency_hz)
            if probability > 0.0 and generator.random() < probability:
                events.append(
                    MotorEvent(
                        event_time_s=event_time_s,
                        availability_time_s=event_time_s,
                        wingbeat_phase_rad=path.preferred_phase_rad,
                        semantics=BridgeSignalSemantics.SEEDED_SYNTHETIC,
                        generator_seed=channel_seed,
                    )
                )
            event_time_s += period_s
        return tuple(events)


@dataclass(frozen=True)
class CausalBridgeResult:
    descending: DescendingTrace
    wing_motor: BridgeWingMotorTrace


class CausalNeuralBridge:
    """Safe end-to-end facade that applies every intervention exactly once.

    NOD1 and DN interventions belong to the encoder boundary; MN and muscle
    interventions belong to the VNC boundary.  The lower-level classes remain
    usable independently, but callers with a shared intervention list should
    use this facade to avoid applying a DN scaling operation twice.
    """

    def __init__(
        self,
        encoder: Optional[CircuitToDNEncoder] = None,
        motor_network: Optional[VNCMotorNetwork] = None,
    ) -> None:
        self.encoder = encoder if encoder is not None else CircuitToDNEncoder()
        self.motor_network = (
            motor_network if motor_network is not None else VNCMotorNetwork()
        )

    def run(
        self,
        trace: CircuitOutputTrace,
        *,
        seed: int = 0,
        interventions: Sequence[BridgeIntervention] = (),
        exact_motor_events: Sequence[ExactMotorEvent] = (),
    ) -> CausalBridgeResult:
        encoder_interventions = tuple(
            intervention
            for intervention in interventions
            if intervention.target_type in (BridgeTargetType.NOD1, BridgeTargetType.DN)
        )
        motor_interventions = tuple(
            intervention
            for intervention in interventions
            if intervention.target_type in (BridgeTargetType.MN, BridgeTargetType.MUSCLE)
        )
        descending = self.encoder.encode(trace, encoder_interventions)
        wing_motor = self.motor_network.run(
            descending,
            seed=seed,
            interventions=motor_interventions,
            exact_motor_events=exact_motor_events,
        )
        return CausalBridgeResult(descending=descending, wing_motor=wing_motor)


__all__ = [
    "AvailabilityQueue",
    "AvailableValue",
    "BridgeIntervention",
    "BridgeInterventionMode",
    "BridgeSignalSemantics",
    "BridgeTargetType",
    "BridgeWingMotorTrace",
    "CausalBridgeResult",
    "CausalNeuralBridge",
    "CircuitToDNEncoder",
    "CircuitToDNEncoderConfig",
    "DescendingChannelTrace",
    "DescendingTrace",
    "ExactMotorEvent",
    "MotorChannelTrace",
    "MotorEvent",
    "MotorPathway",
    "TimedRateSample",
    "VNCMotorNetwork",
    "VNCMotorNetworkConfig",
    "default_steering_pathways",
    "mirror_side",
]
