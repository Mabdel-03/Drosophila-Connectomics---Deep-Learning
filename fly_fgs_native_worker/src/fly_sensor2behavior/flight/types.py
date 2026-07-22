"""Typed values shared by the exploratory flight dynamics engine.

The types in this module deliberately contain no connectome identifiers.  Atlas
crosswalks and their evidence belong at an input boundary; the flight engine
only receives motor commands with explicit signal provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Tuple

import numpy as np


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"


class MuscleClass(str, Enum):
    ASYNCHRONOUS_POWER = "asynchronous_power"
    STEERING = "steering"
    TENSION = "tension"


class MotorSignalKind(str, Enum):
    """How sub-wingbeat motor events were obtained."""

    EXACT_SPIKES = "exact_spikes"
    INFERRED_RATE = "inferred_rate"
    SEEDED_SYNTHETIC_SPIKES = "seeded_synthetic_spikes"


class ValidationStatus(str, Enum):
    EXPLORATORY = "exploratory"
    CALIBRATED = "calibrated"
    VALIDATED = "validated"


class PerturbationTarget(str, Enum):
    PATHWAY = "pathway"
    DN = "dn"
    MN = "mn"
    MUSCLE = "muscle"
    ENVIRONMENT = "environment"


class PerturbationMode(str, Enum):
    SILENCE = "silence"
    ACTIVATE = "activate"
    SCALE = "scale"
    GUST = "gust"


@dataclass(frozen=True)
class MotorCommand:
    """One motor-neuron population's command to one named muscle.

    ``rate_hz`` describes the firing rate of each motor unit, not wingbeat
    frequency.  For asynchronous power muscles, multiple low-rate units are
    deterministically splayed in phase.  For steering muscles, inferred events
    are emitted at ``preferred_phase_rad`` and therefore remain visibly
    inferred rather than being presented as measured spike timing.
    """

    neuron_id: str
    muscle: str
    side: Side
    muscle_class: MuscleClass
    signal_kind: MotorSignalKind
    rate_hz: float = 0.0
    # Runtime events occur at ``spike_times_s``. When an upstream exact event
    # crossed a causal latency boundary, its original measurement timestamp is
    # retained separately instead of relabelling its availability as the spike.
    spike_times_s: Tuple[float, ...] = ()
    measurement_spike_times_s: Tuple[float, ...] = ()
    motor_units: int = 1
    preferred_phase_rad: float = 0.0
    provenance: str = "unspecified"
    generator_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.side, Side):
            raise TypeError("side must be a Side enum value")
        if not isinstance(self.muscle_class, MuscleClass):
            raise TypeError("muscle_class must be a MuscleClass enum value")
        if not isinstance(self.signal_kind, MotorSignalKind):
            raise TypeError("signal_kind must be a MotorSignalKind enum value")
        if not self.neuron_id or not self.muscle:
            raise ValueError("neuron_id and muscle must be non-empty")
        if self.rate_hz < 0.0 or not np.isfinite(self.rate_hz):
            raise ValueError("rate_hz must be finite and non-negative")
        if self.motor_units < 1:
            raise ValueError("motor_units must be at least one")
        times = tuple(float(t) for t in self.spike_times_s)
        measurement_times = tuple(float(t) for t in self.measurement_spike_times_s)
        if any((not np.isfinite(t)) or t < 0.0 for t in times):
            raise ValueError("spike times must be finite and non-negative")
        if any((not np.isfinite(t)) or t < 0.0 for t in measurement_times):
            raise ValueError("measurement spike times must be finite and non-negative")
        if tuple(sorted(times)) != times:
            raise ValueError("spike times must be sorted")
        if tuple(sorted(measurement_times)) != measurement_times:
            raise ValueError("measurement spike times must be sorted")
        if self.signal_kind in (
            MotorSignalKind.EXACT_SPIKES,
            MotorSignalKind.SEEDED_SYNTHETIC_SPIKES,
        ) and self.rate_hz != 0.0:
            raise ValueError("spike commands cannot also specify rate_hz")
        if self.signal_kind is MotorSignalKind.INFERRED_RATE and times:
            raise ValueError("inferred-rate commands cannot contain exact spikes")
        if self.signal_kind is MotorSignalKind.INFERRED_RATE and measurement_times:
            raise ValueError("inferred-rate commands cannot contain spike measurements")
        if measurement_times:
            if len(measurement_times) != len(times):
                raise ValueError(
                    "measurement spike times must match runtime spike times"
                )
            if any(
                measured > available
                for measured, available in zip(measurement_times, times)
            ):
                raise ValueError(
                    "a spike measurement cannot occur after its runtime availability"
                )
        if self.signal_kind is MotorSignalKind.EXACT_SPIKES and self.generator_seed is not None:
            raise ValueError("exact spike commands cannot carry a generator seed")
        if self.signal_kind is MotorSignalKind.SEEDED_SYNTHETIC_SPIKES:
            if self.generator_seed is None or self.generator_seed < 0:
                raise ValueError("seeded synthetic spike commands require a non-negative seed")
        if self.signal_kind is MotorSignalKind.INFERRED_RATE and self.generator_seed is not None:
            raise ValueError("inferred-rate commands cannot carry a spike generator seed")
        object.__setattr__(self, "spike_times_s", times)
        object.__setattr__(
            self,
            "measurement_spike_times_s",
            measurement_times if measurement_times else times,
        )


@dataclass(frozen=True)
class Perturbation:
    """A bounded, deterministic perturbation applied during an episode.

    ``SCALE`` magnitude is dimensionless. ``ACTIVATE`` is normalized drive for
    DNs/muscles and an added rate in Hz for inferred-rate MNs. A ``GUST`` vector
    is wind velocity in m/s and is multiplied by magnitude. Exact-spike MN
    stimulation should be represented by an exact ``MotorCommand`` rather than
    silently manufacturing event times here.
    """

    target_type: PerturbationTarget
    target: str
    mode: PerturbationMode
    start_s: float
    end_s: float
    magnitude: float = 1.0
    vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.target_type, PerturbationTarget):
            raise TypeError("target_type must be a PerturbationTarget enum value")
        if not isinstance(self.mode, PerturbationMode):
            raise TypeError("mode must be a PerturbationMode enum value")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("perturbation target must be a non-empty string")
        if self.start_s < 0.0 or self.end_s <= self.start_s:
            raise ValueError("perturbation interval must have end > start >= 0")
        if not np.isfinite(self.magnitude):
            raise ValueError("perturbation magnitude must be finite")
        if self.mode in (
            PerturbationMode.SCALE,
            PerturbationMode.ACTIVATE,
            PerturbationMode.GUST,
        ) and self.magnitude < 0.0:
            raise ValueError("scale, activation, and gust magnitudes must be non-negative")
        if (
            self.target_type is PerturbationTarget.ENVIRONMENT
        ) != (self.mode is PerturbationMode.GUST):
            raise ValueError("environment perturbations and gust mode must be paired")
        if len(self.vector) != 3 or not np.all(np.isfinite(self.vector)):
            raise ValueError("perturbation vector must contain three finite values")

    def active(self, time_s: float) -> bool:
        return self.start_s <= time_s < self.end_s


@dataclass(frozen=True)
class IndividualMuscleState:
    """One named muscle's instantaneous state at the hinge boundary.

    ``phase_effect`` is a signed, dimensionless mechanical modulation.  Force
    and activation remain non-negative.  The optional virtual moment arm is an
    explicitly exploratory mapping parameter, not reconstructed anatomy.
    """

    muscle: str
    side: Side
    muscle_class: MuscleClass
    activation: float
    force_n: float
    phase_effect: float = 0.0
    resonance_scale: float = 1.0
    virtual_moment_arm_m: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.muscle:
            raise ValueError("individual muscle name must be non-empty")
        if self.side is Side.BILATERAL:
            raise ValueError("individual muscles must have left or right side")
        values = (
            self.activation,
            self.force_n,
            self.phase_effect,
            self.resonance_scale,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("individual muscle state must be finite")
        if self.activation < 0.0 or self.activation > 1.5 or self.force_n < 0.0:
            raise ValueError("individual muscle activation and force must be non-negative")
        if self.resonance_scale <= 0.0:
            raise ValueError("individual muscle resonance scale must be positive")
        if self.virtual_moment_arm_m is not None:
            if not np.isfinite(self.virtual_moment_arm_m) or self.virtual_moment_arm_m <= 0.0:
                raise ValueError("virtual moment arm must be finite and positive")


@dataclass(frozen=True)
class MuscleSnapshot:
    power_left: float
    power_right: float
    steering_left: float
    steering_right: float
    tension_left: float
    tension_right: float
    individual: Mapping[str, IndividualMuscleState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_keys = set()
        for key, state in self.individual.items():
            expected = "%s:%s" % (state.side.value, state.muscle)
            if key != expected:
                raise ValueError("individual muscle key must be %s" % expected)
            if key in expected_keys:
                raise ValueError("individual muscle keys must be unique")
            expected_keys.add(key)


@dataclass(frozen=True)
class WingKinematics:
    """Prescribed left/right wing state in SI units and radians."""

    phase_rad: float
    frequency_hz: float
    stroke_rad: np.ndarray
    stroke_velocity_rad_s: np.ndarray
    stroke_acceleration_rad_s2: np.ndarray
    angle_of_attack_rad: np.ndarray
    deviation_rad: np.ndarray
    generalized_torque_n_m: np.ndarray
    wing_axis_torque_n_m: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.phase_rad) or not np.isfinite(self.frequency_hz):
            raise ValueError("wing phase and frequency must be finite")
        if self.frequency_hz <= 0.0:
            raise ValueError("wing frequency must be positive")
        for name in (
            "stroke_rad",
            "stroke_velocity_rad_s",
            "stroke_acceleration_rad_s2",
            "angle_of_attack_rad",
            "deviation_rad",
            "generalized_torque_n_m",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError("%s must contain two finite values" % name)
            object.__setattr__(self, name, value.copy())
        if self.wing_axis_torque_n_m is not None:
            axis = np.asarray(self.wing_axis_torque_n_m, dtype=float)
            if axis.shape != (6,) or not np.all(np.isfinite(axis)):
                raise ValueError("wing_axis_torque_n_m must contain six finite values")
            object.__setattr__(self, "wing_axis_torque_n_m", axis.copy())


@dataclass(frozen=True)
class AerodynamicWrench:
    force_body_n: np.ndarray
    torque_body_n_m: np.ndarray
    left_force_body_n: np.ndarray
    right_force_body_n: np.ndarray
    mechanical_power_w: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "force_body_n",
            "torque_body_n_m",
            "left_force_body_n",
            "right_force_body_n",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("%s must contain three finite values" % name)
            object.__setattr__(self, name, value.copy())
        if not np.isfinite(self.mechanical_power_w):
            raise ValueError("mechanical_power_w must be finite")


@dataclass
class RigidBodyState:
    """Six-degree-of-freedom body state; quaternion maps body to world."""

    position_world_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    velocity_world_m_s: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    quaternion_body_to_world: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    )
    angular_velocity_body_rad_s: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "position_world_m",
            "velocity_world_m_s",
            "angular_velocity_body_rad_s",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("%s must contain three finite values" % name)
            setattr(self, name, value.copy())
        quaternion = np.asarray(self.quaternion_body_to_world, dtype=float)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError(
                "quaternion_body_to_world must contain four finite values"
            )
        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-6):
            raise ValueError("quaternion_body_to_world must be unit normalized")
        self.quaternion_body_to_world = quaternion.copy()

    def copy(self) -> "RigidBodyState":
        return RigidBodyState(
            self.position_world_m.copy(),
            self.velocity_world_m_s.copy(),
            self.quaternion_body_to_world.copy(),
            self.angular_velocity_body_rad_s.copy(),
        )


@dataclass(frozen=True)
class EpisodeDiagnostics:
    status: ValidationStatus
    exact_spike_count: int
    inferred_spike_count: int
    physics_steps: int
    final_time_s: float
    integrated_aerodynamic_impulse_n_s: Tuple[float, float, float]
    warnings: Tuple[str, ...]
    metrics: Mapping[str, float]
    physics_backend: str = "reduced_order"
    aerodynamic_owner: str = "reduced_order_quasi_steady"
    physics_provenance: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FlightEpisodeOutput:
    time_s: np.ndarray
    position_world_m: np.ndarray
    velocity_world_m_s: np.ndarray
    quaternion_body_to_world: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    wing_stroke_rad: np.ndarray
    wing_angle_of_attack_rad: np.ndarray
    aerodynamic_force_body_n: np.ndarray
    aerodynamic_torque_body_n_m: np.ndarray
    motor_event_times_s: Mapping[str, np.ndarray]
    motor_event_phases_rad: Mapping[str, np.ndarray]
    muscle_activation: Mapping[str, np.ndarray]
    diagnostics: EpisodeDiagnostics
    muscle_force_n: Mapping[str, np.ndarray] = field(default_factory=dict)
    muscle_phase_effect: Mapping[str, np.ndarray] = field(default_factory=dict)
    muscle_work_j: Mapping[str, np.ndarray] = field(default_factory=dict)
    measured_wing_joint_angle_rad: Optional[np.ndarray] = None
    measured_wing_joint_velocity_rad_s: Optional[np.ndarray] = None
    measured_wing_joint_order: Tuple[str, ...] = ()
    whole_fly_com_position_world_m: Optional[np.ndarray] = None
    ground_contact_count: Optional[np.ndarray] = None
    external_actuator_torque_n_m: Optional[np.ndarray] = None
    physics_time_s: Optional[np.ndarray] = None
    ground_contact_transition_point_count: Optional[np.ndarray] = None
    external_actuator_torque_physics_n_m: Optional[np.ndarray] = None
    measured_wing_joint_angle_physics_rad: Optional[np.ndarray] = None
    measured_wing_joint_velocity_physics_rad_s: Optional[np.ndarray] = None
    aerodynamic_force_body_physics_n: Optional[np.ndarray] = None
    aerodynamic_torque_body_physics_n_m: Optional[np.ndarray] = None
