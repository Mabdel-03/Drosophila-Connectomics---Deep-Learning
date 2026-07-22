"""Deterministic open-loop flight episodes for neural-to-motor experiments.

The engine is intentionally a dependency-light scientific scaffold.  It does
not execute FlyBody or MuJoCo; ``ExternalFlightPhysicsAdapter`` defines that
future worker boundary.  The included body and aerodynamic models are labelled
exploratory and support reproducible development, perturbation logic, and
timestep-convergence checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from .aerodynamics import QuasiSteadyAerodynamics
from .clock import MultiRateClock
from .hinge import VirtualWingHinge, WingHinge
from .muscles import (
    AsynchronousPowerMuscle,
    PhaseCodedSteeringMuscle,
    TensionMuscle,
    ThoraxOscillator,
)
from .rigid_body import ExternalFlightPhysicsAdapter, RigidBody6DOF, quaternion_to_matrix
from .signals import ExplicitSpikeCursor, PhaseLockedSpikeGenerator, SplayedSpikeGenerator
from .types import (
    AerodynamicWrench,
    EpisodeDiagnostics,
    FlightEpisodeOutput,
    MotorCommand,
    MotorSignalKind,
    IndividualMuscleState,
    MuscleClass,
    MuscleSnapshot,
    Perturbation,
    PerturbationMode,
    PerturbationTarget,
    RigidBodyState,
    Side,
    ValidationStatus,
)


@dataclass(frozen=True)
class FlightSimulationConfig:
    duration_s: float = 0.100
    physics_dt_s: float = 0.0001
    neural_dt_s: float = 0.005
    logging_dt_s: float = 0.001
    seed: int = 0
    # ``None`` selects the explicit exploratory airborne baseline. An empty
    # tuple means no motor commands and must never be replaced implicitly.
    motor_commands: Optional[Tuple[MotorCommand, ...]] = None
    dn_drives: Mapping[str, float] = field(default_factory=dict)
    # Calibrated adapters may map normalized DN drive to an MN rate increment.
    # Keys are DN names, then exact ``MotorCommand.neuron_id`` values; values
    # are Hz per normalized drive and must never be raw connectome counts.
    dn_motor_rate_gains_hz: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    perturbations: Tuple[Perturbation, ...] = ()
    vch_dch_factor: float = 1.0
    pathway_dn_names: Tuple[str, ...] = (
        "DNp26",
        "DNae002",
        "DNbe001",
        "DNa04",
        "DNg32",
        "DNge107",
        "DNp07",
    )
    dng02_power_rate_gain: float = 0.65
    initial_state: Optional[RigidBodyState] = None

    def __post_init__(self) -> None:
        clock_values = (
            self.duration_s,
            self.physics_dt_s,
            self.neural_dt_s,
            self.logging_dt_s,
        )
        if any(not np.isfinite(value) for value in clock_values):
            raise ValueError("duration and clock timesteps must be finite")
        if self.duration_s <= 0.0 or self.physics_dt_s <= 0.0:
            raise ValueError("duration and physics timestep must be positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.neural_dt_s < self.physics_dt_s or self.logging_dt_s < self.physics_dt_s:
            raise ValueError("neural and logging timesteps cannot be below physics dt")
        if self.vch_dch_factor < 0.0 or not np.isfinite(self.vch_dch_factor):
            raise ValueError("vch_dch_factor must be finite and non-negative")
        if (
            not np.isfinite(self.dng02_power_rate_gain)
            or self.dng02_power_rate_gain < 0.0
        ):
            raise ValueError("dng02_power_rate_gain must be finite and non-negative")
        if self.motor_commands is not None:
            commands = tuple(self.motor_commands)
            if len({command.neuron_id for command in commands}) != len(commands):
                raise ValueError("motor command neuron_id values must be unique")
            object.__setattr__(self, "motor_commands", commands)
        if any(not np.isfinite(float(value)) for value in self.dn_drives.values()):
            raise ValueError("DN drives must be finite")
        if (
            any(not isinstance(name, str) or not name for name in self.pathway_dn_names)
            or len(set(self.pathway_dn_names)) != len(self.pathway_dn_names)
        ):
            raise ValueError("pathway_dn_names must contain unique non-empty strings")
        for dn_name, targets in self.dn_motor_rate_gains_hz.items():
            if not isinstance(dn_name, str) or not dn_name:
                raise ValueError("DN-to-MN gain keys must be non-empty DN names")
            if any(not isinstance(name, str) or not name for name in targets):
                raise ValueError("DN-to-MN targets must be non-empty MN identifiers")
            if any(not np.isfinite(float(value)) for value in targets.values()):
                raise ValueError("DN-to-MN rate gains must be finite")


def default_motor_commands() -> Tuple[MotorCommand, ...]:
    """Return an explicitly exploratory, bilaterally symmetric motor program."""

    commands: List[MotorCommand] = []
    for side in (Side.LEFT, Side.RIGHT):
        commands.extend(
            [
                MotorCommand(
                    "MN-DLM-%s" % side.value,
                    "DLM",
                    side,
                    MuscleClass.ASYNCHRONOUS_POWER,
                    MotorSignalKind.INFERRED_RATE,
                    rate_hz=5.0,
                    motor_units=6,
                    provenance="exploratory baseline; rate-derived timing",
                ),
                MotorCommand(
                    "MN-DVM-%s" % side.value,
                    "DVM",
                    side,
                    MuscleClass.ASYNCHRONOUS_POWER,
                    MotorSignalKind.INFERRED_RATE,
                    rate_hz=5.0,
                    motor_units=3,
                    provenance="exploratory baseline; rate-derived timing",
                ),
                MotorCommand(
                    "MN-tp1-%s" % side.value,
                    "tp1",
                    side,
                    MuscleClass.TENSION,
                    MotorSignalKind.INFERRED_RATE,
                    rate_hz=10.0,
                    motor_units=1,
                    provenance="exploratory baseline tension rate",
                ),
            ]
        )
        # Keep individual steering channels all the way to the hinge.  The
        # phases are explicit exploratory defaults and are replaced by a
        # calibrated registry entry rather than being inferred from synapses.
        for muscle, preferred_phase in (
            ("iv2", 0.15 * np.pi),
            ("i1", 0.45 * np.pi),
            ("iv1", 0.75 * np.pi),
            ("b3", 1.10 * np.pi),
            ("b1", 0.0),
        ):
            commands.append(
                MotorCommand(
                    "MN-%s-%s" % (muscle, side.value),
                    muscle,
                    side,
                    MuscleClass.STEERING,
                    MotorSignalKind.INFERRED_RATE,
                    rate_hz=0.0,
                    preferred_phase_rad=preferred_phase,
                    provenance=(
                        "exploratory baseline; no tonic steering; phase is an "
                        "uncalibrated virtual-hinge parameter"
                    ),
                )
            )
    return tuple(commands)


class _MotorRuntime:
    def __init__(self, command: MotorCommand) -> None:
        self.command = command
        self.cursor = (
            ExplicitSpikeCursor(command.spike_times_s)
            if command.signal_kind
            in (
                MotorSignalKind.EXACT_SPIKES,
                MotorSignalKind.SEEDED_SYNTHETIC_SPIKES,
            )
            else None
        )
        self.splayed = None
        self.phase_locked = None
        if command.signal_kind is MotorSignalKind.INFERRED_RATE:
            if command.muscle_class is MuscleClass.STEERING:
                self.phase_locked = PhaseLockedSpikeGenerator(
                    command.preferred_phase_rad
                )
            else:
                self.splayed = SplayedSpikeGenerator(command.motor_units)


class FlightEpisodeRunner:
    """Run one deterministic exploratory flight episode."""

    def __init__(
        self,
        hinge: Optional[WingHinge] = None,
        aerodynamics: Optional[QuasiSteadyAerodynamics] = None,
        physics_adapter: Optional[ExternalFlightPhysicsAdapter] = None,
    ) -> None:
        self.hinge = hinge if hinge is not None else VirtualWingHinge()
        if physics_adapter is not None and aerodynamics is not None:
            raise ValueError(
                "external physics owns aerodynamics; do not also provide analytic aerodynamics"
            )
        self.physics_adapter = physics_adapter
        self.aerodynamics = None if physics_adapter is not None else (
            aerodynamics if aerodynamics is not None else QuasiSteadyAerodynamics()
        )
        if physics_adapter is not None:
            backend_name = getattr(physics_adapter, "backend_name", "")
            aerodynamic_owner = getattr(physics_adapter, "aerodynamic_owner", "")
            if not backend_name or not aerodynamic_owner:
                raise ValueError(
                    "external physics adapter must declare backend_name and aerodynamic_owner"
                )

    @staticmethod
    def _pathway_factor(config: FlightSimulationConfig, time_s: float) -> float:
        factor = config.vch_dch_factor
        for perturbation in config.perturbations:
            if not perturbation.active(time_s):
                continue
            if perturbation.target_type is not PerturbationTarget.PATHWAY:
                continue
            if perturbation.target.lower() not in ("vch/dch", "vch+dch", "vch_dch"):
                continue
            if perturbation.mode is PerturbationMode.SILENCE:
                factor = 0.0
            elif perturbation.mode is PerturbationMode.SCALE:
                factor *= perturbation.magnitude
            elif perturbation.mode is PerturbationMode.ACTIVATE:
                factor += perturbation.magnitude
        return max(0.0, factor)

    @staticmethod
    def _dn_drive(
        name: str, config: FlightSimulationConfig, time_s: float
    ) -> float:
        value = float(config.dn_drives.get(name, 0.0))
        if name in config.pathway_dn_names:
            value *= FlightEpisodeRunner._pathway_factor(config, time_s)
        for perturbation in config.perturbations:
            if (
                not perturbation.active(time_s)
                or perturbation.target_type is not PerturbationTarget.DN
                or perturbation.target != name
            ):
                continue
            if perturbation.mode is PerturbationMode.SILENCE:
                value = 0.0
            elif perturbation.mode is PerturbationMode.SCALE:
                value *= perturbation.magnitude
            elif perturbation.mode is PerturbationMode.ACTIVATE:
                value += perturbation.magnitude
        return value

    @staticmethod
    def _motor_modifiers(
        command: MotorCommand,
        config: FlightSimulationConfig,
        time_s: float,
    ) -> Tuple[float, float, bool]:
        scale = 1.0
        added_rate = 0.0
        silent = False
        for perturbation in config.perturbations:
            if (
                not perturbation.active(time_s)
                or perturbation.target_type is not PerturbationTarget.MN
                or perturbation.target != command.neuron_id
            ):
                continue
            if perturbation.mode is PerturbationMode.SILENCE:
                silent = True
            elif perturbation.mode is PerturbationMode.SCALE:
                scale *= perturbation.magnitude
            elif perturbation.mode is PerturbationMode.ACTIVATE:
                added_rate += perturbation.magnitude
        return max(0.0, scale), max(0.0, added_rate), silent

    @staticmethod
    def _mapped_dn_rate_increment(
        command: MotorCommand,
        config: FlightSimulationConfig,
        time_s: float,
    ) -> float:
        increment = 0.0
        for dn_name, target_gains in config.dn_motor_rate_gains_hz.items():
            gain = float(target_gains.get(command.neuron_id, 0.0))
            increment += gain * FlightEpisodeRunner._dn_drive(
                dn_name, config, time_s
            )
        return increment

    @staticmethod
    def _muscle_perturbation(
        side: Side,
        muscle_name: str,
        value: float,
        config: FlightSimulationConfig,
        time_s: float,
    ) -> float:
        result = value
        aliases = {muscle_name, "%s:%s" % (side.value, muscle_name)}
        for perturbation in config.perturbations:
            if (
                not perturbation.active(time_s)
                or perturbation.target_type is not PerturbationTarget.MUSCLE
                or perturbation.target not in aliases
            ):
                continue
            if perturbation.mode is PerturbationMode.SILENCE:
                result = 0.0
            elif perturbation.mode is PerturbationMode.SCALE:
                result *= perturbation.magnitude
            elif perturbation.mode is PerturbationMode.ACTIVATE:
                result += perturbation.magnitude
        return float(result)

    @staticmethod
    def _wind_world(
        config: FlightSimulationConfig, time_s: float
    ) -> np.ndarray:
        wind = np.zeros(3, dtype=float)
        for perturbation in config.perturbations:
            if (
                perturbation.active(time_s)
                and perturbation.target_type is PerturbationTarget.ENVIRONMENT
                and perturbation.mode is PerturbationMode.GUST
            ):
                wind += perturbation.magnitude * np.asarray(
                    perturbation.vector, dtype=float
                )
        return wind

    def run(self, config: FlightSimulationConfig) -> FlightEpisodeOutput:
        # The seed is consumed even though default generators are deterministic;
        # this keeps stochastic calibrated components reproducible when injected.
        # Reserve a deterministic generator for injected calibrated components.
        # The bundled reduced-order components themselves do not sample noise.
        _rng = np.random.default_rng(config.seed)
        clock = MultiRateClock(
            config.duration_s,
            config.physics_dt_s,
            {"neural": config.neural_dt_s, "log": config.logging_dt_s},
        )
        commands = (
            default_motor_commands()
            if config.motor_commands is None
            else config.motor_commands
        )
        command_ids = {command.neuron_id for command in commands}
        muscle_targets = {command.muscle for command in commands}
        muscle_targets.update(
            "%s:%s" % (command.side.value, command.muscle)
            for command in commands
        )
        known_dns = (
            set(config.dn_drives)
            | set(config.dn_motor_rate_gains_hz)
            | set(config.pathway_dn_names)
            | {"DNg02"}
        )
        mapped_targets = {
            neuron_id
            for targets in config.dn_motor_rate_gains_hz.values()
            for neuron_id in targets
        }
        unknown_mapped_targets = mapped_targets - command_ids
        if unknown_mapped_targets:
            raise ValueError(
                "DN-to-MN gains target unknown motor commands: %s"
                % ", ".join(sorted(unknown_mapped_targets))
            )
        pathway_aliases = {"vch/dch", "vch+dch", "vch_dch"}
        for perturbation in config.perturbations:
            if (
                perturbation.target_type is PerturbationTarget.PATHWAY
                and perturbation.target.lower() not in pathway_aliases
            ):
                raise ValueError("pathway perturbation targets an unknown pathway")
            if (
                perturbation.target_type is PerturbationTarget.DN
                and perturbation.target not in known_dns
            ):
                raise ValueError("DN perturbation targets an unused descending neuron")
            if (
                perturbation.target_type is PerturbationTarget.MN
                and perturbation.target not in command_ids
            ):
                raise ValueError("MN perturbation targets an unknown motor command")
            if (
                perturbation.target_type is PerturbationTarget.MUSCLE
                and perturbation.target not in muscle_targets
            ):
                raise ValueError("muscle perturbation targets an unknown muscle")
        for command in commands:
            if any(time_s >= config.duration_s for time_s in command.spike_times_s):
                raise ValueError("motor spike times must lie in [0, duration_s)")
            if command.signal_kind in (
                MotorSignalKind.EXACT_SPIKES,
                MotorSignalKind.SEEDED_SYNTHETIC_SPIKES,
            ):
                for perturbation in config.perturbations:
                    if (
                        perturbation.target_type is PerturbationTarget.MN
                        and perturbation.target == command.neuron_id
                        and perturbation.mode
                        in (PerturbationMode.SCALE, PerturbationMode.ACTIVATE)
                        and perturbation.start_s < config.duration_s
                    ):
                        raise ValueError(
                            "exact or seeded-spike MN commands support silencing only; "
                            "scaling or activation would infer new event timing"
                        )
        runtimes = [_MotorRuntime(command) for command in commands]

        power_models: Dict[Tuple[Side, str], AsynchronousPowerMuscle] = {}
        steering_models: Dict[Tuple[Side, str], PhaseCodedSteeringMuscle] = {}
        tension_models: Dict[Tuple[Side, str], TensionMuscle] = {}
        for command in commands:
            key = (command.side, command.muscle)
            if command.side is Side.BILATERAL:
                raise ValueError("expand bilateral commands into left and right commands")
            if command.muscle_class is MuscleClass.ASYNCHRONOUS_POWER:
                power_models.setdefault(key, AsynchronousPowerMuscle())
            elif command.muscle_class is MuscleClass.STEERING:
                steering_models.setdefault(
                    key, PhaseCodedSteeringMuscle(command.preferred_phase_rad)
                )
            elif command.muscle_class is MuscleClass.TENSION:
                tension_models.setdefault(key, TensionMuscle())

        oscillator = ThoraxOscillator()
        if self.physics_adapter is None:
            body: Optional[RigidBody6DOF] = RigidBody6DOF(
                initial_state=config.initial_state
            )
            current_state = body.state.copy()
            physics_backend = "reduced_order"
            aerodynamic_owner = "reduced_order_quasi_steady"
            physics_provenance: Mapping[str, object] = {
                "engine": "fly_sensor2behavior internal reduced-order dynamics",
                "aerodynamic_model": "internal quasi-steady exploratory model",
            }
        else:
            body = None
            if config.initial_state is not None:
                current_state = config.initial_state.copy()
            else:
                default_state_factory = getattr(
                    self.physics_adapter, "default_initial_state", None
                )
                current_state = (
                    default_state_factory()
                    if callable(default_state_factory)
                    else RigidBodyState()
                )
            current_state.validate()
            self.physics_adapter.reset(current_state)
            physics_backend = str(self.physics_adapter.backend_name)
            aerodynamic_owner = str(self.physics_adapter.aerodynamic_owner)
            provenance_reader = getattr(
                self.physics_adapter, "provenance_metadata", None
            )
            if callable(provenance_reader):
                raw_physics_provenance = provenance_reader()
                if not isinstance(raw_physics_provenance, Mapping):
                    raise ValueError(
                        "external physics provenance_metadata must return a mapping"
                    )
                physics_provenance = dict(raw_physics_provenance)
            else:
                physics_provenance = {
                    "engine": physics_backend,
                    "status": "external adapter supplied no detailed runtime provenance",
                }
        dt_s = config.physics_dt_s
        log_ratio = clock.ratios["log"]
        exact_count = 0
        inferred_count = 0
        impulse = np.zeros(3, dtype=float)
        power_integral = 0.0

        times: List[float] = [0.0]
        positions: List[np.ndarray] = [current_state.position_world_m.copy()]
        velocities: List[np.ndarray] = [current_state.velocity_world_m_s.copy()]
        quaternions: List[np.ndarray] = [current_state.quaternion_body_to_world.copy()]
        angular_velocities: List[np.ndarray] = [
            current_state.angular_velocity_body_rad_s.copy()
        ]
        strokes: List[np.ndarray] = [np.zeros(2, dtype=float)]
        angles: List[np.ndarray] = [
            np.full(2, np.deg2rad(45.0), dtype=float)
        ]
        forces: List[np.ndarray] = [np.zeros(3, dtype=float)]
        torques: List[np.ndarray] = [np.zeros(3, dtype=float)]
        aerodynamic_forces_physics: Optional[List[np.ndarray]] = (
            [np.zeros(3, dtype=float)]
            if self.physics_adapter is not None
            else None
        )
        aerodynamic_torques_physics: Optional[List[np.ndarray]] = (
            [np.zeros(3, dtype=float)]
            if self.physics_adapter is not None
            else None
        )
        wing_state_reader = (
            getattr(self.physics_adapter, "wing_joint_state", None)
            if self.physics_adapter is not None
            else None
        )
        measured_wing_angles: Optional[List[np.ndarray]] = None
        measured_wing_velocities: Optional[List[np.ndarray]] = None
        measured_wing_angles_physics: Optional[List[np.ndarray]] = None
        measured_wing_velocities_physics: Optional[List[np.ndarray]] = None
        latest_measured_wing_angle: Optional[np.ndarray] = None
        latest_measured_wing_velocity: Optional[np.ndarray] = None
        measured_wing_order: Tuple[str, ...] = ()
        com_reader = (
            getattr(self.physics_adapter, "whole_fly_com_position_m", None)
            if self.physics_adapter is not None
            else None
        )
        whole_fly_com_positions: Optional[List[np.ndarray]] = None
        contact_reader = (
            getattr(self.physics_adapter, "ground_contact_count", None)
            if self.physics_adapter is not None
            else None
        )
        ground_contact_counts: Optional[List[int]] = None
        ground_contact_transition_point_counts: Optional[List[int]] = None
        latest_ground_contact_count = 0
        ground_contact_transition_count = 0
        ground_contact_transition_count_first_50ms = 0
        maximum_ground_contact_count = 0
        first_ground_contact_transition_start_s: Optional[float] = None
        first_ground_contact_transition_end_s: Optional[float] = None
        if callable(contact_reader):
            raw_contact_count = contact_reader()
            if (
                isinstance(raw_contact_count, (bool, np.bool_))
                or not isinstance(raw_contact_count, (int, np.integer))
                or int(raw_contact_count) < 0
            ):
                raise ValueError(
                    "external ground contact count must be a non-negative integer"
                )
            latest_ground_contact_count = int(raw_contact_count)
            ground_contact_counts = [latest_ground_contact_count]
            ground_contact_transition_point_counts = [latest_ground_contact_count]
            maximum_ground_contact_count = latest_ground_contact_count
            if latest_ground_contact_count > 0:
                first_ground_contact_transition_start_s = 0.0
                first_ground_contact_transition_end_s = 0.0
        external_actuator_torques: Optional[List[np.ndarray]] = None
        external_actuator_torque_physics: Optional[List[np.ndarray]] = None
        latest_external_actuator_torque_n_m: Optional[np.ndarray] = None
        if self.physics_adapter is not None:
            initial_actuator_torque = getattr(
                self.physics_adapter, "last_actuator_torque_n_m", None
            )
            if initial_actuator_torque is not None:
                initial_actuator_torque = np.asarray(
                    initial_actuator_torque, dtype=float
                )
                if (
                    initial_actuator_torque.shape != (6,)
                    or not np.all(np.isfinite(initial_actuator_torque))
                ):
                    raise ValueError(
                        "external actuator torque telemetry must contain six finite axes"
                    )
                latest_external_actuator_torque_n_m = (
                    initial_actuator_torque.copy()
                )
                external_actuator_torques = [initial_actuator_torque.copy()]
                external_actuator_torque_physics = [
                    initial_actuator_torque.copy()
                ]
        if callable(com_reader):
            initial_com = np.asarray(com_reader(), dtype=float)
            if initial_com.shape != (3,) or not np.all(np.isfinite(initial_com)):
                raise ValueError("external whole-fly COM must contain three finite values")
            whole_fly_com_positions = [initial_com.copy()]
        if callable(wing_state_reader):
            measured_angle, measured_velocity = wing_state_reader()
            measured_angle = np.asarray(measured_angle, dtype=float)
            measured_velocity = np.asarray(measured_velocity, dtype=float)
            if (
                measured_angle.shape != (6,)
                or measured_velocity.shape != (6,)
                or not np.all(np.isfinite(measured_angle))
                or not np.all(np.isfinite(measured_velocity))
            ):
                raise ValueError("external measured wing state must contain six finite axes")
            measured_wing_angles = [measured_angle.copy()]
            measured_wing_velocities = [measured_velocity.copy()]
            measured_wing_angles_physics = [measured_angle.copy()]
            measured_wing_velocities_physics = [measured_velocity.copy()]
            latest_measured_wing_angle = measured_angle.copy()
            latest_measured_wing_velocity = measured_velocity.copy()
            measured_wing_order = tuple(
                getattr(self.physics_adapter, "wing_joint_order", ())
            )
            if len(measured_wing_order) != 6:
                raise ValueError("external measured wing state requires six axis names")
        activation_history: MutableMapping[str, List[float]] = {}
        force_history: MutableMapping[str, List[float]] = {}
        phase_effect_history: MutableMapping[str, List[float]] = {}
        work_history: MutableMapping[str, List[float]] = {}
        cumulative_work: MutableMapping[str, float] = {}
        for (side, name), model in power_models.items():
            label = "%s:%s" % (side.value, name)
            activation_history[label] = [model.activation]
            force_history[label] = [model.force_n]
            phase_effect_history[label] = [0.0]
            work_history[label] = [0.0]
            cumulative_work[label] = 0.0
        for (side, name), model in steering_models.items():
            label = "%s:%s" % (side.value, name)
            activation_history[label] = [model.activation]
            force_history[label] = [model.force_n]
            phase_effect_history[label] = [model.phase_effect]
            work_history[label] = [0.0]
            cumulative_work[label] = 0.0
        for (side, name), model in tension_models.items():
            label = "%s:%s" % (side.value, name)
            activation_history[label] = [model.activation]
            force_history[label] = [0.0]
            phase_effect_history[label] = [0.0]
            work_history[label] = [0.0]
            cumulative_work[label] = 0.0
        motor_event_times: MutableMapping[str, List[float]] = {
            command.neuron_id: [] for command in commands
        }
        motor_event_phases: MutableMapping[str, List[float]] = {
            command.neuron_id: [] for command in commands
        }

        maximum_absolute_stroke = 0.0
        maximum_external_actuator_torque_n_m = 0.0
        frequency_integral = 0.0
        neural_update_count = 0
        external_wrench_telemetry = self.physics_adapter is not None and callable(
            getattr(self.physics_adapter, "aerodynamic_wrench", None)
        )
        held_rate_hz: MutableMapping[str, float] = {
            command.neuron_id: 0.0 for command in commands
        }
        held_silent: MutableMapping[str, bool] = {
            command.neuron_id: False for command in commands
        }
        for tick in clock:
            t0 = tick.time_s
            phase_start = oscillator.phase_rad
            phase_end_estimate = phase_start + (
                2.0 * np.pi * oscillator.frequency_hz * dt_s
            )
            event_map: Dict[Tuple[Side, str], List[float]] = {
                key: [] for key in steering_models
            }
            event_counts: Dict[Tuple[Side, str], int] = {}
            if "neural" in tick.due:
                neural_update_count += 1
                dng02 = self._dn_drive("DNg02", config, t0)
                for runtime in runtimes:
                    command = runtime.command
                    rate_scale, added_rate, silent = self._motor_modifiers(
                        command, config, t0
                    )
                    added_rate += self._mapped_dn_rate_increment(command, config, t0)
                    if command.muscle_class is MuscleClass.ASYNCHRONOUS_POWER:
                        rate_scale *= max(
                            0.0, 1.0 + config.dng02_power_rate_gain * dng02
                        )
                    held_silent[command.neuron_id] = silent
                    if command.signal_kind is MotorSignalKind.INFERRED_RATE:
                        held_rate_hz[command.neuron_id] = (
                            0.0
                            if silent
                            else max(0.0, command.rate_hz * rate_scale + added_rate)
                        )

            for runtime in runtimes:
                command = runtime.command
                key = (command.side, command.muscle)
                if command.signal_kind in (
                    MotorSignalKind.EXACT_SPIKES,
                    MotorSignalKind.SEEDED_SYNTHETIC_SPIKES,
                ):
                    event_times = runtime.cursor.events(t0, t0 + dt_s)
                    if held_silent[command.neuron_id]:
                        event_times = ()
                    count = len(event_times)
                    if command.signal_kind is MotorSignalKind.EXACT_SPIKES:
                        exact_count += count
                    else:
                        inferred_count += count
                    event_counts[key] = event_counts.get(key, 0) + count
                    for event_time in event_times:
                        if command.signal_kind is MotorSignalKind.SEEDED_SYNTHETIC_SPIKES:
                            wrapped_phase = command.preferred_phase_rad % (2.0 * np.pi)
                        else:
                            event_phase = phase_start + 2.0 * np.pi * oscillator.frequency_hz * (
                                event_time - t0
                            )
                            wrapped_phase = event_phase % (2.0 * np.pi)
                        event_map.setdefault(key, []).append(wrapped_phase)
                        motor_event_times[command.neuron_id].append(event_time)
                        motor_event_phases[command.neuron_id].append(wrapped_phase)
                else:
                    rate = held_rate_hz[command.neuron_id]
                    if command.muscle_class is MuscleClass.STEERING:
                        phases = runtime.phase_locked.step(
                            rate, dt_s, phase_start, phase_end_estimate
                        )
                        event_map.setdefault(key, []).extend(phases)
                        count = len(phases)
                        for event_phase in phases:
                            phase_delta = (
                                event_phase - phase_start % (2.0 * np.pi)
                            ) % (2.0 * np.pi)
                            event_time = t0 + phase_delta / max(
                                2.0 * np.pi * oscillator.frequency_hz, 1e-12
                            )
                            motor_event_times[command.neuron_id].append(
                                min(event_time, t0 + dt_s)
                            )
                            motor_event_phases[command.neuron_id].append(event_phase)
                    else:
                        units = runtime.splayed.step(rate, dt_s)
                        count = len(units)
                        for _unit in units:
                            motor_event_times[command.neuron_id].append(t0 + 0.5 * dt_s)
                            motor_event_phases[command.neuron_id].append(
                                (0.5 * (phase_start + phase_end_estimate))
                                % (2.0 * np.pi)
                            )
                    inferred_count += count
                    event_counts[key] = event_counts.get(key, 0) + count

            power_values: Dict[Tuple[Side, str], float] = {}
            omega = 2.0 * np.pi * oscillator.frequency_hz
            for key, model in power_models.items():
                muscle_name = key[1].upper()
                sign = 1.0 if muscle_name.startswith("DLM") else -1.0
                stretch = sign * np.sin(phase_start)
                shortening = sign * omega * np.cos(phase_start) / max(omega, 1.0)
                model.step(
                    event_counts.get(key, 0), stretch, shortening, dt_s
                )
                value = self._muscle_perturbation(
                    key[0], key[1], model.activation, config, t0
                )
                power_values[key] = float(np.clip(value, 0.0, 1.5))

            steering_values: Dict[Tuple[Side, str], float] = {}
            steering_activations: Dict[Tuple[Side, str], float] = {}
            for key, model in steering_models.items():
                model.step(event_map.get(key, ()), dt_s)
                activation = self._muscle_perturbation(
                    key[0], key[1], model.activation, config, t0
                )
                activation = float(np.clip(activation, 0.0, 1.5))
                steering_activations[key] = activation
                # A perturbation changes the non-negative muscle drive once.
                # Phase memory retains its sign/timing; multiplying here makes
                # silence/scale linear instead of applying the perturbation to
                # both force and phase a second time at the hinge.
                steering_values[key] = activation * model.phase_effect

            tension_values: Dict[Tuple[Side, str], float] = {}
            tension_activations: Dict[Tuple[Side, str], float] = {}
            for key, model in tension_models.items():
                model.step(event_counts.get(key, 0), dt_s)
                activation = self._muscle_perturbation(
                    key[0], key[1], model.activation, config, t0
                )
                activation = float(np.clip(activation, 0.0, 1.5))
                tension_activations[key] = activation
                parameters = model.parameters
                tension_values[key] = 1.0 + parameters.resonance_gain * (
                    activation - parameters.baseline_activation
                )

            individual_states: Dict[str, IndividualMuscleState] = {}
            for key, model in power_models.items():
                label = "%s:%s" % (key[0].value, key[1])
                activation = power_values[key]
                individual_states[label] = IndividualMuscleState(
                    muscle=key[1],
                    side=key[0],
                    muscle_class=MuscleClass.ASYNCHRONOUS_POWER,
                    activation=activation,
                    force_n=model.parameters.max_isometric_force_n * activation,
                    virtual_moment_arm_m=self.hinge.parameters.power_virtual_moment_arm_m
                    if isinstance(self.hinge, VirtualWingHinge)
                    else None,
                )
            for key, model in steering_models.items():
                label = "%s:%s" % (key[0].value, key[1])
                activation = steering_activations[key]
                individual_states[label] = IndividualMuscleState(
                    muscle=key[1],
                    side=key[0],
                    muscle_class=MuscleClass.STEERING,
                    activation=activation,
                    force_n=model.parameters.max_force_n * activation,
                    phase_effect=model.phase_effect,
                    virtual_moment_arm_m=self.hinge.parameters.steering_virtual_moment_arm_m
                    if isinstance(self.hinge, VirtualWingHinge)
                    else None,
                )
            for key, model in tension_models.items():
                label = "%s:%s" % (key[0].value, key[1])
                activation = tension_activations[key]
                individual_states[label] = IndividualMuscleState(
                    muscle=key[1],
                    side=key[0],
                    muscle_class=MuscleClass.TENSION,
                    activation=activation,
                    force_n=0.0,
                    resonance_scale=tension_values[key],
                )

            def side_mean(
                values: Mapping[Tuple[Side, str], float],
                side: Side,
                default: float,
            ) -> float:
                selected = [value for (item_side, _), value in values.items() if item_side is side]
                return float(np.mean(selected)) if selected else default

            snapshot = MuscleSnapshot(
                power_left=side_mean(power_values, Side.LEFT, 0.0),
                power_right=side_mean(power_values, Side.RIGHT, 0.0),
                steering_left=side_mean(steering_values, Side.LEFT, 0.0),
                steering_right=side_mean(steering_values, Side.RIGHT, 0.0),
                tension_left=side_mean(tension_values, Side.LEFT, 1.0),
                tension_right=side_mean(tension_values, Side.RIGHT, 1.0),
                individual=individual_states,
            )
            tension_scale = 0.5 * (snapshot.tension_left + snapshot.tension_right)
            oscillator.step(
                0.5 * (snapshot.power_left + snapshot.power_right),
                tension_scale,
                dt_s,
            )
            # Evaluate the rapidly oscillating aerodynamic state at the phase
            # midpoint; left-endpoint sampling otherwise creates a first-order
            # bias even though the slow muscle states use exact relaxation.
            phase_midpoint = 0.5 * (phase_start + oscillator.phase_rad)
            wings = self.hinge.evaluate(
                phase_midpoint, oscillator.frequency_hz, snapshot
            )
            maximum_absolute_stroke = max(
                maximum_absolute_stroke, float(np.max(np.abs(wings.stroke_rad)))
            )
            frequency_integral += oscillator.frequency_hz * dt_s
            rotation = quaternion_to_matrix(current_state.quaternion_body_to_world)
            wind_world = self._wind_world(config, t0)
            air_velocity_body = rotation.T.dot(
                current_state.velocity_world_m_s - wind_world
            )
            if self.physics_adapter is None:
                assert self.aerodynamics is not None and body is not None
                wrench = self.aerodynamics.evaluate(
                    wings,
                    air_velocity_body,
                    current_state.angular_velocity_body_rad_s,
                )
                current_state = body.step(
                    wrench.force_body_n, wrench.torque_body_n_m, dt_s
                )
            else:
                # The external backend owns fluid/contact forces.  The zero
                # inputs make double application impossible; reviewed adapters
                # may return their own root-total fluid telemetry after stepping.
                zero_force = np.zeros(3, dtype=float)
                zero_torque = np.zeros(3, dtype=float)
                ambient_wind_setter = getattr(
                    self.physics_adapter, "set_ambient_air_velocity_world", None
                )
                if callable(ambient_wind_setter):
                    ambient_wind_setter(wind_world)
                elif np.any(np.abs(wind_world) > 0.0):
                    raise ValueError(
                        "selected external physics backend does not implement ambient wind"
                    )
                current_state = self.physics_adapter.step(
                    wings, zero_force, zero_torque, dt_s
                )
                if ground_contact_counts is not None:
                    raw_contact_count = contact_reader()
                    if (
                        isinstance(raw_contact_count, (bool, np.bool_))
                        or not isinstance(raw_contact_count, (int, np.integer))
                        or int(raw_contact_count) < 0
                    ):
                        raise ValueError(
                            "external ground contact count must be a non-negative integer"
                        )
                    latest_ground_contact_count = int(raw_contact_count)
                    maximum_ground_contact_count = max(
                        maximum_ground_contact_count,
                        latest_ground_contact_count,
                    )
                    if latest_ground_contact_count > 0:
                        ground_contact_transition_count += 1
                        if (tick.index + 1) * dt_s <= 0.050 + 1.0e-12:
                            ground_contact_transition_count_first_50ms += 1
                        if first_ground_contact_transition_end_s is None:
                            first_ground_contact_transition_start_s = tick.index * dt_s
                            first_ground_contact_transition_end_s = (
                                tick.index + 1
                            ) * dt_s
                    assert ground_contact_transition_point_counts is not None
                    ground_contact_transition_point_counts.append(
                        latest_ground_contact_count
                    )
                if external_wrench_telemetry:
                    wrench = self.physics_adapter.aerodynamic_wrench()
                else:
                    wrench = AerodynamicWrench(
                        force_body_n=zero_force,
                        torque_body_n_m=zero_torque,
                        left_force_body_n=zero_force.copy(),
                        right_force_body_n=zero_force.copy(),
                        mechanical_power_w=0.0,
                    )
                current_state.validate()
                wrench.validate()
                actuator_torque = getattr(
                    self.physics_adapter, "last_actuator_torque_n_m", None
                )
                if (
                    external_actuator_torque_physics is not None
                    and actuator_torque is None
                ):
                    raise ValueError(
                        "external actuator torque telemetry disappeared during an episode"
                    )
                if actuator_torque is not None:
                    actuator_torque = np.asarray(actuator_torque, dtype=float)
                    if (
                        actuator_torque.shape != (6,)
                        or not np.all(np.isfinite(actuator_torque))
                    ):
                        raise ValueError(
                            "external actuator torque telemetry must contain six finite axes"
                        )
                    latest_external_actuator_torque_n_m = actuator_torque.copy()
                    maximum_external_actuator_torque_n_m = max(
                        maximum_external_actuator_torque_n_m,
                        float(np.max(np.abs(actuator_torque))),
                    )
                    if external_actuator_torque_physics is not None:
                        external_actuator_torque_physics.append(
                            actuator_torque.copy()
                        )
                if measured_wing_angles_physics is not None:
                    assert measured_wing_velocities_physics is not None
                    measured_angle, measured_velocity = wing_state_reader()
                    measured_angle = np.asarray(measured_angle, dtype=float)
                    measured_velocity = np.asarray(measured_velocity, dtype=float)
                    if (
                        measured_angle.shape != (6,)
                        or measured_velocity.shape != (6,)
                        or not np.all(np.isfinite(measured_angle))
                        or not np.all(np.isfinite(measured_velocity))
                    ):
                        raise ValueError(
                            "external measured wing state must contain six finite axes"
                        )
                    latest_measured_wing_angle = measured_angle.copy()
                    latest_measured_wing_velocity = measured_velocity.copy()
                    measured_wing_angles_physics.append(
                        latest_measured_wing_angle.copy()
                    )
                    measured_wing_velocities_physics.append(
                        latest_measured_wing_velocity.copy()
                    )
            if aerodynamic_forces_physics is not None:
                assert aerodynamic_torques_physics is not None
                aerodynamic_forces_physics.append(wrench.force_body_n.copy())
                aerodynamic_torques_physics.append(wrench.torque_body_n_m.copy())
            impulse += wrench.force_body_n * dt_s
            power_integral += wrench.mechanical_power_w * dt_s
            for label, state in individual_states.items():
                side_index = 0 if state.side is Side.LEFT else 1
                moment_arm = state.virtual_moment_arm_m or 0.0
                modulation = (
                    state.phase_effect
                    if state.muscle_class is MuscleClass.STEERING
                    else 1.0
                )
                instantaneous_power = (
                    state.force_n
                    * moment_arm
                    * wings.stroke_velocity_rad_s[side_index]
                    * modulation
                )
                cumulative_work[label] += float(instantaneous_power * dt_s)
            if (tick.index + 1) % log_ratio == 0:
                times.append((tick.index + 1) * dt_s)
                positions.append(current_state.position_world_m.copy())
                velocities.append(current_state.velocity_world_m_s.copy())
                quaternions.append(current_state.quaternion_body_to_world.copy())
                angular_velocities.append(
                    current_state.angular_velocity_body_rad_s.copy()
                )
                strokes.append(wings.stroke_rad.copy())
                angles.append(wings.angle_of_attack_rad.copy())
                forces.append(wrench.force_body_n.copy())
                torques.append(wrench.torque_body_n_m.copy())
                if measured_wing_angles is not None:
                    assert measured_wing_velocities is not None
                    assert latest_measured_wing_angle is not None
                    assert latest_measured_wing_velocity is not None
                    measured_wing_angles.append(
                        latest_measured_wing_angle.copy()
                    )
                    measured_wing_velocities.append(
                        latest_measured_wing_velocity.copy()
                    )
                if whole_fly_com_positions is not None:
                    sampled_com = np.asarray(com_reader(), dtype=float)
                    if sampled_com.shape != (3,) or not np.all(np.isfinite(sampled_com)):
                        raise ValueError(
                            "external whole-fly COM must contain three finite values"
                        )
                    whole_fly_com_positions.append(sampled_com.copy())
                if ground_contact_counts is not None:
                    ground_contact_counts.append(latest_ground_contact_count)
                if external_actuator_torques is not None:
                    assert latest_external_actuator_torque_n_m is not None
                    external_actuator_torques.append(
                        latest_external_actuator_torque_n_m.copy()
                    )
                for key, history in activation_history.items():
                    state = individual_states[key]
                    history.append(state.activation)
                    force_history[key].append(state.force_n)
                    phase_effect_history[key].append(state.phase_effect)
                    work_history[key].append(cumulative_work[key])

        stroke_array = np.asarray(strokes, dtype=float)
        if self.physics_adapter is None:
            warnings = [
                "Exploratory reduced-order NumPy physics; FlyBody/MuJoCo was not executed.",
                "Virtual hinge and aerodynamic coefficients require dataset-specific calibration.",
            ]
        else:
            warnings = [
                "%s physics executed; neuromuscular actuation remains exploratory."
                % physics_backend,
            ]
            if external_wrench_telemetry:
                warnings.append(
                    "Root-total external fluid wrench is recorded; reviewed per-wing "
                    "force decomposition is unavailable."
                )
            else:
                warnings.append(
                    "External aerodynamic wrench telemetry is unavailable in this adapter version."
                )
            body_state_reference = getattr(
                self.physics_adapter, "body_state_reference", None
            )
            if body_state_reference:
                warnings.append("Body-state reference: %s." % body_state_reference)
            if ground_contact_counts is not None:
                if maximum_ground_contact_count:
                    warnings.append(
                        "Ground contact was present at reset or detected during {} "
                        "physics transitions; airborne and contact-dominated "
                        "transitions must be interpreted separately.".format(
                            ground_contact_transition_count
                        )
                    )
                else:
                    warnings.append(
                        "Exact MuJoCo reset/transition telemetry detected no "
                        "ground-contact points."
                    )
        if inferred_count:
            warnings.append(
                "One or more motor spike trains were inferred from rates, not measured."
            )
        metrics = {
            "mean_wingbeat_frequency_hz": float(
                frequency_integral / config.duration_s
            ),
            "maximum_absolute_stroke_rad": maximum_absolute_stroke,
            "final_altitude_m": float(current_state.position_world_m[2]),
            "final_speed_m_s": float(np.linalg.norm(current_state.velocity_world_m_s)),
            "maximum_logged_angular_speed_rad_s": float(
                np.max(np.linalg.norm(np.asarray(angular_velocities), axis=1))
            ),
            "mean_mechanical_power_w": float(power_integral / config.duration_s),
            "dng02_normalized_drive": float(config.dn_drives.get("DNg02", 0.0)),
            "physics_dt_s": float(dt_s),
            "neural_update_count": float(neural_update_count),
            "maximum_external_actuator_torque_n_m": float(
                maximum_external_actuator_torque_n_m
            ),
            "minimum_root_altitude_m": float(
                np.min(np.asarray(positions, dtype=float)[:, 2])
            ),
        }
        measured_angle_array: Optional[np.ndarray] = None
        measured_velocity_array: Optional[np.ndarray] = None
        measured_angle_physics_array: Optional[np.ndarray] = None
        measured_velocity_physics_array: Optional[np.ndarray] = None
        whole_fly_com_array: Optional[np.ndarray] = None
        if measured_wing_angles is not None:
            assert measured_wing_velocities is not None
            measured_angle_array = np.asarray(measured_wing_angles, dtype=float)
            measured_velocity_array = np.asarray(
                measured_wing_velocities, dtype=float
            )
            assert measured_wing_angles_physics is not None
            assert measured_wing_velocities_physics is not None
            measured_angle_physics_array = np.asarray(
                measured_wing_angles_physics, dtype=float
            )
            measured_velocity_physics_array = np.asarray(
                measured_wing_velocities_physics, dtype=float
            )
            measured_stroke = measured_angle_array[:, (0, 3)]
            centered_measured = measured_stroke - np.mean(
                measured_stroke, axis=0, keepdims=True
            )
            centered_desired = stroke_array - np.mean(
                stroke_array, axis=0, keepdims=True
            )
            metrics["maximum_measured_wing_excursion_rad"] = float(
                np.max(np.ptp(measured_angle_physics_array, axis=0))
            )
            metrics["desired_measured_stroke_rms_rad"] = float(
                np.sqrt(np.mean((centered_desired - centered_measured) ** 2))
            )
        if whole_fly_com_positions is not None:
            whole_fly_com_array = np.asarray(whole_fly_com_positions, dtype=float)
            metrics["final_whole_fly_com_altitude_m"] = float(
                whole_fly_com_array[-1, 2]
            )
            metrics["minimum_whole_fly_com_altitude_m"] = float(
                np.min(whole_fly_com_array[:, 2])
            )
        ground_contact_array: Optional[np.ndarray] = None
        ground_contact_transition_point_array: Optional[np.ndarray] = None
        external_actuator_torque_array: Optional[np.ndarray] = None
        external_actuator_torque_physics_array: Optional[np.ndarray] = None
        aerodynamic_force_physics_array: Optional[np.ndarray] = None
        aerodynamic_torque_physics_array: Optional[np.ndarray] = None
        physics_time_array: Optional[np.ndarray] = None
        if ground_contact_counts is not None:
            ground_contact_array = np.asarray(ground_contact_counts, dtype=np.int64)
            assert ground_contact_transition_point_counts is not None
            ground_contact_transition_point_array = np.asarray(
                ground_contact_transition_point_counts, dtype=np.int64
            )
            metrics["ground_contact_telemetry_available"] = 1.0
            metrics["initial_ground_contact_count"] = float(
                ground_contact_array[0]
            )
            metrics["ground_contact_transition_count"] = float(
                ground_contact_transition_count
            )
            metrics["ground_contact_transition_count_first_50ms"] = float(
                ground_contact_transition_count_first_50ms
            )
            metrics["maximum_ground_contact_count"] = float(
                maximum_ground_contact_count
            )
            metrics["ground_contact_occurred"] = float(
                maximum_ground_contact_count > 0
            )
            metrics[
                "first_ground_contact_transition_start_s_or_duration_s"
            ] = float(
                config.duration_s
                if first_ground_contact_transition_start_s is None
                else first_ground_contact_transition_start_s
            )
            metrics[
                "first_ground_contact_transition_end_s_or_duration_s"
            ] = float(
                config.duration_s
                if first_ground_contact_transition_end_s is None
                else first_ground_contact_transition_end_s
            )
        if external_actuator_torques is not None:
            external_actuator_torque_array = np.asarray(
                external_actuator_torques, dtype=float
            )
            assert external_actuator_torque_physics is not None
            external_actuator_torque_physics_array = np.asarray(
                external_actuator_torque_physics, dtype=float
            )
        if aerodynamic_forces_physics is not None:
            assert aerodynamic_torques_physics is not None
            aerodynamic_force_physics_array = np.asarray(
                aerodynamic_forces_physics, dtype=float
            )
            aerodynamic_torque_physics_array = np.asarray(
                aerodynamic_torques_physics, dtype=float
            )
        if (
            ground_contact_transition_point_array is not None
            or external_actuator_torque_physics_array is not None
            or measured_angle_physics_array is not None
            or aerodynamic_force_physics_array is not None
        ):
            physics_time_array = np.arange(
                clock.step_count + 1, dtype=float
            ) * dt_s
        expected_physics_samples = clock.step_count + 1
        if (
            ground_contact_transition_point_array is not None
            and ground_contact_transition_point_array.shape
            != (expected_physics_samples,)
        ):
            raise RuntimeError("ground-contact transition telemetry lost a physics sample")
        if (
            external_actuator_torque_physics_array is not None
            and external_actuator_torque_physics_array.shape
            != (expected_physics_samples, 6)
        ):
            raise RuntimeError("actuator-torque telemetry lost a physics sample")
        if (
            measured_angle_physics_array is not None
            and (
                measured_angle_physics_array.shape
                != (expected_physics_samples, 6)
                or measured_velocity_physics_array is None
                or measured_velocity_physics_array.shape
                != (expected_physics_samples, 6)
            )
        ):
            raise RuntimeError("measured-wing telemetry lost a physics sample")
        if (
            aerodynamic_force_physics_array is not None
            and (
                aerodynamic_force_physics_array.shape
                != (expected_physics_samples, 3)
                or aerodynamic_torque_physics_array is None
                or aerodynamic_torque_physics_array.shape
                != (expected_physics_samples, 3)
            )
        ):
            raise RuntimeError("aerodynamic-wrench telemetry lost a physics sample")
        diagnostics = EpisodeDiagnostics(
            status=ValidationStatus.EXPLORATORY,
            exact_spike_count=exact_count,
            inferred_spike_count=inferred_count,
            physics_steps=clock.step_count,
            final_time_s=config.duration_s,
            integrated_aerodynamic_impulse_n_s=tuple(float(x) for x in impulse),
            warnings=tuple(warnings),
            metrics=metrics,
            physics_backend=physics_backend,
            aerodynamic_owner=aerodynamic_owner,
            physics_provenance=physics_provenance,
        )
        return FlightEpisodeOutput(
            time_s=np.asarray(times, dtype=float),
            position_world_m=np.asarray(positions, dtype=float),
            velocity_world_m_s=np.asarray(velocities, dtype=float),
            quaternion_body_to_world=np.asarray(quaternions, dtype=float),
            angular_velocity_body_rad_s=np.asarray(angular_velocities, dtype=float),
            wing_stroke_rad=stroke_array,
            wing_angle_of_attack_rad=np.asarray(angles, dtype=float),
            aerodynamic_force_body_n=np.asarray(forces, dtype=float),
            aerodynamic_torque_body_n_m=np.asarray(torques, dtype=float),
            motor_event_times_s={
                key: np.asarray(values, dtype=float)
                for key, values in motor_event_times.items()
            },
            motor_event_phases_rad={
                key: np.asarray(values, dtype=float)
                for key, values in motor_event_phases.items()
            },
            muscle_activation={
                key: np.asarray(history, dtype=float)
                for key, history in activation_history.items()
            },
            diagnostics=diagnostics,
            muscle_force_n={
                key: np.asarray(history, dtype=float)
                for key, history in force_history.items()
            },
            muscle_phase_effect={
                key: np.asarray(history, dtype=float)
                for key, history in phase_effect_history.items()
            },
            muscle_work_j={
                key: np.asarray(history, dtype=float)
                for key, history in work_history.items()
            },
            measured_wing_joint_angle_rad=measured_angle_array,
            measured_wing_joint_velocity_rad_s=measured_velocity_array,
            measured_wing_joint_order=measured_wing_order,
            whole_fly_com_position_world_m=whole_fly_com_array,
            ground_contact_count=ground_contact_array,
            external_actuator_torque_n_m=external_actuator_torque_array,
            physics_time_s=physics_time_array,
            ground_contact_transition_point_count=(
                ground_contact_transition_point_array
            ),
            external_actuator_torque_physics_n_m=(
                external_actuator_torque_physics_array
            ),
            measured_wing_joint_angle_physics_rad=(
                measured_angle_physics_array
            ),
            measured_wing_joint_velocity_physics_rad_s=(
                measured_velocity_physics_array
            ),
            aerodynamic_force_body_physics_n=aerodynamic_force_physics_array,
            aerodynamic_torque_body_physics_n_m=(
                aerodynamic_torque_physics_array
            ),
        )


def convergence_metrics(
    coarse: FlightEpisodeOutput, fine: FlightEpisodeOutput
) -> Mapping[str, float]:
    """Compare integrated impulse and final state under timestep refinement.

    A naive relative position error is ill-conditioned in hover because the
    reference displacement approaches zero.  The standardized body-state score
    therefore uses biologically interpretable scales (2.5 mm body length,
    1 m/s translation, 200 rad/s rotation, and one radian attitude).  Raw
    relative values are also returned for transparency.
    """

    coarse_impulse = np.asarray(
        coarse.diagnostics.integrated_aerodynamic_impulse_n_s, dtype=float
    )
    fine_impulse = np.asarray(
        fine.diagnostics.integrated_aerodynamic_impulse_n_s, dtype=float
    )

    def relative_error(first: np.ndarray, second: np.ndarray) -> float:
        denominator = max(float(np.linalg.norm(second)), 1e-15)
        return float(np.linalg.norm(first - second) / denominator)

    position_difference = float(
        np.linalg.norm(coarse.position_world_m[-1] - fine.position_world_m[-1])
    )
    velocity_difference = float(
        np.linalg.norm(coarse.velocity_world_m_s[-1] - fine.velocity_world_m_s[-1])
    )
    angular_velocity_difference = float(
        np.linalg.norm(
            coarse.angular_velocity_body_rad_s[-1]
            - fine.angular_velocity_body_rad_s[-1]
        )
    )
    coarse_q = coarse.quaternion_body_to_world[-1]
    fine_q = fine.quaternion_body_to_world[-1]
    quaternion_dot = float(
        np.clip(abs(np.dot(coarse_q, fine_q)), 0.0, 1.0)
    )
    attitude_difference_rad = 2.0 * np.arccos(quaternion_dot)
    standardized_components = np.array(
        [
            position_difference / 2.5e-3,
            velocity_difference / 1.0,
            angular_velocity_difference / 200.0,
            attitude_difference_rad / 1.0,
        ],
        dtype=float,
    )
    return {
        "aerodynamic_impulse_relative_error": relative_error(
            coarse_impulse, fine_impulse
        ),
        "final_position_relative_error": relative_error(
            coarse.position_world_m[-1], fine.position_world_m[-1]
        ),
        "final_velocity_relative_error": relative_error(
            coarse.velocity_world_m_s[-1], fine.velocity_world_m_s[-1]
        ),
        "final_angular_velocity_relative_error": relative_error(
            coarse.angular_velocity_body_rad_s[-1],
            fine.angular_velocity_body_rad_s[-1],
        ),
        "final_position_body_length_error": float(standardized_components[0]),
        "final_velocity_standardized_error": float(standardized_components[1]),
        "final_angular_velocity_standardized_error": float(
            standardized_components[2]
        ),
        "final_attitude_standardized_error": float(standardized_components[3]),
        "standardized_body_state_error": float(
            np.linalg.norm(standardized_components)
        ),
    }
