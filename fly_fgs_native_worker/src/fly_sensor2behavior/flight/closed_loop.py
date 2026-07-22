"""Exploratory closed-loop optic-flow stabilization for software validation.

This module deliberately implements only a one-degree-of-freedom yaw plant.
It is a causal integration test for the path

``rendered panorama -> retina -> motion/NOD1 surrogate -> DN/MN surrogate
-> muscle yaw torque -> body yaw -> rendered panorama``.

It is **not** a FlyBody simulation, a calibrated Drosophila controller, or
evidence of biological yaw recovery.  The retina and motion detector are the
analytic software-test models from :mod:`fly_sensor2behavior.vision`; all
neural gains and the yaw torque gain are declared exploratory parameters.
Self-supervised recovery tests therefore validate causality, sign symmetry,
determinism, and closed-loop software behavior only.

Every dimensional value is SI.  Each stage is connected by an availability
queue and zero-order hold.  A value can affect the plant only after all of its
declared sensor, motion-processing, descending, motor, and muscle latencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..schema import EyeSide, RetinalFrame
from ..vision.motion import MotionFrame, ReichardtMotionDetector
from ..vision.retina import PanoramicRetina
from ..vision.stimulus import AnalyticGratingScene, AnalyticScene
from .bridge import AvailabilityQueue, AvailableValue
from .types import ValidationStatus


_TIME_TOLERANCE_S = 1e-12


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


@dataclass(frozen=True)
class YawTorquePulse:
    """Half-open external yaw-torque pulse used as a software perturbation."""

    start_s: float
    end_s: float
    torque_n_m: float

    def __post_init__(self) -> None:
        start_s = _finite(self.start_s, "start_s")
        end_s = _finite(self.end_s, "end_s")
        _finite(self.torque_n_m, "torque_n_m")
        if start_s < 0.0 or end_s <= start_s:
            raise ValueError("torque pulse must satisfy end_s > start_s >= 0")

    def value_at(self, time_s: float) -> float:
        return self.torque_n_m if self.start_s <= time_s < self.end_s else 0.0


@dataclass(frozen=True)
class ReducedYawLoopConfig:
    """Configuration for the explicitly reduced, exploratory yaw loop.

    ``optic_flow_to_yaw_torque_n_m`` encodes the software sign convention:
    with the current panoramic receptor ordering, positive body yaw produces
    a negative Reichardt population response, so a positive gain opposes the
    perturbation.  This is not a fitted muscle moment arm or physiological
    synaptic strength.
    """

    duration_s: float = 0.250
    physics_dt_s: float = 0.0001
    retinal_frame_interval_s: float = 0.002
    retinal_exposure_s: float = 0.002
    retinal_sensor_latency_s: float = 0.001
    motion_processing_latency_s: float = 0.001
    descending_latency_s: float = 0.003
    motor_latency_s: float = 0.002
    muscle_activation_latency_s: float = 0.002
    motion_filter_tau_s: float = 0.010
    receptor_count: int = 64
    yaw_inertia_kg_m2: float = 1.8e-12
    passive_yaw_damping_n_m_s: float = 0.0
    optic_flow_to_yaw_torque_n_m: float = 8.0e-9
    maximum_yaw_torque_n_m: float = 2.0e-10
    initial_yaw_world_rad: float = 0.0
    initial_yaw_rate_rad_s: float = 3.0
    feedback_enabled: bool = True
    torque_pulses: Tuple[YawTorquePulse, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        positive = (
            "duration_s",
            "physics_dt_s",
            "retinal_frame_interval_s",
            "retinal_exposure_s",
            "motion_filter_tau_s",
            "yaw_inertia_kg_m2",
            "maximum_yaw_torque_n_m",
        )
        for name in positive:
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError("%s must be positive" % name)
        nonnegative = (
            "retinal_sensor_latency_s",
            "motion_processing_latency_s",
            "descending_latency_s",
            "motor_latency_s",
            "muscle_activation_latency_s",
            "passive_yaw_damping_n_m_s",
            "optic_flow_to_yaw_torque_n_m",
        )
        for name in nonnegative:
            if _finite(getattr(self, name), name) < 0.0:
                raise ValueError("%s must be non-negative" % name)
        _finite(self.initial_yaw_world_rad, "initial_yaw_world_rad")
        _finite(self.initial_yaw_rate_rad_s, "initial_yaw_rate_rad_s")
        if self.retinal_exposure_s > self.retinal_frame_interval_s:
            raise ValueError("retinal exposure cannot exceed the frame interval")
        if self.physics_dt_s > self.retinal_frame_interval_s:
            raise ValueError("physics timestep cannot exceed retinal frame interval")
        step_count = self.duration_s / self.physics_dt_s
        if not math.isclose(step_count, round(step_count), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("duration_s must be an integer multiple of physics_dt_s")
        frame_steps = self.retinal_frame_interval_s / self.physics_dt_s
        if not math.isclose(frame_steps, round(frame_steps), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "retinal_frame_interval_s must be an integer multiple of physics_dt_s"
            )
        if (
            isinstance(self.receptor_count, bool)
            or not isinstance(self.receptor_count, int)
            or self.receptor_count < 4
        ):
            raise ValueError("receptor_count must be an integer of at least four")
        if not isinstance(self.feedback_enabled, bool):
            raise TypeError("feedback_enabled must be boolean")
        if any(not isinstance(pulse, YawTorquePulse) for pulse in self.torque_pulses):
            raise TypeError("torque_pulses must contain YawTorquePulse values")
        if any(pulse.end_s > self.duration_s + _TIME_TOLERANCE_S for pulse in self.torque_pulses):
            raise ValueError("torque pulses must lie within the episode duration")

    @property
    def total_feedback_latency_s(self) -> float:
        """Latency after exposure end, excluding exposure duration itself."""

        return (
            self.retinal_sensor_latency_s
            + self.motion_processing_latency_s
            + self.descending_latency_s
            + self.motor_latency_s
            + self.muscle_activation_latency_s
        )


@dataclass(frozen=True)
class ReducedYawFeedbackEvent:
    """One image-derived feedback sample and its complete causal timing chain."""

    retinal_measurement_time_s: float
    retinal_availability_time_s: float
    motion_availability_time_s: float
    descending_availability_time_s: float
    motor_availability_time_s: float
    muscle_torque_availability_time_s: float
    optic_flow_response: float
    nod1_surrogate_drive: float
    requested_yaw_torque_n_m: float

    def timing_is_causal(self) -> bool:
        times = (
            self.retinal_measurement_time_s,
            self.retinal_availability_time_s,
            self.motion_availability_time_s,
            self.descending_availability_time_s,
            self.motor_availability_time_s,
            self.muscle_torque_availability_time_s,
        )
        return all(
            current + _TIME_TOLERANCE_S >= previous
            for previous, current in zip(times, times[1:])
        )


@dataclass(frozen=True)
class ReducedYawLoopResult:
    """Traces from the yaw-only feedback scaffold.

    ``causal_timing_violations`` checks both event ordering and the actual
    zero-order-held torque trace.  A downstream self-supervised evaluator
    should require an empty result before interpreting recovery metrics.
    """

    time_s: np.ndarray
    yaw_world_rad: np.ndarray
    yaw_rate_rad_s: np.ndarray
    optic_flow_response: np.ndarray
    descending_drive: np.ndarray
    motor_drive: np.ndarray
    control_yaw_torque_n_m: np.ndarray
    disturbance_yaw_torque_n_m: np.ndarray
    feedback_events: Tuple[ReducedYawFeedbackEvent, ...]
    status: ValidationStatus = ValidationStatus.EXPLORATORY
    backend: str = "reduced_analytic_yaw_only"
    validation_scope: str = "self_supervised_software_invariants_only"
    warnings: Tuple[str, ...] = (
        "Not FlyBody/MuJoCo: yaw-only analytic plant.",
        "Retina, motion/NOD1, DN/MN, and muscle-torque mappings are uncalibrated reduced surrogates.",
        "Yaw recovery is a software-control invariant, not biological validation.",
    )

    def causal_timing_violations(self, *, atol_n_m: float = 1e-18) -> Tuple[str, ...]:
        violations = []
        if any(not event.timing_is_causal() for event in self.feedback_events):
            violations.append("feedback event availability precedes an upstream stage")
        ordered = sorted(
            self.feedback_events,
            key=lambda event: event.muscle_torque_availability_time_s,
        )
        event_index = 0
        expected_torque_n_m = 0.0
        for sample_index, time_s in enumerate(self.time_s):
            while (
                event_index < len(ordered)
                and ordered[event_index].muscle_torque_availability_time_s
                <= float(time_s) + _TIME_TOLERANCE_S
            ):
                expected_torque_n_m = ordered[event_index].requested_yaw_torque_n_m
                event_index += 1
            observed = float(self.control_yaw_torque_n_m[sample_index])
            if not math.isclose(observed, expected_torque_n_m, rel_tol=0.0, abs_tol=atol_n_m):
                violations.append(
                    "control torque at %.9g s does not match causally available zero-order hold"
                    % float(time_s)
                )
                break
        return tuple(violations)

    def integrated_absolute_yaw_rate_rad(self) -> float:
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(integrate(np.abs(self.yaw_rate_rad_s), self.time_s))


class ReducedClosedLoopYawSimulator:
    """Run the deterministic causal yaw-only software-validation loop."""

    def __init__(self, scene: Optional[AnalyticScene] = None) -> None:
        self.scene = scene if scene is not None else AnalyticGratingScene(
            spatial_frequency_cycles_per_rad=1.0,
            angular_velocity_rad_s=0.0,
            phase_rad=0.0,
            mean_luminance=0.5,
            contrast=0.8,
        )

    @staticmethod
    def _disturbance_torque(config: ReducedYawLoopConfig, time_s: float) -> float:
        return float(sum(pulse.value_at(time_s) for pulse in config.torque_pulses))

    def run(self, config: ReducedYawLoopConfig = ReducedYawLoopConfig()) -> ReducedYawLoopResult:
        retina = PanoramicRetina(
            receptor_count=config.receptor_count,
            sensor_latency_s=config.retinal_sensor_latency_s,
        )
        detector = ReichardtMotionDetector(
            temporal_tau_s=config.motion_filter_tau_s,
            output_latency_s=config.motion_processing_latency_s,
        )

        retinal_queue = AvailabilityQueue[RetinalFrame]()
        motion_queue = AvailabilityQueue[MotionFrame]()
        descending_queue = AvailabilityQueue[float]()
        motor_queue = AvailabilityQueue[float]()
        torque_queue = AvailabilityQueue[float]()

        processed_retinal = None
        processed_motion = None
        processed_descending = None
        processed_motor = None
        next_exposure_end_s = config.retinal_exposure_s
        events = []

        step_count = int(round(config.duration_s / config.physics_dt_s))
        time_s = np.arange(step_count + 1, dtype=float) * config.physics_dt_s
        yaw = np.zeros(step_count + 1, dtype=float)
        yaw_rate = np.zeros(step_count + 1, dtype=float)
        optic_flow = np.zeros(step_count + 1, dtype=float)
        descending_drive = np.zeros(step_count + 1, dtype=float)
        motor_drive = np.zeros(step_count + 1, dtype=float)
        control_torque = np.zeros(step_count + 1, dtype=float)
        disturbance_torque = np.zeros(step_count + 1, dtype=float)
        yaw[0] = config.initial_yaw_world_rad
        yaw_rate[0] = config.initial_yaw_rate_rad_s

        for index, current_time_s in enumerate(time_s):
            while next_exposure_end_s <= current_time_s + _TIME_TOLERANCE_S:
                exposure_start_s = next_exposure_end_s - config.retinal_exposure_s
                frame = retina.sample(
                    self.scene,
                    exposure_start_s=exposure_start_s,
                    exposure_end_s=next_exposure_end_s,
                    body_yaw_rad=float(yaw[index]),
                    eye_side=EyeSide.BINOCULAR,
                )
                retinal_queue.push(
                    AvailableValue(
                        measurement_time_s=frame.measurement_time_s,
                        availability_time_s=frame.availability_time_s,
                        value=frame,
                    )
                )
                next_exposure_end_s += config.retinal_frame_interval_s

            retinal = retinal_queue.advance(float(current_time_s))
            if retinal is not None and retinal is not processed_retinal:
                motion = detector.update(
                    retinal.value,
                    current_time_s=float(current_time_s),
                )
                motion_queue.push(
                    AvailableValue(
                        measurement_time_s=motion.measurement_time_s,
                        availability_time_s=motion.availability_time_s,
                        value=motion,
                    )
                )
                processed_retinal = retinal

            motion = motion_queue.advance(float(current_time_s))
            if motion is not None and motion is not processed_motion:
                raw_drive = float(motion.value.population_response)
                nod1_drive = raw_drive if config.feedback_enabled else 0.0
                descending_available_s = (
                    motion.availability_time_s + config.descending_latency_s
                )
                motor_available_s = descending_available_s + config.motor_latency_s
                torque_available_s = (
                    motor_available_s + config.muscle_activation_latency_s
                )
                requested_torque = float(
                    np.clip(
                        config.optic_flow_to_yaw_torque_n_m * nod1_drive,
                        -config.maximum_yaw_torque_n_m,
                        config.maximum_yaw_torque_n_m,
                    )
                )
                descending_queue.push(
                    AvailableValue(
                        measurement_time_s=motion.measurement_time_s,
                        availability_time_s=descending_available_s,
                        value=nod1_drive,
                    )
                )
                events.append(
                    ReducedYawFeedbackEvent(
                        retinal_measurement_time_s=motion.value.measurement_time_s,
                        retinal_availability_time_s=(
                            motion.value.availability_time_s
                            - config.motion_processing_latency_s
                        ),
                        motion_availability_time_s=motion.value.availability_time_s,
                        descending_availability_time_s=descending_available_s,
                        motor_availability_time_s=motor_available_s,
                        muscle_torque_availability_time_s=torque_available_s,
                        optic_flow_response=raw_drive,
                        nod1_surrogate_drive=nod1_drive,
                        requested_yaw_torque_n_m=requested_torque,
                    )
                )
                processed_motion = motion

            descending = descending_queue.advance(float(current_time_s))
            if descending is not None and descending is not processed_descending:
                motor_queue.push(
                    AvailableValue(
                        measurement_time_s=descending.measurement_time_s,
                        availability_time_s=(
                            descending.availability_time_s + config.motor_latency_s
                        ),
                        value=descending.value,
                    )
                )
                processed_descending = descending

            motor = motor_queue.advance(float(current_time_s))
            if motor is not None and motor is not processed_motor:
                torque_queue.push(
                    AvailableValue(
                        measurement_time_s=motor.measurement_time_s,
                        availability_time_s=(
                            motor.availability_time_s
                            + config.muscle_activation_latency_s
                        ),
                        value=float(
                            np.clip(
                                config.optic_flow_to_yaw_torque_n_m * motor.value,
                                -config.maximum_yaw_torque_n_m,
                                config.maximum_yaw_torque_n_m,
                            )
                        ),
                    )
                )
                processed_motor = motor

            torque = torque_queue.advance(float(current_time_s))
            optic_flow[index] = (
                0.0 if motion is None else motion.value.population_response
            )
            descending_drive[index] = 0.0 if descending is None else descending.value
            motor_drive[index] = 0.0 if motor is None else motor.value
            control_torque[index] = 0.0 if torque is None else torque.value
            disturbance_torque[index] = self._disturbance_torque(
                config, float(current_time_s)
            )

            if index == step_count:
                continue
            angular_acceleration_rad_s2 = (
                control_torque[index]
                + disturbance_torque[index]
                - config.passive_yaw_damping_n_m_s * yaw_rate[index]
            ) / config.yaw_inertia_kg_m2
            dt_s = config.physics_dt_s
            yaw[index + 1] = (
                yaw[index]
                + yaw_rate[index] * dt_s
                + 0.5 * angular_acceleration_rad_s2 * dt_s * dt_s
            )
            yaw_rate[index + 1] = (
                yaw_rate[index] + angular_acceleration_rad_s2 * dt_s
            )

        return ReducedYawLoopResult(
            time_s=time_s,
            yaw_world_rad=yaw,
            yaw_rate_rad_s=yaw_rate,
            optic_flow_response=optic_flow,
            descending_drive=descending_drive,
            motor_drive=motor_drive,
            control_yaw_torque_n_m=control_torque,
            disturbance_yaw_torque_n_m=disturbance_torque,
            feedback_events=tuple(events),
        )


__all__ = [
    "ReducedClosedLoopYawSimulator",
    "ReducedYawFeedbackEvent",
    "ReducedYawLoopConfig",
    "ReducedYawLoopResult",
    "YawTorquePulse",
]
