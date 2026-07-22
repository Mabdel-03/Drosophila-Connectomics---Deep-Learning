"""Swappable virtual wing-hinge interface and exploratory implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .types import MuscleClass, MuscleSnapshot, Side, WingKinematics


class WingHinge(Protocol):
    """Boundary to a reduced hinge, anatomical hinge, or FlyBody adapter."""

    def evaluate(
        self,
        phase_rad: float,
        frequency_hz: float,
        muscles: MuscleSnapshot,
    ) -> WingKinematics:
        ...


@dataclass(frozen=True)
class VirtualWingHingeParameters:
    baseline_stroke_amplitude_rad: float = 2.05
    reference_power_activation: float = 0.62
    power_amplitude_exponent: float = 0.35
    minimum_amplitude_scale: float = 0.45
    maximum_amplitude_scale: float = 1.45
    steering_amplitude_gain: float = 0.10
    baseline_angle_of_attack_rad: float = np.deg2rad(45.0)
    steering_pitch_gain_rad: float = np.deg2rad(11.0)
    steering_deviation_gain_rad: float = np.deg2rad(7.0)
    muscle_torque_scale_n_m: float = 1.8e-9
    power_virtual_moment_arm_m: float = 55.0e-6
    steering_virtual_moment_arm_m: float = 45.0e-6
    # Local FlyBody wing axes are ordered yaw/stroke, roll/deviation,
    # pitch/rotation.  These coefficients are visible exploratory parameters;
    # they are not inferred from structural synapse counts.
    steering_axis_weights: tuple = (
        ("iv2", (0.70, 0.20, 0.10)),
        ("i1", (0.45, -0.15, 0.40)),
        ("iv1", (0.35, 0.45, -0.20)),
        ("b3", (0.55, -0.35, 0.10)),
        ("b1", (0.60, 0.00, 0.25)),
    )
    default_steering_axis_weights: tuple = (0.50, 0.00, 0.00)

    def __post_init__(self) -> None:
        scalar_names = (
            "baseline_stroke_amplitude_rad",
            "reference_power_activation",
            "power_amplitude_exponent",
            "minimum_amplitude_scale",
            "maximum_amplitude_scale",
            "steering_amplitude_gain",
            "baseline_angle_of_attack_rad",
            "steering_pitch_gain_rad",
            "steering_deviation_gain_rad",
            "muscle_torque_scale_n_m",
            "power_virtual_moment_arm_m",
            "steering_virtual_moment_arm_m",
        )
        if not np.all(
            np.isfinite([float(getattr(self, name)) for name in scalar_names])
        ):
            raise ValueError("virtual hinge scalar parameters must be finite")
        if (
            self.baseline_stroke_amplitude_rad <= 0.0
            or self.reference_power_activation <= 0.0
            or self.power_amplitude_exponent <= 0.0
            or self.minimum_amplitude_scale <= 0.0
            or self.maximum_amplitude_scale < self.minimum_amplitude_scale
            or self.muscle_torque_scale_n_m <= 0.0
            or self.power_virtual_moment_arm_m <= 0.0
            or self.steering_virtual_moment_arm_m <= 0.0
        ):
            raise ValueError("virtual hinge parameters are outside valid ranges")
        names = []
        for item in self.steering_axis_weights:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("steering axis weights must be name/vector pairs")
            name, weights = item
            if not isinstance(name, str) or not name or name.lower() in names:
                raise ValueError("steering axis weight names must be unique")
            parsed = np.asarray(weights, dtype=float)
            if parsed.shape != (3,) or not np.all(np.isfinite(parsed)):
                raise ValueError("steering axis weights must contain three finite values")
            names.append(name.lower())
        default = np.asarray(self.default_steering_axis_weights, dtype=float)
        if default.shape != (3,) or not np.all(np.isfinite(default)):
            raise ValueError("default steering axis weights must contain three finite values")


class VirtualWingHinge:
    """Reduced muscle/phase-to-wing map.

    The gains are explicit parameters so calibration can replace them without
    changing the simulator.  This component prescribes wing kinematics and is
    not presented as the sclerite-resolved hinge of Melis et al. (2024).
    """

    validation_status = "exploratory"

    def __init__(
        self, parameters: VirtualWingHingeParameters = VirtualWingHingeParameters()
    ) -> None:
        self.parameters = parameters
        self._steering_axis_weights = {
            str(name).lower(): np.asarray(weights, dtype=float)
            for name, weights in parameters.steering_axis_weights
        }
        for name, weights in self._steering_axis_weights.items():
            if weights.shape != (3,) or not np.all(np.isfinite(weights)):
                raise ValueError("invalid steering axis weights for %s" % name)

    def evaluate(
        self,
        phase_rad: float,
        frequency_hz: float,
        muscles: MuscleSnapshot,
    ) -> WingKinematics:
        p = self.parameters
        power = np.array([muscles.power_left, muscles.power_right], dtype=float)
        steering = np.array(
            [muscles.steering_left, muscles.steering_right], dtype=float
        )
        normalized_power = np.maximum(power, 1e-9) / p.reference_power_activation
        amplitude_scale = np.power(normalized_power, p.power_amplitude_exponent)
        amplitude_scale = np.clip(
            amplitude_scale, p.minimum_amplitude_scale, p.maximum_amplitude_scale
        )
        amplitude_scale *= np.clip(
            1.0 + p.steering_amplitude_gain * steering, 0.75, 1.25
        )
        amplitude = p.baseline_stroke_amplitude_rad * amplitude_scale
        omega = 2.0 * np.pi * frequency_hz
        stroke = amplitude * np.sin(phase_rad)
        stroke_velocity = amplitude * omega * np.cos(phase_rad)
        stroke_acceleration = -amplitude * omega * omega * np.sin(phase_rad)

        # Phase effect is signed. Mirrored musculature is handled by keeping
        # anatomical left/right channels separate rather than silently flipping
        # dataset labels here.
        angle_of_attack = np.clip(
            p.baseline_angle_of_attack_rad + p.steering_pitch_gain_rad * steering,
            np.deg2rad(15.0),
            np.deg2rad(75.0),
        )
        deviation = p.steering_deviation_gain_rad * steering * np.sin(
            phase_rad + np.pi / 2.0
        )
        torque = (
            p.muscle_torque_scale_n_m
            * power
            * np.cos(phase_rad)
            * (1.0 + 0.2 * steering)
        )
        axis_torque = np.zeros(6, dtype=float)
        for state in muscles.individual.values():
            side_offset = 0 if state.side is Side.LEFT else 3
            if state.muscle_class is MuscleClass.ASYNCHRONOUS_POWER:
                moment_arm = (
                    state.virtual_moment_arm_m
                    if state.virtual_moment_arm_m is not None
                    else p.power_virtual_moment_arm_m
                )
                local = np.array(
                    [state.force_n * moment_arm * np.cos(phase_rad), 0.0, 0.0],
                    dtype=float,
                )
            elif state.muscle_class is MuscleClass.STEERING:
                moment_arm = (
                    state.virtual_moment_arm_m
                    if state.virtual_moment_arm_m is not None
                    else p.steering_virtual_moment_arm_m
                )
                weights = self._steering_axis_weights.get(
                    state.muscle.lower(),
                    np.asarray(p.default_steering_axis_weights, dtype=float),
                )
                local = state.force_n * moment_arm * state.phase_effect * weights
            else:
                # Tension muscles alter resonance through the thorax oscillator;
                # the first virtual hinge does not invent a direct wing torque.
                local = np.zeros(3, dtype=float)
            axis_torque[side_offset : side_offset + 3] += local
        return WingKinematics(
            phase_rad=float(phase_rad),
            frequency_hz=float(frequency_hz),
            stroke_rad=stroke,
            stroke_velocity_rad_s=stroke_velocity,
            stroke_acceleration_rad_s2=stroke_acceleration,
            angle_of_attack_rad=angle_of_attack,
            deviation_rad=deviation,
            generalized_torque_n_m=torque,
            wing_axis_torque_n_m=axis_torque,
        )
