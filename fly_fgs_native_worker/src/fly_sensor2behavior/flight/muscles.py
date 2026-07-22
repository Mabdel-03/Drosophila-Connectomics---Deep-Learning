"""Reduced flight-muscle dynamics with explicit physiological distinctions.

These models are exploratory, literature-constrained reduced-order components;
they are not anatomical reconstructions.  In particular, asynchronous DLM/DVM
motor spikes drive a calcium/stretch state rather than one contraction or one
wingbeat per spike.  Steering output depends on spike phase, and tension muscle
activation changes thoracic resonance.

Relevant experimental constraints include Melis et al. (Nature 2024,
doi:10.1038/s41586-024-07293-4) and O'Sullivan et al. (Nature 2023,
doi:10.1038/s41586-023-06099-0).  Numerical defaults below remain exploratory
until fitted to a declared calibration dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _exp_relax(current: float, target: float, dt_s: float, tau_s: float) -> float:
    return target + (current - target) * np.exp(-dt_s / tau_s)


@dataclass(frozen=True)
class AsynchronousPowerParameters:
    calcium_tau_s: float = 0.045
    activation_tau_s: float = 0.012
    resting_calcium: float = 0.32
    calcium_per_spike: float = 0.18
    half_activation: float = 0.42
    hill_coefficient: float = 4.0
    stretch_gain: float = 0.32
    shortening_deactivation_gain: float = 0.05
    max_isometric_force_n: float = 90.0e-6

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, name)) for name in self.__dataclass_fields__)
        if not np.all(np.isfinite(values)):
            raise ValueError("asynchronous power parameters must be finite")
        if self.calcium_tau_s <= 0.0 or self.activation_tau_s <= 0.0:
            raise ValueError("asynchronous power time constants must be positive")
        if (
            self.resting_calcium < 0.0
            or self.calcium_per_spike < 0.0
            or self.half_activation <= 0.0
            or self.hill_coefficient <= 0.0
            or self.shortening_deactivation_gain < 0.0
            or self.max_isometric_force_n <= 0.0
        ):
            raise ValueError("asynchronous power parameters are outside valid ranges")


class AsynchronousPowerMuscle:
    """Calcium- and stretch-activated indirect flight muscle population."""

    def __init__(
        self,
        parameters: AsynchronousPowerParameters = AsynchronousPowerParameters(),
        initial_calcium: float = 0.48,
    ) -> None:
        self.parameters = parameters
        self.calcium = float(initial_calcium)
        self.activation = self._hill(self.calcium)
        self.force_n = parameters.max_isometric_force_n * self.activation

    def _hill(self, calcium: float) -> float:
        p = self.parameters
        numerator = max(calcium, 0.0) ** p.hill_coefficient
        denominator = numerator + p.half_activation ** p.hill_coefficient
        return numerator / denominator if denominator > 0.0 else 0.0

    def step(
        self,
        spike_count: int,
        stretch: float,
        shortening_velocity_s: float,
        dt_s: float,
        activation_bias: float = 0.0,
    ) -> float:
        if spike_count < 0 or dt_s <= 0.0:
            raise ValueError("spike_count must be non-negative and dt positive")
        p = self.parameters
        self.calcium = _exp_relax(
            self.calcium, p.resting_calcium, dt_s, p.calcium_tau_s
        )
        self.calcium += p.calcium_per_spike * float(spike_count)
        calcium_activation = self._hill(self.calcium)
        stretch_multiplier = max(0.05, 1.0 + p.stretch_gain * stretch)
        shortening_multiplier = max(
            0.05, 1.0 - p.shortening_deactivation_gain * max(shortening_velocity_s, 0.0)
        )
        target = np.clip(
            calcium_activation * stretch_multiplier * shortening_multiplier
            + activation_bias,
            0.0,
            1.5,
        )
        self.activation = _exp_relax(
            self.activation, float(target), dt_s, p.activation_tau_s
        )
        self.force_n = p.max_isometric_force_n * self.activation
        return self.force_n


@dataclass(frozen=True)
class SteeringParameters:
    calcium_tau_s: float = 0.010
    phase_memory_tau_s: float = 0.008
    calcium_per_spike: float = 0.25
    phase_impulse: float = 0.55
    tonic_tau_s: float = 0.020
    max_force_n: float = 35.0e-6

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, name)) for name in self.__dataclass_fields__)
        if not np.all(np.isfinite(values)):
            raise ValueError("steering parameters must be finite")
        if (
            self.calcium_tau_s <= 0.0
            or self.phase_memory_tau_s <= 0.0
            or self.tonic_tau_s <= 0.0
            or self.calcium_per_spike < 0.0
            or self.phase_impulse < 0.0
            or self.max_force_n <= 0.0
        ):
            raise ValueError("steering parameters are outside valid ranges")


class PhaseCodedSteeringMuscle:
    """Synchronous steering muscle with a signed phase-dependent effect."""

    def __init__(
        self,
        preferred_phase_rad: float,
        parameters: SteeringParameters = SteeringParameters(),
    ) -> None:
        self.preferred_phase_rad = float(preferred_phase_rad) % (2.0 * np.pi)
        self.parameters = parameters
        self.activation = 0.0
        self.phase_effect = 0.0
        self.force_n = 0.0

    def step(
        self,
        spike_phases_rad: Sequence[float],
        dt_s: float,
        tonic_drive: float = 0.0,
    ) -> float:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        p = self.parameters
        self.activation *= np.exp(-dt_s / p.calcium_tau_s)
        self.phase_effect *= np.exp(-dt_s / p.phase_memory_tau_s)
        for phase in spike_phases_rad:
            delta = float(phase) - self.preferred_phase_rad
            self.activation += p.calcium_per_spike
            # Phase changes a muscle's mechanical effect even though its force
            # and calcium remain non-negative.
            self.phase_effect += p.phase_impulse * np.cos(delta)
        tonic_target = float(np.clip(tonic_drive, 0.0, 1.0))
        self.activation = _exp_relax(
            self.activation, tonic_target, dt_s, p.tonic_tau_s
        )
        self.activation = float(np.clip(self.activation, 0.0, 1.5))
        self.phase_effect = float(np.clip(self.phase_effect, -1.5, 1.5))
        self.force_n = p.max_force_n * self.activation
        return self.phase_effect


@dataclass(frozen=True)
class TensionParameters:
    activation_tau_s: float = 0.025
    activation_per_spike: float = 0.18
    decay_tau_s: float = 0.060
    baseline_activation: float = 0.25
    resonance_gain: float = 0.10

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, name)) for name in self.__dataclass_fields__)
        if not np.all(np.isfinite(values)):
            raise ValueError("tension parameters must be finite")
        if (
            self.activation_tau_s <= 0.0
            or self.decay_tau_s <= 0.0
            or self.activation_per_spike < 0.0
            or not 0.0 <= self.baseline_activation <= 1.5
            or self.resonance_gain < 0.0
        ):
            raise ValueError("tension parameters are outside valid ranges")


class TensionMuscle:
    """Slow tension-muscle state that shifts thoracic resonance."""

    def __init__(
        self, parameters: TensionParameters = TensionParameters()
    ) -> None:
        self.parameters = parameters
        self.activation = parameters.baseline_activation
        self.excitation = 0.0

    def step(
        self,
        spike_count: int,
        dt_s: float,
        activation_bias: float = 0.0,
    ) -> float:
        if spike_count < 0 or dt_s <= 0.0:
            raise ValueError("spike_count must be non-negative and dt positive")
        p = self.parameters
        self.excitation *= np.exp(-dt_s / p.decay_tau_s)
        self.excitation += p.activation_per_spike * spike_count
        target = np.clip(
            p.baseline_activation + self.excitation + activation_bias,
            0.0,
            1.5,
        )
        self.activation = _exp_relax(
            self.activation, float(target), dt_s, p.activation_tau_s
        )
        self.activation = float(np.clip(self.activation, 0.0, 1.5))
        return 1.0 + p.resonance_gain * (
            self.activation - p.baseline_activation
        )


@dataclass(frozen=True)
class ThoraxOscillatorParameters:
    natural_frequency_hz: float = 200.0
    frequency_relaxation_s: float = 0.004
    power_frequency_gain: float = 0.05
    reference_power_activation: float = 0.62
    minimum_frequency_hz: float = 120.0
    maximum_frequency_hz: float = 260.0

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, name)) for name in self.__dataclass_fields__)
        if not np.all(np.isfinite(values)):
            raise ValueError("thorax oscillator parameters must be finite")
        if (
            self.frequency_relaxation_s <= 0.0
            or self.minimum_frequency_hz <= 0.0
            or self.maximum_frequency_hz < self.minimum_frequency_hz
            or not self.minimum_frequency_hz
            <= self.natural_frequency_hz
            <= self.maximum_frequency_hz
            or not 0.0 <= self.reference_power_activation <= 1.5
        ):
            raise ValueError("thorax oscillator parameters are outside valid ranges")


class ThoraxOscillator:
    """Autonomous stretch-activated thorax/wingbeat oscillator."""

    def __init__(
        self,
        parameters: ThoraxOscillatorParameters = ThoraxOscillatorParameters(),
        initial_phase_rad: float = 0.0,
    ) -> None:
        self.parameters = parameters
        self.phase_rad = float(initial_phase_rad)
        self.frequency_hz = parameters.natural_frequency_hz

    def step(
        self,
        power_activation: float,
        tension_resonance_scale: float,
        dt_s: float,
    ) -> float:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        p = self.parameters
        power_delta = power_activation - p.reference_power_activation
        target_frequency = p.natural_frequency_hz * (
            tension_resonance_scale + p.power_frequency_gain * power_delta
        )
        target_frequency = float(
            np.clip(target_frequency, p.minimum_frequency_hz, p.maximum_frequency_hz)
        )
        self.frequency_hz = _exp_relax(
            self.frequency_hz,
            target_frequency,
            dt_s,
            p.frequency_relaxation_s,
        )
        old_phase = self.phase_rad
        self.phase_rad += 2.0 * np.pi * self.frequency_hz * dt_s
        return old_phase
