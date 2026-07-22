"""Deterministic one-dimensional analytic visual scenes.

The scene classes return normalized luminance only.  They are intended for
software and causal-model validation; they do not model radiometry, optics,
photon noise, or a calibrated Drosophila eye.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


_TWO_PI = 2.0 * math.pi


def wrap_angle_rad(value):
    """Wrap scalar/array angles to ``[-pi, pi)`` deterministically."""

    return (np.asarray(value, dtype=float) + math.pi) % _TWO_PI - math.pi


class AnalyticScene(Protocol):
    def luminance(self, azimuth_world_rad: np.ndarray, time_s: float) -> np.ndarray:
        ...


def _validate_luminance_parameters(mean: float, contrast: float) -> None:
    if not math.isfinite(mean) or not 0.0 <= mean <= 1.0:
        raise ValueError("mean_luminance must lie in [0, 1]")
    if not math.isfinite(contrast) or not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must lie in [0, 1]")
    if mean * (1.0 - contrast) < 0.0 or mean * (1.0 + contrast) > 1.0:
        raise ValueError("mean_luminance and contrast must remain in [0, 1]")


@dataclass(frozen=True)
class UniformScene:
    luminance_level: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.luminance_level) or not 0.0 <= self.luminance_level <= 1.0:
            raise ValueError("luminance_level must lie in [0, 1]")

    def luminance(self, azimuth_world_rad: np.ndarray, time_s: float) -> np.ndarray:
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        return np.full_like(np.asarray(azimuth_world_rad, dtype=float), self.luminance_level)


@dataclass(frozen=True)
class AnalyticGratingScene:
    """A sinusoidal panoramic grating translated at a declared angular speed."""

    spatial_frequency_cycles_per_rad: float = 2.0
    angular_velocity_rad_s: float = 0.0
    phase_rad: float = 0.0
    mean_luminance: float = 0.5
    contrast: float = 0.8

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.spatial_frequency_cycles_per_rad)
            or self.spatial_frequency_cycles_per_rad <= 0.0
        ):
            raise ValueError("spatial frequency must be finite and positive")
        if not math.isfinite(self.angular_velocity_rad_s):
            raise ValueError("angular velocity must be finite")
        if not math.isfinite(self.phase_rad):
            raise ValueError("phase_rad must be finite")
        _validate_luminance_parameters(self.mean_luminance, self.contrast)

    def luminance(self, azimuth_world_rad: np.ndarray, time_s: float) -> np.ndarray:
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        phase = _TWO_PI * self.spatial_frequency_cycles_per_rad * (
            np.asarray(azimuth_world_rad, dtype=float)
            - self.angular_velocity_rad_s * time_s
        ) + (self.phase_rad % _TWO_PI)
        return self.mean_luminance * (1.0 + self.contrast * np.sin(phase))


@dataclass(frozen=True)
class FigureGroundScene:
    """Continuous-onset figure over a moving sinusoidal background.

    Before ``figure_onset_s`` the figure centre is advected with the ground.
    At onset it changes velocity without changing position, preventing the
    nonphysical position jump in the audited legacy implementation.
    """

    background_velocity_rad_s: float = 0.0
    figure_velocity_rad_s: float = 1.0
    figure_onset_s: float = 0.0
    figure_initial_center_rad: float = 0.0
    figure_width_rad: float = 0.6
    spatial_frequency_cycles_per_rad: float = 2.0
    phase_rad: float = 0.0
    mean_luminance: float = 0.5
    contrast: float = 0.8

    def __post_init__(self) -> None:
        finite = (
            self.background_velocity_rad_s,
            self.figure_velocity_rad_s,
            self.figure_onset_s,
            self.figure_initial_center_rad,
            self.figure_width_rad,
            self.spatial_frequency_cycles_per_rad,
            self.phase_rad,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("figure-ground parameters must be finite")
        if self.figure_onset_s < 0.0:
            raise ValueError("figure_onset_s must be non-negative")
        if not 0.0 < self.figure_width_rad <= _TWO_PI:
            raise ValueError("figure_width_rad must lie in (0, 2*pi]")
        if self.spatial_frequency_cycles_per_rad <= 0.0:
            raise ValueError("spatial frequency must be positive")
        _validate_luminance_parameters(self.mean_luminance, self.contrast)

    def figure_center_rad(self, time_s: float) -> float:
        if time_s <= self.figure_onset_s:
            return self.figure_initial_center_rad + self.background_velocity_rad_s * time_s
        onset_position = (
            self.figure_initial_center_rad
            + self.background_velocity_rad_s * self.figure_onset_s
        )
        return onset_position + self.figure_velocity_rad_s * (
            time_s - self.figure_onset_s
        )

    def luminance(self, azimuth_world_rad: np.ndarray, time_s: float) -> np.ndarray:
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        azimuth = np.asarray(azimuth_world_rad, dtype=float)
        ground_coordinate = azimuth - self.background_velocity_rad_s * time_s
        if time_s >= self.figure_onset_s:
            centre = self.figure_center_rad(time_s)
            inside = np.abs(wrap_angle_rad(azimuth - centre)) <= 0.5 * self.figure_width_rad
            # Anchor both centre and texture phase to the background at onset,
            # then advect the figure at its independent velocity.
            figure_coordinate = (
                azimuth
                - self.background_velocity_rad_s * self.figure_onset_s
                - self.figure_velocity_rad_s * (time_s - self.figure_onset_s)
            )
            coordinate = np.where(inside, figure_coordinate, ground_coordinate)
        else:
            coordinate = ground_coordinate
        phase = (
            _TWO_PI * self.spatial_frequency_cycles_per_rad * coordinate
            + (self.phase_rad % _TWO_PI)
        )
        result = self.mean_luminance * (1.0 + self.contrast * np.sin(phase))
        return result


@dataclass(frozen=True)
class RandomColumnScene:
    """Independently sampled panoramic columns with a stable seeded texture."""

    seed: int
    column_count: int = 64
    angular_velocity_rad_s: float = 0.0
    low_luminance: float = 0.1
    high_luminance: float = 0.9

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.column_count < 2:
            raise ValueError("column_count must be at least two")
        if not math.isfinite(self.angular_velocity_rad_s):
            raise ValueError("angular_velocity_rad_s must be finite")
        if not 0.0 <= self.low_luminance < self.high_luminance <= 1.0:
            raise ValueError("column luminance bounds must satisfy 0 <= low < high <= 1")

    @property
    def texture(self) -> np.ndarray:
        generator = np.random.default_rng(self.seed)
        choices = generator.integers(0, 2, size=self.column_count)
        return np.where(choices == 0, self.low_luminance, self.high_luminance)

    def luminance(self, azimuth_world_rad: np.ndarray, time_s: float) -> np.ndarray:
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        phase = (
            wrap_angle_rad(
                np.asarray(azimuth_world_rad, dtype=float)
                - self.angular_velocity_rad_s * time_s
            )
            + math.pi
        ) / _TWO_PI
        indices = np.floor(phase * self.column_count).astype(int) % self.column_count
        return self.texture[indices]


__all__ = [
    "AnalyticGratingScene",
    "AnalyticScene",
    "FigureGroundScene",
    "RandomColumnScene",
    "UniformScene",
    "wrap_angle_rad",
]
