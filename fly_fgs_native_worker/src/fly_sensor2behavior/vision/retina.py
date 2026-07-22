"""Causal normalized-luminance sampling on a declared ommatidial lattice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..schema import (
    Confidence,
    ConfidenceLevel,
    EvidenceTier,
    EyeSide,
    Provenance,
    RetinalFrame,
    RetinalSignalKind,
)
from .stimulus import AnalyticScene


@dataclass(frozen=True)
class PanoramicRetina:
    """A deterministic 1-D test retina, not a calibrated compound-eye model."""

    receptor_count: int = 64
    field_of_view_rad: float = 2.0 * math.pi
    sensor_latency_s: float = 0.001
    integration_subsamples: int = 5

    def __post_init__(self) -> None:
        if self.receptor_count < 4:
            raise ValueError("receptor_count must be at least four")
        if not math.isfinite(self.field_of_view_rad) or not 0.0 < self.field_of_view_rad <= 2.0 * math.pi:
            raise ValueError("field_of_view_rad must lie in (0, 2*pi]")
        if not math.isfinite(self.sensor_latency_s) or self.sensor_latency_s < 0.0:
            raise ValueError("sensor_latency_s must be finite and non-negative")
        if self.integration_subsamples < 1:
            raise ValueError("integration_subsamples must be positive")

    @property
    def azimuth_body_rad(self) -> np.ndarray:
        half = 0.5 * self.field_of_view_rad
        return np.linspace(
            -half,
            half,
            self.receptor_count,
            endpoint=False,
            dtype=float,
        )

    @property
    def directions_body(self) -> Tuple[Tuple[float, float, float], ...]:
        return tuple(
            (float(math.cos(angle)), float(math.sin(angle)), 0.0)
            for angle in self.azimuth_body_rad
        )

    def sample(
        self,
        scene: AnalyticScene,
        *,
        exposure_start_s: float,
        exposure_end_s: float,
        body_yaw_rad: float = 0.0,
        eye_side: EyeSide = EyeSide.BINOCULAR,
    ) -> RetinalFrame:
        if not math.isfinite(exposure_start_s) or exposure_start_s < 0.0:
            raise ValueError("exposure_start_s must be finite and non-negative")
        if not math.isfinite(exposure_end_s) or exposure_end_s <= exposure_start_s:
            raise ValueError("exposure_end_s must be greater than exposure_start_s")
        if not math.isfinite(body_yaw_rad):
            raise ValueError("body_yaw_rad must be finite")
        times = np.linspace(
            exposure_start_s,
            exposure_end_s,
            self.integration_subsamples,
            dtype=float,
        )
        azimuth_world = self.azimuth_body_rad + body_yaw_rad
        samples = np.mean(
            np.vstack([scene.luminance(azimuth_world, float(time_s)) for time_s in times]),
            axis=0,
        )
        confidence = Confidence(
            tier=EvidenceTier.MODEL_INFERENCE,
            level=ConfidenceLevel.LOW,
            score=0.2,
            basis="analytic 1-D software-validation retina; optics/radiometry uncalibrated",
        )
        provenance = Provenance(
            source_uri="urn:fly-sensor2behavior:vision:analytic-retina:v1",
            method="finite-exposure normalized-luminance sampling",
            notes=(
                "Image-derived samples; downstream motion code is not given the scene's "
                "declared angular velocity."
            ),
        )
        return RetinalFrame(
            measurement_time_s=0.5 * (exposure_start_s + exposure_end_s),
            availability_time_s=exposure_end_s + self.sensor_latency_s,
            exposure_start_s=exposure_start_s,
            exposure_end_s=exposure_end_s,
            eye_side=eye_side,
            signal_kind=RetinalSignalKind.NORMALIZED_LUMINANCE,
            unit="1",
            samples=tuple(float(value) for value in samples),
            ommatidial_directions_body=self.directions_body,
            provenance=provenance,
            confidence=confidence,
            calibration_id="analytic-panorama-v1-uncalibrated",
        )


__all__ = ["PanoramicRetina"]
