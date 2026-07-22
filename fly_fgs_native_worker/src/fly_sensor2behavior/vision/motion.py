"""Causal image-derived motion detection for self-supervised validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..schema import RetinalFrame, RetinalSignalKind


@dataclass(frozen=True)
class MotionFrame:
    measurement_time_s: float
    availability_time_s: float
    local_direction_response: Tuple[float, ...]
    population_response: float
    source_frame_count: int
    semantics: str = "causal_image_derived_reichardt_surrogate"


class ReichardtMotionDetector:
    """Minimal causal adjacent-receptor correlator.

    The detector sees only retinal samples and their timestamps.  It cannot
    access the analytic scene or its ground-truth velocity.  Its sign is a
    declared software convention and requires later physiological calibration.
    """

    def __init__(self, *, temporal_tau_s: float = 0.010, output_latency_s: float = 0.001) -> None:
        if not math.isfinite(temporal_tau_s) or temporal_tau_s <= 0.0:
            raise ValueError("temporal_tau_s must be finite and positive")
        if not math.isfinite(output_latency_s) or output_latency_s < 0.0:
            raise ValueError("output_latency_s must be finite and non-negative")
        self.temporal_tau_s = float(temporal_tau_s)
        self.output_latency_s = float(output_latency_s)
        self._last: Optional[RetinalFrame] = None
        self._delayed: Optional[np.ndarray] = None
        self._frame_count = 0

    def reset(self) -> None:
        self._last = None
        self._delayed = None
        self._frame_count = 0

    def update(self, frame: RetinalFrame, *, current_time_s: Optional[float] = None) -> MotionFrame:
        if frame.signal_kind is not RetinalSignalKind.NORMALIZED_LUMINANCE:
            raise ValueError("Reichardt detector currently requires normalized luminance")
        if current_time_s is not None and frame.availability_time_s > current_time_s + 1e-15:
            raise ValueError("retinal frame is not yet causally available")
        current = np.asarray(frame.samples, dtype=float)
        if self._last is not None:
            if frame.measurement_time_s <= self._last.measurement_time_s:
                raise ValueError("retinal frames must have increasing measurement times")
            if len(current) != len(self._last.samples):
                raise ValueError("retinal receptor count cannot change during a run")
        self._frame_count += 1
        if self._last is None:
            self._delayed = current.copy()
            local = np.zeros_like(current)
        else:
            dt_s = frame.measurement_time_s - self._last.measurement_time_s
            alpha = 1.0 - math.exp(-dt_s / self.temporal_tau_s)
            assert self._delayed is not None
            delayed = self._delayed + alpha * (np.asarray(self._last.samples) - self._delayed)
            # Opponent adjacent-pair correlation.  Periodic wrap is appropriate
            # only for this declared panoramic software-test retina.
            local = delayed * np.roll(current, -1) - np.roll(delayed, -1) * current
            self._delayed = delayed
        self._last = frame
        return MotionFrame(
            measurement_time_s=frame.measurement_time_s,
            availability_time_s=frame.availability_time_s + self.output_latency_s,
            local_direction_response=tuple(float(value) for value in local),
            population_response=float(np.mean(local)),
            source_frame_count=self._frame_count,
        )


__all__ = ["MotionFrame", "ReichardtMotionDetector"]
