"""Versioned anti-aliasing and named torque signal products.

The authoritative channel is formed offline from every completed MuJoCo
internal step.  Filtering is linear phase, delay compensated, and never
overwrites the raw sample table.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


PAPER_SIGNAL_SCHEMA_VERSION = "paper_torque_signal.v2"


@dataclass(frozen=True)
class PaperFIRSpecification:
    input_rate_hz: int
    output_rate_hz: int = 1000
    passband_edge_hz: float = 350.0
    stopband_edge_hz: float = 500.0
    maximum_passband_ripple_db: float = 0.01
    minimum_stopband_attenuation_db: float = 100.0
    taps: int = 0
    group_delay_input_samples: int = 0
    window: str = "kaiser"

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def design_paper_antialias_fir(
    input_rate_hz: int,
    *,
    output_rate_hz: int = 1000,
    passband_edge_hz: float = 350.0,
    stopband_edge_hz: float = 500.0,
    attenuation_db: float = 100.0,
) -> Tuple[np.ndarray, PaperFIRSpecification]:
    """Design the released odd-length Kaiser low-pass FIR."""

    if (
        isinstance(input_rate_hz, bool)
        or not isinstance(input_rate_hz, int)
        or input_rate_hz <= 0
    ):
        raise ValueError("input_rate_hz must be a positive integer")
    if input_rate_hz % output_rate_hz:
        raise ValueError("raw and synchronized rates must have an integer ratio")
    if not 0.0 < passband_edge_hz < stopband_edge_hz <= output_rate_hz / 2.0:
        raise ValueError("FIR edges are inconsistent with the output Nyquist rate")
    transition_hz = stopband_edge_hz - passband_edge_hz
    normalized_angular_width = 2.0 * math.pi * transition_hz / input_rate_hz
    # Kaiser order estimates are asymptotic.  A three-decibel design margin
    # makes the measured response, rather than the approximation, satisfy the
    # released >=100 dB stopband contract on both 40 and 80 kHz grids.
    design_attenuation_db = attenuation_db + 3.0
    order = int(
        math.ceil(
            (design_attenuation_db - 8.0)
            / (2.285 * normalized_angular_width)
        )
    )
    tap_count = order + 1
    if tap_count % 2 == 0:
        tap_count += 1
    beta = 0.1102 * (design_attenuation_db - 8.7)
    center = (tap_count - 1) / 2.0
    samples = np.arange(tap_count, dtype=float) - center
    cutoff_hz = 0.5 * (passband_edge_hz + stopband_edge_hz)
    taps = (
        2.0
        * cutoff_hz
        / input_rate_hz
        * np.sinc(2.0 * cutoff_hz / input_rate_hz * samples)
        * np.kaiser(tap_count, beta)
    )
    taps /= np.sum(taps)
    spec = PaperFIRSpecification(
        input_rate_hz=input_rate_hz,
        output_rate_hz=output_rate_hz,
        passband_edge_hz=passband_edge_hz,
        stopband_edge_hz=stopband_edge_hz,
        maximum_passband_ripple_db=0.01,
        minimum_stopband_attenuation_db=attenuation_db,
        taps=tap_count,
        group_delay_input_samples=(tap_count - 1) // 2,
    )
    return taps, spec


def measure_fir_response(
    taps: Sequence[float],
    sample_rate_hz: float,
    *,
    passband_edge_hz: float = 350.0,
    stopband_edge_hz: float = 500.0,
    fft_points: int = 1 << 20,
) -> Mapping[str, float]:
    """Return reproducible passband and stopband acceptance measurements."""

    coefficients = np.asarray(taps, dtype=float)
    response = np.fft.rfft(coefficients, n=fft_points)
    frequency = np.fft.rfftfreq(fft_points, d=1.0 / sample_rate_hz)
    magnitude = np.abs(response)
    passband = magnitude[frequency <= passband_edge_hz]
    stopband = magnitude[frequency >= stopband_edge_hz]
    ripple_db = 20.0 * math.log10(
        max(float(np.max(passband)), 1.0e-300)
        / max(float(np.min(passband)), 1.0e-300)
    )
    attenuation_db = -20.0 * math.log10(
        max(float(np.max(stopband)), 1.0e-300)
    )
    return {
        "passband_ripple_db": ripple_db,
        "stopband_attenuation_db": attenuation_db,
    }


def _fft_convolve_same(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    count = len(signal) + len(kernel) - 1
    fft_count = 1 << (count - 1).bit_length()
    full = np.fft.irfft(
        np.fft.rfft(signal, fft_count) * np.fft.rfft(kernel, fft_count),
        fft_count,
    )[:count]
    start = (len(kernel) - 1) // 2
    return full[start : start + len(signal)]


def synchronize_raw_torque(
    raw_time_s: Sequence[float],
    raw_torque_n_m: np.ndarray,
    *,
    trial_duration_s: float,
    input_rate_hz: int,
    output_rate_hz: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Filter raw padded trials and sample exact synchronized centers."""

    time = np.asarray(raw_time_s, dtype=float)
    torque = np.asarray(raw_torque_n_m, dtype=float)
    if time.ndim != 1 or torque.ndim not in (1, 2):
        raise ValueError("raw torque must be one- or two-dimensional")
    if torque.shape[-1] != len(time):
        raise ValueError("raw torque and time dimensions disagree")
    if len(time) < 2 or not np.all(np.isfinite(time)) or not np.all(np.isfinite(torque)):
        raise ValueError("raw torque samples must be finite and non-empty")
    observed_dt = np.diff(time)
    expected_dt = 1.0 / input_rate_hz
    if not np.allclose(observed_dt, expected_dt, rtol=0.0, atol=1.0e-11):
        raise ValueError("raw completed-state timestamps are not uniform")
    taps, specification = design_paper_antialias_fir(
        input_rate_hz, output_rate_hz=output_rate_hz
    )
    response = measure_fir_response(taps, input_rate_hz)
    if response["passband_ripple_db"] > specification.maximum_passband_ripple_db:
        raise ValueError("anti-alias FIR passband ripple exceeds its contract")
    if response["stopband_attenuation_db"] < specification.minimum_stopband_attenuation_db:
        raise ValueError("anti-alias FIR stopband attenuation misses its contract")
    rows = torque[None, :] if torque.ndim == 1 else torque
    filtered = np.stack(
        [_fft_convolve_same(row, taps) for row in rows], axis=0
    )
    synchronized_time = np.arange(
        0.0,
        trial_duration_s - 0.5 / output_rate_hz,
        1.0 / output_rate_hz,
        dtype=float,
    )
    synchronized = np.stack(
        [np.interp(synchronized_time, time, row) for row in filtered], axis=0
    )
    if torque.ndim == 1:
        synchronized = synchronized[0]
    receipt: Dict[str, Any] = {
        "schema_version": PAPER_SIGNAL_SCHEMA_VERSION,
        "method": "linear_phase_kaiser_polyphase_equivalent_offline_fir",
        "group_delay_compensated": True,
        "sample_timestamp_semantics": "completed integration state",
        "sample_center_timestamps_explicit": True,
        "fir": specification.to_dict(),
        "measured_response": dict(response),
    }
    return synchronized_time, synchronized, receipt


def _zero_phase_lowpass(signal: np.ndarray, rate_hz: int, cutoff_hz: float) -> np.ndarray:
    tap_count = 401
    center = (tap_count - 1) / 2.0
    samples = np.arange(tap_count, dtype=float) - center
    taps = (
        2.0
        * cutoff_hz
        / rate_hz
        * np.sinc(2.0 * cutoff_hz / rate_hz * samples)
        * np.hanning(tap_count)
    )
    taps /= np.sum(taps)
    rows = signal[None, :] if signal.ndim == 1 else signal
    output = np.stack([_fft_convolve_same(row, taps) for row in rows], axis=0)
    return output[0] if signal.ndim == 1 else output


def named_torque_products(
    synchronized_torque_n_m: np.ndarray,
    *,
    synchronized_rate_hz: int = 1000,
    wingbeat_frequency_hz: float = 200.0,
) -> Mapping[str, np.ndarray]:
    """Produce fixed sensitivity channels without choosing a best match."""

    torque = np.asarray(synchronized_torque_n_m, dtype=float)
    products: Dict[str, np.ndarray] = {
        "paper_comparison": torque.copy(),
    }
    for cutoff in (10.0, 25.0, 50.0):
        products["lowpass_{}hz".format(int(cutoff))] = _zero_phase_lowpass(
            torque, synchronized_rate_hz, cutoff
        )
    window = int(round(synchronized_rate_hz / wingbeat_frequency_hz))
    if window < 1:
        raise ValueError("wingbeat frequency exceeds synchronized sample rate")
    kernel = np.ones(window, dtype=float) / window
    rows = torque[None, :] if torque.ndim == 1 else torque
    averaged = np.stack(
        [_fft_convolve_same(row, kernel) for row in rows], axis=0
    )
    products["wingbeat_averaged"] = averaged[0] if torque.ndim == 1 else averaged
    return products


__all__ = [
    "PAPER_SIGNAL_SCHEMA_VERSION",
    "PaperFIRSpecification",
    "design_paper_antialias_fir",
    "measure_fir_response",
    "named_torque_products",
    "synchronize_raw_torque",
]
