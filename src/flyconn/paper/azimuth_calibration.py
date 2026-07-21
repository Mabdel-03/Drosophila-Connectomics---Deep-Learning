"""Absolute (p, q) hex-lattice -> visual azimuth/elevation calibration (SECONDARY evidence).

The differential RF test (``geometry_hex.differential_rf``) is the PRIMARY, calibration-free
gate. This module supplies a *corroborating* absolute-degree reading so the FD3 receptive
field can be quoted against Egelhaaf's numbers (peak 40-50 deg, frontal margin ~20 deg,
lateral reach ~100 deg, half-max width ~62 deg). There is **no ground-truth degree field
anywhere in the data** — only the hex (p, q) lattice — so this map is an explicit model with
a stated error budget, and a claim built on it can never exceed CONFIRMED_WITH_CAVEAT.

Model: along the principal azimuth axis (p), assume an approximately LINEAR column-index ->
azimuth mapping anchored at two landmarks: the frontal eye margin (most-frontal column p_lo
~ -10 deg, the contralateral edge of the monocular field per Beersma et al. 1977 / Egelhaaf
FD1 frontal boundary) and the caudal pole (most-lateral column p_hi ~ +170 deg). The slope
is (az_hi - az_lo) / (p_hi - p_lo). Elevation (q) is treated coarsely the same way over a
~+/-90 deg band; only azimuth is used for FD3's horizontal RF claims.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Eye-coordinate landmarks (degrees). Azimuth 0 = frontal midline; + = lateral/ipsilateral.
# Frontal margin slightly contralateral (~-10 deg) and caudal pole ~+170 deg — the fly
# monocular horizontal field spans ~180 deg (Beersma, Stavenga & Kuiper 1977; consistent
# with Egelhaaf 1985 FD1 frontal boundary ~-10 deg and FD4 reach beyond +120 deg).
AZ_FRONTAL_DEG = -10.0
AZ_CAUDAL_DEG = 170.0


@lru_cache(maxsize=1)
def _p_extent() -> tuple[float, float]:
    """(p_lo, p_hi) — the frontal and caudal extremes of the lattice p-axis.

    Taken from the actual column lattice so the anchor columns are data-derived, not
    guessed. p_lo (most frontal) maps to AZ_FRONTAL_DEG, p_hi (most caudal) to AZ_CAUDAL_DEG.
    """
    from .geometry_hex import _pq_map

    pq = _pq_map()
    return (float(pq["p"].min()), float(pq["p"].max()))


def pq_to_azimuth(p: float | np.ndarray, q=None, eye=None) -> float | np.ndarray:
    """Map hex column p -> approximate visual azimuth in degrees (linear two-anchor model).

    ``q``/``eye`` accepted for signature symmetry but unused (azimuth is p-driven).
    """
    p_lo, p_hi = _p_extent()
    if p_hi == p_lo:
        return np.nan if np.ndim(p) == 0 else np.full_like(np.asarray(p, float), np.nan)
    slope = (AZ_CAUDAL_DEG - AZ_FRONTAL_DEG) / (p_hi - p_lo)
    return AZ_FRONTAL_DEG + (np.asarray(p, dtype=float) - p_lo) * slope


def width_p_to_degrees(width_p: float) -> float:
    """Convert a p-axis FWHM (lattice units) to degrees under the linear azimuth model."""
    p_lo, p_hi = _p_extent()
    if p_hi == p_lo:
        return float("nan")
    slope = (AZ_CAUDAL_DEG - AZ_FRONTAL_DEG) / (p_hi - p_lo)
    return float(abs(width_p) * slope)


def calibration_error_budget() -> dict:
    """The honest uncertainty on the absolute-degree map (printed verbatim in claim notes).

    Combined in quadrature. The linear-model term dominates because real ommatidial
    sampling is non-uniform (denser frontally), so a linear column->azimuth map is least
    accurate mid-field.
    """
    terms = {
        "linearity_deg": 12.5,    # non-uniform sampling vs linear model (+/-10-15)
        "landmark_deg": 6.5,      # frontal-margin / caudal-pole column identification (+/-5-8)
        "eye_span_deg": 7.5,      # literature FOV spread 150-180 deg (+/-5-10)
        "dropout_deg": 1.5,       # T4/T5 (p,q) dropout centroid bias (+/-1-2)
    }
    combined = float(np.sqrt(sum(v ** 2 for v in terms.values())))
    return {**terms, "combined_deg": round(combined, 1),
            "model": "linear two-anchor p->azimuth; azimuth is p-driven, q->elevation coarse",
            "verdict_ceiling": "CONFIRMED_WITH_CAVEAT (no ground-truth degree field exists)"}
