"""Spatial geometry helpers for the retinotopy (family D) and cable-distance (family E)
analyses.

Synapse positions arrive in **nanometres** (the public ``synapses_nt_v1`` ``pt_position``
schema — verified empirically, see ``fw_access.POSITION_NM_PER_UNIT``). We convert to
micrometres once, here, and every downstream distance is in um. The paper's family-E
result is stated to be "sign-identical under a Euclidean metric", so the primary
distances here are Euclidean between synapse points; geodesic (skeleton) distance is an
optional upgrade handled separately.

``scaling_self_test`` asserts the nm->um conversion is sane by checking that VCH's input
synapse field spans a physically plausible size (tens of um), pinning the units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fw_access import POSITION_NM_PER_UNIT


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
def positions_to_um(xyz_nm: np.ndarray) -> np.ndarray:
    """(N,3) nanometre coords -> (N,3) micrometres (single /1000 scaling)."""
    return np.asarray(xyz_nm, dtype=float) * POSITION_NM_PER_UNIT / 1000.0


def syn_positions_um(df: pd.DataFrame, side: str) -> np.ndarray:
    """Extract the (N,3) um positions of the pre- or post-synaptic point of each synapse."""
    cols = [f"{side}_pt_position_{a}" for a in ("x", "y", "z")]
    return positions_to_um(df[cols].to_numpy())


# ---------------------------------------------------------------------------
# Centroids & patch radius (family D)
# ---------------------------------------------------------------------------
def centroid(points_um: np.ndarray) -> np.ndarray:
    """Mean position (um) of a synapse cloud."""
    return np.asarray(points_um, dtype=float).mean(axis=0)


def patch_radius_um(points_um: np.ndarray, metric: str = "median") -> float:
    """Spread of a synapse cloud about its centroid, in um.

    ``median`` (default) = median distance of points to their centroid — robust, and
    what the paper means by 'input-patch radius'. ``rms`` = root-mean-square distance.
    """
    pts = np.asarray(points_um, dtype=float)
    if len(pts) == 0:
        return float("nan")
    d = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    return float(np.sqrt((d ** 2).mean())) if metric == "rms" else float(np.median(d))


def frac_within(points_um: np.ndarray, radius_um: float) -> float:
    """Fraction of a synapse cloud lying within ``radius_um`` of its centroid."""
    pts = np.asarray(points_um, dtype=float)
    if len(pts) == 0:
        return float("nan")
    d = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    return float((d <= radius_um).mean())


# ---------------------------------------------------------------------------
# Synapse-to-synapse Euclidean distance (family E primary)
# ---------------------------------------------------------------------------
def min_pairwise_distance_um(points_a_um: np.ndarray, points_b_um: np.ndarray) -> float:
    """Minimum Euclidean distance (um) between two synapse clouds.

    Used for "how close is the nearest VCH->T4a synapse to a T4a->LLPC1 output
    terminal" — the closeness signature of presynaptic/terminal gating.
    """
    a = np.asarray(points_a_um, dtype=float)
    b = np.asarray(points_b_um, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # Chunked to bound memory if either cloud is large.
    best = np.inf
    for i in range(0, len(a), 2048):
        chunk = a[i:i + 2048]
        d = np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def median_nearest_distance_um(points_um: np.ndarray, target_um: np.ndarray) -> float:
    """Median over ``points`` of each point's distance to its NEAREST ``target`` point (um).

    This is the paper's "synapse -> nearest output terminal" metric: each input synapse
    is scored by its closest terminal, then the population is summarised by the median.
    Distinct from ``min_pairwise_distance_um`` (a single global minimum), which would be
    dominated by whichever lone synapse happens to sit near the terminal.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(points_um, dtype=float)
    tgt = np.asarray(target_um, dtype=float)
    if len(pts) == 0 or len(tgt) == 0:
        return float("nan")
    d, _ = cKDTree(tgt).query(pts)
    return float(np.median(d))


def two_field_separation_um(points_a_um: np.ndarray, points_b_um: np.ndarray) -> float:
    """Distance (um) between the centroids of two synapse clouds (e.g. motion vs form
    dendritic fields on one LLPC1)."""
    if len(points_a_um) == 0 or len(points_b_um) == 0:
        return float("nan")
    return float(np.linalg.norm(centroid(points_a_um) - centroid(points_b_um)))


def cloud_extent_um(points_um: np.ndarray) -> dict:
    """Bounding-box extent + centroid of a synapse cloud (a skeleton-free morphology proxy).

    Returns per-axis span and centroid in um. The input-synapse cloud of a tangential cell
    approximates its dendritic arbor; the output-synapse cloud approximates its axonal field.
    """
    pts = np.asarray(points_um, dtype=float)
    if len(pts) == 0:
        return {"n": 0}
    lo, hi, ctr = pts.min(0), pts.max(0), pts.mean(0)
    return {
        "n": int(len(pts)),
        "centroid": [round(float(v), 1) for v in ctr],
        "span_x": round(float(hi[0] - lo[0]), 1),   # medio-lateral
        "span_y": round(float(hi[1] - lo[1]), 1),   # dorso-ventral
        "span_z": round(float(hi[2] - lo[2]), 1),   # antero-posterior
    }


def fraction_beyond_x_um(points_um: np.ndarray, x_threshold_um: float, *, greater: bool) -> float:
    """Fraction of a cloud on one side of a medio-lateral (x) plane — e.g. fraction of axon
    output that has crossed the brain midline (skeleton-free laterality proxy)."""
    pts = np.asarray(points_um, dtype=float)
    if len(pts) == 0:
        return float("nan")
    m = pts[:, 0] > x_threshold_um if greater else pts[:, 0] < x_threshold_um
    return float(m.mean())


# ---------------------------------------------------------------------------
# Self-test (pins the units)
# ---------------------------------------------------------------------------
def scaling_self_test(src) -> dict:
    """Sanity-check voxel->um scaling against a real synapse cloud.

    Pulls VCH's input synapses and asserts the optic-lobe synapse cloud spans a
    plausible physical size (tens of um, not nm or mm). Returns the measured span so
    callers can log it; raises AssertionError on a gross units error.
    """
    VCH = 720575940627706398
    df = src.synapses(post_ids=[VCH])
    pts = syn_positions_um(df, "post")
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    # VCH is a centrifugal cell spanning the optic lobe and reaching the central brain;
    # its input field diagonal is ~hundreds of um. Demand 50-600 um (nm scaling); the
    # 4nm-voxel mis-scaling would give >1 mm and trip this.
    assert 50.0 < span < 600.0, f"nm->um scaling looks wrong: VCH input span = {span} um"
    # Median nearest-neighbour synapse spacing must be sub-micron-to-few-micron.
    from scipy.spatial import cKDTree
    sub = pts if len(pts) <= 3000 else pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
    nn = cKDTree(sub).query(sub, k=2)[0][:, 1]
    nn_med = float(np.median(nn))
    assert 0.2 < nn_med < 5.0, f"nm->um scaling looks wrong: NN synapse spacing = {nn_med} um"
    return {"vch_input_span_um": round(span, 1), "nn_spacing_um": round(nn_med, 3), "n_syn": int(len(df))}
