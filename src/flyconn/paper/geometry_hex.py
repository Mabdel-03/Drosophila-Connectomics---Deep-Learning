"""Hex-lattice receptive-field helpers for the FD3 / Family-K analysis.

Where ``geometry.py`` works in nanometre/micrometre Euclidean space (families D, E),
this module works in the optic-lobe **retinotopic hex lattice** (p, q) — the natural frame
for a *receptive field*. A lobula-plate tangential cell (Nod / FD type) has no (p, q) of
its own, but its presynaptic **T4/T5 inputs do** (``retinotopy_columns.parquet`` covers
100% of T4/T5). So the cell's receptive field is the synapse-weighted (p, q) cloud of the
T4/T5 cells that drive it.

Axis convention (pinned by the ``assert_frontal`` self-test, analogous to
``geometry.scaling_self_test``): the principal **p** axis is the azimuth proxy and **q**
the elevation proxy. The sign of p is fixed empirically by Nod1 (= Egelhaaf FD1, known to
be FRONTAL) landing at negative p; therefore **more lateral = more positive p**. Every
differential test below is relational (lateral-of / wider-than / gap-relative-to the FD1
anchor), so it is invariant to any monotone (p, q) -> visual-angle warp; the absolute
degree map lives separately in ``azimuth_calibration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import fw_access as FW
from .derive import common as CM


# ---------------------------------------------------------------------------
# Retinotopy lookup (root_id -> p, q, eye), cached once per process.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _pq_map() -> pd.DataFrame:
    """root_id-indexed [p, q, eye] from the FlyWire visual-columns parquet.

    Raises FileNotFoundError (clear message) if the columns map was never fetched — the
    differential RF test requires the faithful hex map, not the pos_grid fallback.
    """
    from ..data_prep.retinotopy import _load_columns_map

    cmap = _load_columns_map()
    if cmap is None:
        raise FileNotFoundError(
            "geometry_hex needs retinotopy_columns.parquet (root_id->p,q,eye); "
            "run flyconn.data_prep.retinotopy.fetch_visual_columns() first."
        )
    return cmap.set_index("root_id")[["p", "q", "eye"]]


# ---------------------------------------------------------------------------
# Build a cell's receptive-field point cloud from its T4/T5 inputs.
# ---------------------------------------------------------------------------
@dataclass
class RFCloud:
    """A synapse-weighted (p, q) receptive-field cloud for one cell (or cell side)."""

    p: np.ndarray              # per-input-column hex p
    q: np.ndarray              # per-input-column hex q
    w: np.ndarray              # synapse weight (T4/T5 -> cell synapses)
    n_inputs: int              # number of distinct T4/T5 inputs with a (p,q)
    n_inputs_total: int        # T4/T5 inputs before dropping those lacking (p,q)
    total_syn: int             # total T4/T5 synapse weight retained
    dropout_frac: float        # fraction of T4/T5-input synapse weight lost to missing (p,q)

    @property
    def ok(self) -> bool:
        return self.n_inputs >= 2 and self.total_syn > 0


def input_cell_pq(src, meta, post_root: int, syn=None) -> RFCloud:
    """Synapse-weighted (p, q) cloud of the canonical T4/T5 cells presynaptic to ``post_root``.

    Reuses the family helpers: pull input synapses, count per presynaptic partner
    (``partner_counts``), attach metadata + T4/T5 flags (``attach_meta``), keep canonical
    T4/T5, then join each input's (p, q) from the retinotopy map. Synapse-weighted.

    ``syn`` (optional): a pre-pulled input-synapse frame (e.g. the cell type's inputs pulled
    once) to filter to this cell, avoiding a redundant per-cell pull. If None, pulls here.
    """
    if syn is None:
        syn = src.synapses(post_ids=[int(post_root)])
    syn = syn[syn["post_pt_root_id"] == int(post_root)]
    counts = CM.attach_meta(CM.partner_counts(syn, "pre_pt_root_id"), meta)
    t45 = counts[counts["is_t4t5"]].copy()
    n_total = int(t45["syn"].sum())
    if t45.empty:
        return RFCloud(np.array([]), np.array([]), np.array([]), 0, 0, 0, float("nan"))
    pq = _pq_map()
    j = t45.join(pq, on="root_id", how="inner").dropna(subset=["p", "q"])
    kept_syn = int(j["syn"].sum())
    dropout = 1.0 - (kept_syn / n_total) if n_total else float("nan")
    return RFCloud(
        p=j["p"].to_numpy(dtype=float),
        q=j["q"].to_numpy(dtype=float),
        w=j["syn"].to_numpy(dtype=float),
        n_inputs=int(len(j)),
        n_inputs_total=int(len(t45)),
        total_syn=kept_syn,
        dropout_frac=float(dropout),
    )


# ---------------------------------------------------------------------------
# Cloud statistics (hex analogs of geometry.centroid / patch_radius / frac_within).
# ---------------------------------------------------------------------------
def pq_centroid(cloud: RFCloud) -> tuple[float, float]:
    """Synapse-weighted (p, q) centroid."""
    if not cloud.ok:
        return (float("nan"), float("nan"))
    return (float(np.average(cloud.p, weights=cloud.w)),
            float(np.average(cloud.q, weights=cloud.w)))


def pq_width_p(cloud: RFCloud) -> float:
    """Half-max width of the RF along the azimuth (p) axis, as an FWHM proxy.

    Synapse-weighted standard deviation along p, scaled to FWHM (2.355 * sigma). Robust to
    the lattice being coarse; comparable across cells in the same frame.
    """
    if not cloud.ok:
        return float("nan")
    cp, _ = pq_centroid(cloud)
    var = float(np.average((cloud.p - cp) ** 2, weights=cloud.w))
    return float(2.3548 * np.sqrt(var))


def pq_patch_radius(cloud: RFCloud) -> float:
    """Synapse-weighted median radial distance of the cloud about its centroid (lattice units).

    The small-field signature: a figure-detecting cell pools a *bounded* retinotopic patch,
    so this is small relative to the in-degree-preserving null (see derive k_fd3_lpt42).
    """
    if not cloud.ok:
        return float("nan")
    cp, cq = pq_centroid(cloud)
    d = np.hypot(cloud.p - cp, cloud.q - cq)
    # weighted median
    order = np.argsort(d)
    d_s, w_s = d[order], cloud.w[order]
    cw = np.cumsum(w_s)
    half = cw[-1] / 2.0
    return float(d_s[np.searchsorted(cw, half)])


def pq_band_occupancy(cloud: RFCloud, p_lo: float, p_hi: float) -> float:
    """Fraction of synapse weight whose input column p falls in [p_lo, p_hi].

    Used for the frontal-gap test: with the frontal band defined from the FD1 anchor's
    location, FD3 should have ~zero occupancy there while FD1 has substantial occupancy.
    """
    if not cloud.ok:
        return float("nan")
    sel = (cloud.p >= p_lo) & (cloud.p <= p_hi)
    return float(cloud.w[sel].sum() / cloud.w.sum())


def pq_q_span_frac(cloud: RFCloud, q_full_range: float) -> float:
    """Vertical (elevation = q) span of the cloud as a fraction of the lattice q-range.

    FD3's excitatory RF "covers the entire vertical extent of the visual field"; a high
    fraction is consistent with that. ``q_full_range`` is the lattice's total q extent.
    """
    if not cloud.ok or not q_full_range:
        return float("nan")
    # 5th-95th percentile span (robust to a few stray inputs), synapse-weighted via repeat.
    span = float(np.percentile(cloud.q, 95) - np.percentile(cloud.q, 5))
    return float(span / q_full_range)


# ---------------------------------------------------------------------------
# Differential RF test (PRIMARY verdict gate) — candidate vs the FD1=Nod1 anchor.
# ---------------------------------------------------------------------------
@dataclass
class DifferentialRF:
    cand_centroid_p: float
    ref_centroid_p: float
    centroid_offset_p: float       # cand - ref; >0 means candidate is MORE LATERAL
    cand_width_p: float
    ref_width_p: float
    width_ratio: float             # cand / ref; >1 means candidate is WIDER
    frontal_band: tuple[float, float]
    cand_frontal_occ: float        # candidate occupancy of the FD1 frontal band (expect ~0)
    ref_frontal_occ: float         # FD1 occupancy of its own frontal band (expect high)
    more_lateral: bool
    wider: bool
    has_frontal_gap: bool
    notes: str = ""


def differential_rf(cand: RFCloud, ref: RFCloud, *, gap_drop: float = 0.5) -> DifferentialRF:
    """Compare a candidate RF cloud to the FD1=Nod1 reference cloud in the shared frame.

    The frontal band is the most-frontal stripe where FD1 is excited: from the lattice
    frontal pole up to the FD1 centroid (everything at or frontal of FD1's peak). FD3's
    distinguishing signature: its centroid is more lateral (offset_p > 0), it is wider
    (width_ratio > 1), and it has a FRONTAL GAP — its occupancy of that frontal band is a
    small fraction of FD1's (the only-FD-cell feature). The gap is scored RELATIVELY (a
    large drop vs FD1), not as an absolute near-zero, because FD3's frontal margin is a
    *shift* (Egelhaaf: FD3 margin ~20 deg vs FD1 ~-10 deg), not a complete absence.
    """
    cp, _ = pq_centroid(cand)
    rp, _ = pq_centroid(ref)
    cw = pq_width_p(cand)
    rw = pq_width_p(ref)
    offset = cp - rp
    ratio = cw / rw if rw else float("nan")
    # Frontal band = from the lattice frontal pole to the FD1 centroid (the stripe FD1 peaks
    # in and frontal of). Using -inf as the lower edge captures "the most frontal part".
    band = (-np.inf, rp)
    cand_occ = pq_band_occupancy(cand, *band)
    ref_occ = pq_band_occupancy(ref, *band)
    # Gap present iff the candidate occupies the frontal band at <= gap_drop * FD1's share
    # (a relative drop) AND FD1 itself substantially occupies it.
    gap = bool(np.isfinite(cand_occ) and np.isfinite(ref_occ) and ref_occ >= 0.4
               and cand_occ <= gap_drop * ref_occ)
    return DifferentialRF(
        cand_centroid_p=cp, ref_centroid_p=rp, centroid_offset_p=offset,
        cand_width_p=cw, ref_width_p=rw, width_ratio=ratio,
        frontal_band=band, cand_frontal_occ=cand_occ, ref_frontal_occ=ref_occ,
        more_lateral=bool(offset > 0),
        wider=bool(np.isfinite(ratio) and ratio > 1.0),
        has_frontal_gap=gap,
        notes=f"frontal band = p <= FD1 centroid ({rp:.1f}); gap = cand occ <= "
              f"{gap_drop:g}x FD1 occ",
    )


# ---------------------------------------------------------------------------
# Axis self-test — pins the p-sign convention (analogous to geometry.scaling_self_test).
# ---------------------------------------------------------------------------
def assert_frontal_anchor(nod1_cloud: RFCloud) -> dict:
    """Assert the FD1=Nod1 anchor lands FRONTAL (negative p), pinning 'lateral = +p'.

    If Nod1's centroid p is not clearly frontal, the p<->q (azimuth<->elevation) axis
    assignment is suspect and the differential test must not run. Raises AssertionError,
    matching the fail-fast behaviour of geometry.scaling_self_test.
    """
    cp, cq = pq_centroid(nod1_cloud)
    assert nod1_cloud.ok, "Nod1 RF cloud is empty/degenerate; cannot pin the axis convention"
    assert cp < 0, (
        f"axis-convention self-test FAILED: Nod1 (=FD1, frontal) centroid p={cp:.2f} is not "
        f"negative; p<->q may be swapped or the sign flipped. Refusing to score the RF."
    )
    return {"nod1_centroid_p": round(cp, 3), "nod1_centroid_q": round(cq, 3),
            "convention": "lateral = +p (Nod1=FD1 sits frontal at p<0)"}


def lattice_q_range() -> float:
    """Total q-extent of the optic-lobe hex lattice (for q-span normalisation)."""
    pq = _pq_map()
    return float(pq["q"].max() - pq["q"].min())
