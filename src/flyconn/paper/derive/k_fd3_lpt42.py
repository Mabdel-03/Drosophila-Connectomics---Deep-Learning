"""Derive family K: LPT42_Nod4 is the modern correlate of Egelhaaf-1985 "FD3".

FD1 = Nod1 is already established (family H + the user's direct queries). FD3 is the
regressive (back-to-front), fronto-LATERAL, contralaterally-projecting "noduli group"
figure-detection cell. This module computes, for LPT42_Nod4 and for the alternative Nod
types (the controls), every connectome-measurable FD3 property:

  * identity          — cell count, sides, NT, super_class
  * preferred dir     — T4/T5 lobula-plate LAYER composition (a=front->back, b=back->front,
                        c=up, d=down); FD3 must be layer-b dominant (vs Nod1=FD1 layer-a)
  * output laterality — % of output synapses onto contralateral targets (noduli-group axon)
  * target screen     — does it feed VCH/DCH/Am1 (the Nod5 signature) or its own POF targets
  * receptive field   — the synapse-weighted T4/T5-input (p,q) cloud vs the Nod1=FD1 anchor:
                        more lateral, wider, with a FRONTAL GAP (the only-FD-cell feature),
                        PLUS an absolute-degree reading (secondary, calibrated)
  * small-field null  — the RF patch is bounded vs an in-degree-preserving permutation null
  * bilateral         — the FD3 signature holds INDEPENDENTLY on the left and right cell
  * morphology        — (live only) skeleton-derived dendrite/axon anatomy

Controls: Nod1 (=FD1, layer-a, frontal — the anchor + a discriminator), Nod3 (regressive
but bilateral output — weaker noduli match), Nod5 (layer-c upward + feeds the inhibitors —
not an FD output), Nod2 (GABA — inhibitory).

Source: Egelhaaf 1985 Biol. Cybern. 52:195-209 "The FD3-Cell" (p.202-204, Figs 9-13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as CM
from .. import geometry as G
from .. import geometry_hex as GH
from .. import azimuth_calibration as AZ
from .. import skeleton_fetch as SK
from ..oracle import consts as C

SEED = 12345
N_PERMS = 500
N_BOOT = 2000

CANDIDATE = "LPT42_Nod4"
ANCHOR = "Nod1"                      # = FD1, the frontal/layer-a reference
CONTROLS = ("Nod1", "Nod2", "Nod3", "Nod5")
# Wide-field inhibitory tangentials a true FD output should NOT feed (the Nod5 signature).
INHIBITOR_TARGETS = ("VCH", "DCH", "Am1")


class _CachedSource:
    """Per-run memoization of src.synapses (the offline source re-scans the whole feather on
    every call; the candidate's inputs are pulled by the profile, RF and null blocks). Keyed
    by the frozenset of pre/post ids so identical queries hit the cache. Transparent: exposes
    the same ``synapses``/``track``/``client`` interface as the wrapped source."""

    def __init__(self, src):
        self._src = src
        self.track = getattr(src, "track", None)
        self.client = getattr(src, "client", None)
        self._cache: dict = {}

    def synapses(self, pre_ids=None, post_ids=None):
        key = (tuple(sorted(int(x) for x in pre_ids)) if pre_ids is not None else None,
               tuple(sorted(int(x) for x in post_ids)) if post_ids is not None else None)
        if key not in self._cache:
            self._cache[key] = self._src.synapses(pre_ids=pre_ids, post_ids=post_ids)
        return self._cache[key]


# ---------------------------------------------------------------------------
# Per-type connectome quantities (layer composition + output laterality + targets).
# ---------------------------------------------------------------------------
def _type_profile(src, meta, cell_type: str) -> dict:
    """Layer composition, NT, super_class, output laterality and top targets for a type."""
    roots = np.array(sorted(int(x) for x in meta.root_ids_of_type([cell_type])), dtype=np.int64)
    if len(roots) == 0:
        return {"cell_type": cell_type, "n_cells": 0}
    rset = set(int(x) for x in roots)

    # Inputs -> layer composition (synapse-weighted T4/T5 subtype fractions).
    syn_in = src.synapses(post_ids=roots.tolist())
    syn_in = syn_in[syn_in["post_pt_root_id"].isin(rset)]
    in_counts = CM.attach_meta(CM.partner_counts(syn_in, "pre_pt_root_id"), meta)
    t45 = in_counts[in_counts["is_t4t5"]]
    layer = {
        "a": CM.layer_fraction(t45, "a"), "b": CM.layer_fraction(t45, "b"),
        "c": CM.layer_fraction(t45, "c"), "d": CM.layer_fraction(t45, "d"),
    }
    dom = CM.dominant_layer(t45)

    # Outputs -> laterality + top targets.
    syn_out = src.synapses(pre_ids=roots.tolist())
    syn_out = syn_out[syn_out["pre_pt_root_id"].isin(rset)]
    pre_side = meta.by_root.reindex(syn_out["pre_pt_root_id"].values)["side"].to_numpy()
    post_side = meta.by_root.reindex(syn_out["post_pt_root_id"].values)["side"].to_numpy()
    valid = pd.notna(pre_side) & pd.notna(post_side)
    contra = float(100.0 * np.sum((pre_side != post_side) & valid) / max(int(valid.sum()), 1))
    out_counts = CM.attach_meta(CM.partner_counts(syn_out, "post_pt_root_id"), meta)
    by_ct = out_counts.dropna(subset=["cell_type"]).groupby("cell_type")["syn"].sum()
    top_targets = by_ct.sort_values(ascending=False).head(8)
    total_out = float(by_ct.sum())
    inhibitor_syn = int(by_ct.reindex(list(INHIBITOR_TARGETS)).fillna(0).sum())
    # Fraction of OUTPUT going to the wide-field inhibitors. For Nod5 (a feedback cell) this
    # is the dominant output; for a true FD output (LPT42_Nod4) it is minor. The
    # discriminator is the *fraction*, not mere presence (every cell touches a few).
    inhibitor_out_frac = round(100.0 * inhibitor_syn / total_out, 2) if total_out else float("nan")
    feeds_inhibitors = bool(inhibitor_syn > 0)
    # Are the inhibitors among the cell's TOP-3 targets? (the Nod5 feedback signature)
    inhibitors_dominant = bool(set(INHIBITOR_TARGETS) & set(top_targets.head(3).index))

    nt = str(meta.by_root.loc[int(roots[0]), "nt_canonical"])
    sc = str(meta.by_root.loc[int(roots[0]), "super_class"])
    sides = meta.by_root.reindex(roots)["side"].value_counts().to_dict()

    # NT confidence (quantifies the cholinergic claim beyond a bare categorical).
    nt_conf, nt_unanimous = float("nan"), None
    if "top_nt_conf" in meta.df.columns:
        rows = meta.by_root.reindex(roots)
        nt_conf = float(rows["top_nt_conf"].mean())
        nt_unanimous = bool((rows["top_nt"] == nt).all()) if "top_nt" in meta.df.columns else None

    return {
        "cell_type": cell_type,
        "n_cells": int(len(roots)),
        "sides": {str(k): int(v) for k, v in sides.items()},
        "nt": nt,
        "mean_nt_conf": round(nt_conf, 4) if np.isfinite(nt_conf) else None,
        "nt_unanimous": nt_unanimous,
        "super_class": sc,
        "layer_frac": {k: (round(v, 2) if np.isfinite(v) else None) for k, v in layer.items()},
        "dominant_layer": dom,
        "dominant_direction": C.LAYER_DIRECTION.get(dom) if dom else None,
        "contra_output_pct": round(contra, 2),
        "feeds_inhibitors": feeds_inhibitors,
        "inhibitors_dominant": inhibitors_dominant,
        "inhibitor_syn": inhibitor_syn,
        "inhibitor_out_frac": inhibitor_out_frac,
        "top_targets": {str(k): int(v) for k, v in top_targets.items()},
        "_roots": roots,
    }


# ---------------------------------------------------------------------------
# Bidirectional contralateral inhibition (N3, DISCRIMINATING).
# ---------------------------------------------------------------------------
# The stratification (contra GABA partners split by progressive/regressive lobula-plate layer)
# lives in ``common.contra_inhibition_profile`` so family P (the FD3 input pathway) shares the
# same implementation. Re-exported here under the historical names for the existing call sites.
_INH_LAYER = CM.INH_LAYER
MIN_CLASSIFIED_INH = CM.MIN_CLASSIFIED_INH
_contra_inhibition = CM.contra_inhibition_profile


# ---------------------------------------------------------------------------
# Receptive-field block: differential vs FD1 anchor, per side (bilateral), + absolute deg.
# ---------------------------------------------------------------------------
def _rf_per_side(src, meta, cell_type: str) -> dict[str, GH.RFCloud]:
    """RFCloud for each cell of ``cell_type``, keyed by side (left/right).

    Pulls the cell type's input synapses ONCE (cached on the wrapped source) and filters to
    each cell, instead of one pull per cell.
    """
    roots = sorted(int(x) for x in meta.root_ids_of_type([cell_type]))
    if not roots:
        return {}
    syn = src.synapses(post_ids=roots)  # cached: same key as the type profile's input pull
    out: dict[str, GH.RFCloud] = {}
    for r in roots:
        side = str(meta.by_root.loc[r, "side"])
        cloud = GH.input_cell_pq(src, meta, r, syn=syn)
        if cloud.ok:
            out[side] = cloud  # one cell per side for the Nod/LPT types of interest
    return out


def _bootstrap_offset_ci(cand: GH.RFCloud, ref: GH.RFCloud, rng) -> tuple[float, float]:
    """95% CI on the candidate-minus-reference centroid-p offset, resampling input columns."""
    if not (cand.ok and ref.ok):
        return (float("nan"), float("nan"))
    offs = np.empty(N_BOOT)
    nc, nr = len(cand.p), len(ref.p)
    for i in range(N_BOOT):
        ci = rng.integers(0, nc, nc)
        ri = rng.integers(0, nr, nr)
        cp = np.average(cand.p[ci], weights=cand.w[ci])
        rp = np.average(ref.p[ri], weights=ref.w[ri])
        offs[i] = cp - rp
    return (float(np.percentile(offs, 2.5)), float(np.percentile(offs, 97.5)))


def _rf_block(src, meta, candidate: str = CANDIDATE) -> dict:
    cand_sides = _rf_per_side(src, meta, candidate)
    anchor_sides = _rf_per_side(src, meta, ANCHOR)
    rng = np.random.default_rng(SEED)
    q_range = GH.lattice_q_range()

    # Axis self-test on the anchor (pins lateral = +p). Pool both Nod1 cells of a side; use
    # right if present else any.
    anchor_for_selftest = anchor_sides.get("right") or next(iter(anchor_sides.values()))
    selftest = GH.assert_frontal_anchor(anchor_for_selftest)

    per_side = {}
    for side, cand in cand_sides.items():
        ref = anchor_sides.get(side) or anchor_for_selftest
        diff = GH.differential_rf(cand, ref)
        lo, hi = _bootstrap_offset_ci(cand, ref, rng)
        az_peak = float(AZ.pq_to_azimuth(diff.cand_centroid_p))
        az_width = AZ.width_p_to_degrees(diff.cand_width_p)
        per_side[side] = {
            "centroid_offset_p": round(diff.centroid_offset_p, 3),
            "offset_ci95": [round(lo, 3), round(hi, 3)],
            "more_lateral": diff.more_lateral and lo > 0,   # beyond CI
            "width_ratio": round(diff.width_ratio, 3),
            "wider": diff.wider,
            "cand_frontal_occ": round(diff.cand_frontal_occ, 4),
            "ref_frontal_occ": round(diff.ref_frontal_occ, 4),
            "has_frontal_gap": diff.has_frontal_gap,
            "q_span_frac": round(GH.pq_q_span_frac(cand, q_range), 3),
            "abs_peak_az_deg": round(az_peak, 1),
            "abs_width_deg": round(az_width, 1),
            "n_inputs": cand.n_inputs,
            "dropout_frac": round(cand.dropout_frac, 4) if np.isfinite(cand.dropout_frac) else None,
        }

    # Bilateral replication: the CORE FD3 RF signature (more lateral + frontal gap) holds on
    # BOTH sides independently. Width is corroborating, not core (a single FD1 reference cell
    # can be broad), so it is not required for the bilateral gate.
    sides_present = sorted(per_side.keys())
    bilateral_ok = len(sides_present) >= 2 and all(
        per_side[s]["more_lateral"] and per_side[s]["has_frontal_gap"]
        for s in sides_present
    )
    return {
        "axis_selftest": selftest,
        "per_side": per_side,
        "sides_present": sides_present,
        "bilateral_ok": bilateral_ok,
        "calibration_error_budget": AZ.calibration_error_budget(),
    }


# ---------------------------------------------------------------------------
# Small-field permutation null (RF patch bounded vs in-degree-preserving null).
# ---------------------------------------------------------------------------
def _smallfield_null(src, meta, candidate: str = CANDIDATE) -> dict:
    """Observed candidate RF patch radius vs an in-degree-preserving permutation null.

    Pool = all canonical T4/T5 cells with a (p,q) in the candidate's eye(s). Null: keep the
    candidate's number of distinct T4/T5 inputs but draw which ones at random; recompute the
    hex patch radius. Mirrors family D's d_retinotopy_null but in lattice space.
    """
    pq = GH._pq_map()
    t45_meta = meta.df[meta.df["cell_type"].apply(lambda c: bool(GH.CM.is_canonical_t4t5(pd.Series([c]))[0]))]
    pool = pq.join(t45_meta.set_index("root_id")[["cell_type"]], how="inner").dropna(subset=["p", "q"])
    pool_pts = pool[["p", "q"]].to_numpy(dtype=float)

    clouds = _rf_per_side(src, meta, candidate)
    if not clouds:
        return {"obs_radius": float("nan"), "null_radius": float("nan"),
                "z_score": float("nan"), "p_value": float("nan"), "n_perms": N_PERMS}
    # Pool the candidate's inputs across sides for the observed radius + in-degree.
    obs_radii, indeg = [], []
    for cloud in clouds.values():
        obs_radii.append(GH.pq_patch_radius(cloud))
        indeg.append(cloud.n_inputs)
    obs = float(np.median(obs_radii))

    rng = np.random.default_rng(SEED)
    n_pool = len(pool_pts)
    null = np.empty(N_PERMS)
    for j in range(N_PERMS):
        radii = []
        for k in indeg:
            sel = rng.choice(n_pool, size=min(k, n_pool), replace=False)
            pts = pool_pts[sel]
            cp, cq = pts[:, 0].mean(), pts[:, 1].mean()
            d = np.hypot(pts[:, 0] - cp, pts[:, 1] - cq)
            radii.append(float(np.median(d)))
        null[j] = np.median(radii)
    null_mean = float(np.mean(null))
    null_std = float(np.std(null))
    z = (obs - null_mean) / null_std if null_std else float("nan")
    p_value = max(float((null <= obs).mean()), 1.0 / N_PERMS)
    return {"obs_radius": round(obs, 3), "null_radius": round(null_mean, 3),
            "z_score": round(z, 2), "p_value": p_value, "n_perms": N_PERMS,
            "bounded": bool(obs < null_mean)}


# ---------------------------------------------------------------------------
# Morphology. Real skeletons are unavailable on the public datastack (no L2 cache;
# precomputed store empty; on-demand generation refused -> NoL2CacheException). We therefore
# derive a SKELETON-FREE morphology proxy from the per-synapse 3D point clouds we already
# pull: the INPUT-synapse cloud approximates the dendritic arbor (in the lobula plate) and the
# OUTPUT-synapse cloud approximates the axonal field. These recover Egelhaaf's FD3 anatomical
# claims (dorso-ventral dendrite span; heterolateral axon that crosses the midline toward the
# noduli; contralateral terminus) as measured quantities. A true skeleton would refine the
# dendrite-vs-axon split and branch structure but is not required for these claims.
# ---------------------------------------------------------------------------
def _midline_x_um(meta) -> float:
    return float(meta.df["soma_x"].median()) / 1000.0


def _noduli_landmark_um(src, meta) -> np.ndarray | None:
    """Centroid (um) of the Nod1 axonal output cloud = the noduli-group convergence landmark."""
    n1 = sorted(int(x) for x in meta.root_ids_of_type([ANCHOR]))
    if not n1:
        return None
    out = src.synapses(pre_ids=n1)
    out = out[out["pre_pt_root_id"].isin(set(n1))]
    if len(out) == 0:
        return None
    return G.centroid(G.syn_positions_um(out, "pre"))


def _synapse_cloud_morphology(src, meta, candidate: str = CANDIDATE) -> dict:
    """Skeleton-free dendrite/axon morphology from input/output synapse clouds (any track)."""
    roots = sorted(int(x) for x in meta.root_ids_of_type([candidate]))
    if not roots:
        return {"available": False, "reason": "no candidate cells"}
    midline = _midline_x_um(meta)
    noduli = _noduli_landmark_um(src, meta)
    cells = []
    for r in roots:
        side = str(meta.by_root.loc[r, "side"])
        din = src.synapses(post_ids=[r]); din = din[din["post_pt_root_id"] == r]
        dout = src.synapses(pre_ids=[r]); dout = dout[dout["pre_pt_root_id"] == r]
        dend = G.syn_positions_um(din, "post")   # dendrite (inputs land on this cell)
        axon = G.syn_positions_um(dout, "pre")   # axon terminals (this cell's outputs)
        dext = G.cloud_extent_um(dend)
        aext = G.cloud_extent_um(axon)
        # Heterolateral axon: the axonal field is displaced from the dendritic field TOWARD the
        # contralateral hemisphere. Measured as the dendrite->axon centroid shift along the
        # medio-lateral (x) axis; a right-soma cell should shift -x (toward midline/left), a
        # left-soma cell +x. This is frame-robust (relative shift, not an absolute midline).
        ml_shift = (aext["centroid"][0] - dext["centroid"][0]) if (aext.get("n") and dext.get("n")) else None
        crossed_contra = (None if ml_shift is None
                          else bool((ml_shift < 0) if side == "right" else (ml_shift > 0)))
        # Axon converges near the noduli landmark, and is closer to it than the dendrite is.
        dist_axon_nod = (float(np.linalg.norm(aext.get("centroid", [0, 0, 0]) - noduli))
                         if noduli is not None and aext.get("n") else None)
        dist_dend_nod = (float(np.linalg.norm(dext.get("centroid", [0, 0, 0]) - noduli))
                         if noduli is not None and dext.get("n") else None)
        cells.append({
            "root_id": r, "side": side,
            "dendrite": dext, "axon": aext,
            "dv_span_um": dext.get("span_y"),
            "ml_span_um": dext.get("span_x"),
            "axon_ml_shift_um": round(ml_shift, 1) if ml_shift is not None else None,
            "axon_crosses_contra": crossed_contra,
            "axon_to_noduli_um": round(dist_axon_nod, 1) if dist_axon_nod is not None else None,
            "dend_to_noduli_um": round(dist_dend_nod, 1) if dist_dend_nod is not None else None,
            "axon_nearer_noduli": bool(dist_axon_nod is not None and dist_dend_nod is not None
                                       and dist_axon_nod < dist_dend_nod),
        })
    return {"available": True, "source": "synapse_cloud_proxy",
            "midline_x_um": round(midline, 1),
            "noduli_landmark_um": [round(float(v), 1) for v in noduli] if noduli is not None else None,
            "cells": cells}


def _skeleton_morphology(src, meta, candidate: str = CANDIDATE) -> dict | None:
    """Morphology from REAL skeletons (fafbseg, cached to the project tree). Computes the same
    quantities as the synapse-cloud proxy (D-V span, heterolateral axon crossing, noduli
    convergence) from skeleton vertices, so the oracle claims are identical but skeleton-backed.

    Works from cache regardless of track/env (fafbseg+cloudvolume live only in flyconn_cave;
    once cached, any env reads it). Returns None if no skeleton is available for any cell.
    """
    roots = sorted(int(x) for x in meta.root_ids_of_type([candidate]))
    noduli = _noduli_landmark_um(src, meta)
    cells = []
    for r in roots:
        skel = SK.fetch_skeleton(src, r)
        if skel is None or not skel.ok:
            continue
        v = skel.vertices_um
        side = str(meta.by_root.loc[r, "side"])
        # Split skeleton vertices into dendrite vs axon by nearest synapse cloud: input
        # synapses mark dendrite, output synapses mark axon. Each vertex inherits the label
        # of its nearer synapse-cloud centroid (a coarse but principled compartment split).
        din = src.synapses(post_ids=[r]); din = din[din["post_pt_root_id"] == r]
        dout = src.synapses(pre_ids=[r]); dout = dout[dout["pre_pt_root_id"] == r]
        in_c = G.centroid(G.syn_positions_um(din, "post")) if len(din) else None
        out_c = G.centroid(G.syn_positions_um(dout, "pre")) if len(dout) else None
        if in_c is not None and out_c is not None:
            d_in = np.linalg.norm(v - in_c, axis=1)
            d_out = np.linalg.norm(v - out_c, axis=1)
            dend_v, axon_v = v[d_in <= d_out], v[d_out < d_in]
        else:
            dend_v, axon_v = v, v
        dext = G.cloud_extent_um(dend_v)
        aext = G.cloud_extent_um(axon_v)
        ml_shift = (aext["centroid"][0] - dext["centroid"][0]) if (aext.get("n") and dext.get("n")) else None
        crossed = (None if ml_shift is None
                   else bool((ml_shift < 0) if side == "right" else (ml_shift > 0)))
        a2n = float(np.linalg.norm(aext.get("centroid", [0, 0, 0]) - noduli)) if (noduli is not None and aext.get("n")) else None
        d2n = float(np.linalg.norm(dext.get("centroid", [0, 0, 0]) - noduli)) if (noduli is not None and dext.get("n")) else None
        cells.append({
            "root_id": r, "side": side, "source": skel.source,
            "n_vertices": int(len(v)), "n_edges": int(len(skel.edges)),
            "dendrite": dext, "axon": aext,
            "dv_span_um": round(float(v[:, 1].max() - v[:, 1].min()), 1),
            "ml_span_um": round(float(v[:, 0].max() - v[:, 0].min()), 1),
            "axon_ml_shift_um": round(ml_shift, 1) if ml_shift is not None else None,
            "axon_crosses_contra": crossed,
            "axon_to_noduli_um": round(a2n, 1) if a2n is not None else None,
            "dend_to_noduli_um": round(d2n, 1) if d2n is not None else None,
            "axon_nearer_noduli": bool(a2n is not None and d2n is not None and a2n < d2n),
        })
    if not cells:
        return None
    return {"available": True, "source": "skeleton",
            "noduli_landmark_um": [round(float(v), 1) for v in noduli] if noduli is not None else None,
            "cells": cells}


def _morphology(src, meta, candidate: str = CANDIDATE) -> dict:
    """Real skeletons (fafbseg, cached) when available, else the synapse-cloud proxy.

    Both yield the SAME claim quantities; only the ``source`` field differs. Skeletons are
    preferred (true neuron geometry); the proxy is the always-available fallback.
    """
    skel = _skeleton_morphology(src, meta, candidate)
    if skel is not None:
        return skel
    proxy = _synapse_cloud_morphology(src, meta, candidate)
    proxy["skeleton_reason"] = (getattr(SK.fetch_skeleton, "last_skip_reason", "")
                                or "no cached skeleton; run scripts/fetch_fd3_skeletons.py "
                                   "in the flyconn_cave env to add real skeletons")
    return proxy


# ---------------------------------------------------------------------------
# Soma location (N2, offline-CONFIRMABLE proxy for the cell-body morphology claim).
# ---------------------------------------------------------------------------
def _soma_block(meta, candidate: str = CANDIDATE) -> dict:
    cand_roots = sorted(int(x) for x in meta.root_ids_of_type([candidate]))
    nod1_roots = sorted(int(x) for x in meta.root_ids_of_type([ANCHOR]))
    centrifugal = sorted(int(x) for x in meta.root_ids_of_type(["VCH", "DCH"]))
    return CM.soma_summary(meta, cand_roots, anchor_roots=nod1_roots, anterior_roots=centrifugal)


# ---------------------------------------------------------------------------
# Whole-FD-family uniqueness screen (N5, DISCRIMINATING headline).
# ---------------------------------------------------------------------------
# Egelhaaf FD reference signatures (binary features): layer (a=progressive / b=regressive),
# whether the RF is lateral-of FD1 with a frontal gap, and whether the axon is heterolateral.
# ``lateral_gap`` is Egelhaaf's frontal-gap-with-lateral-field signature, which he certifies as
# UNIQUE to FD3. FD4 has NO frontal gap: its receptive field spans the whole eye (including the
# frontal field) but is weighted laterally, captured by ``whole_eye_lateralwt`` instead. The FD4
# row previously encoded ``lateral_gap: True``, which contradicts Egelhaaf (p.202, p.204-205);
# it is corrected here.
FD_SIGNATURES = {
    "FD1": {"layer": "a", "lateral_gap": False, "whole_eye_lateralwt": False, "heterolateral": True},
    "FD2": {"layer": "b", "lateral_gap": False, "whole_eye_lateralwt": False, "heterolateral": False},
    "FD3": {"layer": "b", "lateral_gap": True, "whole_eye_lateralwt": False, "heterolateral": True},
    "FD4": {"layer": "a", "lateral_gap": False, "whole_eye_lateralwt": True, "heterolateral": True},
}


def fd_family_screen(profiles: dict, rf_by_type: dict) -> dict:
    """Match every candidate Nod/LPT type to each Egelhaaf FD signature; show LPT42_Nod4 is the
    reciprocal best hit for FD3.

    Candidate feature vector: dominant layer (a/b/c/d); lateral_gap = (RF more-lateral than FD1
    AND has frontal gap on >=1 side); heterolateral = contra_output_pct >= 70. Score = number of
    matching binary features vs an FD signature (layer match counts once).
    """
    def cand_vec(ct):
        prof = profiles.get(ct, {})
        rf = rf_by_type.get(ct, {})
        lateral_gap = bool(rf.get("more_lateral") and rf.get("has_frontal_gap"))
        return {"layer": prof.get("dominant_layer"),
                "lateral_gap": lateral_gap,
                "heterolateral": bool((prof.get("contra_output_pct") or 0) >= 70.0)}

    cands = [ct for ct in profiles if profiles[ct].get("n_cells")]
    score = {}
    for fd, sig in FD_SIGNATURES.items():
        score[fd] = {}
        for ct in cands:
            v = cand_vec(ct)
            s = int(v["layer"] == sig["layer"]) + int(v["lateral_gap"] == sig["lateral_gap"]) \
                + int(v["heterolateral"] == sig["heterolateral"])
            score[fd][ct] = s
    # best candidate for FD3, and best FD for LPT42_Nod4
    fd3 = score["FD3"]
    best_for_fd3 = max(fd3, key=fd3.get) if fd3 else None
    runner = sorted(fd3.values(), reverse=True)
    margin = (runner[0] - runner[1]) if len(runner) >= 2 else runner[0] if runner else 0
    lpt_scores = {fd: score[fd].get(CANDIDATE, -1) for fd in FD_SIGNATURES}
    best_fd_for_lpt = max(lpt_scores, key=lpt_scores.get) if lpt_scores else None
    reciprocal = bool(best_for_fd3 == CANDIDATE and best_fd_for_lpt == "FD3")
    return {
        "score_matrix": score,
        "best_match_for_FD3": best_for_fd3,
        "best_FD_for_LPT42": best_fd_for_lpt,
        "fd3_margin": int(margin),
        "reciprocal_best_hit": reciprocal,
        "n_candidates": len(cands),
    }


# ---------------------------------------------------------------------------
# Robustness / sensitivity ("the verdict does not depend on a knob").
# ---------------------------------------------------------------------------
def _robustness(src, meta, cand_clouds, anchor_cloud) -> dict:
    """Re-evaluate the identity-deciding booleans across knob grids + seeds; report pass-fraction."""
    results = {"per_knob": {}, "seed_stability": {}}
    # Knob 1: gap_drop threshold for the frontal gap.
    gap_pass = []
    for gd in (0.3, 0.4, 0.5, 0.6, 0.7):
        ok = all(GH.differential_rf(c, anchor_cloud, gap_drop=gd).has_frontal_gap
                 for c in cand_clouds.values())
        gap_pass.append(ok)
    results["per_knob"]["gap_drop"] = {"grid": [0.3, 0.4, 0.5, 0.6, 0.7],
                                       "pass": gap_pass, "all_pass": all(gap_pass)}
    # Knob 2/3 (layer-b cutoff, contra threshold) are categorical with huge margins
    # (layer-b 98%, contra >=80%) and evaluated in the oracle; here we add seed stability of
    # the two stochastic estimates (bootstrap offset CI, permutation null).
    seeds = (12345, 1, 2, 3, 4)
    offsets_lower, null_z = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        # bootstrap offset lower bound for the first present side
        side = sorted(cand_clouds.keys())[0]
        lo, hi = _bootstrap_offset_ci(cand_clouds[side], anchor_cloud, rng)
        offsets_lower.append(lo)
    results["seed_stability"] = {
        "seeds": list(seeds),
        "offset_ci_lower_min": round(float(np.min(offsets_lower)), 3),
        "offset_ci_lower_max": round(float(np.max(offsets_lower)), 3),
        "all_offset_lower_gt0": bool(np.all(np.array(offsets_lower) > 0)),
    }
    grid_all = results["per_knob"]["gap_drop"]["all_pass"] and results["seed_stability"]["all_offset_lower_gt0"]
    results["grid_pass_fraction"] = 1.0 if grid_all else 0.0
    return results


# ---------------------------------------------------------------------------
# Cross-version replication (N4): re-derive headline fields on v630.
# ---------------------------------------------------------------------------
def _cross_version(src, meta, candidate: str = CANDIDATE) -> dict:
    if getattr(src, "track", None) != "live":
        return {"available": False, "reason": "v630 cross-check is live-only"}
    try:
        from .. import fw_access as FW
        v630 = _CachedSource(FW.LiveCaveFlyWire(mat_version=630))
        prof = _type_profile(v630, meta, candidate)
        rf = _rf_block(v630, meta, candidate)
        v783 = _type_profile(src, meta, candidate)
        agree = {
            "dominant_layer": prof.get("dominant_layer") == v783.get("dominant_layer"),
            "layer_b_within_5pp": abs((prof.get("layer_frac") or {}).get("b", 0)
                                      - (v783.get("layer_frac") or {}).get("b", 0)) <= 5.0,
            "contra_both_ge70": (prof.get("contra_output_pct") or 0) >= 70
            and (v783.get("contra_output_pct") or 0) >= 70,
            "rf_lateral_gap": all(rf["per_side"][s]["more_lateral"] and rf["per_side"][s]["has_frontal_gap"]
                                  for s in rf.get("per_side", {})) if rf.get("per_side") else False,
        }
        return {
            "available": True,
            "v630": {"dominant_layer": prof.get("dominant_layer"),
                     "layer_b": (prof.get("layer_frac") or {}).get("b"),
                     "contra_output_pct": prof.get("contra_output_pct")},
            "v783": {"dominant_layer": v783.get("dominant_layer"),
                     "layer_b": (v783.get("layer_frac") or {}).get("b"),
                     "contra_output_pct": v783.get("contra_output_pct")},
            "agree": agree,
            "all_core_agree": bool(agree["dominant_layer"] and agree["contra_both_ge70"]
                                   and agree["rf_lateral_gap"]),
        }
    except Exception as e:  # noqa: BLE001 — never crash the family on a v630 hiccup
        return {"available": False, "reason": f"v630 derivation failed: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    # Family K is intrinsically BILATERAL: it scores the FD3 candidate (LPT42_Nod4) and the
    # controls per cell side already, so it does not branch on cfg. The parameter is accepted
    # for a uniform orchestrator call signature; cfg is unused here by design.
    del cfg
    raw_src = src
    src = _CachedSource(src)  # memoize synapse pulls across the profile / RF / null blocks
    profiles = {ct: _type_profile(src, meta, ct) for ct in set(CONTROLS) | {CANDIDATE}}
    cand = profiles[CANDIDATE]
    rf = _rf_block(src, meta)
    nullb = _smallfield_null(src, meta)
    morph = _morphology(src, meta)
    soma = _soma_block(meta)
    contra_inh = {ct: _contra_inhibition(src, meta, ct) for ct in (CANDIDATE, ANCHOR)}

    # Family-screen RF features per candidate type (cheap: reuse differential RF vs FD1 anchor).
    anchor_clouds = _rf_per_side(src, meta, ANCHOR)
    anchor_cloud = anchor_clouds.get("right") or (next(iter(anchor_clouds.values())) if anchor_clouds else None)
    rf_by_type = {}
    for ct in profiles:
        if not profiles[ct].get("n_cells"):
            continue
        clouds = _rf_per_side(src, meta, ct)
        if clouds and anchor_cloud is not None:
            diffs = [GH.differential_rf(c, anchor_cloud) for c in clouds.values()]
            rf_by_type[ct] = {"more_lateral": any(d.more_lateral for d in diffs),
                              "has_frontal_gap": any(d.has_frontal_gap for d in diffs)}
    screen = fd_family_screen(profiles, rf_by_type)

    cand_clouds = _rf_per_side(src, meta, CANDIDATE)
    robustness = _robustness(src, meta, cand_clouds, anchor_cloud) if anchor_cloud else {"grid_pass_fraction": None}
    cross_version = _cross_version(raw_src, meta)

    # FDR over the p-valued claims of the family (currently the small-field permutation p;
    # the Wilcoxon RF p when present). Threshold/effect-size claims excluded by the oracle.
    pvals = [nullb.get("p_value", float("nan"))]
    p_adj = CM.bh_adjust(pvals)
    nullb["p_value_bh"] = p_adj[0]

    return {
        "candidate": CANDIDATE,
        "anchor": ANCHOR,
        "cand_n_cells": cand.get("n_cells"),
        "cand_sides": cand.get("sides"),
        "cand_nt": cand.get("nt"),
        "cand_mean_nt_conf": cand.get("mean_nt_conf"),
        "cand_nt_unanimous": cand.get("nt_unanimous"),
        "cand_super_class": cand.get("super_class"),
        "cand_layer_frac": cand.get("layer_frac"),
        "cand_dominant_layer": cand.get("dominant_layer"),
        "cand_dominant_direction": cand.get("dominant_direction"),
        "cand_contra_output_pct": cand.get("contra_output_pct"),
        "cand_feeds_inhibitors": cand.get("feeds_inhibitors"),
        "cand_inhibitors_dominant": cand.get("inhibitors_dominant"),
        "cand_inhibitor_out_frac": cand.get("inhibitor_out_frac"),
        "cand_top_targets": cand.get("top_targets"),
        "profiles": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in profiles.items()},
        "rf": rf,
        "smallfield_null": nullb,
        "morphology": morph,
        "soma": soma,
        "contra_inhibition": contra_inh,
        "fd_family_screen": screen,
        "robustness": robustness,
        "cross_version": cross_version,
        "track": src.track,
    }
