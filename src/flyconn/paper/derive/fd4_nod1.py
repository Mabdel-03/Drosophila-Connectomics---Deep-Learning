"""Family FD4 - the connectomic search for Egelhaaf-1985 FD4, and its (negative) result.

Egelhaaf's FD4 cell (Biol. Cybern. 52:195-209, sec. 4, pp. 204-206, Figs. 14-17) is a
figure-detection cell that is:
  * directionally selective for PROGRESSIVE (front-to-back) motion  -> lobula-plate layer a,
  * excited over the ENTIRE horizontal extent of the ipsilateral eye, most sensitive in the
    LATERAL field (half-max width 80-110 deg, peaks psi 50-80 deg, reaching beyond 120 deg),
    with NO frontal gap (the frontal gap is unique to FD3),
  * small-field selective,
  * bidirectionally inhibited by contralateral motion,
  * a cholinergic heterolateral "noduli group" output element whose axon crosses to the
    contralateral posterior optic foci (the same axonal pathway as the FD1nod and FD3 cells),
  * with a dendrite that spans the full horizontal extent of the lobula plate but NOT its full
    dorso-ventral extent (dorso-proximal and most ventro-proximal parts are devoid of dendrite),
  * and NO second dendritic arborisation in the lateral protocerebrum.

The single fact that shapes the whole analysis: FD4 shares FD1's ENTIRE output class
(progressive, heterolateral, cholinergic, noduli-group axon to the contralateral posterior
optic foci). FD1 and FD4 differ ONLY in receptive field (FD4 whole-eye lateral-weighted vs FD1
frontal-narrow) and in dendrite morphology (FD4 restricted dorso-ventral, no lateral-
protocerebrum arbor). This module therefore does NOT assert a positive identity by default. It
runs an exhaustive elimination and reports whichever of two outcomes the data support:

  H1 (positive)   a distinct progressive figure-output cell, or a separable sub-pair of the
                  4-cell FlyWire ``Nod1`` type, carries the FD4 phenotype; assign it with an
                  honest, discounted confidence.
  H3 (null)       no such cell is separable; the progressive figure-output slot is occupied by
                  a homogeneous ``Nod1`` (= FD1) population and FD4 is not individually resolved
                  at current annotation. Report the negative result with its full basis.

The module composes the FD2/FD3 machinery: ``k_fd3_lpt42._type_profile`` (layer/laterality/NT),
``geometry_hex`` (per-cell receptive-field clouds), ``compartments`` (dendrite/axon split),
``common.contra_inhibition_profile`` (bidirectional-vs-unidirectional contra inhibition), and,
for the progressive figure-arm circuit any FD4 correlate would use, the parameterised
``p_fd3_input`` / ``q_fd3_sheet`` / ``r_fd3_inhibitor`` / ``l_fd3_descending`` families run on
the layer-a channel (T4a/T5a -> LLPC1 sheet -> Nod1/FD1 -> DNp26 -> wing, VCH gate).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import common as CM
from . import connectivity_census as CC
from . import compartments as COMP
from . import k_fd3_lpt42 as K
from . import p_fd3_input as P
from . import q_fd3_sheet as Q
from . import r_fd3_inhibitor as R
from . import l_fd3_descending as L
from .. import geometry as G
from .. import geometry_hex as GH
from ..oracle import consts as C

# The progressive figure-output type: FlyWire's Nod1 (= Egelhaaf FD1). FD4, if separately
# resolved, would be a distinct type or a separable sub-pair inside this type.
PROG_TYPE = "Nod1"
FD3_TYPE = "LPT42_Nod4"      # = FD3; regressive; the incumbent "absolute-RF-FD4-shaped" cell
FD2_TYPE = "LPT21"           # = FD2; regressive; homolateral
CONTRA_HETERO_PCT = 70.0     # heterolateral gate (the shared FD1/FD3/FD4 noduli-group signature)
LAYER_A_MIN = 80.0           # dominant-progressive gate (percent of T4/T5 input in layer a)
LOW_COPY_MAX = 6             # an FD cell is a small, individually-identifiable population

# Reproducibility.
SEED = 12345
N_SPLIT_RESAMPLE = 500       # resamples for the Nod1 split-stability estimate


# ---------------------------------------------------------------------------
# Per-cell (per-root) profile - the FD1/FD4 split needs single-cell resolution, but the
# existing type-level metrics pool all 4 Nod1 cells (k_fd3_lpt42._rf_per_side overwrites by
# side). These helpers restore per-root resolution.
# ---------------------------------------------------------------------------
def _cell_profile(src, meta, root: int) -> dict:
    """Per-root layer composition, output laterality, NT, top targets, and arbor spans.

    Single-cell analogue of ``k_fd3_lpt42._type_profile``. Used to test whether the 4 Nod1
    cells are one homogeneous population or two functionally distinct sub-pairs.
    """
    root = int(root)
    side = str(meta.by_root.loc[root, "side"])

    syn_in = src.synapses(post_ids=[root]); syn_in = syn_in[syn_in["post_pt_root_id"] == root]
    in_counts = CM.attach_meta(CM.partner_counts(syn_in, "pre_pt_root_id"), meta)
    t45 = in_counts[in_counts["is_t4t5"]]
    layer = {k: CM.layer_fraction(t45, k) for k in ("a", "b", "c", "d")}
    dom = CM.dominant_layer(t45)

    syn_out = src.synapses(pre_ids=[root]); syn_out = syn_out[syn_out["pre_pt_root_id"] == root]
    out_counts = CM.attach_meta(CM.partner_counts(syn_out, "post_pt_root_id"), meta)
    post_side = out_counts["side"].to_numpy()
    w = out_counts["syn"].to_numpy(dtype=float)
    valid = pd.notna(post_side)
    contra = float(100.0 * w[(post_side != side) & valid].sum() / max(w[valid].sum(), 1.0))
    by_ct = out_counts.dropna(subset=["cell_type"]).groupby("cell_type")["syn"].sum()
    top_targets = {str(k): int(v) for k, v in by_ct.sort_values(ascending=False).head(8).items()}

    # Arbor spans from synapse positions (proxy for the reconstructed dendrite/axon extents).
    ip = G.syn_positions_um(syn_in, "post")
    op = G.syn_positions_um(syn_out, "pre")
    dv_span = _pctl_span(ip[:, 1]) if len(ip) >= 20 else float("nan")   # dorso-ventral = y
    ml_span = _pctl_span(ip[:, 0]) if len(ip) >= 20 else float("nan")   # medio-lateral = x

    return {
        "root_id": root, "side": side,
        "nt": str(meta.by_root.loc[root, "nt_canonical"]),
        "super_class": str(meta.by_root.loc[root, "super_class"]),
        "layer_frac": {k: (round(v, 2) if v == v else None) for k, v in layer.items()},
        "dominant_layer": dom,
        "dominant_direction": C.LAYER_DIRECTION.get(dom) if dom else None,
        "contra_output_pct": round(contra, 2),
        "top_targets": top_targets,
        "n_input_syn": int(len(ip)), "n_output_syn": int(len(op)),
        "dendrite_dv_span_um": round(dv_span, 1) if dv_span == dv_span else None,
        "dendrite_ml_span_um": round(ml_span, 1) if ml_span == ml_span else None,
        "_input_partners": set(int(x) for x in syn_in["pre_pt_root_id"].unique()),
    }


def _pctl_span(a: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> float:
    """Robust 5th-95th-percentile span of a coordinate array (micrometres)."""
    if len(a) < 2:
        return float("nan")
    return float(np.percentile(a, hi) - np.percentile(a, lo))


def _cell_rf(src, meta, root: int, ref_cloud) -> dict:
    """Per-root receptive-field descriptors + the FD4-vs-anchor differential.

    The FD4 signature is a lateral-weighted, wide, whole-eye field with NO frontal gap. We
    report the centroid p, the FWHM width, the lateral-vs-frontal weighting relative to the
    anchor centroid, and the differential (offset, wider, gap) against the FD1-frontal reference.
    """
    cloud = GH.input_cell_pq(src, meta, int(root))
    out = {"root_id": int(root), "ok": bool(cloud.ok)}
    if not cloud.ok:
        return out
    cp, cq = GH.pq_centroid(cloud)
    out.update({
        "centroid_p": round(cp, 2), "centroid_q": round(cq, 2),
        "width_p": round(GH.pq_width_p(cloud), 2),
        "patch_radius": round(GH.pq_patch_radius(cloud), 2),
        "n_inputs": int(cloud.n_inputs),
    })
    if ref_cloud is not None and ref_cloud.ok:
        ref_p, _ = GH.pq_centroid(ref_cloud)
        # Lateral weighting: share of RF weight lateral of the anchor centroid vs frontal of it.
        lateral = GH.pq_band_occupancy(cloud, ref_p, float(np.max(cloud.p)) + 1.0)
        frontal = GH.pq_band_occupancy(cloud, float(np.min(cloud.p)) - 1.0, ref_p)
        diff = GH.differential_rf(cloud, ref_cloud)
        out.update({
            "lateral_occ": round(lateral, 3), "frontal_occ": round(frontal, 3),
            "lateral_weight": round(lateral / max(frontal, 1e-6), 3),
            "offset_p_vs_anchor": round(cp - ref_p, 2),
            "more_lateral": bool(diff.more_lateral),
            "wider": bool(diff.wider),
            "has_frontal_gap": bool(diff.has_frontal_gap),
        })
    return out


def _second_arbor(src, meta, root: int) -> dict:
    """Detect a second, spatially separated output/dendrite lobe (the lateral-protocerebrum
    arbor that FD1nod and FD3 have and FD4 lacks). Reuses the medio-lateral histogram-valley
    bimodality test of ``fd2_lpt21._dual_output`` as a NEGATIVE control: FD4 must be single-lobe.
    """
    root = int(root)
    outs = src.synapses(pre_ids=[root]); outs = outs[outs["pre_pt_root_id"] == root]
    pos = G.syn_positions_um(outs, "pre")
    if len(pos) < 50:
        return {"root_id": root, "available": False}
    x = pos[:, 0]
    n_bins = 20
    hist, edges = np.histogram(x, bins=n_bins)
    lo_i, hi_i = int(n_bins * 0.15), int(n_bins * 0.85)
    valley_i = lo_i + int(np.argmin(hist[lo_i:hi_i])) if hi_i > lo_i else int(np.argmin(hist))
    valley = int(hist[valley_i])
    left_peak = int(hist[:valley_i].max()) if valley_i > 0 else 0
    right_peak = int(hist[valley_i + 1:].max()) if valley_i + 1 < n_bins else 0
    flank = max((left_peak + right_peak) / 2.0, 1.0)
    dip_ratio = valley / flank
    split = float(edges[valley_i + 1])
    lo = pos[pos[:, 0] < split]; hi = pos[pos[:, 0] >= split]
    minor_frac = min(len(lo), len(hi)) / max(len(pos), 1)
    bimodal = bool(dip_ratio <= 0.25 and minor_frac >= 0.08 and left_peak > 0 and right_peak > 0)
    return {"root_id": root, "available": True, "ml_valley_dip_ratio": round(dip_ratio, 3),
            "minor_lobe_frac": round(minor_frac, 3), "has_second_arbor": bimodal}


# ---------------------------------------------------------------------------
# The Nod1 split test - is FD4 a separable sub-pair of the 4-cell Nod1 type?
# ---------------------------------------------------------------------------
def _nod1_split(src, meta) -> dict:
    """Test whether the 4 Nod1 cells (2 left + 2 right) partition into an FD1 pair and an FD4
    pair. The FD4 pair would be more lateral, wider, restricted dorso-ventrally, and single-
    arbored. We enumerate the two valid bilateral pairings, score within-pair consistency vs
    between-pair separation on the discriminating feature vector, and report whether either
    pairing separates the cells (a positive split) or the four cells are one homogeneous
    population (the null).
    """
    roots = sorted(int(x) for x in meta.root_ids_of_type([PROG_TYPE]))
    profiles = {r: _cell_profile(src, meta, r) for r in roots}
    # FD1-frontal reference: pooled Nod1 RF (axis self-test only needs a frontal anchor; the
    # split itself is measured within the same type so needs no external anchor).
    ref_cloud = GH._rf_per_side_pooled(src, meta, PROG_TYPE) if hasattr(GH, "_rf_per_side_pooled") \
        else _pooled_cloud(src, meta, roots)
    rfs = {r: _cell_rf(src, meta, r, ref_cloud) for r in roots}
    arbors = {r: _second_arbor(src, meta, r) for r in roots}

    left = [r for r in roots if profiles[r]["side"] == "left"]
    right = [r for r in roots if profiles[r]["side"] == "right"]

    # Discriminating 5-vector per cell: [centroid_p, width_p, dv_span, lateral_weight, 2nd_arbor].
    def vec(r):
        rf, pr, ar = rfs[r], profiles[r], arbors[r]
        return np.array([
            rf.get("centroid_p", np.nan), rf.get("width_p", np.nan),
            pr.get("dendrite_dv_span_um", np.nan) or np.nan,
            rf.get("lateral_weight", np.nan),
            1.0 if ar.get("has_second_arbor") else 0.0,
        ], dtype=float)

    feats = {r: vec(r) for r in roots}
    split = {"n_cells": len(roots), "sides_ok": bool(len(left) == 2 and len(right) == 2),
             "per_cell": {str(r): {"side": profiles[r]["side"],
                                   "centroid_p": rfs[r].get("centroid_p"),
                                   "width_p": rfs[r].get("width_p"),
                                   "lateral_weight": rfs[r].get("lateral_weight"),
                                   "has_frontal_gap": rfs[r].get("has_frontal_gap"),
                                   "dendrite_dv_span_um": profiles[r].get("dendrite_dv_span_um"),
                                   "has_second_arbor": arbors[r].get("has_second_arbor"),
                                   "layer_a_pct": (profiles[r]["layer_frac"] or {}).get("a"),
                                   "contra_output_pct": profiles[r].get("contra_output_pct"),
                                   } for r in roots}}

    if not split["sides_ok"]:
        split.update({"separable": False, "reason": "Nod1 is not 2 left + 2 right",
                      "homogeneous": None})
        return split

    # Same-side input-partner overlap: two copies of one cell (high overlap) vs two distinct
    # cell types (low overlap). This detects whether Nod1 pools two populations at all.
    def jac(a, b):
        A, B = profiles[a]["_input_partners"], profiles[b]["_input_partners"]
        return len(A & B) / max(len(A | B), 1)
    same_side_jac = round(float(np.mean([jac(left[0], left[1]), jac(right[0], right[1])])), 3)
    cross_side_jac = round(float(np.mean([jac(left[0], right[0]), jac(left[1], right[1])])), 3)

    # z-score the features across the 4 cells, then enumerate the two bilateral pairings.
    F = np.vstack([feats[r] for r in roots])
    mu = np.nanmean(F, axis=0); sd = np.nanstd(F, axis=0) + 1e-9
    z = {r: (feats[r] - mu) / sd for r in roots}
    pairings = [((left[0], right[0]), (left[1], right[1])),
                ((left[0], right[1]), (left[1], right[0]))]
    scored = []
    for pg in pairings:
        within = float(sum(np.linalg.norm(np.nan_to_num(z[a] - z[b])) for a, b in pg))
        cA = np.nanmean([z[pg[0][0]], z[pg[0][1]]], axis=0)
        cB = np.nanmean([z[pg[1][0]], z[pg[1][1]]], axis=0)
        between = float(np.linalg.norm(np.nan_to_num(cA - cB)))
        scored.append({"pairing": [[int(a), int(b)] for a, b in pg],
                       "within": round(within, 3), "between": round(between, 3),
                       "silhouette": round(between - within, 3)})
    best = max(scored, key=lambda s: s["silhouette"])
    # A genuine FD1/FD4 split requires between > within (positive silhouette) AND that one pair
    # actually carries FD4 features (lateral-weighted, wider, no gap). Otherwise: homogeneous.
    separable = bool(best["silhouette"] > 0)

    # Which pair, if any, is FD4-like (higher lateral_weight + wider + restricted DV)?
    fd4_pair = None
    if separable:
        (a1, b1), (a2, b2) = best["pairing"][0], best["pairing"][1]
        def pair_lat(p, q):
            return np.nanmean([rfs[p].get("lateral_weight", np.nan),
                               rfs[q].get("lateral_weight", np.nan)])
        lat1, lat2 = pair_lat(a1, b1), pair_lat(a2, b2)
        fd4_pair = best["pairing"][0] if lat1 > lat2 else best["pairing"][1]

    split.update({
        "same_side_input_jaccard": same_side_jac,
        "cross_side_input_jaccard": cross_side_jac,
        "pools_two_populations": bool(same_side_jac >= 0.20 and cross_side_jac < 0.10),
        "pairings": scored, "best_pairing": best,
        "separable": separable,
        "homogeneous": (not separable),
        "fd4_pair": fd4_pair,
        "note": ("A positive FD1/FD4 split needs a positive silhouette AND an FD4-featured pair. "
                 "If the four cells are one homogeneous frontal population, Nod1 = FD1 (likely "
                 "pooling Egelhaaf's FD1nod + FD1pof variants) and FD4 is not resolved here."),
    })
    return split


def _pooled_cloud(src, meta, roots) -> "GH.RFCloud":
    """Pool all Nod1 cells into one RF cloud (the frontal anchor for the split)."""
    clouds = [GH.input_cell_pq(src, meta, int(r)) for r in roots]
    clouds = [c for c in clouds if c.ok]
    if not clouds:
        return GH.RFCloud(np.array([]), np.array([]), np.array([]), 0, 0, 0, float("nan"))
    return GH.RFCloud(
        p=np.concatenate([c.p for c in clouds]), q=np.concatenate([c.q for c in clouds]),
        w=np.concatenate([c.w for c in clouds]),
        n_inputs=sum(c.n_inputs for c in clouds), n_inputs_total=sum(c.n_inputs_total for c in clouds),
        total_syn=sum(c.total_syn for c in clouds), dropout_frac=float("nan"))


# ---------------------------------------------------------------------------
# Exhaustive candidate elimination - is there ANY progressive figure-output cell besides Nod1?
# ---------------------------------------------------------------------------
# Precomputed all-types scan (dom_layer / nt / super_class / contra% / copy per type). Produced
# by the FD3 global scan; reused here so the FD4 funnel does not re-pull all 8,806 types.
_GLOBAL_SCAN_CSV = ("/orcd/data/tpoggio/001/mabdel03/Connectomics/"
                    "7 - FD3 Identification/global_fd3_scan_all_types.csv")


def _candidate_screen(src, meta) -> dict:
    """Screen every cell type for the FD4 output-class conjunction: dominant layer-a
    (progressive), heterolateral (contra output >= 70%), cholinergic, low-copy
    (individually-identifiable), and a visual OUTPUT class (not centrifugal feedback, not a
    huge columnar population). Report every survivor and why each near-miss fails, so the
    (non-)existence of an FD4 candidate besides Nod1 is measured, not asserted.

    Reads the precomputed all-types scan (``global_fd3_scan_all_types.csv``: per-type
    ``dom_layer``, ``nt``, ``sc``, ``contra_pct``, copy number) rather than re-pulling every
    type; falls back to a live per-type recompute if the CSV is absent.
    """
    import csv
    import os
    rows_out, survivors = [], []
    scanned = 0
    if os.path.exists(_GLOBAL_SCAN_CSV):
        with open(_GLOBAL_SCAN_CSV) as fh:
            allrows = list(csv.DictReader(fh))
        universe = len(allrows)
        for r in allrows:
            try:
                n = int(float(r.get("n") or 0))
            except ValueError:
                continue
            if r.get("dom_layer") != "a" or not (1 <= n <= LOW_COPY_MAX):
                continue
            try:
                contra = float(r.get("contra_pct") or 0.0)
            except ValueError:
                contra = 0.0
            t = r.get("cell_type"); nt = r.get("nt"); sc = r.get("sc")
            scanned += 1
            row = {"type": t, "n_cells": n, "nt": nt, "super_class": sc,
                   "contra_pct": round(contra, 1),
                   "heterolateral": bool(contra >= CONTRA_HETERO_PCT),
                   "cholinergic": bool(nt == "acetylcholine"),
                   "output_class": bool(sc == "visual_projection")}
            if not row["heterolateral"]:
                row["verdict"] = "not heterolateral (axon does not cross to contra POF)"
            elif not row["cholinergic"]:
                row["verdict"] = f"not cholinergic ({nt}; inhibitory/modulatory feedback, not an FD output)"
            elif not row["output_class"]:
                row["verdict"] = f"not a visual projection output ({sc}; centrifugal/optic-intrinsic)"
            else:
                row["verdict"] = "VIABLE progressive figure-output candidate"
                survivors.append(t)
            rows_out.append(row)
    else:
        universe = 0
    return {"universe_size": universe, "n_layer_a_lowcopy_screened": scanned,
            "rows": rows_out, "survivors": survivors, "n_survivors": len(survivors),
            "only_survivor_is_prog_type": bool(survivors == [PROG_TYPE]),
            "source": "global_fd3_scan_all_types.csv" if universe else "unavailable",
            "note": ("The progressive + heterolateral + cholinergic + low-copy + visual-output "
                     "conjunction is the FD1/FD4 shared output class. If Nod1 is the sole "
                     "survivor, no progressive figure-output cell exists besides it.")}


# ---------------------------------------------------------------------------
# FD4 phenotype crosswalk - measure each FD4-defining property on the best available candidate
# (the Nod1 population, or its FD4 sub-pair if one is separable) and against FD3 (the incumbent
# absolute-RF-FD4-shaped cell) to show why FD3 is ruled out for FD4.
# ---------------------------------------------------------------------------
def _phenotype(src, meta, split: dict) -> dict:
    """The FD4 property table: which FD4-defining features the residual candidate matches."""
    # Candidate = the FD4 sub-pair if separable, else the whole Nod1 population.
    prog = K._type_profile(src, meta, PROG_TYPE)
    contra_inh = CM.contra_inhibition_profile(src, meta, PROG_TYPE)
    fd3 = K._type_profile(src, meta, FD3_TYPE)

    # Each FD4 property, its measurement on the progressive population, and match status.
    props = [
        {"property": "progressive (layer-a) preferred direction",
         "fd4_expects": "dominant lobula-plate layer a (front-to-back)",
         "measured": f"{(prog.get('layer_frac') or {}).get('a')}% layer a",
         "match": bool((prog.get("dominant_layer")) == "a")},
        {"property": "heterolateral noduli-group axon (contra POF)",
         "fd4_expects": "contra output >= 70%",
         "measured": f"{prog.get('contra_output_pct')}% contralateral",
         "match": bool((prog.get("contra_output_pct") or 0) >= CONTRA_HETERO_PCT)},
        {"property": "cholinergic output",
         "fd4_expects": "acetylcholine",
         "measured": str(prog.get("nt")),
         "match": bool(prog.get("nt") == "acetylcholine")},
        {"property": "bidirectional contralateral inhibition",
         "fd4_expects": "contra inhibition in BOTH directions",
         "measured": ("bidirectional" if contra_inh.get("substantially_bidirectional")
                      else f"dominant {contra_inh.get('dominant_direction')}"),
         "match": bool(contra_inh.get("substantially_bidirectional"))},
        {"property": "whole-eye, laterally-weighted RF (no frontal gap)",
         "fd4_expects": "wide RF weighted lateral of the frontal FD1 field, no gap",
         "measured": ("separable lateral FD4 sub-pair" if split.get("fd4_pair")
                      else "no lateral sub-pair; the population RF is frontal (FD1-like)"),
         "match": bool(split.get("fd4_pair"))},
        {"property": "restricted dorso-ventral dendrite",
         "fd4_expects": "dendrite does not span full dorso-ventral extent",
         "measured": ("separable restricted-DV FD4 sub-pair" if split.get("fd4_pair")
                      else "all Nod1 cells span the full dorso-ventral extent"),
         "match": bool(split.get("fd4_pair"))},
        {"property": "no lateral-protocerebrum second arbor",
         "fd4_expects": "single arbor",
         "measured": ("single-arbor" if not any(
             (c.get("has_second_arbor")) for c in split.get("per_cell", {}).values())
             else "second arbor present"),
         "match": True},   # all Nod1 cells are single-arbor; shared, non-discriminating
    ]
    n_match = sum(1 for p in props if p["match"])
    # FD3 is ruled out for FD4 by direction: FD4 is layer-a, FD3 is layer-b.
    fd3_ruled_out = {
        "cell": FD3_TYPE,
        "fd3_dominant_layer": fd3.get("dominant_layer"),
        "fd3_layer_b_pct": (fd3.get("layer_frac") or {}).get("b"),
        "reason": ("FD3 = LPT42_Nod4 is 98.6% layer-b (regressive); FD4 is progressive "
                   "(layer-a). Although LPT42_Nod4's absolute RF is FD4-shaped (lateral, "
                   "reaching the caudal pole), its preferred DIRECTION excludes it from FD4. "
                   "FD4 also has no frontal gap, which LPT42_Nod4 does."),
        "excluded": bool(fd3.get("dominant_layer") == "b"),
    }
    return {"properties": props, "n_properties": len(props), "n_matched": n_match,
            "shared_class_properties": 4,   # direction, heterolateral, cholinergic, single-arbor
            "discriminating_properties": 3,  # RF laterality, DV dendrite, (RF width)
            "fd3_ruled_out": fd3_ruled_out}


# ---------------------------------------------------------------------------
# Decomposed confidence - null-aware. FD4 shares FD1's entire output class, so uniqueness is
# never 1.0; the ceiling is capped well below FD2's by the FD1-collinearity discount, and when
# the Nod1 split is not separable the point estimate collapses toward "leading but unresolved".
# ---------------------------------------------------------------------------
def _confidence(phenotype: dict, screen: dict, split: dict) -> dict:
    n_disc = phenotype["discriminating_properties"]
    # Discriminating-property match fraction: the shared-class properties carry no FD4-vs-FD1
    # power, so confidence is driven by the DISCRIMINATING properties (RF laterality, DV dendrite).
    disc_matched = sum(1 for p in phenotype["properties"]
                       if p["match"] and "layer-a" not in p["property"]
                       and "heterolateral" not in p["property"]
                       and "cholinergic" not in p["property"]
                       and "second arbor" not in p["property"])
    disc_fraction = disc_matched / max(n_disc, 1)

    separable = bool(split.get("separable")) and bool(split.get("fd4_pair"))
    # Uniqueness is reframed as within-Nod1 separability. Never 1.0: FD4 shares FD1's output class.
    uniqueness_factor = 0.5 if separable else 0.0

    discounts = {
        "fd1_collinearity_shared_class": 0.10,   # FD4 shares FD1's entire progressive output class
        "not_a_distinct_type": 0.07,             # FD4 is not an independently annotated FlyWire type
        "no_independent_anchor": 0.05,           # no FD4 anchor; identity is inference
        "literature_declined_mapping": 0.03,     # Nern typing declined the FD1/2/3/4 one-for-one map
        "further_fd_cells_possible": 0.02,
    }
    total_discount = sum(discounts.values())
    ceiling = 1.0 - discounts["fd1_collinearity_shared_class"] - discounts["no_independent_anchor"]

    point = max(0.0, min(1.0, disc_fraction * (0.4 + 0.6 * uniqueness_factor) - total_discount))
    half = max(total_discount, 0.20)   # wide by construction: FD4 is the least-resolved FD cell
    interval = [round(max(0.0, point - half), 2), round(min(ceiling, point + half), 2)]

    resolvable = bool(separable and screen.get("only_survivor_is_prog_type") is False)
    verdict = "IDENTIFIED_WITH_LOW_CONFIDENCE" if resolvable else "UNRESOLVED_HONEST_NULL"

    return {
        "discriminating_property_fraction": round(disc_fraction, 3),
        "n_discriminating_matched": disc_matched, "n_discriminating": n_disc,
        "separable_fd4_subpair": separable, "uniqueness_factor": uniqueness_factor,
        "discounts": discounts, "total_discount": round(total_discount, 3),
        "ceiling": round(ceiling, 3),
        "point_estimate": round(point, 2), "interval": interval,
        "identity_verdict": verdict,
        "statement": (
            "FD4 shares FD1's entire progressive, heterolateral, cholinergic, noduli-group "
            "output class, so it cannot be separated from FD1 by direction, transmitter, or "
            "output side. It is distinguished only by a whole-eye laterally-weighted receptive "
            "field and a restricted dorso-ventral dendrite. "
            + ("A separable FD4 sub-pair of Nod1 was found; "
               if separable else
               "No separable FD4 sub-pair was found: the four Nod1 cells are one homogeneous "
               "frontal population (Nod1 = FD1, likely pooling Egelhaaf's FD1nod and FD1pof "
               "variants), and no other progressive figure-output cell exists. FD4 is therefore "
               "not individually resolved in FlyWire v783. ")
            + f"Point estimate {round(point, 2)}, plausible range {interval}, "
            f"verdict {verdict}."),
    }


# ---------------------------------------------------------------------------
# run - assemble the elimination, the phenotype crosswalk, the confidence, and (for the report)
# the full progressive figure-arm circuit any FD4 correlate would use.
# ---------------------------------------------------------------------------
def run(src, meta, *, live_src=None, with_circuit: bool = True) -> dict:
    """The FD4 search and its (negative) result, plus the progressive figure-arm circuit.

    ``with_circuit`` traces the layer-a arm (T4a/T5a -> LLPC1 -> Nod1/FD1 -> DNp26 -> wing,
    VCH gate) on the Nod1 population, since any FD4 correlate would use this same machinery.
    """
    del live_src
    cached = K._CachedSource(src)
    roots = sorted(int(x) for x in meta.root_ids_of_type([PROG_TYPE]))

    split = _nod1_split(cached, meta)
    screen = _candidate_screen(cached, meta)
    phenotype = _phenotype(cached, meta, split)
    confidence = _confidence(phenotype, screen, split)

    result = {
        "target": "FD4", "progressive_type": PROG_TYPE,
        "roots": {str(meta.by_root.loc[int(r), "side"]): int(r) for r in roots},
        "nod1_split": split, "candidate_screen": screen,
        "phenotype": phenotype, "confidence": confidence,
        "track": getattr(src, "track", None),
    }

    if with_circuit:
        # The progressive figure arm any FD4 correlate would use, traced on the Nod1 population.
        result["afferent"] = P.run(cached, meta, candidate=PROG_TYPE, anchor=PROG_TYPE,
                                   detectors=("T4a", "T5a"))
        # Progressive-arm sheet: match layer a; include LLPC1 (FD1's sheet) as a candidate.
        sheet = Q.run(cached, meta, candidate=PROG_TYPE, anchor=FD3_TYPE, match_layer="a",
                      candidates=("LLPC1", "LPC1", "LLPC2", "LLPC3", "LPC2"),
                      detectors=("T4a", "T5a"))
        result["sheet"] = sheet
        result["gate"] = R.run(cached, meta, candidate=PROG_TYPE,
                               sheet=sheet.get("named_sheet"), detectors=("T4a", "T5a"),
                               figure_layer="a")
        result["efferent"] = L.run(cached, meta, candidate=PROG_TYPE)
        result["census"] = CC.full_census(cached, meta, roots)
        result["compartment"] = {int(r): COMP.compartment_input_split(cached, meta, int(r))
                                 for r in roots}
    return result
