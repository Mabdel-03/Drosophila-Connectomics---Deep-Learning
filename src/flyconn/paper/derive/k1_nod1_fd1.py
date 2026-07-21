"""Derive family K1 (Stage-8 Phase 0): audit the RIGHT-side Nod1 = Egelhaaf-1985 FD1 anchor.

The project (and family K) TREAT Nod1 as the FD1 anchor — `k_fd3_lpt42.ANCHOR = "Nod1"` —
but never validated it the way LPT42_Nod4=FD3 was validated. Neither the Ziyin paper nor
Egelhaaf states "Nod1 = FD1"; it is Ziyin's interpretation, and there is a real cell-class
tension (Egelhaaf's FD1 is a lobula-plate TANGENTIAL cell; FlyWire Nod1 carries
super_class = visual_projection). This module TESTS Nod1 against Egelhaaf's FD1 criteria
instead of assuming them, and confronts the cell-class question head-on.

Egelhaaf 1985 FD1 (Biol. Cybern. 52:195-209, "The FD1-Cell", p.197-201, Figs 1-6):
  * excited by PROGRESSIVE (front-to-back) small-field motion  -> layer-a dominant
  * excitatory RF in the FRONTAL part of the IPSILATERAL eye (peak az ~10 deg, half-max
    width ~43 deg; frontal margin ~ -10 deg at the eye's edge)
  * inhibited by wide-field motion (the FD small-field signature)
  * dominant variant FD1nod = "noduli group": axon -> CONTRALATERAL posterior optic foci
    (a heterolateral output element); a second variant FD1pof projects ipsilaterally
  * an excitatory OUTPUT element of the optic lobe (cholinergic in the modern circuit)

Reuses the family-K machinery (``_type_profile``, ``_rf_per_side``, the RF/azimuth helpers,
``_synapse_cloud_morphology``) so the measured quantities are identical in definition to the
FD3 audit. The verdict logic differs: here Nod1 is the SUBJECT, scored against the absolute
FD1 spec and against the alternative candidates the two papers actually raise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as CM
from . import k_fd3_lpt42 as K  # reuse _type_profile, _rf_per_side, _synapse_cloud_morphology
from .. import geometry_hex as GH
from .. import azimuth_calibration as AZ
from ..oracle import consts as C

SUBJECT = "Nod1"  # the cell asserted to be FD1

# The alternatives the papers actually raise as FD-circuit elements (the "what else could
# FD1 be?" controls). VCH is the cell the Ziyin paper ties to FD selectivity; the LLPC1
# sheet is the retinotopic first-figure layer; Nod2/Nod5 are sibling Nod-types.
ALTERNATIVES = ("VCH", "LLPC1", "Nod2", "Nod5", "LPT42_Nod4")

# Egelhaaf FD1 reference numbers (for the absolute, calibrated RF reading; CWC at best
# because the azimuth calibration itself anchors its frontal pole to FD1).
FD1_PEAK_AZ_DEG = 10.0
FD1_HALFMAX_WIDTH_DEG = 43.0


def _frontality(cloud: GH.RFCloud) -> dict:
    """Is the cell's RF FRONTAL in its own eye? (the absolute FD1-frontal test).

    Reports the synapse-weighted centroid p, the frontal-band occupancy (fraction of RF
    weight in the most-frontal third of the lattice p-range), and the calibrated azimuth
    peak/width. Frontality is judged on p (lattice-frame, no calibration needed); degrees
    are corroborating (CWC) because the AZ calibration uses FD1 as its frontal anchor.
    """
    if not cloud.ok:
        return {"ok": False}
    cp, cq = GH.pq_centroid(cloud)
    p_lo, p_hi = AZ._p_extent()
    frontal_third_hi = p_lo + (p_hi - p_lo) / 3.0
    frontal_occ = GH.pq_band_occupancy(cloud, p_lo, frontal_third_hi)
    # is the centroid in the frontal half of the p-range?
    frontal_centroid = bool(cp < (p_lo + p_hi) / 2.0)
    az_peak = float(AZ.pq_to_azimuth(cp))
    az_width = float(AZ.width_p_to_degrees(GH.pq_width_p(cloud)))
    return {
        "ok": True,
        "centroid_p": round(cp, 3),
        "centroid_q": round(cq, 3),
        "frontal_third_occ": round(float(frontal_occ), 4),
        "frontal_centroid": frontal_centroid,
        "abs_peak_az_deg": round(az_peak, 1),
        "abs_width_deg": round(az_width, 1),
        "n_inputs": cloud.n_inputs,
        "dropout_frac": round(cloud.dropout_frac, 4) if np.isfinite(cloud.dropout_frac) else None,
    }


def _fd1_criteria_for_type(src, meta, cell_type: str) -> dict:
    """Score a cell type against Egelhaaf's FD1 criteria (used for Nod1 AND the alternatives).

    Returns the connectome-measurable FD1 features so the oracle can decide which candidate
    is the best FD1 match. For wide-field / sheet candidates (VCH, LLPC1) the RF block is
    pooled across the type's cells; frontality is reported but interpreted with care (a
    retinotopic sheet has no single RF; VCH is wide-field, not a frontal patch).
    """
    prof = K._type_profile(src, meta, cell_type)
    if not prof.get("n_cells"):
        return {"cell_type": cell_type, "n_cells": 0}

    # RF: pool the type's cells per side; use the side with the cleanest cloud.
    clouds = K._rf_per_side(src, meta, cell_type)
    frontal = {}
    for side, cloud in clouds.items():
        frontal[side] = _frontality(cloud)
    # representative RF (prefer right, else any present)
    rep_side = "right" if "right" in frontal else (next(iter(frontal), None))
    rep = frontal.get(rep_side, {"ok": False}) if rep_side else {"ok": False}

    layer = prof.get("layer_frac") or {}
    dom = prof.get("dominant_layer")
    contra = prof.get("contra_output_pct")
    nt = prof.get("nt")
    sc = prof.get("super_class")

    # FD1 binary criteria.
    is_layer_a = bool(dom == "a")
    is_progressive = is_layer_a  # layer-a == front-to-back == progressive
    is_excitatory = bool(nt == "acetylcholine")
    is_heterolateral = bool((contra or 0) >= 70.0)
    is_frontal = bool(rep.get("ok") and rep.get("frontal_centroid")
                      and (rep.get("frontal_third_occ") or 0) >= 0.30)
    # cell class: Egelhaaf FD1 is a lobula-plate tangential cell. FlyWire tangential cells
    # carry super_class in {optic, visual_centrifugal}; visual_projection is a VPN (downstream).
    is_tangential_class = sc in ("optic", "visual_centrifugal")

    n_fd1_features = int(is_progressive) + int(is_excitatory) + int(is_heterolateral) + int(is_frontal)
    return {
        "cell_type": cell_type,
        "n_cells": prof.get("n_cells"),
        "sides": prof.get("sides"),
        "nt": nt,
        "mean_nt_conf": prof.get("mean_nt_conf"),
        "super_class": sc,
        "dominant_layer": dom,
        "dominant_direction": prof.get("dominant_direction"),
        "layer_a_pct": layer.get("a"),
        "contra_output_pct": contra,
        "rf_frontal": frontal,
        "rep_side": rep_side,
        "is_progressive_layer_a": is_progressive,
        "is_excitatory_ach": is_excitatory,
        "is_heterolateral_contra": is_heterolateral,
        "is_frontal_rf": is_frontal,
        "is_tangential_cell_class": is_tangential_class,
        "n_fd1_functional_features": n_fd1_features,  # of 4 (excl. cell class, scored separately)
    }


def run(src, meta) -> dict:
    """Audit Nod1 vs FD1 on the RIGHT hemisphere, and score the alternatives.

    Headline outputs:
      * subject (Nod1) feature vector vs the FD1 spec
      * the alternatives (VCH, LLPC1 sheet, Nod2, Nod5, LPT42_Nod4) scored the same way
      * the cell-class verdict (tangential FD1 vs visual_projection VPN) — the named caveat
      * whether Nod1 is the best FUNCTIONAL FD1 match among the candidates
    """
    src = K._CachedSource(src)
    subject = _fd1_criteria_for_type(src, meta, SUBJECT)
    alts = {ct: _fd1_criteria_for_type(src, meta, ct) for ct in ALTERNATIVES}

    # Axis self-test: Nod1 must sit frontal in the lattice (pins lateral=+p). If this fails,
    # the whole p<->azimuth convention is suspect; record it rather than asserting (Phase 0
    # is an audit, not a gate that should crash).
    clouds = K._rf_per_side(src, meta, SUBJECT)
    anchor_for_selftest = clouds.get("right") or (next(iter(clouds.values())) if clouds else None)
    selftest = {"available": False}
    if anchor_for_selftest is not None and anchor_for_selftest.ok:
        cp, cq = GH.pq_centroid(anchor_for_selftest)
        selftest = {"available": True, "nod1_centroid_p": round(cp, 3),
                    "nod1_centroid_q": round(cq, 3), "frontal": bool(cp < 0)}

    # Best FUNCTIONAL FD1 match among all candidates (the 4 functional features; cell class
    # handled separately because no candidate is BOTH a tangential cell AND the frontal
    # excitatory readout — that is the whole tension).
    candidates = {SUBJECT: subject, **alts}
    scored = {ct: v.get("n_fd1_functional_features", -1) for ct, v in candidates.items()
              if v.get("n_cells")}
    best_functional = max(scored, key=scored.get) if scored else None
    ranked = sorted(scored.values(), reverse=True)
    margin = (ranked[0] - ranked[1]) if len(ranked) >= 2 else (ranked[0] if ranked else 0)

    # The cell-class confrontation: Nod1 is visual_projection (a VPN), NOT a lobula-plate
    # tangential cell. So the honest verdict is "functional/anatomical FD1-role correlate",
    # not "literally Egelhaaf's tangential FD1". Record the explicit statement.
    nod1_is_tangential = subject.get("is_tangential_cell_class", False)
    cell_class_verdict = ("tangential_FD1" if nod1_is_tangential
                          else "FD1_role_correlate_not_tangential")

    # Morphology proxy: does Nod1's axon cross to the contralateral side (the FD1nod
    # noduli-group signature)? Reuse the synapse-cloud morphology, but for SUBJECT.
    morph = _nod1_morphology(src, meta)

    return {
        "subject": SUBJECT,
        "subject_features": subject,
        "alternatives": alts,
        "axis_selftest": selftest,
        "best_functional_fd1_match": best_functional,
        "functional_match_margin": int(margin),
        "nod1_is_best_functional": bool(best_functional == SUBJECT),
        "nod1_is_tangential_cell": nod1_is_tangential,
        "cell_class_verdict": cell_class_verdict,
        "fd1_peak_az_ref_deg": FD1_PEAK_AZ_DEG,
        "fd1_width_ref_deg": FD1_HALFMAX_WIDTH_DEG,
        "calibration_error_budget": AZ.calibration_error_budget(),
        "morphology": morph,
        "track": src.track,
    }


def _nod1_morphology(src, meta) -> dict:
    """Skeleton-free axon-crossing check for Nod1 (the FD1nod noduli-group signature).

    Mirrors k_fd3_lpt42._synapse_cloud_morphology but for SUBJECT=Nod1: per cell, the
    dendrite->axon medio-lateral centroid shift and whether it crosses toward the
    contralateral hemisphere.
    """
    from .. import geometry as G
    roots = sorted(int(x) for x in meta.root_ids_of_type([SUBJECT]))
    if not roots:
        return {"available": False, "reason": "no Nod1 cells"}
    midline = float(meta.df["soma_x"].median()) / 1000.0
    cells = []
    for r in roots:
        side = str(meta.by_root.loc[r, "side"])
        din = src.synapses(post_ids=[r]); din = din[din["post_pt_root_id"] == r]
        dout = src.synapses(pre_ids=[r]); dout = dout[dout["pre_pt_root_id"] == r]
        dend = G.syn_positions_um(din, "post")
        axon = G.syn_positions_um(dout, "pre")
        dext = G.cloud_extent_um(dend)
        aext = G.cloud_extent_um(axon)
        ml_shift = (aext["centroid"][0] - dext["centroid"][0]) if (aext.get("n") and dext.get("n")) else None
        crossed = (None if ml_shift is None
                   else bool((ml_shift < 0) if side == "right" else (ml_shift > 0)))
        cells.append({
            "root_id": r, "side": side,
            "dv_span_um": dext.get("span_y"),
            "axon_ml_shift_um": round(ml_shift, 1) if ml_shift is not None else None,
            "axon_crosses_contra": crossed,
        })
    any_cross = any(c["axon_crosses_contra"] for c in cells if c["axon_crosses_contra"] is not None)
    return {"available": True, "source": "synapse_cloud_proxy",
            "midline_x_um": round(midline, 1), "any_axon_crosses_contra": bool(any_cross),
            "cells": cells}
