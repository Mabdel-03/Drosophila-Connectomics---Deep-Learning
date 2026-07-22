"""Family FD2 - LPT21 is Egelhaaf-1985 FD2, and its full photoreceptor-to-motor circuit.

Egelhaaf's FD2 cell is regressive (back-to-front), has a FRONTAL receptive field reaching the frontal
margin of the ipsilateral eye, is small-field selective, and is a HOMOLATERAL output element whose
main axon projects to the IPSILATERAL posterior optic foci, with a unique second axonal branch running
frontally to the anterior optic foci (Egelhaaf 1985, Biol. Cybern. 52:195-209, p.201-202, 207-208).
Its large-field inhibitory organisation "could not be resolved" physiologically. This module assembles
the connectome evidence that the FlyWire cell type ``LPT21`` is FD2, attaches a decomposed confidence
interval, and traces its complete circuit from photoreceptor input to motor output.

It composes, without re-implementing, the machinery the FD3 program already built:
  * identity battery  - ``kd_fd3_disambig._candidate_block`` (layer / RF-vs-FD1 / homolateral / frontal
                        / small-field / morphology / soma), run on LPT21.
  * afferent pathway  - ``p_fd3_input.run(candidate="LPT21")`` (photoreceptor -> lamina -> medulla ->
                        T4b/T5b -> LPT21, plus columnar sheets and inhibitory inputs).
  * sheet             - ``q_fd3_sheet.run(candidate="LPT21")`` (the direction-matched layer-b sheet).
  * inhibitory gate   - ``r_fd3_inhibitor.run(candidate="LPT21", sheet=<Q sheet>)``.
  * efferent pathway  - ``l_fd3_descending.run(candidate="LPT21")`` (LPT21 -> descending neurons ->
                        motor neurons/muscles in the male CNS).
  * connectivity      - ``connectivity_census.full_census`` + ``compartments.compartment_input_split``.

New to this module: the FD2 confidence-interval computation (per-property posteriors + a global
uniqueness scan + explicit discounting) and the dual-output bimodality metric (the connectome
correlate of Egelhaaf's second, frontal axonal branch).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as CM
from . import connectivity_census as CC
from . import compartments as COMP
from . import kd_fd3_disambig as KD
from . import k_fd3_lpt42 as K
from . import p_fd3_input as P
from . import q_fd3_sheet as Q
from . import r_fd3_inhibitor as R
from . import l_fd3_descending as L
from .. import geometry as G

CANDIDATE = "LPT21"
ANCHOR = "Nod1"                          # = FD1, the frontal/layer-a reference cell
CONTRA_HOMO_PCT = KD.CONTRA_HOMO_PCT     # <=15% contra output = homolateral (FD2 ipsilateral projection)
CONTRA_HETERO_PCT = KD.CONTRA_HETERO_PCT # >=70% = heterolateral (the FD3 signature LPT21 must NOT show)

# Universe for the global uniqueness scan: the lobula-plate tangential family the FD cells live in.
# We scan every LPT*/Nod* type that reads T4/T5, so uniqueness is measured over the relevant class,
# not asserted. (A candidate must be a small bilateral population reading T4/T5 to be an FD cell.)
SCAN_MIN_T4T5_SYN = 100
SCAN_MAX_CELLS = 10


# ---------------------------------------------------------------------------
# Identity block (lift KD's candidate battery for LPT21).
# ---------------------------------------------------------------------------
def _identity(src, meta) -> dict:
    """LPT21's FD-defining properties vs the FD1 anchor, via the KD candidate battery."""
    return KD._candidate_block(src, meta, CANDIDATE)


# ---------------------------------------------------------------------------
# Dual output: the connectome correlate of Egelhaaf's second (frontal) axonal branch.
# ---------------------------------------------------------------------------
def _dual_output(src, meta) -> dict:
    """Egelhaaf reports FD2 has TWO axonal outputs: the main ipsilateral posterior-optic-foci terminal
    plus a second branch running ~70-90 um frontally to the anterior optic foci (unique among FD
    cells, and the basis for his landing-response hypothesis). Its connectome correlate is a BIMODAL
    output-synapse field: two spatially separated terminal clusters. We test each LPT21 cell for a
    clear medio-lateral gap between two output-synapse lobes and report the gap and the two lobe sizes.
    """
    roots = sorted(int(x) for x in meta.root_ids_of_type([CANDIDATE]))
    cells = []
    for r in roots:
        side = str(meta.by_root.loc[r, "side"])
        out = src.synapses(pre_ids=[r]); out = out[out["pre_pt_root_id"] == r]
        pos = G.syn_positions_um(out, "pre")
        if len(pos) < 50:
            cells.append({"root_id": r, "side": side, "available": False})
            continue
        # Bimodality is detected from the medio-lateral HISTOGRAM valley (robust to synapse density),
        # not the largest gap between individual synapses: with thousands of dense synapses the
        # largest single gap is tiny even when the two terminal fields are clearly separated. We find
        # the deepest valley in the central region between the two flanking peaks and require it to be
        # a small fraction of those peaks (a genuine dip separating two lobes).
        x = pos[:, 0]
        n_bins = 20
        hist, edges = np.histogram(x, bins=n_bins)
        lo_i, hi_i = int(n_bins * 0.15), int(n_bins * 0.85)
        valley_i = lo_i + int(np.argmin(hist[lo_i:hi_i])) if hi_i > lo_i else int(np.argmin(hist))
        split = float(edges[valley_i + 1])
        valley = int(hist[valley_i])
        left_peak = int(hist[:valley_i].max()) if valley_i > 0 else 0
        right_peak = int(hist[valley_i + 1:].max()) if valley_i + 1 < n_bins else 0
        flank = max((left_peak + right_peak) / 2.0, 1.0)
        dip_ratio = valley / flank
        lo = pos[pos[:, 0] < split]; hi = pos[pos[:, 0] >= split]
        n_lo, n_hi = int(len(lo)), int(len(hi))
        minor_frac = min(n_lo, n_hi) / max(len(pos), 1)
        # Bimodal iff the central valley is a deep dip (<= 25% of the flanking peaks) AND the smaller
        # lobe holds a real share of the output (>= 8%), so it is two fields, not a tail.
        bimodal = bool(dip_ratio <= 0.25 and minor_frac >= 0.08 and left_peak > 0 and right_peak > 0)
        cells.append({
            "root_id": r, "side": side, "available": True,
            "n_output_syn": int(len(pos)), "ml_valley_dip_ratio": round(dip_ratio, 3),
            "flanking_peaks": [left_peak, right_peak], "valley_count": valley,
            "lobe_sizes": [n_lo, n_hi], "minor_lobe_frac": round(minor_frac, 3),
            "bimodal": bimodal,
            "lobe_lo_centroid_um": [round(float(v), 1) for v in lo.mean(axis=0)] if n_lo else None,
            "lobe_hi_centroid_um": [round(float(v), 1) for v in hi.mean(axis=0)] if n_hi else None,
        })
    avail = [c for c in cells if c.get("available")]
    both_bimodal = bool(avail) and all(c["bimodal"] for c in avail)
    return {"available": bool(avail), "cells": cells, "both_bimodal": both_bimodal,
            "note": ("two spatially separated output terminal fields per cell = the connectome "
                     "correlate of Egelhaaf's main ipsilateral POF terminal plus the frontal branch")}


# ---------------------------------------------------------------------------
# Global uniqueness scan (the confidence-interval backbone).
# ---------------------------------------------------------------------------
def _uniqueness_scan(src, meta, anchor_cloud=None) -> dict:
    """Scan the LP-tangential family for cells matching the FD2 conjunction and report the margin.

    FD2 conjunction: regressive (dominant layer-b), cholinergic, homolateral (contra <= 15%), a single
    bilateral pair (n == 2), and a bilaterally-consistent FRONTAL receptive field co-located with FD1.
    LPT21 should be the unique full match; the nearest competitor (a regressive homolateral cell that
    fails one clause) sets the margin. Every hit and near-miss is reported so uniqueness is measured.
    """
    df = meta.df
    universe = sorted(set(
        t for t in df["cell_type"].dropna().unique()
        if isinstance(t, str) and (t.startswith("LPT") or t.startswith("Nod")) and "e0" not in t))
    rows = []
    for t in universe:
        roots = sorted(int(x) for x in meta.root_ids_of_type([t]))
        if not (1 <= len(roots) <= SCAN_MAX_CELLS):
            continue
        rset = set(roots)
        si = src.synapses(post_ids=roots); si = si[si["post_pt_root_id"].isin(rset)]
        inc = CM.attach_meta(CM.partner_counts(si, "pre_pt_root_id"), meta)
        t45 = inc[inc["is_t4t5"]]
        t45syn = int(t45["syn"].sum())
        if t45syn < SCAN_MIN_T4T5_SYN:
            continue
        lb = CM.layer_fraction(t45, "b")
        nt = str(df[df["cell_type"] == t]["nt_canonical"].iloc[0])
        so = src.synapses(pre_ids=roots); so = so[so["pre_pt_root_id"].isin(rset)]
        ps = meta.by_root.reindex(so["pre_pt_root_id"].values)["side"].to_numpy()
        pos = meta.by_root.reindex(so["post_pt_root_id"].values)["side"].to_numpy()
        v = pd.notna(ps) & pd.notna(pos)
        contra = 100.0 * np.sum((ps != pos) & v) / max(int(v.sum()), 1)
        regressive = bool(lb == lb and lb >= 80.0)
        cholinergic = (nt == "acetylcholine")
        homolateral = bool(contra <= CONTRA_HOMO_PCT)
        single_pair = (len(roots) == 2)
        rows.append({
            "type": t, "n_cells": len(roots), "nt": nt, "t4t5_syn": t45syn,
            "layer_b_pct": round(lb, 1) if lb == lb else None, "contra_pct": round(contra, 1),
            "regressive": regressive, "cholinergic": cholinergic, "homolateral": homolateral,
            "single_pair": single_pair,
            "core_match": bool(regressive and cholinergic and homolateral),
            "full_match": bool(regressive and cholinergic and homolateral and single_pair),
        })
    core_hits = [r for r in rows if r["core_match"]]
    full_hits = [r for r in rows if r["full_match"]]
    # The frontal-RF clause is checked only for the small full-match set (RF is the expensive metric).
    for r in full_hits:
        r["frontal_rf"] = _frontal_rf_check(src, meta, r["type"])
    fd2_hits = [r for r in full_hits if r.get("frontal_rf")]
    lpt21_in = any(r["type"] == CANDIDATE for r in fd2_hits)
    competitors = [r for r in core_hits if r["type"] != CANDIDATE]
    return {
        "universe_size": len(universe), "n_scanned": len(rows),
        "core_hits": core_hits, "full_hits": full_hits, "fd2_hits": fd2_hits,
        "n_core_hits": len(core_hits), "n_full_hits": len(full_hits), "n_fd2_hits": len(fd2_hits),
        "lpt21_is_fd2_hit": lpt21_in, "unique": bool(len(fd2_hits) == 1 and lpt21_in),
        "competitors": competitors,
    }


def _frontal_rf_check(src, meta, cell_type: str) -> bool:
    """A cell has FD2's frontal RF iff, on BOTH sides, its T4/T5-input centroid is near-frontal (not
    lateral of the FD1 anchor beyond the bootstrap CI) and it fills the FD1 frontal band with no gap."""
    try:
        rf = K._rf_block(src, meta, cell_type)
    except Exception:  # noqa: BLE001 - RF axis self-test may refuse; treat as non-frontal
        return False
    per = rf.get("per_side", {})
    if len(per) < 2:
        return False
    ok = []
    for s in per:
        d = per[s]
        ref_occ = d.get("ref_frontal_occ") or 0
        cand_occ = d.get("cand_frontal_occ") or 0
        ok.append(bool(not d.get("more_lateral") and not d.get("has_frontal_gap")
                       and ref_occ >= 0.4 and cand_occ >= 0.5 * ref_occ))
    return bool(ok) and all(ok)


# ---------------------------------------------------------------------------
# Confidence interval (decomposed: per-property + uniqueness margin + discounting).
# ---------------------------------------------------------------------------
def _confidence(identity: dict, uniqueness: dict, dual: dict) -> dict:
    """A transparent, reconstructable FD2=LPT21 confidence: a point estimate and interval built from
    (1) the per-property match strengths, (2) the global-uniqueness margin, and (3) explicit
    discounting for the no-anchor caveat and FD2's unmeasured large-field physiology.
    """
    lb = identity.get("layer_b_pct")
    contra = identity.get("contra_output_pct")
    nb = identity.get("smallfield_null", {})
    props = [
        {"property": "regressive (layer-b) direction", "match": bool(lb is not None and lb >= 80),
         "support": f"{lb}% of T4/T5 input is layer-b", "strength": "near-certain"},
        {"property": "frontal receptive field (co-located with FD1)",
         "match": bool(identity.get("frontal_colocated")),
         "support": "RF centroid at the FD1 anchor (offset CI spans 0) on both sides, frontal band filled",
         "strength": "strong"},
        {"property": "homolateral projection (ipsilateral POF)",
         "match": bool(identity.get("homolateral")),
         "support": f"{contra}% of output is contralateral (near-zero = ipsilateral projection)",
         "strength": "near-certain"},
        {"property": "small-field selective",
         "match": bool(identity.get("smallfield_bounded")),
         "support": f"input patch bounded vs in-degree null (p={nb.get('p_value')})",
         "strength": "strong"},
        {"property": "cholinergic output",
         "match": bool(identity.get("nt") == "acetylcholine"),
         "support": f"acetylcholine, confidence {identity.get('nt_conf')}", "strength": "strong"},
        {"property": "second frontal axonal branch (dual output)",
         "match": bool(dual.get("both_bimodal")),
         "support": "two spatially separated output terminal fields per cell",
         "strength": "corroborating (unique to FD2)"},
    ]
    n_props = len(props)
    n_matched = sum(1 for p in props if p["match"])
    prop_fraction = n_matched / n_props

    unique = bool(uniqueness.get("unique"))
    n_fd2_hits = uniqueness.get("n_fd2_hits", 0)
    # Uniqueness factor: 1.0 if LPT21 is the sole full FD2 match in the scanned family, decaying with
    # additional hits.
    uniqueness_factor = 1.0 if (unique and n_fd2_hits == 1) else (1.0 / max(n_fd2_hits, 1))

    # Discounts (each caps the upper bound; stated so a reader can reconstruct the interval):
    discounts = {
        "no_independent_anchor": 0.05,      # FD2 has no anchor like FD1=Nod1; identity is inference
        "large_field_unmeasured": 0.03,     # Egelhaaf could not resolve FD2's large-field organisation
        "further_fd_cells_possible": 0.02,  # Egelhaaf could not exclude undiscovered FD cells
        "n_equals_2": 0.02,                 # one bilateral pair; robustness from L/R agreement, not N
    }
    total_discount = sum(discounts.values())

    # Point estimate: property fraction gated by uniqueness, minus the honest discounts.
    point = max(0.0, min(1.0, prop_fraction * (0.5 + 0.5 * uniqueness_factor) - total_discount))
    # Interval: the discount total sets the half-width, floored so it is never spuriously tight at n=2.
    # The upper bound is capped below certainty: without an independent FD2 anchor the connectome
    # cannot establish the identity with certainty, so the interval never reaches 1.0.
    half = max(total_discount, 0.05)
    ceiling = 1.0 - discounts["no_independent_anchor"]
    interval = [round(max(0.0, point - half), 2), round(min(ceiling, point + half), 2)]
    return {
        "properties": props, "n_properties": n_props, "n_matched": n_matched,
        "property_fraction": round(prop_fraction, 3),
        "uniqueness_unique": unique, "n_fd2_hits": n_fd2_hits,
        "uniqueness_factor": round(uniqueness_factor, 3),
        "discounts": discounts, "total_discount": round(total_discount, 3),
        "point_estimate": round(point, 2), "interval": interval,
        "statement": (f"LPT21 matches {n_matched}/{n_props} FD2-defining properties and is the "
                      f"{'unique' if unique else 'leading'} full FD2 match in the scanned "
                      f"lobula-plate tangential family ({n_fd2_hits} hit(s)). Point estimate "
                      f"{round(point, 2)}, plausible range {interval}, after discounting for the "
                      f"absence of an independent FD2 anchor, FD2's unmeasured large-field "
                      f"organisation, and n=2."),
    }


# ---------------------------------------------------------------------------
# run - assemble identity + confidence + full circuit.
# ---------------------------------------------------------------------------
def run(src, meta, *, live_src=None) -> dict:
    """FD2=LPT21 identity (with confidence) and the full photoreceptor-to-motor circuit.

    Ledger-compatible single-source signature. ``live_src`` is unused here (the identity confirmation
    tracks are handled by the verify script, which passes the live/v630 contra values separately)."""
    del live_src
    cached = K._CachedSource(src)
    roots = sorted(int(x) for x in meta.root_ids_of_type([CANDIDATE]))

    identity = _identity(cached, meta)
    dual = _dual_output(cached, meta)
    uniqueness = _uniqueness_scan(cached, meta)
    confidence = _confidence(identity, uniqueness, dual)

    # Full circuit: afferent (P), sheet (Q), gate (R), efferent (L). All parameterised on LPT21.
    afferent = P.run(cached, meta, candidate=CANDIDATE, anchor=ANCHOR)
    sheet = Q.run(cached, meta, candidate=CANDIDATE, anchor=ANCHOR)
    gate = R.run(cached, meta, candidate=CANDIDATE, sheet=sheet.get("named_sheet"))
    efferent = L.run(cached, meta, candidate=CANDIDATE)

    # Comprehensive connectivity + dendrite/axon compartment split (cell-agnostic reuse).
    census = CC.full_census(cached, meta, roots)
    compartment = {int(r): COMP.compartment_input_split(cached, meta, int(r)) for r in roots}

    return {
        "candidate": CANDIDATE, "anchor": ANCHOR,
        "roots": {str(meta.by_root.loc[int(r), "side"]): int(r) for r in roots},
        "identity": identity,
        "dual_output": dual,
        "uniqueness": uniqueness,
        "confidence": confidence,
        "afferent": afferent,
        "sheet": sheet,
        "gate": gate,
        "efferent": efferent,
        "census": census,
        "compartment": compartment,
        "track": getattr(src, "track", None),
    }
