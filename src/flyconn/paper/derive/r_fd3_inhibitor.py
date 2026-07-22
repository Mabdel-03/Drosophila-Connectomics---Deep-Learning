"""Derive family R: FD3's wide-field inhibitor(s) — the TWO gates of the FD3 circuit.

The mentor asked which cell occupies VCH's functional slot for FD3 (VCH pools wide-field motion,
feeds back onto the detectors, gates the sheet, and inhibits the figure cell). This family
screens FD3's inhibitory inputs and characterizes the winner as a functional-ROLE homolog of VCH.

Measured (v783): the OPPONENT gate is **LPi14** — GABAergic, lobula-plate INTRINSIC (super_class
optic, NOT centrifugal like VCH), that pools wide-field T4/T5 (72.6% of its input) tuned 94.8%
layer-a (PROGRESSIVE — the OPPONENT direction to FD3's layer-b), and:
  LPi14 -> FD3 961 syn (direct inhibition), -> T4b/T5b 7004 syn (feedback onto detectors),
  -> LPC1 sheet 6631 syn (gates the sheet). FD3 -> LPi14 only ~5 syn (NOT reciprocal with FD3).
So LPi14 is a functional-ROLE homolog of VCH, not a cell-type homolog: different class (intrinsic
vs centrifugal), opponent tuning (layer-a for a layer-b figure cell), non-reciprocal with the FD
cell.

Per the mentor: we ALSO search the full LPi panel for a REGRESSIVE (same-direction) wide-field
inhibitor matching Egelhaaf's predicted ipsilateral surround. Measured (v783): the same-direction
gate is **LPi12** — GABAergic optic LPi, tuned 99.6% layer-b (SAME direction as FD3), that pools
wide-field T4/T5 (66.5%) and DOMINATES detector-level feedback (-> T4b/T5b 18,678 syn, 2.7x LPi14)
but touches FD3 only weakly (-> FD3 68 syn = 2.5% of FD3's GABA input) and barely gates the sheet
(-> LPC1 165 syn). So the two gates are COMPLEMENTARY and act at DIFFERENT circuit nodes: LPi14 is
the opponent gate at the FD3/sheet node; LPi12 is the same-direction gate at the T4b/T5b DETECTOR
node. LPi12 does NOT replace LPi14 (its FD3/sheet contacts are minor slices of the totals). The
prior UNVERIFIABLE ``R.same_direction_surround`` verdict was an artifact of LPi12 never being in
the candidate set — the full-panel sweep here closes that gap.

CRITICAL departure from Family B: the screen does NOT require ``visual_centrifugal`` — that is the
whole point (FD3's gate is an optic LPi). Mirrors a_vch_loop + f_sheet_regulation + b_inhibitor_screen.
"""

from __future__ import annotations

import numpy as np

from . import common as CM
from . import q_fd3_sheet as Q
from .k_fd3_lpt42 import _CachedSource
from ..oracle import consts as C

FD3 = "LPT42_Nod4"
INHIBITOR = "LPi14"                 # the opponent/progressive gate (FD3/sheet node)
SURROUND = "LPi12"                  # the same-direction/regressive gate (detector node)
DETECTORS = ("T4b", "T5b")         # FD3's layer-b detectors
# Candidate wide-field inhibitors of FD3: the FULL LPi panel (LPi01..LPi15) so any same-direction
# surround is actually scored, plus the retained non-LPi controls (central GABA + the sole
# centrifugal control cLP03). NO centrifugal requirement (unlike Family B).
_LPI_PANEL = tuple(f"LPi{i:02d}" for i in range(1, 16))
_CONTROLS = ("PLP142", "Am1", "cLP03", "LLPt", "CT1")
CANDIDATES = _LPI_PANEL + _CONTROLS
# floors
FD3_INHIB_SYN_FLOOR = 100          # VCH-role: direct inhibition of FD3 (the opponent-gate floor)
FD3_SURROUND_SYN_FLOOR = 20        # surround-role: weaker direct-FD3 floor (LPi12 -> FD3 = 68)
DETECTOR_FEEDBACK_FLOOR = 1000
SHEET_GATE_FLOOR = 1000
WIDEFIELD_T4T5_FRAC_FLOOR = 40.0   # % of the cell's input that is T4/T5 (wide-field pooling)
# floor-sensitivity grid for the direct-FD3 threshold (report verdict-vs-threshold)
FD3_FLOOR_GRID = (30, 50, 68, 100, 165, 300)
# enrichment null (LPi12 -> FD3 vs a size-matched random LP-tangential target)
N_ENRICH_PERMS = 2000
ENRICH_SEED = 12345


def _fd3_roots(meta, candidate: str = FD3):
    return sorted(int(x) for x in meta.root_ids_of_type([candidate]))


def _type_roots(meta, ct):
    return sorted(int(x) for x in meta.root_ids_of_type([ct]))


def _to_targets(src, roots, target_set) -> int:
    """Total synapses from a cell type onto a target root set (e.g. the sheet, or the detectors)."""
    if not roots or not target_set:
        return 0
    out = src.synapses(pre_ids=list(roots)); out = out[out["pre_pt_root_id"].isin(set(roots))]
    return int(len(out[out["post_pt_root_id"].isin(target_set)]))


def _reached(src, roots, target_set) -> int:
    """Distinct target cells contacted by a cell type."""
    if not roots or not target_set:
        return 0
    out = src.synapses(pre_ids=list(roots)); out = out[out["pre_pt_root_id"].isin(set(roots))]
    return int(out[out["post_pt_root_id"].isin(target_set)]["post_pt_root_id"].nunique())


def _sink_frac(to_target: int, nt: str, target_gaba_total: int) -> float | None:
    """Sink-normalized fraction: the candidate's synapses onto a target as a share of that target's
    total GABAergic (inhibitory) input. Only meaningful for a GABA candidate — the denominator is
    GABA-only, so a glutamate candidate's count over a GABA pool is a category error (returns None).
    (Guards against the numerator/denominator NT-mismatch flagged in the Stage-7 audit.)"""
    if nt != "gaba":
        return None
    denom = target_gaba_total or 1
    return round(100.0 * to_target / denom, 2)


def _widefield_profile(src, meta, roots) -> dict:
    """Mirror of a_vch_loop: the cell's T4/T5 pooling + its layer tuning + reciprocity with T4/T5."""
    rset = set(roots)
    ins = src.synapses(post_ids=roots); ins = ins[ins["post_pt_root_id"].isin(rset)]
    ic = CM.attach_meta(CM.partner_counts(ins, "pre_pt_root_id"), meta)
    total_in = int(ic["syn"].sum())
    t45_in = ic[ic["is_t4t5"]]
    t45_in_syn = int(t45_in["syn"].sum())
    outs = src.synapses(pre_ids=roots); outs = outs[outs["pre_pt_root_id"].isin(rset)]
    oc = CM.attach_meta(CM.partner_counts(outs, "post_pt_root_id"), meta)
    t45_out = oc[oc["is_t4t5"]]
    recip = set(t45_in["root_id"]) & set(t45_out["root_id"])
    return {
        "total_in_syn": total_in,
        "t4t5_in_syn": t45_in_syn,
        "t4t5_in_frac": round(100.0 * t45_in_syn / total_in, 1) if total_in else float("nan"),
        "t4t5_in_neurons": int(t45_in["root_id"].nunique()),
        "layer_a_pct": round(CM.layer_a_fraction(t45_in), 1) if len(t45_in) else float("nan"),
        "layer_b_pct": round(CM.layer_b_fraction(t45_in), 1) if len(t45_in) else float("nan"),
        "dominant_layer": CM.dominant_layer(t45_in) if len(t45_in) else None,
        "t4t5_reciprocal_n": int(len(recip)),
    }


def _inhibition_denominators(src, meta, fd3_set, det_set) -> dict:
    """Sink-side totals used to normalize a candidate's contact: FD3's total GABAergic input, and
    the detector pool's total GABAergic input. A candidate's ``-> FD3`` in RAW synapses confounds
    the candidate's cell number and its output budget; expressing it as a fraction of these totals
    answers "how much of FD3's (or the detectors') inhibition comes from this cell" — the only
    normalization that separates an FD3-gate from a detector-gate. Computed once per run."""
    fi = src.synapses(post_ids=list(fd3_set)); fi = fi[fi["post_pt_root_id"].isin(fd3_set)]
    fic = CM.attach_meta(CM.partner_counts(fi, "pre_pt_root_id"), meta)
    fd3_gaba = int(fic.loc[fic["nt_canonical"] == "gaba", "syn"].sum())
    di = src.synapses(post_ids=list(det_set)); di = di[di["post_pt_root_id"].isin(det_set)]
    dic = CM.attach_meta(CM.partner_counts(di, "pre_pt_root_id"), meta)
    det_gaba = int(dic.loc[dic["nt_canonical"] == "gaba", "syn"].sum())
    return {"fd3_total_gaba_in": fd3_gaba, "detector_total_gaba_in": det_gaba}


def _out_budget_and_fd3_rank(src, meta, roots, fd3_set) -> dict:
    """The candidate's total output synapse budget, and where FD3 ranks among its targets.

    Hub control: LPi14 is an extreme connectivity hub, so a large ``-> FD3`` count can be spillover
    from a big output budget. Source-normalized fractions (of the candidate's own output) and FD3's
    rank among the candidate's postsynaptic targets strip that hubness out."""
    rset = set(roots)
    out = src.synapses(pre_ids=list(roots)); out = out[out["pre_pt_root_id"].isin(rset)]
    total_out = int(len(out))
    by_target = out.groupby("post_pt_root_id").size().sort_values(ascending=False)
    # rank FD3's pair among the candidate's targets (1 = top); report the best of the two FD3 cells
    ranks = [int((by_target.index.get_loc(r)) + 1) for r in fd3_set if r in by_target.index]
    fd3_rank = min(ranks) if ranks else None
    # n_out_targets = distinct postsynaptic cells (len of the grouped index); do NOT use nunique()
    # on by_target, which counts distinct synapse-COUNT values, not distinct targets.
    return {"total_out_syn": total_out,
            "fd3_target_rank": fd3_rank, "n_out_targets": int(len(by_target))}


def _screen_candidate(src, meta, ct, fd3_set, sheet_set, det_set, denom,
                      figure_layer: str = "b") -> dict:
    roots = _type_roots(meta, ct)
    if not roots:
        return {"cell_type": ct, "present": False}
    wf = _widefield_profile(src, meta, roots)
    nt = str(meta.by_root.loc[roots[0], "nt_canonical"])
    sc = str(meta.by_root.loc[roots[0], "super_class"])
    to_fd3 = _to_targets(src, roots, fd3_set)
    to_detectors = _to_targets(src, roots, det_set)
    to_sheet = _to_targets(src, roots, sheet_set)
    sheet_reached = _reached(src, roots, sheet_set)
    budget = _out_budget_and_fd3_rank(src, meta, roots, fd3_set)
    n = len(roots)
    tot_out = budget["total_out_syn"] or 1
    fd3_gaba = denom.get("fd3_total_gaba_in") or 1
    det_gaba = denom.get("detector_total_gaba_in") or 1
    # direction label relative to the FIGURE cell's own layer: an inhibitor reading the OPPOSITE
    # horizontal layer is the OPPONENT gate; one reading the SAME layer is the same-direction
    # surround. FD2/FD3 are layer-b (opponent=a); FD1/FD4 are layer-a (opponent=b).
    opp = "a" if figure_layer == "b" else "b"
    dom = wf["dominant_layer"]
    direction = ("opponent" if dom == opp else "same_direction" if dom == figure_layer
                 else "vertical" if dom in ("c", "d") else "unknown")
    # VCH-role pass: wide-field pooling + inhibits FD3 + gates the sheet + feeds back on detectors.
    # NT must be inhibitory-capable (gaba or glutamate); NO centrifugal requirement.
    is_inhibitory_nt = nt in ("gaba", "glutamate")
    vch_role = bool(is_inhibitory_nt
                    and wf["t4t5_in_frac"] >= WIDEFIELD_T4T5_FRAC_FLOOR
                    and to_fd3 >= FD3_INHIB_SYN_FLOOR
                    and to_sheet >= SHEET_GATE_FLOOR
                    and to_detectors >= DETECTOR_FEEDBACK_FLOOR)
    # SURROUND-role pass (Egelhaaf's same-direction detector gate): wide-field pooling + strong
    # detector feedback + a WEAK direct-FD3 contact; NO sheet-gate requirement. This is the role
    # LPi12 fills — it dominates the detector node without gating the FD3/sheet node.
    surround_role = bool(is_inhibitory_nt
                         and wf["t4t5_in_frac"] >= WIDEFIELD_T4T5_FRAC_FLOOR
                         and to_detectors >= DETECTOR_FEEDBACK_FLOOR
                         and to_fd3 >= FD3_SURROUND_SYN_FLOOR)
    return {
        "cell_type": ct, "present": True, "n_cells": n,
        "nt": nt, "super_class": sc, "is_centrifugal": sc == "visual_centrifugal",
        "to_fd3_syn": to_fd3, "to_detectors_syn": to_detectors,
        "to_sheet_syn": to_sheet, "sheet_reached": sheet_reached,
        "direction": direction, **wf, **budget,
        # (1) per-cell normalization
        "to_fd3_per_cell": round(to_fd3 / n, 1), "to_detectors_per_cell": round(to_detectors / n, 1),
        # (2) source-normalized (fraction of the candidate's OWN output)
        "to_fd3_frac_of_output": round(100.0 * to_fd3 / tot_out, 2),
        "to_detectors_frac_of_output": round(100.0 * to_detectors / tot_out, 2),
        "to_sheet_frac_of_output": round(100.0 * to_sheet / tot_out, 2),
        # (3) sink-normalized (fraction of the TARGET's total GABA inhibition) — the decisive metric.
        # GABA-only (numerator and denominator both GABA); None for a glutamate candidate.
        "frac_of_fd3_inhibition": _sink_frac(to_fd3, nt, fd3_gaba),
        "frac_of_detector_inhibition": _sink_frac(to_detectors, nt, det_gaba),
        "vch_role": vch_role,
        "surround_role": surround_role,
    }


def _per_fd3_cell_syn(src, meta, ct, fd3) -> dict:
    """Synapses from a candidate type onto EACH FD3 cell (left/right agreement gate at n=2)."""
    roots = _type_roots(meta, ct)
    if not roots:
        return {}
    rset = set(roots)
    out = src.synapses(pre_ids=roots); out = out[out["pre_pt_root_id"].isin(rset)]
    out = out[out["post_pt_root_id"].isin(set(fd3))]
    by = out.groupby("post_pt_root_id").size().to_dict()
    return {str(r): int(by.get(r, 0)) for r in fd3}


def _floor_sweep(present: dict, det_floor: int) -> list[dict]:
    """Verdict-vs-threshold for the direct-FD3 floor. For each grid value, which cells clear a
    surround-style gate: wide-field pooling + detector-feedback (>= det_floor) + direct-FD3 >= floor.
    This gate deliberately does NOT require gating the sheet (the same-direction surround does not),
    so it shows how threshold-dependent the "inhibits FD3 above floor" call is for the surround
    (LPi12 -> FD3 = 68 offline / 107 live straddles the 100 floor)."""
    rows = []
    for floor in FD3_FLOOR_GRID:
        passing = sorted(
            ct for ct, r in present.items()
            if r.get("nt") in ("gaba", "glutamate")
            and (r.get("t4t5_in_frac") or 0) >= WIDEFIELD_T4T5_FRAC_FLOOR
            and (r.get("to_detectors_syn") or 0) >= det_floor
            and (r.get("to_fd3_syn") or 0) >= floor)
        rows.append({"fd3_floor": floor, "n_pass": len(passing), "pass": passing})
    return rows


def _fd3_enrichment_null(src, meta, ct, fd3, fd3_set) -> dict:
    """Is ``ct`` -> FD3 enriched over what a size-matched random LP-tangential target would receive?

    In-degree-preserving target permutation (mirrors k_fd3_lpt42/q_fd3_sheet nulls): the candidate
    sends a fixed number of output synapses; ask whether FD3 captures more of them than a random
    optic/visual_projection target with a comparable input budget. obs << null (few) => the
    candidate's FD3 contact is spillover from a promiscuous cell (a detector-gate that grazes FD3),
    not an FD3-specific input. obs >> null => a genuine, if minor, FD3-specific co-input."""
    roots = _type_roots(meta, ct)
    if not roots:
        return {"available": False}
    rset = set(roots)
    out = src.synapses(pre_ids=roots); out = out[out["pre_pt_root_id"].isin(rset)]
    obs = int(len(out[out["post_pt_root_id"].isin(fd3_set)]))
    # candidate output distributed over its targets; how concentrated onto FD3-sized targets?
    by_target = out.groupby("post_pt_root_id").size()
    # candidate pool of "FD3-like" alternative targets: optic/visual_projection cells the candidate
    # actually contacts, excluding FD3 itself (size-matched by being real postsynaptic partners).
    tgt_meta = meta.by_root.reindex(by_target.index)
    pool_mask = tgt_meta["super_class"].isin(["optic", "visual_projection"]).to_numpy()
    pool = by_target[pool_mask]
    pool = pool[~pool.index.isin(fd3_set)]
    if len(pool) < 5:
        return {"available": False, "reason": "too few comparable targets"}
    vals = pool.to_numpy()
    k = len(fd3)  # FD3 is a pair; draw k targets and sum their captured synapses
    rng = np.random.default_rng(ENRICH_SEED)
    null = np.empty(N_ENRICH_PERMS)
    for i in range(N_ENRICH_PERMS):
        null[i] = vals[rng.choice(len(vals), size=min(k, len(vals)), replace=False)].sum()
    null_mean = float(np.mean(null)); null_std = float(np.std(null))
    z = (obs - null_mean) / null_std if null_std else float("nan")
    # one-sided enrichment p (obs is high) with the 1/N floor
    p_enrich = max(float((null >= obs).mean()), 1.0 / N_ENRICH_PERMS)
    return {"available": True, "cell_type": ct, "obs_to_fd3": obs,
            "null_mean": round(null_mean, 2), "null_std": round(null_std, 2),
            "z_score": round(z, 2), "p_enrichment": p_enrich, "n_perms": N_ENRICH_PERMS,
            "enriched": bool(obs > null_mean)}


def _winner_block(present, named, fd3_to_named) -> dict:
    win = present.get(named, {})
    return {
        "cell_type": named, "nt": win.get("nt"), "super_class": win.get("super_class"),
        "is_centrifugal": win.get("is_centrifugal"), "direction": win.get("direction"),
        "layer_a_pct": win.get("layer_a_pct"), "layer_b_pct": win.get("layer_b_pct"),
        "t4t5_in_frac": win.get("t4t5_in_frac"),
        "to_fd3_syn": win.get("to_fd3_syn"), "to_detectors_syn": win.get("to_detectors_syn"),
        "to_sheet_syn": win.get("to_sheet_syn"), "t4t5_reciprocal_n": win.get("t4t5_reciprocal_n"),
        "frac_of_fd3_inhibition": win.get("frac_of_fd3_inhibition"),
        "frac_of_detector_inhibition": win.get("frac_of_detector_inhibition"),
        "to_fd3_frac_of_output": win.get("to_fd3_frac_of_output"),
        "to_detectors_frac_of_output": win.get("to_detectors_frac_of_output"),
        "fd3_target_rank": win.get("fd3_target_rank"),
        "fd3_to_inhibitor_syn": fd3_to_named,   # ~5: NOT reciprocal with FD3
    }


def run(src, meta, cfg: C.SideConfig = C.RIGHT, *, candidate: str = FD3,
        sheet: str | None = None, detectors: tuple = DETECTORS, figure_layer: str = "b",
        roots: list[int] | None = None) -> dict:
    # ``detectors`` and ``figure_layer`` select the arm: layer-b "T4b"/"T5b" for FD2/FD3,
    # layer-a "T4a"/"T5a" for the FD1/FD4 progressive arm (opponent/same-direction labels flip
    # accordingly). ``roots`` overrides the figure-cell type lookup for a sub-set of a type.
    del cfg
    src = _CachedSource(src)
    fd3 = [int(x) for x in roots] if roots is not None else _fd3_roots(meta, candidate)
    fd3_set = set(fd3)
    det_set = set(r for ct in detectors for r in _type_roots(meta, ct))
    # the named sheet-set (from Family Q). Defaults to FD3's LPC1 (Q.NAMED_SHEET); the FD2/FD4
    # orchestrators pass the re-derived direction-matched sheet.
    sheet_roots = set(_type_roots(meta, sheet or Q.NAMED_SHEET))
    denom = _inhibition_denominators(src, meta, fd3_set, det_set)

    screen = {ct: _screen_candidate(src, meta, ct, fd3_set, sheet_roots, det_set, denom,
                                    figure_layer=figure_layer)
              for ct in CANDIDATES}
    present = {ct: r for ct, r in screen.items() if r.get("present")}

    # OPPONENT gate (LPi14): fills the full VCH role (inhibits FD3 + gates sheet + feeds detectors).
    vch_pass = sorted([ct for ct, r in present.items() if r["vch_role"]])
    opponent = sorted([ct for ct in vch_pass if present[ct]["direction"] == "opponent"],
                      key=lambda ct: present[ct]["to_fd3_syn"], reverse=True)
    named = opponent[0] if opponent else (vch_pass[0] if vch_pass else None)

    # SAME-DIRECTION surround (LPi12): regressive wide-field cell dominating the DETECTOR node.
    # Uses the weaker surround_role gate (no sheet requirement, low direct-FD3 floor) and is ranked
    # by detector feedback (its dominant contact), not by -> FD3.
    surround_pass = sorted([ct for ct, r in present.items()
                            if r["surround_role"] and r["direction"] == "same_direction"],
                           key=lambda ct: present[ct]["to_detectors_syn"], reverse=True)
    same_direction = surround_pass
    sd_named = surround_pass[0] if surround_pass else None

    # non-reciprocity of the opponent winner with FD3 (the honest VCH departure)
    fd3_to_named = 0
    if named:
        so = src.synapses(pre_ids=fd3); so = so[so["pre_pt_root_id"].isin(fd3_set)]
        nm_set = set(_type_roots(meta, named))
        fd3_to_named = int(len(so[so["post_pt_root_id"].isin(nm_set)]))

    # per-FD3-cell L/R split for the two named gates (n=2 agreement gate)
    per_cell = {ct: _per_fd3_cell_syn(src, meta, ct, fd3)
                for ct in {named, sd_named} if ct}

    # floor sensitivity + the same-direction gate's FD3-specificity null
    floor_sweep = _floor_sweep(present, DETECTOR_FEEDBACK_FLOOR)
    enrichment = _fd3_enrichment_null(src, meta, sd_named, fd3, fd3_set) if sd_named else {"available": False}

    return {
        "candidate": candidate,
        "named_inhibitor": named,
        "same_direction_inhibitor": sd_named,
        "screen": screen,
        "denominators": denom,
        "pass_set": vch_pass,
        "opponent_gates": opponent,              # LPi14 etc (progressive; suppress via opponency)
        "same_direction_gates": same_direction,  # LPi12 etc (Egelhaaf's regressive detector surround)
        "winner": _winner_block(present, named, fd3_to_named),
        "same_direction_winner": _winner_block(present, sd_named, 0) if sd_named else None,
        "per_fd3_cell_syn": per_cell,
        "floor_sweep": floor_sweep,
        "same_direction_enrichment": enrichment,
        "track": src.track,
    }
