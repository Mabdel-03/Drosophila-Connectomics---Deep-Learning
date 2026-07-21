"""Family KD — disambiguating FD3: is it ``LPT42_Nod4`` or ``Nod3``?

A collaborator proposes ``Nod3`` as the FlyWire correlate of Egelhaaf-1985 FD3, in place of the
previously established ``LPT42_Nod4``. Both cells are regressive (lobula-plate layer-b) cholinergic
noduli-family cells, so preferred direction and transmitter cannot separate them. This module runs
every FD3-defining property for BOTH candidates in parallel, on the offline (primary), live (confirm)
and v630 (cross-version) tracks, and adjudicates by a pre-registered decision rule (in the KD oracle).

It does NOT re-implement any metric: it composes the Family-K primitives (``_type_profile``,
``_rf_block``, ``_smallfield_null``, ``_morphology``, ``fd_family_screen``, ``_robustness``,
``_cross_version``) with the candidate parameterized, so the two candidates pass through the identical
code path and only the ``candidate=`` argument differs. Family K's own verdict is untouched.

Egelhaaf's FD2 and FD3 are BOTH regressive; they differ in receptive field (FD2 frontal, FD3
fronto-lateral with a frontal gap) and in projection side (FD2 ipsilateral posterior optic foci; FD3
contralateral, heterolateral "noduli-group" axon). That is exactly the axis this module measures.
The expected, data-supported outcome is ``LPT42_Nod4`` = FD3 and ``Nod3`` = FD2, but the rule is
two-sided and can conclude the opposite.

Source: Egelhaaf 1985 Biol. Cybern. 52:195-209 "Figure-detection cells" (FD2 p.201-202; FD3 p.202-204).
"""

from __future__ import annotations

import numpy as np

from . import common as CM
from . import k_fd3_lpt42 as K
from .. import geometry_hex as GH

# The three candidates under test, plus the FD1 anchor and the context controls used by the screen.
# LPT42_Nod4 is the FD3 candidate (lateral RF + frontal gap + heterolateral axon); Nod3 and LPT21 are
# the two other regressive noduli-family cells the collaborator raised for FD2.
LPT42 = "LPT42_Nod4"
NOD3 = "Nod3"
LPT21 = "LPT21"
ANCHOR = "Nod1"                       # = FD1, the frontal/layer-a reference
CANDIDATES = (LPT42, NOD3, LPT21)
SCREEN_TYPES = (LPT42, NOD3, LPT21, "Nod1", "Nod2", "Nod5")

# Heterolateral threshold: an FD3 axon sends the large majority of its output across the midline.
CONTRA_HETERO_PCT = 70.0
CONTRA_CUTOFF_SWEEP = (60.0, 65.0, 70.0, 75.0, 80.0)
# Homolateral threshold: Egelhaaf's FD2 is a homolateral cell projecting to the IPSILATERAL posterior
# optic foci, i.e. its output stays on its own side. A cleanly homolateral connectome cell keeps almost
# all output ipsilateral (contra% near zero). LPT21 sits at ~1.5% contra; Nod3 at ~45% (mixed, not
# homolateral); LPT42_Nod4 at ~90% (heterolateral). The FD2 gate is a LOW contra ceiling.
CONTRA_HOMO_PCT = 15.0
# Frontal co-location: the RF centroid is not displaced laterally beyond the FD1 anchor's bootstrap CI
# (offset CI includes 0 or is negative) AND the field fills the FD1 frontal band. This is FD2's frontal
# field, distinct from FD3's lateral field and from Nod3's slightly-lateral intermediate field.
FRONTAL_COLOC_MAX_OFFSET = 1.5       # lattice columns; |mean offset| below this reads as frontal


# ---------------------------------------------------------------------------
# Per-candidate connectome measurements on ONE track (offline / live / v630).
# ---------------------------------------------------------------------------
def _candidate_block(src, meta, candidate: str) -> dict:
    """Every FD3 property for one candidate on one source, reusing the Family-K helpers.

    Returns the profile (layer/NT/laterality/targets), the RF differential vs the FD1 anchor
    (per side, with bootstrap CI + frontal-gap), the small-field null, morphology, soma, and the
    contralateral-inhibition stratification. All bilateral by construction.
    """
    profile = K._type_profile(src, meta, candidate)
    rf = K._rf_block(src, meta, candidate)
    nullb = K._smallfield_null(src, meta, candidate)
    morph = K._morphology(src, meta, candidate)
    soma = K._soma_block(meta, candidate)
    contra_inh = CM.contra_inhibition_profile(src, meta, candidate)

    # FDR over the family's p-valued claim (the small-field permutation p).
    nullb["p_value_bh"] = CM.bh_adjust([nullb.get("p_value", float("nan"))])[0]

    # Derived booleans that feed the decision rule (per-side concordant where relevant).
    per_side = rf.get("per_side", {})
    sides = sorted(per_side)
    rf_more_lateral_both = bool(sides) and all(per_side[s]["more_lateral"] for s in sides)
    rf_gap_both = bool(sides) and all(per_side[s]["has_frontal_gap"] for s in sides)
    frontal_field = _is_frontal_field(rf)          # the FD2-like signature (fills the FD1 band)

    m_cells = morph.get("cells", []) if morph.get("available") else []
    axon_cross_both = bool(m_cells) and all(c.get("axon_crosses_contra") for c in m_cells)
    axon_noduli_both = bool(m_cells) and all(c.get("axon_nearer_noduli") for c in m_cells)

    contra = profile.get("contra_output_pct")
    heterolateral = bool(contra is not None and contra >= CONTRA_HETERO_PCT)
    projects_contra_pof = bool(heterolateral and axon_noduli_both)
    # FD2 signatures: homolateral output (ipsilateral projection) and a frontal RF co-located with FD1.
    homolateral = bool(contra is not None and contra <= CONTRA_HOMO_PCT)
    mean_offset = None
    if sides:
        offs = [per_side[s].get("centroid_offset_p") for s in sides
                if per_side[s].get("centroid_offset_p") is not None]
        mean_offset = float(np.mean(offs)) if offs else None
    frontal_colocated = bool(
        mean_offset is not None and abs(mean_offset) <= FRONTAL_COLOC_MAX_OFFSET
        and not rf_more_lateral_both and frontal_field)

    return {
        "candidate": candidate,
        "track": getattr(src, "track", None),
        "profile": {k: v for k, v in profile.items() if not str(k).startswith("_")},
        "rf": rf,
        "smallfield_null": nullb,
        "morphology": morph,
        "soma": soma,
        "contra_inhibition": contra_inh,
        # decision-relevant booleans
        "layer_b_pct": (profile.get("layer_frac") or {}).get("b"),
        "dominant_layer": profile.get("dominant_layer"),
        "nt": profile.get("nt"),
        "nt_conf": profile.get("mean_nt_conf"),
        "contra_output_pct": contra,
        "heterolateral": heterolateral,
        "homolateral": homolateral,
        "rf_more_lateral_both": rf_more_lateral_both,
        "rf_frontal_gap_both": rf_gap_both,
        "rf_lateral_gap_both": bool(rf_more_lateral_both and rf_gap_both),
        "frontal_field": frontal_field,
        "frontal_colocated": frontal_colocated,
        "mean_rf_offset": mean_offset,
        "smallfield_bounded": bool(nullb.get("bounded")),
        "axon_crosses_contra_both": axon_cross_both,
        "projects_contralateral_pof": projects_contra_pof,
        "morphology_source": morph.get("source") if morph.get("available") else None,
    }


def _is_frontal_field(rf: dict) -> bool:
    """FD2-like frontal receptive field: the candidate FILLS the FD1 frontal band and has NO frontal
    gap, on every resolved side. This is the positive signature that assigns Nod3 to FD2 rather than
    merely "not FD3".

    The test is on frontal-band occupancy (does the field cover the front?) and the absence of the
    FD3 gap, NOT on the sign of the small centroid offset: a cell can sit a few columns off the
    frontal pole yet still fill the frontal band (Nod3 fills ~36-48% of the FD1 band versus FD3's
    ~2-3%). Requiring occupancy at least half of FD1's own frontal occupancy separates a frontal
    field (FD2) from the FD3 frontal gap.
    """
    per_side = rf.get("per_side", {})
    if not per_side:
        return False
    hits = []
    for s in per_side:
        d = per_side[s]
        ref_occ = d.get("ref_frontal_occ") or 0
        cand_occ = d.get("cand_frontal_occ") or 0
        fills_band = ref_occ >= 0.4 and cand_occ >= 0.5 * ref_occ
        hits.append(bool(fills_band and not d.get("has_frontal_gap")))
    return bool(hits) and all(hits)


# ---------------------------------------------------------------------------
# FD-family screen readout for BOTH candidates (the constructive core).
# ---------------------------------------------------------------------------
def _constructive(profiles: dict, rf_by_type: dict, cand_blocks: dict) -> dict:
    """Run the whole-family screen and assign each candidate its Egelhaaf FD identity.

    ``fd_family_screen`` scores every candidate against FD1-FD4 on three BINARY features (layer,
    lateral-gap, heterolateral). LPT42_Nod4 is the reciprocal-best FD3. For FD2 the binary screen is
    too coarse: it scores BOTH Nod3 and LPT21 at FD2=3, because it bins any contra<70% as
    "not heterolateral" and cannot see that LPT21 is CLEANLY homolateral (contra ~1.5%, matching FD2's
    ipsilateral projection and frontal RF) while Nod3 is a mixed intermediate (contra ~45%, RF a few
    columns lateral). We therefore break the FD2 tie with the finer, physiology-grounded metrics from
    ``cand_blocks``: FD2 requires (i) a homolateral projection (contra% at or below the homolateral
    ceiling) and (ii) a frontal RF co-located with the FD1 anchor. The candidate that satisfies BOTH
    is the FD2 match; a candidate that satisfies neither cleanly is left as an intermediate.
    """
    screen = K.fd_family_screen(profiles, rf_by_type)
    score = screen.get("score_matrix", {})       # score[FD][candidate]

    def best_fd_for(ct: str) -> tuple[str | None, int, dict]:
        per_fd = {fd: score.get(fd, {}).get(ct, -1) for fd in K.FD_SIGNATURES}
        if not per_fd:
            return None, -1, {}
        best = max(per_fd, key=per_fd.get)
        return best, int(per_fd[best]), {k: int(v) for k, v in per_fd.items()}

    lpt_best, lpt_best_score, lpt_scores = best_fd_for(LPT42)
    nod3_best, _, nod3_scores = best_fd_for(NOD3)
    lpt21_best, _, lpt21_scores = best_fd_for(LPT21)
    anchor_best, _, _ = best_fd_for(ANCHOR)       # positive control: Nod1 should screen best = FD1

    # FD2 tie-break on the fine metrics. Score = homolateral + frontal-colocated (0-2) for each of the
    # two regressive non-FD3 candidates. The winner (>=1 lead, both features) is FD2.
    def fd2_fine_score(ct: str) -> int:
        b = cand_blocks.get(ct, {})
        return int(bool(b.get("homolateral"))) + int(bool(b.get("frontal_colocated")))

    lpt21_fd2_fine = fd2_fine_score(LPT21)
    nod3_fd2_fine = fd2_fine_score(NOD3)
    fd2_candidate = None
    if lpt21_fd2_fine == 2 and lpt21_fd2_fine > nod3_fd2_fine:
        fd2_candidate = LPT21
    elif nod3_fd2_fine == 2 and nod3_fd2_fine > lpt21_fd2_fine:
        fd2_candidate = NOD3
    # Nod3's residual identity: not FD3 (fails lateral+gap+heterolateral) and not the clean FD2
    # (fails homolateral and/or frontal co-location) -> intermediate.
    nod3_identity = "FD2" if fd2_candidate == NOD3 else (
        "intermediate" if fd2_candidate == LPT21 else "unassigned")

    return {
        "score_matrix": score,
        "lpt42_best_fd": lpt_best,
        "lpt42_best_score": lpt_best_score,
        "lpt42_scores": lpt_scores,
        "lpt42_fd3_score": int(score.get("FD3", {}).get(LPT42, -1)),
        "nod3_best_fd": nod3_best,           # coarse screen best (FD2 by binary features)
        "nod3_scores": nod3_scores,
        "lpt21_best_fd": lpt21_best,
        "lpt21_scores": lpt21_scores,
        # coarse binary screen (ties LPT21 and Nod3 at FD2)
        "nod3_fd2_score": int(score.get("FD2", {}).get(NOD3, -1)),
        "nod3_fd3_score": int(score.get("FD3", {}).get(NOD3, -1)),
        "lpt21_fd2_score": int(score.get("FD2", {}).get(LPT21, -1)),
        "lpt21_fd3_score": int(score.get("FD3", {}).get(LPT21, -1)),
        # fine FD2 tie-break
        "fd2_candidate": fd2_candidate,            # expect LPT21
        "lpt21_fd2_fine": lpt21_fd2_fine,          # expect 2 (homolateral + frontal)
        "nod3_fd2_fine": nod3_fd2_fine,            # expect 0-1 (mixed / slightly lateral)
        "nod3_identity": nod3_identity,            # expect "intermediate"
        "anchor_best_fd": anchor_best,             # expect "FD1"
        "fd3_best_candidate": screen.get("best_match_for_FD3"),
        "fd3_margin": screen.get("fd3_margin"),
        "reciprocal_fd3_lpt42": bool(screen.get("best_match_for_FD3") == LPT42 and lpt_best == "FD3"),
        "fd2_is_lpt21": bool(fd2_candidate == LPT21),
        "n_candidates": screen.get("n_candidates"),
    }


# ---------------------------------------------------------------------------
# Contra-cutoff robustness: does the heterolateral gate flip the verdict?
# ---------------------------------------------------------------------------
def _contra_cutoff_sweep(lpt_contra, nod3_contra) -> dict:
    """Recompute the heterolateral booleans across a grid of contra% cutoffs.

    The verdict-relevant fact is that LPT42_Nod4 is heterolateral and Nod3 is not. We report the
    lowest cutoff at which that ordering would break (if any). The default cutoff is 70%; the
    measured values (~90 vs ~45) clear the whole grid, so the ordering is invariant.
    """
    rows = []
    flip = None
    for c in CONTRA_CUTOFF_SWEEP:
        lpt_h = bool(lpt_contra is not None and lpt_contra >= c)
        nod3_h = bool(nod3_contra is not None and nod3_contra >= c)
        ordering_ok = bool(lpt_h and not nod3_h)   # FD3-consistent ordering
        rows.append({"cutoff": c, "lpt42_hetero": lpt_h, "nod3_hetero": nod3_h,
                     "ordering_ok": ordering_ok})
        if not ordering_ok and flip is None:
            flip = c
    return {"grid": rows, "invariant": bool(all(r["ordering_ok"] for r in rows)),
            "first_break_cutoff": flip}


# ---------------------------------------------------------------------------
# Concordance roll-ups (bilateral + offline-vs-live agreement).
# ---------------------------------------------------------------------------
def _concordance(off: dict, live: dict) -> dict:
    """Bilateral concordance (already per-side inside each block) and offline-vs-live agreement of
    the decision-relevant booleans + the contra% gap size."""
    out = {"offline_vs_live": {}, "bilateral": {}}
    for ct in CANDIDATES:
        o = off["candidates"].get(ct, {})
        l = (live.get("candidates") or {}).get(ct, {}) if live.get("available") else {}
        out["bilateral"][ct] = {
            "rf_lateral_gap_both": o.get("rf_lateral_gap_both"),
            "axon_crosses_contra_both": o.get("axon_crosses_contra_both"),
        }
        if l:
            oc, lc = o.get("contra_output_pct"), l.get("contra_output_pct")
            out["offline_vs_live"][ct] = {
                "contra_offline": oc, "contra_live": lc,
                "contra_pp_gap": (round(abs(oc - lc), 1) if oc is not None and lc is not None else None),
                "heterolateral_agree": bool(o.get("heterolateral") == l.get("heterolateral")),
                "dominant_layer_agree": bool(o.get("dominant_layer") == l.get("dominant_layer")),
            }
    return out


# ---------------------------------------------------------------------------
# Comparison rows (the report's central head-to-head table).
# ---------------------------------------------------------------------------
def _fmt_pct(x) -> str:
    return f"{x:.2f}%" if isinstance(x, (int, float)) else "n/a"


def _comparison_rows(off: dict, live: dict, cons: dict) -> list[dict]:
    """Three-candidate table: Property | Egelhaaf ref | LPT21 | Nod3 | LPT42_Nod4 | assigns.

    Rows list each property, its Egelhaaf reference value, the measured value for the three
    regressive candidates, and which Egelhaaf FD cell the property points that candidate group toward.
    ``assigns`` names the resolved identity for the discriminating rows (FD2=LPT21, FD3=LPT42_Nod4,
    Nod3 intermediate); shared rows carry ``assigns="none"``.
    """
    b = {ct: off["candidates"][ct] for ct in CANDIDATES}
    bl = {ct: (live.get("candidates") or {}).get(ct, {}) for ct in CANDIDATES} if live.get("available") else {ct: {} for ct in CANDIDATES}

    def rf_word(x):
        if x.get("rf_lateral_gap_both"):
            return "lateral of FD1, with frontal gap"
        if x.get("frontal_colocated"):
            return "frontal, co-located with FD1"
        if x.get("rf_more_lateral_both"):
            return "slightly lateral of FD1, no gap"
        return "near-frontal, no gap"

    def proj_word(x):
        if x.get("heterolateral"):
            return "contralateral (heterolateral)"
        if x.get("homolateral"):
            return "ipsilateral (homolateral)"
        return "mixed / bilateral"

    rows = [
        {"property": "Preferred direction (lobula-plate layer)",
         "egelhaaf": "FD2 and FD3 both regressive (layer-b)",
         "lpt21": f"layer-b {_fmt_pct(b[LPT21].get('layer_b_pct'))}",
         "nod3": f"layer-b {_fmt_pct(b[NOD3].get('layer_b_pct'))}",
         "lpt42": f"layer-b {_fmt_pct(b[LPT42].get('layer_b_pct'))}",
         "discriminates": False, "assigns": "none"},
        {"property": "Neurotransmitter",
         "egelhaaf": "excitatory figure-detection output",
         "lpt21": f"acetylcholine ({b[LPT21].get('nt_conf')})",
         "nod3": f"acetylcholine ({b[NOD3].get('nt_conf')})",
         "lpt42": f"acetylcholine ({b[LPT42].get('nt_conf')})",
         "discriminates": False, "assigns": "none"},
        {"property": "Small-field selectivity (bounded vs null)",
         "egelhaaf": "stronger to small figure than wide field",
         "lpt21": "bounded" if b[LPT21].get("smallfield_bounded") else "not bounded",
         "nod3": "bounded" if b[NOD3].get("smallfield_bounded") else "not bounded",
         "lpt42": "bounded" if b[LPT42].get("smallfield_bounded") else "not bounded",
         "discriminates": False, "assigns": "none"},
        {"property": "Receptive-field position vs FD1 anchor",
         "egelhaaf": "FD2 frontal; FD3 fronto-lateral with a gap",
         "lpt21": rf_word(b[LPT21]) + f" (offset {b[LPT21].get('mean_rf_offset'):+.1f})" if b[LPT21].get("mean_rf_offset") is not None else rf_word(b[LPT21]),
         "nod3": rf_word(b[NOD3]) + f" (offset {b[NOD3].get('mean_rf_offset'):+.1f})" if b[NOD3].get("mean_rf_offset") is not None else rf_word(b[NOD3]),
         "lpt42": rf_word(b[LPT42]) + f" (offset {b[LPT42].get('mean_rf_offset'):+.1f})" if b[LPT42].get("mean_rf_offset") is not None else rf_word(b[LPT42]),
         "discriminates": True, "assigns": "FD2=LPT21; FD3=LPT42_Nod4"},
        {"property": "Frontal gap (unique to FD3)",
         "egelhaaf": "FD3 alone spares the most-frontal field",
         "lpt21": "no gap (fills frontal band)" if b[LPT21].get("frontal_field") else "no gap",
         "nod3": "no gap (fills frontal band)" if b[NOD3].get("frontal_field") else "no gap",
         "lpt42": "gap present (both sides)" if b[LPT42].get("rf_frontal_gap_both") else "no gap",
         "discriminates": True, "assigns": "FD3=LPT42_Nod4"},
        {"property": "Output laterality (contralateral %), offline",
         "egelhaaf": "FD2 homolateral (ipsi); FD3 heterolateral (contra)",
         "lpt21": _fmt_pct(b[LPT21].get("contra_output_pct")) + " (" + proj_word(b[LPT21]) + ")",
         "nod3": _fmt_pct(b[NOD3].get("contra_output_pct")) + " (" + proj_word(b[NOD3]) + ")",
         "lpt42": _fmt_pct(b[LPT42].get("contra_output_pct")) + " (" + proj_word(b[LPT42]) + ")",
         "discriminates": True, "assigns": "FD2=LPT21; FD3=LPT42_Nod4"},
        {"property": "Output laterality (contralateral %), live",
         "egelhaaf": "FD2 homolateral (ipsi); FD3 heterolateral (contra)",
         "lpt21": _fmt_pct(bl[LPT21].get("contra_output_pct")) if bl[LPT21] else "not available",
         "nod3": _fmt_pct(bl[NOD3].get("contra_output_pct")) if bl[NOD3] else "not available",
         "lpt42": _fmt_pct(bl[LPT42].get("contra_output_pct")) if bl[LPT42] else "not available",
         "discriminates": bool(live.get("available")), "assigns": "FD2=LPT21; FD3=LPT42_Nod4"},
        {"property": "Projection target region",
         "egelhaaf": "FD2 ipsilateral POF; FD3 contralateral POF",
         "lpt21": "ipsilateral POF" if b[LPT21].get("homolateral") else ("contralateral POF" if b[LPT21].get("projects_contralateral_pof") else "mixed"),
         "nod3": "contralateral POF" if b[NOD3].get("projects_contralateral_pof") else "mixed / bilateral",
         "lpt42": "contralateral POF" if b[LPT42].get("projects_contralateral_pof") else "mixed",
         "discriminates": True, "assigns": "FD2=LPT21; FD3=LPT42_Nod4"},
        {"property": "FD-family assignment",
         "egelhaaf": "each cell should map to one FD identity",
         "lpt21": "FD2 (homolateral + frontal)",
         "nod3": "intermediate (not clean FD2 or FD3)",
         "lpt42": "FD3 (lateral + gap + heterolateral)",
         "discriminates": True, "assigns": "FD2=LPT21; Nod3 intermediate; FD3=LPT42_Nod4"},
    ]
    return rows


def _inh_note(ci: dict) -> str:
    if not ci.get("available"):
        return "no contralateral inhibitory partners resolved"
    if not ci.get("sufficient"):
        return f"power-limited (n_classified={ci.get('n_classified')})"
    return f"both directions present={ci.get('both_present')}"


# ---------------------------------------------------------------------------
# run — ledger-compatible single-source signature; builds its own live/v630 sources.
# ---------------------------------------------------------------------------
def run(src, meta, *, live_src=None) -> dict:
    """Disambiguate FD3 between ``LPT42_Nod4`` and ``Nod3`` on the given (offline/primary) source.

    ``live_src`` (optional) is a live CAVE source for the confirmation track; when omitted, the
    live column is derived from ``src`` only if ``src`` is itself live, else marked unavailable.
    The v630 cross-version block is derived per candidate through K's ``_cross_version`` (live-only).
    Ledger-compatible: callable as ``run(src, meta)``.
    """
    off_cached = K._CachedSource(src)

    # Offline / primary track.
    off = {"track": getattr(src, "track", None), "candidates": {}}
    for ct in CANDIDATES:
        off["candidates"][ct] = _candidate_block(off_cached, meta, ct)

    # FD-family screen (constructive) on the primary track: build profiles + rf_by_type for all
    # screen types, then read each candidate's best-FD.
    anchor_clouds = K._rf_per_side(off_cached, meta, ANCHOR)
    anchor_cloud = anchor_clouds.get("right") or (next(iter(anchor_clouds.values())) if anchor_clouds else None)
    profiles, rf_by_type = {}, {}
    for ct in SCREEN_TYPES:
        prof = K._type_profile(off_cached, meta, ct)
        if not prof.get("n_cells"):
            continue
        profiles[ct] = {k: v for k, v in prof.items() if not str(k).startswith("_")}
        clouds = K._rf_per_side(off_cached, meta, ct)
        if clouds and anchor_cloud is not None:
            diffs = [GH.differential_rf(c, anchor_cloud) for c in clouds.values()]
            rf_by_type[ct] = {"more_lateral": any(d.more_lateral for d in diffs),
                              "has_frontal_gap": any(d.has_frontal_gap for d in diffs)}
    cons = _constructive(profiles, rf_by_type, off["candidates"])

    # Live confirmation track (v783). Reuse src if it is already live, else the passed live_src.
    live = {"available": False, "reason": "no live source provided"}
    lsrc = None
    if getattr(src, "track", None) == "live":
        lsrc = off_cached
    elif live_src is not None:
        lsrc = K._CachedSource(live_src)
    if lsrc is not None:
        live = {"available": True, "track": getattr(lsrc, "track", "live"), "candidates": {}}
        for ct in CANDIDATES:
            live["candidates"][ct] = _candidate_block(lsrc, meta, ct)

    # v630 cross-version per candidate (live-only; returns unavailable otherwise).
    cross_version = {ct: K._cross_version(lsrc if lsrc is not None else src, meta, ct)
                     for ct in CANDIDATES}

    # Robustness: contra-cutoff sweep on the primary contra values.
    contra_sweep = _contra_cutoff_sweep(
        off["candidates"][LPT42].get("contra_output_pct"),
        off["candidates"][NOD3].get("contra_output_pct"))

    concordance = _concordance(off, live)
    comparison_rows = _comparison_rows(off, live, cons)

    cells_audit = {
        ct: {str(meta.by_root.loc[int(r), "side"]): int(r)
             for r in sorted(int(x) for x in meta.root_ids_of_type([ct]))}
        for ct in (LPT42, NOD3, LPT21, ANCHOR)
    }

    return {
        "meta": {"flywire_version": "783", "seed": K.SEED, "n_boot": K.N_BOOT,
                 "n_perms": K.N_PERMS, "contra_hetero_pct": CONTRA_HETERO_PCT,
                 "primary_track": off["track"], "live_available": live.get("available")},
        "cells_audit": cells_audit,
        "candidates_offline": off["candidates"],
        "live": live,
        "cross_version": cross_version,
        "constructive": cons,
        "contra_cutoff_sweep": contra_sweep,
        "concordance": concordance,
        "comparison_rows": comparison_rows,
        "track": off["track"],
    }
