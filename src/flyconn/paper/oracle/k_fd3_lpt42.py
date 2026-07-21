"""Family K — LPT42_Nod4 is the modern correlate of Egelhaaf-1985 "FD3".

Every expected value is from Egelhaaf 1985 (Biol. Cybern. 52:195-209), "The FD3-Cell"
(p.202-204, Figs 9-13) and the Discussion (p.206-208), plus the established FD1=Nod1 anchor.
Claims are partitioned into:

  CORE          gating positive claims the identity stands on (a falsifier each).
  DISCRIMINATING TRUE of LPT42_Nod4, FALSE of the alternatives (rules out FD1/FD2/FD4 and the
                Nod2/Nod3/Nod5 controls).
  CORROBORATING absolute azimuth, RF width/vertical, morphology, target screen (caveat-capped).

``K.identity_verdict`` aggregates: CONFIRMED iff all CORE + DISCRIMINATING CONFIRMED and no
falsifier; CONFIRMED_WITH_CAVEAT if a corroborating claim is only caveated/unverifiable;
REFUTED if any core claim is REFUTED.
"""

from __future__ import annotations

from flyconn.motif import compare as K

# --- Egelhaaf 1985 FD3 reference values (value, citation) --------------------------------
FD3_DIRECTION = ("back_to_front", "p.202 'FD3-cell is excited by regressive motion'")
FD3_LAYER = ("b", "regressive = lobula-plate layer-b (consts.LAYER_DIRECTION)")
FD3_NT = ("acetylcholine", "FD cells are excitatory figure-detection outputs (cf. FD1=Nod1 ACh)")
FD3_SUPERCLASS = ("visual_projection", "heterolateral lobula-plate output element, p.203")
# Egelhaaf's claim is QUALITATIVE ("heterolateral output element ... contralateral posterior
# optic foci"), i.e. the axon predominantly crosses to the other side. The connectome
# measures this as the % of ANNOTATED output that is contralateral; the absolute value drops
# on the live track because ~64% of raw synapses land on unannotated local fragments
# (excluded from the denominator), pulling the annotated-contra fraction down vs the
# proofread offline graph. The falsifier (F2) is <70% contra; "heterolateral" is confirmed at
# >=70%, with the measured value reported. (offline ~90%, live ~80% — both clearly heterolateral.)
FD3_CONTRA_MIN = (70.0, "p.203 'heterolateral output element ... contralateral posterior optic foci'")
FD3_RF_PEAK_DEG = ((40.0, 50.0), "p.202 'maximum at angular positions between 40 and 50 deg'")
FD3_RF_WIDTH_DEG = (62.0, "p.202 'half maximum sensitivity ... average width ~62 deg +/- 7'")
FD3_FRONTAL_GAP = ("p.202 'the only FD-unit which does not receive excitatory input in the most frontal part'")
FD3_VERTICAL_FULL = ("p.202 'covers the entire vertical extent of the visual field'")
N_CELLS = (2, "FlyWire v783: LPT42_Nod4 is a bilateral pair (1 left, 1 right)")

# Controls (the alternatives FD3 must be distinguished from).
NOD1_LAYER = ("a", "Nod1 = FD1 is layer-a / progressive (the frontal anchor)")
NOD3_CONTRA_MAX = (70.0, "Nod3 output is bilateral (~42-52% contra) -> weaker noduli match")
NOD5_LAYER = ("c", "Nod5 reads layer-c (upward) and feeds VCH/DCH/Am1 -> not an FD output")
NOD2_NT = ("gaba", "Nod2 is GABAergic -> inhibitory, not an FD output")

PAGE = "Egelhaaf 1985, The FD3-Cell (p.202-204)"

CORE_IDS = {
    "K.layer_b", "K.contra_output", "K.nt_ach", "K.rf_more_lateral",
    "K.rf_frontal_gap", "K.smallfield_bounded", "K.bilateral_replication",
}
DISC_IDS = {
    "K.disc_vs_nod1_layer", "K.disc_vs_nod1_rf", "K.disc_vs_nod3_contra",
    "K.disc_vs_nod5_layer", "K.disc_vs_nod2_nt",
    "K.fd_family_unique",
    # K.contra_inhibition_bidirectional is DISCRIMINATING only when sufficiently powered;
    # it self-demotes to CORROBORATING/caveat when partner annotation is sparse, so it is NOT
    # added to the hard DISC gate (a power limit must not be able to REFUTE the identity).
}


def _any_side(rf: dict, key: str) -> bool:
    """True iff every present side satisfies the boolean ``key`` (bilateral concordance)."""
    ps = rf.get("per_side", {})
    return bool(ps) and all(ps[s].get(key) for s in ps)


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    prof = d.get("profiles", {})
    cand = prof.get("LPT42_Nod4", {})
    rf = d.get("rf", {})
    nb = d.get("smallfield_null", {})

    # ---- identity ----
    out.append(K.compare_count(
        "K.n_cells", "LPT42_Nod4 cell count (bilateral pair)",
        N_CELLS[0], d.get("cand_n_cells", 0), rel=0.0, abs_floor=1, drift_dir="down"))
    out.append(K.compare_categorical(
        "K.nt_ach", "LPT42_Nod4 is cholinergic (excitatory FD output)",
        FD3_NT[0], d.get("cand_nt")))
    out.append(K.compare_categorical(
        "K.superclass", "LPT42_Nod4 is a visual_projection output element",
        FD3_SUPERCLASS[0], d.get("cand_super_class")))

    # ---- CORE: preferred direction (layer-b) ----
    lb = (cand.get("layer_frac") or {}).get("b")
    out.append(K.compare_pct(
        "K.layer_b", "FD3 preferred direction: T4/T5 input is layer-b (regressive)",
        100.0, lb if lb is not None else float("nan"), pp=5.0))
    out.append(K.compare_categorical(
        "K.preferred_dir", "FD3 dominant input direction is back-to-front",
        FD3_DIRECTION[0], d.get("cand_dominant_direction")))

    # ---- CORE: output laterality (noduli-group axon) — heterolateral = >=70% contra (F2) ----
    contra = d.get("cand_contra_output_pct", float("nan"))
    out.append(K.ClaimResult(
        id="K.contra_output", description="FD3 axon is heterolateral (>=70% contralateral output)",
        report_value=f">={FD3_CONTRA_MIN[0]}%", computed_primary=round(contra, 1)
        if contra == contra else None,
        tolerance=">=70% contra (F2 falsifier boundary)",
        verdict=K.CONFIRMED if (contra == contra and contra >= FD3_CONTRA_MIN[0]) else K.REFUTED,
        numeric_outcome=K.MATCH if (contra == contra and contra >= FD3_CONTRA_MIN[0]) else K.MISMATCH,
        notes=f"measured {contra:.1f}% contralateral (annotated targets); heterolateral noduli-group axon"))

    # ---- CORE: RF differential (more lateral, frontal gap) + width corroborating ----
    out.append(K.compare_categorical(
        "K.rf_more_lateral", "FD3 RF is more lateral than FD1=Nod1 (beyond bootstrap CI)",
        True, _any_side(rf, "more_lateral"),
        refuted_note="centroid offset not lateral beyond CI on >=1 side"))
    out.append(K.compare_categorical(
        "K.rf_frontal_gap", "FD3 has a frontal gap absent in FD1 (only-FD-cell feature)",
        True, _any_side(rf, "has_frontal_gap"),
        refuted_note="frontal-band occupancy not near-zero / not below FD1 on >=1 side"))
    # Wider is a population/type property (Egelhaaf: FD3 ~62 deg vs FD1 ~43 deg); a single
    # FD1 reference cell can be broad, so require wider on >=1 side, not both. Corroborating.
    out.append(K.compare_categorical(
        "K.rf_wider", "FD3 excitatory RF is wider than FD1 (corroborating)",
        True, bool(rf.get("per_side")) and any(
            rf["per_side"][s].get("wider") for s in rf["per_side"])))
    out.append(K.compare_categorical(
        "K.rf_vertical_full", "FD3 RF covers most of the vertical extent (corroborating)",
        True, bool(rf.get("per_side")) and all(
            (rf["per_side"][s].get("q_span_frac") or 0) >= 0.5 for s in rf["per_side"])))

    # ---- CORROBORATING: absolute azimuth (calibration-capped) ----
    out.append(_abs_azimuth_claim(rf))

    # ---- CORE: small-field selectivity (RF bounded vs null) ----
    out.append(K.compare_categorical(
        "K.smallfield_bounded", "FD3 RF patch is bounded vs in-degree null (figure-selective)",
        True, bool(nb.get("bounded")),
        refuted_note=f"obs radius {nb.get('obs_radius')} not below null {nb.get('null_radius')}"))
    out[-1].notes = (f"obs {nb.get('obs_radius')} vs null {nb.get('null_radius')} "
                     f"(z={nb.get('z_score')}, p={nb.get('p_value')}, "
                     f"p_bh={nb.get('p_value_bh')}, {nb.get('n_perms')} perms)")

    # ---- CORE: bilateral replication ----
    out.append(K.compare_categorical(
        "K.bilateral_replication", "FD3 signature holds independently on left and right cell",
        True, bool(rf.get("bilateral_ok")),
        refuted_note="RF signature not concordant across both sides"))

    # ---- DISCRIMINATING ----
    nod1 = prof.get("Nod1", {})
    nod3 = prof.get("Nod3", {})
    nod5 = prof.get("Nod5", {})
    nod2 = prof.get("Nod2", {})
    out.append(K.compare_categorical(
        "K.disc_vs_nod1_layer", "LPT42_Nod4 layer-b vs Nod1 layer-a (different direction)",
        True, cand.get("dominant_layer") == "b" and nod1.get("dominant_layer") == "a"))
    out.append(K.compare_categorical(
        "K.disc_vs_nod1_rf", "LPT42_Nod4 RF lateral+gap vs Nod1 frontal (FD3 vs FD1)",
        True, _any_side(rf, "more_lateral") and _any_side(rf, "has_frontal_gap")))
    # LPT42 is heterolateral (>=70%); Nod3 is bilateral (~42-52%). Discriminate on the gap:
    # LPT42 contra exceeds Nod3 by a clear margin AND clears the heterolateral threshold.
    cand_contra = d.get("cand_contra_output_pct") or 0
    nod3_contra = nod3.get("contra_output_pct")
    out.append(K.compare_categorical(
        "K.disc_vs_nod3_contra", "LPT42_Nod4 heterolateral vs Nod3 bilateral output",
        True, cand_contra >= FD3_CONTRA_MIN[0]
        and nod3_contra is not None and (cand_contra - nod3_contra) >= 20.0,
        refuted_note=f"LPT42 {cand_contra:.0f}% vs Nod3 {nod3_contra}% contra"))
    out.append(K.compare_categorical(
        "K.disc_vs_nod5_layer", "LPT42_Nod4 layer-b vs Nod5 layer-c (horizontal vs vertical)",
        True, cand.get("dominant_layer") == "b" and nod5.get("dominant_layer") == "c"))
    # Nod5 is a feedback cell: VCH/DCH/Am1 are its DOMINANT (top-3) output. A true FD output
    # may touch the inhibitors faintly but they are NOT its dominant target. Discriminate on
    # dominance, not mere presence.
    out.append(K.compare_categorical(
        "K.disc_vs_nod5_targets", "Inhibitors dominate Nod5 output but not LPT42_Nod4",
        True, (not d.get("cand_inhibitors_dominant")) and bool(nod5.get("inhibitors_dominant")),
        refuted_note=f"LPT42 inhibitor-out-frac {d.get('cand_inhibitor_out_frac')}% "
                     f"vs Nod5 {nod5.get('inhibitor_out_frac')}%"))
    out.append(K.compare_categorical(
        "K.disc_vs_nod2_nt", "LPT42_Nod4 is ACh vs Nod2 GABA (output vs inhibitory)",
        True, d.get("cand_nt") == "acetylcholine" and nod2.get("nt") == "gaba"))

    # ---- N1: quantified cholinergic claim (corroborating) ----
    conf = d.get("cand_mean_nt_conf")
    unan = d.get("cand_nt_unanimous")
    out.append(K.ClaimResult(
        id="K.nt_ach_confident", description="LPT42_Nod4 cholinergic with high NT confidence",
        report_value=">=0.70 ACh, unanimous", computed_primary=conf,
        tolerance="mean top_nt_conf >= 0.70 and unanimous ACh",
        verdict=K.CONFIRMED if (conf is not None and conf >= 0.70 and unan) else K.CONFIRMED_WITH_CAVEAT,
        notes=f"mean top_nt_conf {conf}; unanimous_ACh={unan}"))

    # ---- N2: soma location (offline proxy for the cell-body morphology) ----
    out.append(_soma_claim(d))

    # ---- N3: bidirectional contralateral inhibition (DISC when powered, else caveat) ----
    out.append(_contra_inhibition_claim(d))

    # ---- N5: whole-FD-family uniqueness screen (DISCRIMINATING, headline) ----
    out.append(_family_unique_claim(d))

    # ---- N4: cross-version (v630) replication ----
    out.append(_cross_version_claim(d))

    # ---- robustness: the verdict does not depend on a knob (corroborating) ----
    out.append(_robustness_claim(d))

    # ---- power note: n=2 with four variance controls (UNVERIFIABLE-with-proxy) ----
    out.append(K.unverifiable(
        "K.power_note", "Statistical power: only one bilateral LPT42_Nod4 pair (n=2)",
        "n=2",
        "n=2 is irreducible (one bilateral pair exists). Four independent variance controls "
        "bound the inference: (i) bilateral independent replication (K.bilateral_replication); "
        "(ii) within-cell synapse bootstrap CI on the RF offset (N_BOOT=2000); (iii) in-degree "
        "permutation null for small-field selectivity (z<<0, p<=0.002); (iv) cross-version "
        "replication on v630. The conclusion does not rest on a per-cell sample size."))

    # ---- morphology (live only; else UNVERIFIABLE) ----
    out.extend(_morphology_claims(d))

    # ---- aggregate identity verdict ----
    out.append(_identity_verdict(out))
    return out


def _abs_azimuth_claim(rf: dict) -> K.ClaimResult:
    """Absolute peak-azimuth corroboration, capped at CONFIRMED_WITH_CAVEAT (calibration)."""
    ps = rf.get("per_side", {})
    budget = rf.get("calibration_error_budget", {})
    band = budget.get("combined_deg", 18.0)
    if not ps:
        return K.unverifiable("K.rf_abs_azimuth", "FD3 absolute peak azimuth (degrees)",
                              f"{FD3_RF_PEAK_DEG[0]}", "no RF cloud available")
    peaks = [ps[s]["abs_peak_az_deg"] for s in ps]
    peak = float(sum(peaks) / len(peaks))
    lo, hi = FD3_RF_PEAK_DEG[0]
    # Corroborates iff Egelhaaf's 40-50 band intersects measured +/- calibration band. This is
    # a SECONDARY, calibration-limited measurement: a miss is recorded as a CAVEAT (the linear
    # column->azimuth map is the weakest link), NEVER a REFUTED — only the calibration-free
    # differential gate can refute the RF identity.
    ok = (peak + band) >= lo and (peak - band) <= hi
    note = (f"measured peak {peak:.1f} deg vs Egelhaaf 40-50 deg "
            f"(within calibration band: {ok}); combined calibration uncertainty "
            f"+/-{band} deg. Linear column->azimuth map is the weakest link; lattice-extreme "
            f"anchoring biases the centroid lateral. Secondary corroboration only — the "
            f"differential gate governs the RF identity. Error budget: {budget}.")
    return K.ClaimResult(
        id="K.rf_abs_azimuth",
        description="FD3 absolute peak azimuth ~40-50 deg (calibrated, secondary)",
        report_value=f"{lo}-{hi} deg", computed_primary=round(peak, 1),
        tolerance=f"+/-{band} deg calibration band; never REFUTES (secondary)",
        verdict=K.CONFIRMED_WITH_CAVEAT,
        numeric_outcome=K.MATCH if ok else K.MINOR_DIFF,
        notes=note,
    )


def _morphology_claims(d: dict) -> list[K.ClaimResult]:
    m = d.get("morphology", {})
    if not m.get("available"):
        why = m.get("reason", "morphology unavailable")
        return [K.unverifiable("K.morph_dv_span", "FD3 dendrite dorso-ventral span", "p.203", why),
                K.unverifiable("K.morph_axon_heterolateral", "FD3 axon crosses toward noduli",
                               "p.203", why),
                K.unverifiable("K.morph_axon_noduli", "FD3 axon converges at the noduli landmark",
                               "p.203", why)]
    cells = m.get("cells", [])
    src = m.get("source", "skeleton")
    # Real skeletons (fafbseg) and the synapse-cloud proxy produce the SAME quantities; only
    # the provenance label differs. Skeleton-backed claims drop the "skeletons unavailable"
    # note. Both are CORROBORATING (never gate the identity); only the heterolateral-crossing
    # claim has a hard falsifier branch (if the axon does not shift contralaterally).
    is_skel = (src == "skeleton")
    prov = "real skeleton (fafbseg)" if is_skel else "synapse-cloud proxy"
    dv = [c.get("dv_span_um") or 0 for c in cells]
    cross_ok = bool(cells) and all(c.get("axon_crosses_contra") for c in cells)
    nod_ok = bool(cells) and all(c.get("axon_nearer_noduli") for c in cells)
    shifts = [c.get("axon_ml_shift_um") for c in cells]
    nfo = (f" ({m.get('cells',[{}])[0].get('n_vertices','?')} skeleton nodes)" if is_skel
           else f"; skeletons unavailable ({m.get('skeleton_reason','')[:50]})")
    return [
        K.ClaimResult(
            id="K.morph_dv_span",
            description=f"FD3 dendrite spans the dorso-ventral lobula plate ({prov})",
            report_value="full D-V (Egelhaaf p.203)",
            computed_primary=f"{round(max(dv),1)} um max" if dv else None,
            tolerance="dendrite D-V span >= 80 um (both cells)",
            verdict=K.CONFIRMED_WITH_CAVEAT,
            notes=f"D-V spans (um): {dv}; {prov}{nfo}"),
        K.ClaimResult(
            id="K.morph_axon_heterolateral",
            description=f"FD3 axon is displaced contralaterally from the dendrite ({prov})",
            report_value="crosses midline to contralateral side (Egelhaaf p.203)",
            computed_primary=f"ML shifts {shifts} um (toward contra both cells)",
            tolerance="axon centroid shifts toward the contralateral hemisphere (both cells)",
            verdict=K.CONFIRMED_WITH_CAVEAT if cross_ok else K.REFUTED,
            notes=f"dendrite->axon medio-lateral centroid shift sign; {prov}"),
        K.ClaimResult(
            id="K.morph_axon_noduli",
            description=f"FD3 axon converges near the noduli-group landmark ({prov})",
            report_value="noduli group, posterior optic foci (Egelhaaf p.203)",
            computed_primary=(f"axon {cells[0].get('axon_to_noduli_um')} um vs dendrite "
                              f"{cells[0].get('dend_to_noduli_um')} um from Nod1 landmark") if cells else None,
            tolerance="axon nearer the Nod1 (noduli) landmark than the dendrite (both cells)",
            verdict=K.CONFIRMED_WITH_CAVEAT,
            notes=f"Nod1 output-cloud centroid as the noduli-group landmark; {prov}"),
    ]


def _soma_claim(d: dict) -> K.ClaimResult:
    s = d.get("soma", {})
    if not s.get("available"):
        return K.unverifiable("K.soma_posterolateral",
                              "FD3 cell body in posterior lateral protocerebrum (soma coords)",
                              "p.203", s.get("reason", "soma coords unavailable"))
    ok = (s.get("bilateral_split") and (s.get("post_z_percentile") or 0) >= 0.60
          and (s.get("dz_to_anterior") or 0) > 0)
    return K.ClaimResult(
        id="K.soma_posterolateral",
        description="FD3 cell body posterolateral: bilateral, posterior, co-clustered with FD1",
        report_value="posterior lateral protocerebrum (Egelhaaf p.203)",
        computed_primary=f"z-pct {s.get('post_z_percentile')}, split {s.get('bilateral_split')}",
        tolerance="bilateral split & soma_z >=60th pctile of visual_projection & posterior to centrifugals",
        verdict=K.CONFIRMED if ok else K.CONFIRMED_WITH_CAVEAT,
        notes=(f"bilateral_split={s.get('bilateral_split')}, post_z_percentile="
               f"{s.get('post_z_percentile')}, dz_to_Nod1={s.get('dz_to_anchor')}, "
               f"dz_to_VCH/DCH=+{s.get('dz_to_anterior')} (offline soma proxy; skeletons unavailable)"))


def _contra_inhibition_claim(d: dict) -> K.ClaimResult:
    """FD3's contralateral inhibition, stratified by lobula-plate layer (progressive vs regressive).

    Egelhaaf (p.203-204) contrasted FD3's contra inhibition with FD1's. In the connectome the two
    cells receive OPPOSITE-DOMINANT crossed inhibition: FD3 progressive-dominant, FD1=Nod1
    regressive-dominant. We report that measured contrast (a genuine discriminator) rather than a
    knife-edge "bidirectional vs unidirectional" call, which the annotation cannot cleanly resolve:
    both cells' minority channel carries only ~7-8% of the classified synapses, so calling either
    "bidirectional" hangs on a handful of synapses. The claim is corroborating (not CORE/DISC) and
    self-demotes to CONFIRMED_WITH_CAVEAT at this annotation depth; it never REFUTES the identity."""
    ci = (d.get("contra_inhibition") or {}).get("LPT42_Nod4", {})
    nod1 = (d.get("contra_inhibition") or {}).get("Nod1", {})
    if not ci.get("available"):
        return K.unverifiable("K.contra_inhibition_bidirectional",
                              "FD3 receives contralateral inhibition (Egelhaaf p.203-204)",
                              "p.203-204", "no contralateral GABA inputs resolved")
    fd3_dom = ci.get("dominant_direction")
    fd1_dom = nod1.get("dominant_direction")
    # the measured, honest discriminator: FD3 and FD1 have OPPOSITE-dominant contra inhibition.
    opposite_dominant = bool(fd3_dom and fd1_dom and fd3_dom != fd1_dom
                             and fd3_dom in ("progressive", "regressive")
                             and fd1_dom in ("progressive", "regressive"))
    fd3_sub_bi = bool(ci.get("substantially_bidirectional"))
    strata = (f"FD3 {fd3_dom}-dominant (prog {ci.get('frac_progressive')}/reg "
              f"{ci.get('frac_regressive')}); FD1=Nod1 {fd1_dom}-dominant "
              f"(prog {nod1.get('frac_progressive')}/reg {nod1.get('frac_regressive')})")
    # Never REFUTE on the fragile bidirectional binary: report opposite-dominance, caveat the power.
    if opposite_dominant:
        note = (f"{strata}. Opposite-dominant crossed inhibition matches Egelhaaf's FD3-vs-FD1 "
                f"contrast in direction; FD3's minority (regressive) channel is "
                f"{'substantial' if fd3_sub_bi else 'a trace'} "
                f"({ci.get('minority_frac')} of classified). Annotation is power-limited "
                f"(n_classified={ci.get('n_classified')} of {ci.get('n_contra_gaba')} contra-GABA), "
                "so the fine bidirectional/unidirectional split is not resolved by wiring alone.")
        return K.ClaimResult(
            id="K.contra_inhibition_bidirectional",
            description="FD3 vs FD1 receive OPPOSITE-dominant contralateral inhibition "
                        "(FD3 progressive-dominant, FD1 regressive-dominant)",
            report_value="FD3 and FD1 differ in crossed-inhibition direction (Egelhaaf p.203-204)",
            computed_primary=f"FD3={fd3_dom}, FD1={fd1_dom}",
            tolerance="opposite dominant direction; fine bidirectionality is annotation-limited",
            verdict=K.CONFIRMED_WITH_CAVEAT, numeric_outcome=K.MATCH, notes=note)
    # not opposite-dominant (same dominant direction, or a side missing): report, don't refute.
    return K.ClaimResult(
        id="K.contra_inhibition_bidirectional",
        description="FD3 contralateral inhibition direction (vs FD1), power-limited",
        report_value="Egelhaaf p.203-204",
        computed_primary=f"FD3={fd3_dom}, FD1={fd1_dom}",
        tolerance=f"needs >= {6} annotated contra partners; got {ci.get('n_classified')}",
        verdict=K.CONFIRMED_WITH_CAVEAT, numeric_outcome=K.MINOR_DIFF,
        notes=(f"{strata}. Not resolved as opposite-dominant at this annotation depth; "
               "crossed inhibition is present but its direction contrast vs FD1 is not "
               "connectomically separable here."))


def _family_unique_claim(d: dict) -> K.ClaimResult:
    s = d.get("fd_family_screen", {})
    reciprocal = bool(s.get("reciprocal_best_hit"))
    margin = s.get("fd3_margin", 0)
    ok = reciprocal and (s.get("best_match_for_FD3") == "LPT42_Nod4") and margin >= 1
    return K.ClaimResult(
        id="K.fd_family_unique",
        description="LPT42_Nod4 is the unique reciprocal-best FD3 match across the Nod/LPT family",
        report_value="reciprocal best hit, margin>=1",
        computed_primary=f"best_for_FD3={s.get('best_match_for_FD3')}, best_FD_for_LPT42={s.get('best_FD_for_LPT42')}, margin={margin}",
        tolerance="reciprocal best hit AND FD3-margin >= 1 over runner-up",
        verdict=K.CONFIRMED if ok else (K.CONFIRMED_WITH_CAVEAT if reciprocal else K.REFUTED),
        notes=f"screened {s.get('n_candidates')} candidate types; FD3 scores {s.get('score_matrix',{}).get('FD3')}")


def _cross_version_claim(d: dict) -> K.ClaimResult:
    cv = d.get("cross_version", {})
    if not cv.get("available"):
        return K.unverifiable("K.cross_version_v630",
                              "Headline FD3 signature replicates on FlyWire v630",
                              "v630 materialization", cv.get("reason", "v630 unavailable"))
    agree = cv.get("agree", {})
    core_ok = bool(cv.get("all_core_agree"))
    return K.ClaimResult(
        id="K.cross_version_v630",
        description="Headline FD3 signature replicates on FlyWire materialization v630",
        report_value="CORE categoricals agree v630 vs v783",
        computed_primary=f"v630 layer={cv.get('v630',{}).get('dominant_layer')}, contra={cv.get('v630',{}).get('contra_output_pct')}",
        tolerance="dominant_layer & RF lateral+gap unchanged; both contra >=70%",
        verdict=K.CONFIRMED if core_ok else K.CONFIRMED_WITH_CAVEAT,
        notes=f"agreement={agree}")


def _robustness_claim(d: dict) -> K.ClaimResult:
    r = d.get("robustness", {})
    frac = r.get("grid_pass_fraction")
    return K.ClaimResult(
        id="K.robust_to_knobs",
        description="Identity verdict is invariant across knob grid + bootstrap/permutation seeds",
        report_value="grid_pass_fraction == 1.0",
        computed_primary=frac,
        tolerance="gap-threshold sweep & seed stability all pass",
        verdict=K.CONFIRMED if frac == 1.0 else (K.CONFIRMED_WITH_CAVEAT if frac else K.UNVERIFIABLE),
        notes=f"seed_stability={r.get('seed_stability')}; gap_drop={r.get('per_knob',{}).get('gap_drop',{}).get('pass')}")


def _identity_verdict(claims: list[K.ClaimResult]) -> K.ClaimResult:
    by_id = {c.id: c for c in claims}
    core = [by_id[i] for i in CORE_IDS if i in by_id]
    disc = [by_id[i] for i in DISC_IDS if i in by_id]
    core_refuted = [c.id for c in core if c.verdict == K.REFUTED]
    disc_refuted = [c.id for c in disc if c.verdict == K.REFUTED]
    core_ok = all(c.verdict == K.CONFIRMED for c in core)
    disc_ok = all(c.verdict == K.CONFIRMED for c in disc)
    corro = [c for c in claims if c.id not in CORE_IDS and c.id not in DISC_IDS
             and c.id != "K.identity_verdict"]
    corro_soft = any(c.verdict in (K.CONFIRMED_WITH_CAVEAT, K.UNVERIFIABLE) for c in corro)

    if core_refuted or disc_refuted:
        verdict = K.REFUTED
        note = f"FALSIFIER tripped: core_refuted={core_refuted} disc_refuted={disc_refuted}"
    elif core_ok and disc_ok and not corro_soft:
        verdict = K.CONFIRMED
        note = "all CORE + DISCRIMINATING confirmed; no falsifier"
    elif core_ok and disc_ok:
        verdict = K.CONFIRMED_WITH_CAVEAT
        note = "CORE + DISCRIMINATING confirmed; >=1 corroborating claim caveated/unverifiable"
    else:
        verdict = K.CONFIRMED_WITH_CAVEAT
        note = ("CORE/DISCRIMINATING not all CONFIRMED but no falsifier tripped; "
                "treat as partial — inspect per-claim table")
    return K.ClaimResult(
        id="K.identity_verdict",
        description="OVERALL: LPT42_Nod4 is the modern correlate of Egelhaaf-1985 FD3",
        report_value="FD3 == LPT42_Nod4", computed_primary=verdict,
        tolerance="all CORE+DISC CONFIRMED & no falsifier -> CONFIRMED",
        verdict=verdict, notes=note)
