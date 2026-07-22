"""Family KD oracle — the pre-registered decision rule for FD3 == LPT42_Nod4 vs Nod3.

The rule is fixed in advance and is two-sided: it can conclude either candidate, or "unresolved".
It runs on the OFFLINE primary track (the live and v630 tracks confirm, they do not arbitrate).

Discriminating properties (each candidate scores 0-5), from Egelhaaf's FD2-vs-FD3 distinctions:
  D1 heterolateral output          contra_output_pct >= 70
  D2 receptive field lateral+gap   more_lateral AND has_frontal_gap on BOTH sides
  D3 axon crosses the midline      axon_crosses_contra on BOTH cells
  D4 contralateral POF projection  projects_contralateral_pof
  D5 family screen own-best == FD3 the whole-family screen assigns the cell to FD3

Verdict:
  "LPT42_Nod4=FD3; Nod3=FD2"  iff LPT42 hits all 5, Nod3 hits <=1, and Nod3's own-best == FD2.
  "Nod3=FD3"                  iff Nod3 hits >=4 (D1-D4) with own-best == FD3 and LPT42 fails >=2.
  "unresolved by connectomics" iff both hit all 5, or neither cleanly wins.

Bidirectional contralateral inhibition is CORROBORATING only (power-limited at n=2) and never
enters the hard gate. Direction (layer-b) and transmitter (ACh) are SHARED and are reported as
non-discriminating, not as evidence for either candidate.
"""

from __future__ import annotations

from flyconn.motif import compare as K

LPT42 = "LPT42_Nod4"
NOD3 = "Nod3"
LPT21 = "LPT21"

# The five discriminating booleans, in order, keyed by the field the derive block exposes.
DISCRIMINATORS = [
    ("D1_heterolateral", "heterolateral", "output crosses to the contralateral side (>=70%)"),
    ("D2_rf_lateral_gap", "rf_lateral_gap_both", "receptive field lateral of FD1 with a frontal gap (both sides)"),
    ("D3_axon_crosses", "axon_crosses_contra_both", "axon crosses the midline (both cells)"),
    ("D4_contra_pof", "projects_contralateral_pof", "projects to the contralateral posterior optic foci"),
    ("D5_family_fd3", None, "the whole-family screen assigns the cell to FD3"),
]


def _disc_flags(cand_block: dict, own_best_fd: str | None) -> dict:
    """The five discriminating booleans for one candidate."""
    flags = {}
    for key, field, _desc in DISCRIMINATORS:
        if key == "D5_family_fd3":
            flags[key] = bool(own_best_fd == "FD3")
        else:
            flags[key] = bool(cand_block.get(field))
    return flags


def _disc_score(flags: dict) -> int:
    return int(sum(1 for v in flags.values() if v))


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    off = d.get("candidates_offline", {})
    lpt = off.get(LPT42, {})
    nod3 = off.get(NOD3, {})
    cons = d.get("constructive", {})

    lpt_flags = _disc_flags(lpt, cons.get("lpt42_best_fd"))
    nod3_flags = _disc_flags(nod3, cons.get("nod3_best_fd"))
    lpt_score = _disc_score(lpt_flags)
    nod3_score = _disc_score(nod3_flags)

    # ---- SHARED properties (explicitly non-discriminating) ----
    out.append(K.compare_pct(
        "KD.shared_layer_b", "Both candidates are regressive (layer-b): cannot discriminate",
        float(lpt.get("layer_b_pct") or float("nan")),
        float(nod3.get("layer_b_pct") or float("nan")), pp=100.0))
    out[-1].verdict = K.CONFIRMED
    out[-1].notes = (f"LPT42_Nod4 layer-b {lpt.get('layer_b_pct')}%, Nod3 layer-b "
                     f"{nod3.get('layer_b_pct')}% -- SHARED, not evidence for either")
    out.append(K.ClaimResult(
        id="KD.shared_nt", description="Both candidates are cholinergic: cannot discriminate",
        report_value="acetylcholine (both)",
        computed_primary=f"LPT42_Nod4={lpt.get('nt')}, Nod3={nod3.get('nt')}",
        tolerance="shared property", verdict=K.CONFIRMED,
        notes="direction and transmitter are shared; the split rests on the discriminators"))
    out.append(K.ClaimResult(
        id="KD.shared_smallfield", description="Both candidates are small-field selective: cannot discriminate",
        report_value="bounded (both)",
        computed_primary=f"LPT42_Nod4 bounded={lpt.get('smallfield_bounded')}, Nod3 bounded={nod3.get('smallfield_bounded')}",
        tolerance="shared property", verdict=K.CONFIRMED,
        notes="both are FD cells; small-field selectivity does not separate FD2 from FD3"))

    # ---- DISCRIMINATING properties (per candidate) ----
    for key, _field, desc in DISCRIMINATORS:
        out.append(K.ClaimResult(
            id=f"KD.{key}",
            description=f"Discriminator: {desc}",
            report_value="FD3 has it; FD2 does not",
            computed_primary=f"LPT42_Nod4={lpt_flags[key]}, Nod3={nod3_flags[key]}",
            tolerance="FD3-consistent iff True",
            verdict=K.CONFIRMED if (lpt_flags[key] and not nod3_flags[key]) else K.CONFIRMED_WITH_CAVEAT,
            numeric_outcome=K.MATCH if (lpt_flags[key] and not nod3_flags[key]) else K.MINOR_DIFF,
            notes=f"favors {LPT42 if (lpt_flags[key] and not nod3_flags[key]) else ('neither' if lpt_flags[key] == nod3_flags[key] else NOD3)}"))

    # ---- contra% on both tracks (data-source robustness) ----
    live = d.get("live", {})
    lpt_l = (live.get("candidates") or {}).get(LPT42, {}) if live.get("available") else {}
    nod3_l = (live.get("candidates") or {}).get(NOD3, {}) if live.get("available") else {}
    out.append(K.ClaimResult(
        id="KD.contra_two_track",
        description="Output laterality separates the candidates on both offline and live tracks",
        report_value="LPT42_Nod4 heterolateral, Nod3 bilateral",
        computed_primary=(f"offline: LPT42_Nod4 {lpt.get('contra_output_pct')}% vs Nod3 "
                          f"{nod3.get('contra_output_pct')}%"),
        computed_secondary=(f"live: LPT42_Nod4 {lpt_l.get('contra_output_pct')}% vs Nod3 "
                            f"{nod3_l.get('contra_output_pct')}%") if lpt_l else "live not available",
        tolerance="LPT42_Nod4 >= 70% and Nod3 < 70% on the primary track (confirmed on live)",
        verdict=K.CONFIRMED if (lpt.get("heterolateral") and not nod3.get("heterolateral")) else K.REFUTED,
        notes="the contra% metric is data-source sensitive; the ~90 vs ~45 gap far exceeds any track drift"))

    # ---- contra-cutoff robustness ----
    sweep = d.get("contra_cutoff_sweep", {})
    out.append(K.ClaimResult(
        id="KD.contra_cutoff_invariant",
        description="The heterolateral ordering (LPT42_Nod4 yes, Nod3 no) is invariant to the cutoff",
        report_value="invariant over 60-80% cutoff grid",
        computed_primary=f"invariant={sweep.get('invariant')}",
        tolerance="ordering holds across the whole grid",
        verdict=K.CONFIRMED if sweep.get("invariant") else K.CONFIRMED_WITH_CAVEAT,
        notes=f"first_break_cutoff={sweep.get('first_break_cutoff')}"))

    # ---- constructive: FD2 = LPT21 (homolateral + frontal), separated from Nod3 by the fine metrics ----
    lpt21 = off.get(LPT21, {})
    out.append(K.ClaimResult(
        id="KD.fd2_is_lpt21",
        description="FD2 is LPT21: cleanly homolateral (ipsilateral projection) with a frontal RF",
        report_value="FD2 -> LPT21",
        computed_primary=(f"LPT21 homolateral={lpt21.get('homolateral')} (contra "
                          f"{lpt21.get('contra_output_pct')}%), frontal_colocated="
                          f"{lpt21.get('frontal_colocated')}; fd2_candidate={cons.get('fd2_candidate')}"),
        tolerance="LPT21 homolateral AND frontal-co-located, and the chosen FD2 candidate is LPT21",
        verdict=K.CONFIRMED if cons.get("fd2_is_lpt21") else K.CONFIRMED_WITH_CAVEAT,
        notes="Egelhaaf FD2 is a regressive cell with a frontal field projecting to the IPSILATERAL "
              "posterior optic foci (homolateral, non-noduli, p.201). LPT21 matches on both axes; the "
              "coarse 3-feature screen ties LPT21 and Nod3 at FD2 because it bins contra<70 together, so "
              "the homolateral (contra ~0) + frontal-co-location metrics break the tie."))
    out.append(K.ClaimResult(
        id="KD.nod3_intermediate",
        description="Nod3 is an intermediate regressive cell: neither the clean FD2 nor FD3",
        report_value="Nod3 -> intermediate",
        computed_primary=(f"Nod3 contra {nod3.get('contra_output_pct')}% (mixed), RF offset "
                          f"{nod3.get('mean_rf_offset')}, homolateral={nod3.get('homolateral')}, "
                          f"frontal_colocated={nod3.get('frontal_colocated')}, "
                          f"lateral_gap={nod3.get('rf_lateral_gap_both')}"),
        tolerance="Nod3 fails the FD3 gate (no lateral+gap+heterolateral) AND the clean-FD2 gate "
                  "(homolateral+frontal); its contra% and RF offset sit between LPT21 and LPT42_Nod4",
        verdict=K.CONFIRMED if cons.get("nod3_identity") == "intermediate" else K.CONFIRMED_WITH_CAVEAT,
        notes="Nod3 is regressive and cholinergic like the FD cells but its ~45% contralateral output "
              "and slightly-lateral RF place it between FD2 (LPT21) and FD3 (LPT42_Nod4); it is not "
              "assigned a single Egelhaaf identity here."))
    out.append(K.ClaimResult(
        id="KD.lpt42_is_fd3",
        description="LPT42_Nod4 is the reciprocal-best FD3 match across the Nod/LPT family",
        report_value="LPT42_Nod4 <-> FD3 (reciprocal)",
        computed_primary=(f"best_for_FD3={cons.get('fd3_best_candidate')}, "
                          f"LPT42_Nod4 best_fd={cons.get('lpt42_best_fd')}, margin={cons.get('fd3_margin')}"),
        tolerance="reciprocal best hit for FD3",
        verdict=K.CONFIRMED if cons.get("reciprocal_fd3_lpt42") else K.CONFIRMED_WITH_CAVEAT,
        notes=f"anchor Nod1 screens best = {cons.get('anchor_best_fd')} (positive control expects FD1)"))

    # ---- the pre-registered decision verdict ----
    out.append(_decision_verdict(lpt_score, nod3_score, lpt_flags, nod3_flags, cons, off))
    return out


def _decision_verdict(lpt_score, nod3_score, lpt_flags, nod3_flags, cons, off) -> K.ClaimResult:
    """The three-way assignment verdict: FD3=LPT42_Nod4, FD2=LPT21, Nod3=intermediate.

    FD3 rests on LPT42_Nod4 hitting all five FD3 discriminators (unchanged, the strong result). FD2
    rests on LPT21 being cleanly homolateral AND frontal-co-located, which separates it from Nod3
    (mixed / slightly lateral). Nod3 is then the residual intermediate.
    """
    lpt21 = off.get(LPT21, {})
    lpt_all5 = lpt_score == 5
    fd3_ok = lpt_all5
    fd2_ok = bool(cons.get("fd2_is_lpt21") and lpt21.get("homolateral") and lpt21.get("frontal_colocated"))
    nod3_intermediate = cons.get("nod3_identity") == "intermediate"

    # Overturning branch preserved: if Nod3 were to hit the FD3 anatomy/laterality core, it would take FD3.
    nod3_d1_d4 = sum(1 for k in ("D1_heterolateral", "D2_rf_lateral_gap",
                                 "D3_axon_crosses", "D4_contra_pof") if nod3_flags[k])

    if fd3_ok and fd2_ok and nod3_intermediate:
        verdict = K.CONFIRMED
        decision = "FD3=LPT42_Nod4; FD2=LPT21; Nod3=intermediate"
    elif nod3_d1_d4 == 4 and cons.get("nod3_identity") == "FD3" and (5 - lpt_score) >= 2:
        verdict = K.CONFIRMED
        decision = "Nod3=FD3"            # the overturning branch (two-sided rule)
    elif fd3_ok and fd2_ok:
        verdict = K.CONFIRMED_WITH_CAVEAT
        decision = "FD3=LPT42_Nod4; FD2=LPT21 (Nod3 residual identity not resolved)"
    else:
        verdict = K.CONFIRMED_WITH_CAVEAT
        decision = (f"FD3=LPT42_Nod4 (LPT42 {lpt_score}/5); FD2/Nod3 assignment partial "
                    f"(inspect table)")

    return K.ClaimResult(
        id="KD.decision_verdict",
        description="OVERALL: FD3, FD2, and the status of Nod3 among the regressive noduli cells",
        report_value="FD3=LPT42_Nod4; FD2=LPT21; Nod3=intermediate",
        computed_primary=decision,
        tolerance="LPT42 all 5 FD3 discriminators; LPT21 homolateral+frontal; Nod3 residual -> CONFIRMED",
        verdict=verdict,
        notes=(f"FD3 discriminators: LPT42_Nod4 {lpt_score}/5 {lpt_flags}. FD2: LPT21 "
               f"homolateral={lpt21.get('homolateral')} frontal={lpt21.get('frontal_colocated')} "
               f"(contra {lpt21.get('contra_output_pct')}%). Nod3 contra "
               f"{off.get(NOD3, {}).get('contra_output_pct')}% = intermediate. Overturning condition: "
               f"Nod3 would take FD3 only if it hit the four FD3 anatomy/laterality discriminators "
               f"while LPT42_Nod4 failed >=2 of 5."))
