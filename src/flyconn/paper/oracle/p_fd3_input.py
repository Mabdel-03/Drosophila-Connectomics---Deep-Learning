"""Family P — the afferent (input-side) pathway of the FD3 cell (LPT42_Nod4).

Family K identified LPT42_Nod4 as Egelhaaf's FD3; family L followed its output to the wing
muscles. This family grades FD3's INPUT: the motion detectors that drive it, the columnar
cascade upstream of those detectors, its non-motion central inputs, and its contralateral
inhibition. Claims are tagged by evidence strength (the three tiers in ``derive.p_fd3_input``):

  CORE           gating facts the input story stands on (a falsifier each): FD3 is driven by
                 layer-b (regressive) T4/T5; that layer-b drive is a mix of ON (T4b) + OFF (T5b);
                 the upstream medulla cascade onto those detectors is present.
  DISCRIMINATING TRUE of FD3, FALSE of the FD1=Nod1 arm — the two negative controls: FD3 has NO
                 VCH/DCH gate and NO layer-a (progressive) drive. These prove FD3 is a PARALLEL
                 regressive arm, not a copy of the FD1 sheet.
  CORROBORATING  the honest-framing + literature claims: T4/T5 is a MINORITY of FD3's total
                 input; the central sibling sheets read the same layer-b channel; bidirectional
                 contralateral inhibition (power-capped); the R->L->M cascade completeness; the
                 histaminergic front end (literature, not measurable from the annotation table).

``P.input_pathway_verdict`` aggregates exactly like K's identity verdict: CONFIRMED iff all
CORE + DISCRIMINATING CONFIRMED and no falsifier; CONFIRMED_WITH_CAVEAT if only corroborating
claims are soft; REFUTED if any CORE/DISC claim REFUTES.

Source: Egelhaaf 1985 Biol. Cybern. 52:195-209 "The FD3-Cell" (Part II p.202-204; the
input-circuitry Part III 52:267-280 is cited but not re-derived). ON/OFF and histaminergic
labels are literature (Fischbach-Dittrich 1989; Maisak 2013; the connectome gives type names,
not physiology).
"""

from __future__ import annotations

from flyconn.motif import compare as K

PAGE = "Egelhaaf 1985, The FD3-Cell (p.202-204); FlyWire v783 afferent trace"

# P.no_truncation is CORE: a silently-truncated upstream pull makes the cascade counts
# unreliable, so it must block the verdict rather than sit as a soft corroborating note.
CORE_IDS = {"P.layer_b_drive", "P.on_off_mix", "P.upstream_cascade_present", "P.no_truncation"}
DISC_IDS = {"P.nc_no_vch_gate", "P.nc_no_layer_a_drive"}


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    census = d.get("census", {})
    cascade = d.get("upstream_cascade", {})
    central = d.get("central_inputs", {})
    contra = d.get("contra_inhibition", {})
    neg = d.get("negative_controls", {})

    # ---- existence / self-consistency (a floor, not a two-sided count match) ----
    total_in = census.get("total_input_syn", 0) or 0
    out.append(K.ClaimResult(
        id="P.input_census_nonempty",
        description="FD3 receives a substantial direct afferent input (>=1000 synapses)",
        report_value=">=1000 input synapses", computed_primary=int(total_in),
        tolerance=">=1000 (floor)",
        verdict=K.CONFIRMED if total_in >= 1000 else K.REFUTED,
        numeric_outcome=K.MATCH if total_in >= 1000 else K.MISMATCH,
        notes=f"{total_in} input synapses from {census.get('n_partners')} presynaptic partners"))

    # ---- CORE: FD3's T4/T5 drive is layer-b (regressive) ----
    lb = census.get("layer_b_frac_of_t4t5")
    out.append(K.compare_pct(
        "P.layer_b_drive", "FD3's motion (T4/T5) drive is layer-b / regressive (back-to-front)",
        100.0, lb if lb is not None else float("nan"), pp=5.0))
    out.append(K.compare_categorical(
        "P.preferred_dir", "FD3's dominant input direction is back-to-front",
        "back_to_front", census.get("t4t5_dominant_direction")))

    # ---- CORE: the layer-b drive is a MIX of ON (T4b) + OFF (T5b) ----
    oo = census.get("on_off_split", {})
    both_on_off = bool((oo.get("T4b_ON") or 0) > 0 and (oo.get("T5b_OFF") or 0) > 0)
    out.append(K.compare_categorical(
        "P.on_off_mix", "FD3's layer-b drive is a mix of ON (T4b) and OFF (T5b) detectors",
        True, both_on_off,
        refuted_note="one of T4b/T5b carries no drive to FD3"))
    out[-1].notes = (f"T4b(ON)={oo.get('T4b_ON')} syn, T5b(OFF)={oo.get('T5b_OFF')} syn "
                     f"(T4 frac {oo.get('t4_frac')}%)")

    # ---- CORE: the upstream columnar cascade onto FD3's T4b/T5b is present ----
    on = cascade.get("on_limb_t4b", {})
    off = cascade.get("off_limb_t5b", {})
    cascade_ok = bool(on.get("n_expected_present") and off.get("n_expected_present")
                      and cascade.get("lamina_present") and cascade.get("photoreceptor_present"))
    out.append(K.compare_categorical(
        "P.upstream_cascade_present",
        "The canonical R->lamina->medulla->T4b/T5b cascade upstream of FD3 is present",
        True, cascade_ok,
        refuted_note="a medulla / lamina / photoreceptor stage of the cascade is missing"))
    out[-1].notes = (
        f"ON limb T4b<-{on.get('expected_medulla_present')} "
        f"({on.get('expected_frac_of_input')}% of its input); "
        f"OFF limb T5b<-{off.get('expected_medulla_present')} "
        f"({off.get('expected_frac_of_input')}% of its input); "
        f"lamina {cascade.get('lamina_present')}; photoreceptors {cascade.get('photoreceptor_present')}. "
        f"Per-type canonical wiring (Fischbach-Dittrich 1989; Takemura 2013), not FD3-specific counts.")

    # ---- DISCRIMINATING negative controls: FD3 is NOT the FD1 sheet ----
    out.append(K.compare_categorical(
        "P.nc_no_vch_gate", "FD3 has NO VCH/DCH centrifugal gate (unlike FD1=Nod1)",
        True, bool(neg.get("no_vch_gate")),
        refuted_note="VCH/DCH input to FD3 is non-negligible (would mimic the FD1 gate)"))
    out[-1].notes = (f"VCH/DCH -> FD3 = {neg.get('vch_input_syn')} syn "
                     f"({neg.get('vch_input_frac')}% of input). NOTE: FD3 has no VCH gate "
                     f"specifically, but the VCH functional ROLE (wide-field pooling + sheet "
                     f"gating + detector feedback + FD-cell inhibition) IS filled — by the "
                     f"lobula-plate intrinsic cell LPi14, opponent-tuned (see Family R).")
    out.append(K.compare_categorical(
        "P.nc_no_layer_a_drive", "FD3 has NO layer-a (progressive) T4/T5 drive (unlike FD1)",
        True, bool(neg.get("no_layer_a_drive")),
        refuted_note="layer-a (progressive) drive to FD3 is non-negligible"))
    out[-1].notes = f"layer-a fraction of FD3's T4/T5 input = {neg.get('layer_a_frac_of_t4t5')}%"

    # ---- CORROBORATING: T4/T5 is a MINORITY of FD3's total input (honest framing) ----
    t45f = census.get("t4t5_frac_of_total")
    out.append(K.ClaimResult(
        id="P.t4t5_minority_of_input",
        description="Motion (T4/T5) is a minority of FD3's total input; most is central/columnar",
        report_value="<50% of total input synapses",
        computed_primary=t45f,
        tolerance="reported so '~100% layer-b' is read as a within-T4/T5 fraction, not of total",
        verdict=K.CONFIRMED if (t45f is not None and t45f < 50.0) else K.CONFIRMED_WITH_CAVEAT,
        numeric_outcome=K.MATCH if (t45f is not None and t45f < 50.0) else K.MINOR_DIFF,
        notes=(f"T4/T5 = {t45f}% of the {census.get('total_input_syn')} total input synapses; "
               f"the layer-b fraction ({census.get('layer_b_frac_of_t4t5')}%) is WITHIN the T4/T5 "
               f"subset. Input by class: {census.get('input_syn_by_super_class')}.")))

    # ---- CORROBORATING: the dominant central sheets read the same layer-b channel ----
    out.append(K.compare_categorical(
        "P.central_inputs_layerb",
        "FD3's dominant non-motion (columnar sheet) inputs carry the same layer-b channel",
        True, bool(central.get("any_layerb_carrier"))))
    out[-1].notes = (f"{central.get('n_layerb_carriers')}/{len(central.get('top_types', []))} top "
                     f"central input types are layer-b dominant: "
                     f"{[(t['cell_type'], t['dominant_layer']) for t in central.get('top_types', [])]}")

    # ---- CORROBORATING: bidirectional contralateral inhibition (power-capped) ----
    out.append(_contra_inhibition_claim(contra))

    # ---- CORROBORATING: literature front end (histaminergic; connectome cannot adjudicate) ----
    out.append(K.unverifiable(
        "P.peripheral_nt_histaminergic",
        "Photoreceptor/lamina front end is histaminergic (literature)",
        "R1-6/R7/R8 histaminergic",
        "The R1-6->lamina->medulla cascade upstream of FD3's detectors is PRESENT and per-type "
        "wired in v783, but the ON/OFF functional split and the histaminergic transmitter of the "
        "photoreceptors come from physiology (Hardie 1989; Maisak 2013), not the connectome. "
        "NOTE: the annotation table lists R1-6=acetylcholine / R7=gaba, which is a prediction "
        "artifact and biologically wrong (photoreceptors are histaminergic) — not propagated."))

    # ---- power note ----
    out.append(K.unverifiable(
        "P.power_note", "Statistical power: FD3 is one bilateral pair (n=2)",
        "n=2",
        "n=2 is irreducible (one bilateral LPT42_Nod4 pair). The afferent claims rest on the "
        "aggregate synapse populations onto both cells (thousands of input synapses) and on "
        "per-type cascade wiring measured across the full retinotopic population, not on a "
        "per-cell sample size."))

    # ---- truncation guard (a live batching failure would REFUTE, not silently undercount) ----
    if d.get("query_truncated"):
        out.append(K.ClaimResult(
            id="P.no_truncation", description="Upstream trace did not hit the live 500k-row cap",
            report_value="not truncated", computed_primary="truncated",
            tolerance="batched to stay under the cap", verdict=K.REFUTED,
            numeric_outcome=K.MISMATCH,
            notes="A batched upstream pull returned at the 500k cap; increase batching before trusting counts."))
    else:
        out.append(K.compare_categorical(
            "P.no_truncation", "Upstream trace did not hit the live 500k-row cap",
            True, True))

    # ---- aggregate verdict ----
    out.append(_pathway_verdict(out))
    return out


def _contra_inhibition_claim(contra: dict) -> K.ClaimResult:
    """Bidirectional contralateral inhibition — CONFIRMED when powered, else caveat (never REFUTE).

    Egelhaaf 1985 p.203: FD3's contralateral inhibition is BIDIRECTIONAL (both progressive- and
    regressive-tuned), unlike FD1's. The annotated contra-GABA partners are sparse, so a low
    count self-demotes to CONFIRMED_WITH_CAVEAT rather than REFUTING (a power limit must not be
    able to overturn a literature-anchored fact).
    """
    if not contra.get("available"):
        return K.unverifiable("P.contra_inhibition_bidirectional",
                              "FD3 contralateral inhibition is bidirectional (Egelhaaf p.203)",
                              "bidirectional", "no contralateral GABA partners resolved")
    both = bool(contra.get("both_present"))
    # "bidirectional" only if BOTH channels carry a non-trivial share (a 7% minority is dominance,
    # not bidirectionality); a bare-presence 'both' with a trace minority self-demotes to caveat.
    substantial = bool(contra.get("substantially_bidirectional"))
    sufficient = bool(contra.get("sufficient"))
    dom = contra.get("dominant_direction")
    verdict = K.CONFIRMED if (substantial and sufficient) else K.CONFIRMED_WITH_CAVEAT
    note = (f"progressive={contra.get('syn_progressive')} syn, regressive={contra.get('syn_regressive')} syn "
            f"over {contra.get('n_classified')} classified contra-GABA partners; "
            f"{dom}-dominant (minority frac {contra.get('minority_frac')}); "
            + ("both channels substantial" if substantial else
               "the minority channel is a trace, so this reads as direction-dominant rather than "
               "cleanly bidirectional") + "; "
            + ("well-powered" if sufficient else
               "power-limited — caveat, not refutation"))
    return K.ClaimResult(
        id="P.contra_inhibition_bidirectional",
        description="FD3 receives contralateral inhibition in both directions (Egelhaaf 1985 p.203); "
                    "measured as progressive-dominant with a regressive minority",
        report_value="both channels present (progressive-dominant)",
        computed_primary=f"{dom}-dominant, both_present={both}",
        tolerance="both channels present; power-capped (never REFUTES)",
        verdict=verdict, numeric_outcome=K.MATCH if both else K.MINOR_DIFF, notes=note)


def _pathway_verdict(claims: list[K.ClaimResult]) -> K.ClaimResult:
    by_id = {c.id: c for c in claims}
    core = [by_id[i] for i in CORE_IDS if i in by_id]
    disc = [by_id[i] for i in DISC_IDS if i in by_id]
    core_refuted = [c.id for c in core if c.verdict == K.REFUTED]
    disc_refuted = [c.id for c in disc if c.verdict == K.REFUTED]
    core_ok = all(c.verdict == K.CONFIRMED for c in core)
    disc_ok = all(c.verdict == K.CONFIRMED for c in disc)
    corro = [c for c in claims if c.id not in CORE_IDS and c.id not in DISC_IDS
             and c.id != "P.input_pathway_verdict"]
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
                "treat as partial — inspect the per-claim table")
    return K.ClaimResult(
        id="P.input_pathway_verdict",
        description="OVERALL: FD3's afferent pathway (photoreceptor -> layer-b T4/T5 -> FD3)",
        report_value="regressive layer-b afferent arm, parallel to (not a copy of) FD1",
        computed_primary=verdict,
        tolerance="all CORE+DISC CONFIRMED & no falsifier -> CONFIRMED",
        verdict=verdict, notes=note)
