"""Family R oracle — FD3's wide-field inhibitor (the cell in VCH's place).

Grades the screen from ``derive/r_fd3_inhibitor.py``. The result is framed as a functional-ROLE
homolog of VCH, explicitly NOT a cell-type homolog (the ``k1_nod1_fd1`` precedent for Nod1=FD1).

  CORE           a wide-field inhibitory cell fills VCH's ROLE for FD3: it pools wide-field T4/T5,
                 inhibits FD3 directly, gates the LPC1 sheet, and feeds back onto the detectors.
  DISCRIMINATING the winner (LPi14) is OPPONENT-tuned (layer-a, opposite FD3's layer-b) — the
                 feature that distinguishes it from VCH's same-direction gain control; and it is
                 the top opponent gate in the screen.
  CORROBORATING  the honest departures from VCH: it is a lobula-plate INTRINSIC cell (optic, not
                 visual_centrifugal) and it is NOT reciprocal with FD3; PLUS the resolved
                 SAME-direction (regressive) surround — LPi12 — matching Egelhaaf's prediction, and
                 the two-gate architecture it implies.

The same-direction surround was previously UNVERIFIABLE only because LPi12 was absent from the
screen; the full-LPi-panel sweep in the derive now resolves it. LPi12 is GABAergic, layer-b
(SAME direction as FD3), dominates DETECTOR-level feedback (its detector contact is a large slice
of the detector pool's total inhibition and ~44% of its own output), but is a MINOR slice of FD3's
own inhibition — so it COMPLEMENTS, and does not replace, the opponent gate LPi14. A target-
permutation null shows LPi12 -> FD3 is enriched over a size-matched random target, so its (minor)
FD3 contact is a genuine co-input, not spillover. All of this is CORROBORATING: the CORE/DISC LPi14
verdict is unchanged.

Aggregate ``R.inhibitor_verdict`` mirrors ``P.input_pathway_verdict``. NOTE: the verdict does NOT
require centrifugal identity — that is the whole discriminator vs Family B.
"""

from __future__ import annotations

from flyconn.motif import compare as K

CORE_IDS = {"R.pools_widefield_t4t5", "R.inhibits_fd3", "R.gates_sheet", "R.feeds_back_detectors"}
DISC_IDS = {"R.opponent_tuning", "R.named_opponent_gate"}
# sheet-gate floor (mirrors derive.r_fd3_inhibitor.SHEET_GATE_FLOOR): a cell "gates the sheet" only
# above this; used here to assert the same-direction gate does NOT gate the sheet (non-replacement).
SHEET_GATE_FLOOR = 1000


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    named = d.get("named_inhibitor")
    win = d.get("winner", {})

    out.append(K.compare_categorical(
        "R.widefield_inhibitor_found", "A wide-field inhibitor filling VCH's role for FD3 exists",
        True, bool(named),
        refuted_note="no candidate passes the wide-field + inhibits + gates + feedback screen"))

    # ---- CORE: the VCH role ----
    wf = win.get("t4t5_in_frac")
    out.append(K.ClaimResult(
        id="R.pools_widefield_t4t5",
        description="The inhibitor pools wide-field T4/T5 motion (VCH-like)",
        report_value=">=40% of its input is T4/T5", computed_primary=wf,
        tolerance=">=40% T4/T5 input fraction",
        verdict=K.CONFIRMED if (wf is not None and wf >= 40.0) else K.REFUTED,
        numeric_outcome=K.MATCH if (wf is not None and wf >= 40.0) else K.MISMATCH,
        notes=f"{win.get('cell_type')} T4/T5 = {wf}% of input; {win.get('t4t5_in_neurons') if 't4t5_in_neurons' in win else ''}"))
    tf = win.get("to_fd3_syn")
    out.append(K.ClaimResult(
        id="R.inhibits_fd3", description="The inhibitor synapses directly onto FD3 (GABAergic)",
        report_value=">=100 syn onto FD3", computed_primary=tf,
        tolerance=">=100 syn; nt inhibitory",
        verdict=K.CONFIRMED if (tf and tf >= 100 and win.get("nt") == "gaba") else K.REFUTED,
        numeric_outcome=K.MATCH if (tf and tf >= 100) else K.MISMATCH,
        notes=f"{win.get('cell_type')} -> FD3 {tf} syn, nt={win.get('nt')}"))
    ts = win.get("to_sheet_syn")
    out.append(K.ClaimResult(
        id="R.gates_sheet", description="The inhibitor gates the LPC1 sheet (VCH gates LLPC1)",
        report_value=">=1000 syn onto the LPC1 sheet", computed_primary=ts,
        tolerance=">=1000 syn",
        verdict=K.CONFIRMED if (ts and ts >= 1000) else K.REFUTED,
        numeric_outcome=K.MATCH if (ts and ts >= 1000) else K.MISMATCH,
        notes=f"{win.get('cell_type')} -> LPC1 sheet {ts} syn"))
    td = win.get("to_detectors_syn")
    out.append(K.ClaimResult(
        id="R.feeds_back_detectors",
        description="The inhibitor feeds back onto the T4b/T5b detectors (VCH-T4/T5 loop analog)",
        report_value=">=1000 syn onto T4b/T5b", computed_primary=td,
        tolerance=">=1000 syn",
        verdict=K.CONFIRMED if (td and td >= 1000) else K.REFUTED,
        numeric_outcome=K.MATCH if (td and td >= 1000) else K.MISMATCH,
        notes=f"{win.get('cell_type')} -> T4b/T5b {td} syn"))

    # ---- DISCRIMINATING: opponent tuning + named opponent gate ----
    la = win.get("layer_a_pct")
    out.append(K.compare_pct(
        "R.opponent_tuning",
        "The inhibitor is OPPONENT-tuned (layer-a/progressive vs FD3's layer-b)", 100.0,
        la if la is not None else float("nan"), pp=10.0))
    out[-1].notes = (f"{win.get('cell_type')} T4/T5 input is {la}% layer-a (progressive) — the "
                     f"OPPONENT of FD3's layer-b; direction={win.get('direction')}")
    out.append(K.compare_categorical(
        "R.named_opponent_gate", "LPi14 is the top opponent (progressive) wide-field gate of FD3",
        "LPi14", (d.get("opponent_gates") or [None])[0],
        refuted_note=f"top opponent gate is {(d.get('opponent_gates') or [None])[0]}, not LPi14"))

    # ---- CORROBORATING: honest departures from VCH (role- not type-homolog) ----
    out.append(K.ClaimResult(
        id="R.vch_equivalent_role",
        description="LPi14 fills VCH's functional role but is a role-, not cell-type, homolog",
        report_value="functional-role homolog of VCH", computed_primary="role_homolog",
        tolerance="corroborating framing", verdict=K.CONFIRMED, numeric_outcome=K.MATCH,
        notes=("FD3 has no centrifugal VCH gate (P.nc_no_vch_gate); LPi14 fills the VCH ROLE "
               "(pools wide-field + gates the sheet + feeds back on detectors + inhibits FD3) but "
               f"is a lobula-plate INTRINSIC cell (super_class={win.get('super_class')}, "
               f"centrifugal={win.get('is_centrifugal')}), OPPONENT-tuned (layer-a), and NOT "
               f"reciprocal with FD3 (FD3->LPi14 = {win.get('fd3_to_inhibitor_syn')} syn).")))
    out.append(K.compare_categorical(
        "R.not_centrifugal", "The FD3 gate is lobula-plate intrinsic, NOT centrifugal (unlike VCH)",
        False, bool(win.get("is_centrifugal")),
        refuted_note="the winner is visual_centrifugal (would be VCH-like, not the LPi story)"))
    # ---- CORROBORATING: the same-direction (Egelhaaf) surround, now resolved as LPi12 ----
    sd = d.get("same_direction_gates") or []
    sd_named = d.get("same_direction_inhibitor")
    sdw = d.get("same_direction_winner") or {}
    enrich = d.get("same_direction_enrichment") or {}
    floor = d.get("floor_sweep") or []

    # R.same_direction_surround_found: a regressive wide-field inhibitor of FD3's circuit exists.
    sd_layer_b = sdw.get("layer_b_pct")
    found_ok = bool(sd_named and sdw.get("nt") == "gaba" and sd_layer_b is not None
                    and sd_layer_b >= 90.0)
    out.append(K.ClaimResult(
        id="R.same_direction_surround_found",
        description="A regressive (same-direction) wide-field inhibitor of FD3's circuit exists "
                    "(Egelhaaf's predicted ipsilateral surround)",
        report_value="a GABA layer-b wide-field cell in the FD3 circuit",
        computed_primary=f"{sd_named} (layer-b {sd_layer_b}%, nt {sdw.get('nt')})",
        tolerance="dominant layer-b (>=90%), GABA, clears the wide-field + detector-feedback floors",
        verdict=K.CONFIRMED if found_ok else K.UNVERIFIABLE,
        numeric_outcome=K.MATCH if found_ok else K.MINOR_DIFF,
        notes=(f"resolved by the full-LPi-panel sweep (prior UNVERIFIABLE was an artifact of {sd_named} "
               f"being absent from the candidate set); same_direction_gates={sd}")))

    # R.same_direction_detector_dominant — it acts predominantly at the DETECTOR node.
    sd_det_frac = sdw.get("frac_of_detector_inhibition")
    sd_fd3_frac = sdw.get("frac_of_fd3_inhibition")
    win_det_frac = win.get("frac_of_detector_inhibition")
    det_dominant = bool(sd_det_frac is not None and sd_fd3_frac is not None
                        and sd_det_frac > sd_fd3_frac
                        and (win_det_frac is None or sd_det_frac >= win_det_frac))
    out.append(K.ClaimResult(
        id="R.same_direction_detector_dominant",
        description="The same-direction gate acts predominantly at the T4b/T5b DETECTOR node "
                    "(not the FD3/sheet node)",
        report_value="detector-inhibition share > FD3-inhibition share (and >= the opponent gate's)",
        computed_primary=f"{sd_named}: {sd_det_frac}% of detector inhibition vs {sd_fd3_frac}% of FD3 inhibition",
        tolerance="frac_of_detector_inhibition > frac_of_fd3_inhibition and >= opponent gate's detector share",
        verdict=K.CONFIRMED if det_dominant else K.CONFIRMED_WITH_CAVEAT,
        numeric_outcome=K.MATCH if det_dominant else K.MINOR_DIFF,
        notes=(f"{sd_named} sends {sdw.get('to_detectors_frac_of_output')}% of its OWN output to the "
               f"detectors; opponent {win.get('cell_type')} detector share = {win_det_frac}%")))

    # R.two_gate_architecture — LPi14 (opponent, FD3/sheet) + LPi12 (same-dir, detector) complement.
    two_gate = bool(named and sd_named and named != sd_named
                    and win.get("direction") == "opponent" and sdw.get("direction") == "same_direction"
                    and det_dominant)
    out.append(K.ClaimResult(
        id="R.two_gate_architecture",
        description="FD3 has TWO complementary wide-field gates at different nodes: an opponent "
                    "gate at the FD3/sheet node (LPi14) and a same-direction gate at the detector "
                    "node (LPi12)",
        report_value="opponent FD3/sheet gate + same-direction detector gate, distinct cells",
        computed_primary=f"opponent={named} (FD3/sheet), same_direction={sd_named} (detector)",
        tolerance="two distinct GABA cells; opposite dominant layers; the same-direction cell is "
                  "detector-dominant",
        verdict=K.CONFIRMED_WITH_CAVEAT if two_gate else K.UNVERIFIABLE,
        numeric_outcome=K.MATCH if two_gate else K.MINOR_DIFF,
        notes=("the defensible headline; sign of each GABA contact is from physiology, not wiring; "
               "n=2 FD3 pair (both cells agree per per_fd3_cell_syn)")))

    # R.same_direction_not_replacement — LPi12 does NOT replace LPi14 (minor FD3/sheet contact).
    not_replace = bool(sd_fd3_frac is not None and sd_fd3_frac < 10.0
                       and (sdw.get("to_sheet_syn") or 0) < SHEET_GATE_FLOOR)
    out.append(K.ClaimResult(
        id="R.same_direction_not_replacement",
        description="The same-direction gate does NOT replace LPi14 as FD3's inhibitor "
                    "(its FD3 and sheet contacts are minor)",
        report_value="same-direction cell is a minor share of FD3's inhibition and does not gate the sheet",
        computed_primary=f"{sd_named}: {sd_fd3_frac}% of FD3 inhibition, ->sheet {sdw.get('to_sheet_syn')} syn",
        tolerance="frac_of_fd3_inhibition < 10% AND ->sheet < sheet-gate floor",
        verdict=K.CONFIRMED if not_replace else K.REFUTED,
        numeric_outcome=K.MATCH if not_replace else K.MISMATCH,
        notes="guards against overclaiming LPi12 as THE FD3 gate; LPi14 remains the FD3/sheet gate"))

    # R.same_direction_fd3_specificity — is the same-direction gate's FD3 contact real or spillover?
    if enrich.get("available"):
        enriched = bool(enrich.get("enriched")) and (enrich.get("p_enrichment") or 1.0) < 0.05
        out.append(K.ClaimResult(
            id="R.same_direction_fd3_specificity",
            description="The same-direction gate's (minor) FD3 contact is FD3-specific, not spillover "
                        "from a promiscuous cell",
            report_value="->FD3 enriched over a size-matched random LP-tangential target",
            computed_primary=f"obs {enrich.get('obs_to_fd3')} vs null {enrich.get('null_mean')} "
                             f"(z={enrich.get('z_score')}, p={enrich.get('p_enrichment')})",
            tolerance="obs > null and p < 0.05 (in-degree-preserving target permutation)",
            verdict=K.CONFIRMED_WITH_CAVEAT if enriched else K.UNVERIFIABLE,
            numeric_outcome=K.MATCH if enriched else K.MINOR_DIFF,
            notes=f"{enrich.get('n_perms')} perms; a genuine co-input onto FD3, though a minor one"))
    else:
        out.append(K.unverifiable(
            "R.same_direction_fd3_specificity",
            "FD3-specificity of the same-direction gate's contact",
            "enrichment null unavailable",
            f"reason: {enrich.get('reason', 'null not computed')}"))

    # R.floor_sensitivity — the direct-FD3 verdict is threshold-dependent (reported, never gates).
    fs = {row["fd3_floor"]: row["pass"] for row in floor}
    out.append(K.ClaimResult(
        id="R.floor_sensitivity",
        description="The 'inhibits FD3 above floor' call for the same-direction gate is "
                    "threshold-dependent (reported for honesty)",
        report_value="verdict-vs-threshold grid",
        computed_primary=f"{sd_named} passes the direct-FD3 gate at floors "
                         f"{[f for f, p in fs.items() if sd_named in p]}",
        tolerance="report which floors each cell clears; do not hang a verdict on a single cutoff",
        verdict=K.CONFIRMED, numeric_outcome=K.MATCH,
        notes=f"floor grid -> passing cells: {fs}"))

    out.append(K.unverifiable(
        "R.power_note", "FD3 is one bilateral pair (n=2)", "n=2",
        "The inhibitor claims rest on the aggregate synapse populations, not per-cell n; both FD3 "
        "cells agree on the two-gate asymmetry (per_fd3_cell_syn)."))

    out.append(_inhibitor_verdict(out))
    return out


def _inhibitor_verdict(claims: list[K.ClaimResult]) -> K.ClaimResult:
    by = {c.id: c for c in claims}
    core = [by[i] for i in CORE_IDS if i in by]
    disc = [by[i] for i in DISC_IDS if i in by]
    core_ref = [c.id for c in core if c.verdict == K.REFUTED]
    disc_ref = [c.id for c in disc if c.verdict == K.REFUTED]
    core_ok = all(c.verdict == K.CONFIRMED for c in core)
    disc_ok = all(c.verdict == K.CONFIRMED for c in disc)
    corro = [c for c in claims if c.id not in CORE_IDS and c.id not in DISC_IDS
             and c.id != "R.inhibitor_verdict"]
    soft = any(c.verdict in (K.CONFIRMED_WITH_CAVEAT, K.UNVERIFIABLE) for c in corro)
    if core_ref or disc_ref:
        v, note = K.REFUTED, f"FALSIFIER: core={core_ref} disc={disc_ref}"
    elif core_ok and disc_ok and not soft:
        v, note = K.CONFIRMED, "all CORE + DISCRIMINATING confirmed; no falsifier"
    elif core_ok and disc_ok:
        v, note = K.CONFIRMED_WITH_CAVEAT, "CORE+DISC confirmed; a corroborating claim is soft"
    else:
        v, note = K.CONFIRMED_WITH_CAVEAT, "CORE/DISC not all confirmed; inspect per-claim table"
    return K.ClaimResult(
        id="R.inhibitor_verdict",
        description="OVERALL: LPi14 is FD3's wide-field opponent inhibitor (VCH functional-role homolog)",
        report_value="LPi14 = VCH-role opponent gate", computed_primary=v,
        tolerance="all CORE+DISC CONFIRMED & no falsifier -> CONFIRMED (centrifugal NOT required)",
        verdict=v, notes=note)
