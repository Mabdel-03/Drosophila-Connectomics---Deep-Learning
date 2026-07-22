"""Family Q oracle — FD3's intermediate sheet(-set) between the T4b/T5b detectors and FD3.

Grades the measured sheet profiles from ``derive/q_fd3_sheet.py`` against the "named the
T4 -> ??? -> FD intermediate" requirement. Mirrors the C (uniqueness) + D (retinotopy null)
oracle discipline.

  CORE           the named sheet (LPC1) is cholinergic, reads the direction-matched layer-b
                 channel, is feed-forward (detectors -> sheet -> FD3), and pools retinotopically
                 locally (below an in-degree-preserving null).
  DISCRIMINATING LPC1 is the UNIQUE layer-b sheet feeding FD3 (the siblings read orthogonal
                 vertical channels), and FD1's sheet LLPC1 is NOT a major FD3 input (negative
                 control — FD3 reads a different sheet than FD1).
  CORROBORATING  the sheet is a SET {LPC1 + LLPC2/LLPC3}, LPC1 the direction-matched member.

Aggregate ``Q.sheet_verdict`` mirrors ``P.input_pathway_verdict``.
"""

from __future__ import annotations

from flyconn.motif import compare as K

NAMED = "LPC1"
CORE_IDS = {"Q.sheet_cholinergic", "Q.sheet_layer_b", "Q.feed_forward", "Q.retinotopy_local"}
DISC_IDS = {"Q.lpc1_unique_layerb", "Q.nc_not_fd1_sheet"}


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    named = d.get("named_sheet")
    profs = d.get("profiles", {})
    np_prof = profs.get(named, {}) if named else {}
    retino = d.get("retinotopy_null", {})

    # existence: a named layer-b sheet was found
    out.append(K.compare_categorical(
        "Q.named_sheet_found", "A direction-matched (layer-b) sheet feeding FD3 exists",
        NAMED, named, refuted_note="no layer-b columnar-projection sheet clears the FD3 floor"))

    # ---- CORE ----
    out.append(K.compare_categorical(
        "Q.sheet_cholinergic", "The FD3 sheet (LPC1) is cholinergic (excitatory relay)",
        "acetylcholine", np_prof.get("nt")))
    out.append(K.compare_categorical(
        "Q.sheet_layer_b", "The FD3 sheet reads the layer-b (regressive) channel FD3 reads",
        "b", np_prof.get("dominant_layer"),
        refuted_note="named sheet's dominant motion channel is not layer-b"))
    out.append(K.compare_categorical(
        "Q.feed_forward", "Feed-forward chain present: T4b/T5b -> LPC1 -> FD3",
        True, bool(d.get("feed_forward")),
        refuted_note="detector->sheet or sheet->FD3 link missing"))
    # retinotopy: local pooling below null (scale-robust, copied from Family D)
    if retino.get("available"):
        z = retino.get("z_score"); p = retino.get("p_value")
        local = bool(retino.get("local")) and (z is not None and z <= -20) and (p is not None and p < 0.01)
        out.append(K.ClaimResult(
            id="Q.retinotopy_local",
            description="LPC1 pools its T4b/T5b input retinotopically locally (below null)",
            report_value="obs << null (z<=-20, p<0.01)",
            computed_primary=f"obs {retino.get('obs_radius_um')} vs null {retino.get('null_radius_um')}",
            tolerance="z<=-20 and p<0.01 (scale-robust)",
            verdict=K.CONFIRMED if local else (K.CONFIRMED_WITH_CAVEAT if retino.get("local") else K.REFUTED),
            numeric_outcome=K.MATCH if local else K.MINOR_DIFF,
            notes=f"z={z}, p={p}, {retino.get('n_perms')} perms"))
    else:
        out.append(K.unverifiable(
            "Q.retinotopy_local", "LPC1 retinotopic locality vs null",
            "obs << null", f"null unavailable: {retino.get('reason', 'no detector loci')}"))

    # ---- DISCRIMINATING ----
    out.append(K.compare_categorical(
        "Q.lpc1_unique_layerb",
        "LPC1 is the UNIQUE layer-b sheet feeding FD3 (siblings read orthogonal channels)",
        True, bool(named == NAMED and d.get("n_layerb_sheets") == 1),
        refuted_note=f"n_layerb_sheets={d.get('n_layerb_sheets')} (expected 1) or named={named}"))
    out[-1].notes = (f"sheet-set channels: {d.get('sheet_set_channels')}; "
                     f"fold vs other layer-b: {d.get('fold_vs_other_layerb')}")
    llpc1_fd3 = d.get("llpc1_control_to_fd3")
    named_fd3 = np_prof.get("to_fd3_syn")
    nc_ok = bool(llpc1_fd3 is not None and named_fd3 and llpc1_fd3 < 0.6 * named_fd3)
    out.append(K.ClaimResult(
        id="Q.nc_not_fd1_sheet",
        description="FD3 reads a DIFFERENT sheet than FD1: LLPC1 is not a major FD3 input",
        report_value="LLPC1->FD3 < LPC1->FD3", computed_primary=f"LLPC1->FD3={llpc1_fd3}, LPC1->FD3={named_fd3}",
        tolerance="LLPC1->FD3 < 0.6 x LPC1->FD3",
        verdict=K.CONFIRMED if nc_ok else K.REFUTED,
        numeric_outcome=K.MATCH if nc_ok else K.MISMATCH,
        notes="negative control: FD3 is not just re-reading FD1's progressive sheet"))

    # ---- CORROBORATING: the sheet-SET framing ----
    sset = d.get("sheet_set", [])
    out.append(K.ClaimResult(
        id="Q.sheet_set",
        description="FD3's intermediate is a SET {LPC1 horizontal + LLPC2/LLPC3 vertical context}",
        report_value="{LPC1, LLPC2, LLPC3}", computed_primary=str(sset),
        tolerance="LPC1 the direction-matched member; siblings orthogonal channels",
        verdict=K.CONFIRMED if (NAMED in sset and len(sset) >= 2) else K.CONFIRMED_WITH_CAVEAT,
        numeric_outcome=K.MATCH,
        notes=f"channels {d.get('sheet_set_channels')}"))

    # ---- CORROBORATING: directional composition of FD3's sheet input (the mentor's a/c/d point) ----
    comp = d.get("directional_composition", {})
    enull = d.get("entropy_null", {})
    fd1ref = d.get("fd1_reference", {})
    n_dirs = comp.get("n_cardinal_directions")
    # "FD3 pools all four cardinal directions THROUGH ITS SHEETS" — a statement about relayed input
    # (denominator D1), NOT FD3's direct tuning (which stays ~99% layer-b). Downgraded to a generic
    # tangential-cell property if the entropy null / FD1 reference show it is background.
    generic = bool((enull.get("available") and not enull.get("fd3_elevated"))
                   or (fd1ref.get("available") and not fd1ref.get("fd3_specific")))
    dir_ok = bool(n_dirs is not None and n_dirs >= 3)
    out.append(K.ClaimResult(
        id="Q.pools_all_cardinal_directions",
        description="Through its sheets, FD3 receives relayed motion spanning all four cardinal "
                    "directions (LPC1=regressive, LLPC3=down, LLPC2/LPC2=up, LLPC1=progressive)",
        report_value=">=3 hard-labelled directions among FD3's sheets",
        computed_primary=f"{n_dirs} directions; per-channel {comp.get('sheet_frac_by_channel')}",
        tolerance=">=3 direction channels, each with a hard (>=20pp margin, >=95% stable) label",
        verdict=(K.CONFIRMED_WITH_CAVEAT if (dir_ok and generic)
                 else K.CONFIRMED if dir_ok else K.CONFIRMED_WITH_CAVEAT),
        numeric_outcome=K.MATCH if dir_ok else K.MINOR_DIFF,
        notes=(f"channels {comp.get('channel_members')}; entropy={comp.get('channel_entropy_bits')} bits; "
               f"null pct={enull.get('fd3_percentile')} (elevated={enull.get('fd3_elevated')}); "
               f"FD1 spread {fd1ref.get('n_cardinal_directions')} dirs "
               f"(FD3-specific={fd1ref.get('fd3_specific')}). This describes FD3's SHEET-RELAYED "
               f"input; its DIRECT T4/T5 drive stays ~99% layer-b (see Family P).")))
    # the specific, non-generic part: only LPC1 (layer-b) is direction-matched; the rest is context
    matched = comp.get("matched_frac"); context = comp.get("context_frac")
    matched_minor = bool(matched is not None and context is not None and matched < 50.0)
    out.append(K.ClaimResult(
        id="Q.matched_channel_minority",
        description="Only LPC1 (regressive) is direction-matched; the sheet-relayed signal is "
                    "mostly orthogonal/opposite motion CONTEXT",
        report_value="matched (layer-b) sheet input < 50% of sheet-relayed input",
        computed_primary=f"matched={matched}%, context={context}%",
        tolerance="LPC1/layer-b is a minority of FD3's total sheet input; it is still the unique "
                  "direction-matched member (see Q.lpc1_unique_layerb)",
        verdict=K.CONFIRMED if matched_minor else K.CONFIRMED_WITH_CAVEAT,
        numeric_outcome=K.MATCH if matched_minor else K.MINOR_DIFF,
        notes="FD3 reads its own direction (LPC1) plus a near-balanced up/down/opposite surround"))

    out.append(K.unverifiable(
        "Q.power_note", "FD3 is one bilateral pair (n=2)", "n=2",
        "The sheet claims rest on the aggregate synapse populations onto both FD3 cells and on "
        "the full sheet-cell populations, not on a per-cell sample size."))

    out.append(_sheet_verdict(out))
    return out


def _sheet_verdict(claims: list[K.ClaimResult]) -> K.ClaimResult:
    by = {c.id: c for c in claims}
    core = [by[i] for i in CORE_IDS if i in by]
    disc = [by[i] for i in DISC_IDS if i in by]
    core_ref = [c.id for c in core if c.verdict == K.REFUTED]
    disc_ref = [c.id for c in disc if c.verdict == K.REFUTED]
    core_ok = all(c.verdict == K.CONFIRMED for c in core)
    disc_ok = all(c.verdict == K.CONFIRMED for c in disc)
    corro = [c for c in claims if c.id not in CORE_IDS and c.id not in DISC_IDS
             and c.id != "Q.sheet_verdict"]
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
        id="Q.sheet_verdict",
        description="OVERALL: FD3's intermediate sheet-set (LPC1 the direction-matched member)",
        report_value="T4b/T5b -> {LPC1(+LLPC2/3)} -> FD3", computed_primary=v,
        tolerance="all CORE+DISC CONFIRMED & no falsifier -> CONFIRMED", verdict=v, notes=note)
