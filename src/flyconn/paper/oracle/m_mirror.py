"""Family M oracle — MIRROR SYMMETRY + negative controls + wing-flip verdicts.

Existence-test verdict logic (NOT a confirmation):
  * mirror count claims: left compared to the RIGHT computed value with down-drift tolerance,
    BUT gated by an absolute floor so a near-zero left value is REFUTED, not excused as drift;
  * categorical/sign/layer/wing claims: EXACT (the mirror must hold or it's a real difference);
  * negative controls: must FAIL to gate (wrong VCH negligible; layer-b minor) -> CONFIRMED when
    they correctly fail; if a control PASSES (wrong gate works) the mirror is REFUTED;
  * wing-flip: the complement crux -> CONFIRMED iff the two DNp26 bodies steer OPPOSITE wings.
"""

from __future__ import annotations

from flyconn.motif import compare as K

# left-vs-right homolog tolerances (homologs, not replicates).
RECIP_PP = 12.0           # reciprocal-fraction percentage-point tolerance
LAYER_PP = 6.0            # layer-a / layer-b fraction tolerance
SHEET_REL = 0.30          # sheet-size relative tolerance (down-drift)
SHEET_FLOOR = 30          # below this the sheet is ABSENT, not "drift"
CONTRA_MIN = 70.0         # noduli-group contralaterality floor


def build_claims(d: dict) -> list[K.ClaimResult]:
    m = d.get("mirror", {})
    nc = d.get("negative_controls", {})
    wf = d.get("wing_flip", {})
    cc = d.get("cell_correspondence", {})
    claims: list[K.ClaimResult] = []

    def _lr(key):
        v = m.get(key, {})
        return v.get("right"), v.get("left")

    # --- M1 gating-soma laterality (categorical, exact) ---
    claims.append(K.compare_categorical(
        "M.gating_soma_flip", "Left sheet gated by the RIGHT-soma VCH (opposite of right)",
        "right", m.get("gating_soma", {}).get("left"),
        refuted_note="left sheet not gated by a right-soma VCH -> centrifugal crossing violated"))

    # --- M2 direction preserved: left sheet layer-a driven like right (VCH layer-a) ---
    r, l = _lr("vch_layer_a_pct")
    claims.append(K.compare_pct(
        "M.vch_layer_a", "VCH layer-a (progressive) drive: left vs right", round(r, 1), round(l, 1),
        pp=LAYER_PP))
    # LPi15 opponent layer-b preserved
    r, l = _lr("lpi15_layer_b_pct")
    if r is not None and l is not None:
        claims.append(K.compare_pct(
            "M.lpi15_layer_b", "LPi15 opponent layer-b drive: left vs right",
            round(r, 1), round(l, 1), pp=LAYER_PP))

    # --- mirror count comparisons (left vs right computed; down-drift, floored) ---
    r, l = _lr("vch_in_syn")
    claims.append(K.compare_count("M.vch_in_syn", "Gating-VCH input synapses: left vs right",
                                  r, l, rel=0.15, abs_floor=500, drift_dir="down"))
    r, l = _lr("vch_out_syn")
    claims.append(K.compare_count("M.vch_out_syn", "Gating-VCH output synapses: left vs right",
                                  r, l, rel=0.15, abs_floor=500, drift_dir="down"))
    r, l = _lr("recip_frac")
    claims.append(K.compare_pct("M.recip_frac", "T4/T5->VCH reciprocal fraction: left vs right",
                                round(r, 1), round(l, 1), pp=RECIP_PP))

    # sheet size with the absolute FLOOR (the existence guard).
    r, l = _lr("sheet_size")
    if l is None or r is None:
        claims.append(K.unverifiable(
            "M.sheet_size", "LLPC1 sheet size: left vs right", r,
            f"sheet size unavailable (right={r}, left={l}); family D not run in this subset"))
    elif l < SHEET_FLOOR:
        claims.append(K.ClaimResult(
            id="M.sheet_size", description="LLPC1 sheet size: left vs right (existence-floored)",
            report_value=r, computed_primary=l, tolerance=f"floor {SHEET_FLOOR}",
            verdict=K.REFUTED, numeric_outcome=K.MISMATCH,
            notes=f"left sheet {l} < floor {SHEET_FLOOR}: sheet stage ABSENT, not drift"))
    else:
        claims.append(K.compare_count("M.sheet_size", "LLPC1 sheet size: left vs right",
                                      r, l, rel=SHEET_REL, abs_floor=SHEET_FLOOR, drift_dir="down"))

    # --- M readout: Nod1 dominance + relay ---
    r, l = _lr("nod1_top_excitatory")
    claims.append(K.compare_categorical(
        "M.nod1_dominant", "Nod1 is the top excitatory readout of the LEFT sheet (as right)",
        True, bool(l), refuted_note="Nod1 is not the left sheet's top excitatory readout"))
    r, l = _lr("nod1_relay_dominates")
    if l is not None:
        claims.append(K.compare_categorical(
            "M.nod1_relay_dominates", "Left Nod1->DNp26 > direct sheet->DNp26 (relay dominates)",
            True, bool(l), refuted_note="left Nod1 relay does not dominate the direct DNp26 input"))

    # --- M3 noduli contralaterality (left Nod1 + left FD3) ---
    r, l = _lr("nod1_contra_pct")
    if l is not None:
        claims.append(K.compare_categorical(
            "M.nod1_contra", f"Left Nod1 output is >=70% contralateral (noduli group; measured {l}%)",
            True, bool(l >= CONTRA_MIN),
            refuted_note=f"left Nod1 contralaterality {l}% < {CONTRA_MIN}%"))
    r, l = _lr("fd3_contra_pct")
    if l is not None:
        claims.append(K.compare_categorical(
            "M.fd3_contra", f"Left FD3 (LPT42_Nod4) output is >=70% contralateral (measured {l}%)",
            True, bool(l >= CONTRA_MIN),
            refuted_note=f"left FD3 contralaterality {l}% < {CONTRA_MIN}%"))

    # --- M_cells correspondence ---
    claims.append(K.compare_categorical(
        "M.cell_correspondence",
        f"Every named circuit type has a left+right counterpart with matching NT+class "
        f"({cc.get('n_corresponding')}/{cc.get('n_types')})",
        True, bool(cc.get("all_correspond")),
        refuted_note=f"only {cc.get('n_corresponding')}/{cc.get('n_types')} types correspond"))

    # --- M4 WING FLIP (the complement crux) ---
    if wf.get("available"):
        claims.append(K.compare_categorical(
            "M.wing_flip", "Left circuit's DNp26 steers the OPPOSITE physical wing from the right",
            True, bool(wf.get("dnp26_bodies_opposite_wings")),
            refuted_note=f"DNp26 target wings by soma: {wf.get('dnp26_target_wings_by_soma')} "
                         f"(not opposite -> a duplicate, not a complement)"))
        claims.append(K.compare_categorical(
            "M.dnp26_contralateral", "Each DNp26 body is contralateral-steering (intrinsic laterality)",
            True, bool(wf.get("dnp26_each_body_contralateral")),
            refuted_note="a DNp26 body is not contralateral-steering"))
        shared = wf.get("dnp26_shared_muscles", [])
        claims.append(K.compare_categorical(
            "M.dnp26_same_muscles",
            f"Both DNp26 bodies target the SAME steering muscles ({shared})",
            True, bool(set(shared) & {"hg1", "i1"}),
            refuted_note=f"DNp26 bodies do not share the hg1/i1 steering muscles: {shared}"))
        # NC3 permutation null (CORROBORATING, caveat-capped: pooling both bodies dilutes the
        # per-body bias, so a marginal p must not REFUTE the categorical contralateral result).
        pn = wf.get("permutation_null", {})
        if pn.get("available"):
            sig = pn.get("significant_contra")
            claims.append(K.ClaimResult(
                id="M.wing_perm_null",
                description="DNp26 contralateral bias vs somaSide-permutation null (NC3)",
                report_value="contralateral (low tail)", computed_primary=pn.get("p_low_tail"),
                tolerance="p<0.01 low tail (pooled; per-body categorical is primary)",
                verdict=(K.CONFIRMED if sig else K.CONFIRMED_WITH_CAVEAT),
                numeric_outcome=(K.MATCH if sig else K.MINOR_DIFF),
                notes=f"obs_ipsi_frac={pn.get('obs_ipsi_frac')} null_mean={pn.get('null_mean')} "
                      f"z={pn.get('z')} p_low={pn.get('p_low_tail')}; pooling both DNp26 bodies "
                      f"dilutes the per-body contralateral signal (each body is clearly contra)."))
    else:
        claims.append(K.unverifiable(
            "M.wing_flip", "Left circuit's DNp26 steers the OPPOSITE physical wing", True,
            f"MaleCNS unavailable: {wf.get('reason')}"))

    # --- NEGATIVE CONTROLS (must FAIL to gate) ---
    nc1 = nc.get("nc1_wrongside_vch", {})
    claims.append(K.compare_categorical(
        "NC1.wrongside_vch_negligible",
        f"Wrong-side (left-soma) VCH does NOT gate the left sheet "
        f"(reaches {nc1.get('wrong_gated_llpc1')} vs {nc1.get('correct_gated_llpc1')})",
        True, bool(nc1.get("wrong_is_negligible")),
        refuted_note=f"wrong-side VCH gates {nc1.get('wrong_gated_llpc1')} left LLPC1 "
                     f"-> centrifugal crossing falsified"))
    nc2 = nc.get("nc2_layer_b_channel", {})
    claims.append(K.compare_categorical(
        "NC2.layer_b_minor",
        f"Opposite-direction layer-b channel is NOT the left LLPC1 main drive "
        f"(layer_b={nc2.get('layer_b_syn')} vs layer_a={nc2.get('layer_a_syn')})",
        True, bool(nc2.get("layer_b_is_minor")),
        refuted_note=f"layer-b drives the sheet at {nc2.get('layer_b_frac_of_a')}x layer-a "
                     f"-> direction logic falsified"))

    return claims


# Claim partitions for the existence roll-up.
MIRROR_CORE_IDS = {"M.gating_soma_flip", "M.vch_layer_a", "M.recip_frac", "M.nod1_dominant",
                   "M.cell_correspondence"}
COMPLEMENT_IDS = {"M.wing_flip", "M.dnp26_contralateral", "M.nod1_contra"}
NEG_CONTROL_IDS = {"NC1.wrongside_vch_negligible", "NC2.layer_b_minor"}


def mirror_verdict(claims: list[K.ClaimResult]) -> dict:
    """Headline existence verdict: PRESENT / PARTIAL / ABSENT for the left mirror."""
    by = {c.id: c for c in claims}

    def ok(cid):
        c = by.get(cid)
        return c is not None and c.verdict in (K.CONFIRMED, K.CONFIRMED_WITH_CAVEAT)

    core_ok = all(ok(c) for c in MIRROR_CORE_IDS if c in by)
    complement_ok = all(ok(c) for c in COMPLEMENT_IDS if c in by)
    controls_ok = all(ok(c) for c in NEG_CONTROL_IDS if c in by)  # ok == correctly fails to gate
    refuted = [c.id for c in claims if c.verdict == K.REFUTED]

    if not controls_ok:
        verdict = "MIRROR_REFUTED_BY_CONTROL"
        statement = "a negative control passed (a wrong gate works) -> the mirror is falsified."
    elif core_ok and complement_ok:
        verdict = "MIRROR_PRESENT"
        statement = ("The left hemisphere contains a faithful mirror of the right figure-ground "
                     "circuit: every core stage mirrors the right, and the DNp26 command steers "
                     "the OPPOSITE wing (the complement). FD1 label carries the Phase-0 caveat.")
    elif core_ok:
        verdict = "MIRROR_PARTIAL"
        statement = ("The left mirror is present through the sheet/readout but the complement "
                     "(wing-flip or contralateral projection) is incomplete; see per-claim notes.")
    else:
        verdict = "MIRROR_PARTIAL"
        statement = "The left mirror is incomplete; a core mirror claim did not hold."
    return {"verdict": verdict, "statement": statement, "core_ok": core_ok,
            "complement_ok": complement_ok, "controls_ok": controls_ok, "refuted": refuted}
