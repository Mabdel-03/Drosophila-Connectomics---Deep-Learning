"""FD3 full-circuit existence screen — the 7 spine stages, photoreceptor -> muscle.

A cheap PRESENT / WEAK / ABSENT walk of the whole FD3 sensorimotor circuit, modelled on
``existence_screen.py`` (the Stage-8 left-mirror screen). It composes the already-derived
family dicts rather than recomputing: the input tiers (1-4) read Family **P**
(``derive.p_fd3_input``), and the identity/output tiers (5-7) read Families **K** and **L**.
This turns three separately-verified families into one auditable input->output spine with a
single GO / PARTIAL / NO_GO decision, and returns a true negative ("the circuit breaks at
stage X") if a spine stage is missing.

The 7 stages (the FD3 circuit's spine, front to back):
  1. photoreceptor   R1-6/R7/R8 present in the connectome (the eye's entry point)
  2. lamina_medulla  L1/L2/L3 -> Mi/Tm columnar cascade present upstream of FD3's detectors
  3. motion_layer_b  the T4b/T5b elementary-motion detectors driving FD3 are layer-b dominant
  4. fd3_afferent    T4b/T5b + central sheets reach FD3, with bidirectional contra inhibition
  5. identity        FD3 == LPT42_Nod4 (Family K identity verdict is CONFIRMED*)
  6. descending      FD3 reaches a figure-steering descending neuron (Family L)
  7. motor           that descending output resolves to wing-steering muscle (Family L, MaleCNS)

Floors guard against a near-empty stage being read as PRESENT. Stages 1-4 gate the decision
(the input pathway is the new result); 5-7 are the already-established arms, reported for
completeness. Reuses ``derive.p_fd3_input`` (or a passed-in derived dict) and the K/L dicts.
"""

from __future__ import annotations

from flyconn.motif import compare as K

from . import p_fd3_input as P

PRESENT, WEAK, ABSENT = "PRESENT", "WEAK", "ABSENT"
GO, PARTIAL, NO_GO = "GO", "PARTIAL", "NO_GO"

# Stage floors.
CASCADE_MEDULLA_MIN = 2        # >= this many expected medulla types on each limb -> present
LAYER_B_FLOOR_PCT = 50.0       # T4/T5 drive must be majority layer-b
CENTRAL_SYN_FLOOR = 500        # FD3's central-sheet input must clear this (real convergence)


def _label(present: bool, weak: bool = False) -> str:
    return PRESENT if present else (WEAK if weak else ABSENT)


def _stage_photoreceptor(p: dict) -> dict:
    cas = p.get("upstream_cascade", {})
    photo = cas.get("photoreceptor_present", [])
    return {"stage": "photoreceptor", "detail": "R1-6/R7/R8 present (eye entry point)",
            "present_types": photo, "label": _label(bool(photo))}


def _stage_lamina_medulla(p: dict) -> dict:
    cas = p.get("upstream_cascade", {})
    lamina = cas.get("lamina_present", [])
    on = cas.get("on_limb_t4b", {}).get("n_expected_present", 0)
    off = cas.get("off_limb_t5b", {}).get("n_expected_present", 0)
    ok = bool(lamina) and on >= CASCADE_MEDULLA_MIN and off >= CASCADE_MEDULLA_MIN
    weak = bool(lamina) and (on >= 1 or off >= 1)
    return {"stage": "lamina_medulla",
            "detail": "L1/L2/L3 -> Mi/Tm columnar cascade upstream of FD3's T4b/T5b",
            "lamina_present": lamina, "on_limb_medulla": on, "off_limb_medulla": off,
            "label": _label(ok, weak)}


def _stage_motion_layer_b(p: dict) -> dict:
    census = p.get("census", {})
    lb = census.get("layer_b_frac_of_t4t5")
    ok = lb is not None and lb >= LAYER_B_FLOOR_PCT
    oo = census.get("on_off_split", {})
    mix = bool((oo.get("T4b_ON") or 0) > 0 and (oo.get("T5b_OFF") or 0) > 0)
    return {"stage": "motion_layer_b",
            "detail": "FD3's T4b/T5b motion detectors are layer-b (regressive) dominant",
            "layer_b_pct": lb, "on_off_mix": mix, "label": _label(bool(ok and mix), bool(ok))}


def _stage_sheet(q: dict) -> dict:
    """NAMED intermediate sheet: T4b/T5b -> LPC1 -> FD3 (reads Family Q)."""
    named = q.get("named_sheet")
    prof = (q.get("profiles", {}) or {}).get(named, {}) if named else {}
    det_to_sheet = prof.get("layer_b_input_syn", 0) or 0
    sheet_to_fd3 = prof.get("to_fd3_syn", 0) or 0
    ok = bool(q.get("feed_forward") and det_to_sheet > 0 and sheet_to_fd3 >= 500)
    return {"stage": "sheet",
            "detail": f"T4b/T5b -> {named or 'LPC1'} (direction-matched layer-b sheet) -> FD3",
            "named_sheet": named, "det_to_sheet_syn": det_to_sheet, "sheet_to_fd3_syn": sheet_to_fd3,
            "sheet_set": q.get("sheet_set"), "label": _label(ok, bool(sheet_to_fd3 > 0))}


def _stage_fd3_afferent(p: dict) -> dict:
    census = p.get("census", {})
    central = p.get("central_inputs", {})
    contra = p.get("contra_inhibition", {})
    t45 = census.get("t4t5_syn", 0) or 0
    csyn = central.get("total_central_syn", 0) or 0
    ok = bool(t45 > 0 and csyn >= CENTRAL_SYN_FLOOR)
    return {"stage": "fd3_afferent",
            "detail": "T4b/T5b motion + the LPC1 sheet converge on FD3; LPi14 is the wide-field gate",
            "t4t5_syn": t45, "central_syn": csyn,
            "contra_bidirectional": bool(contra.get("both_present")),
            "label": _label(ok, bool(t45 > 0))}


def _stage_widefield_inhibitor(r: dict) -> dict:
    """MODULATORY node: LPi14, the wide-field opponent gate standing in VCH's place (Family R)."""
    win = r.get("winner", {}) or {}
    named = win.get("cell_type")
    to_fd3 = win.get("to_fd3_syn", 0) or 0
    to_sheet = win.get("to_sheet_syn", 0) or 0
    opponent = win.get("direction") == "opponent"
    ok = bool(named and to_fd3 >= 100 and to_sheet >= 1000 and opponent)
    return {"stage": "widefield_inhibitor",
            "detail": (f"{named or 'LPi14'} — wide-field opponent gate (VCH-role; layer-a, "
                       f"inhibits FD3 + gates the sheet)"),
            "name": named, "direction": win.get("direction"),
            "to_fd3_syn": to_fd3, "to_sheet_syn": to_sheet,
            "label": _label(ok, bool(named and to_fd3 > 0))}


def _stage_same_direction_gate(r: dict) -> dict:
    """MODULATORY node: LPi12, the same-direction (regressive) surround gating FD3's DETECTORS
    (Family R). Egelhaaf's predicted ipsilateral surround; acts one stage upstream of FD3."""
    sdw = r.get("same_direction_winner") or {}
    named = sdw.get("cell_type")
    to_det = sdw.get("to_detectors_syn", 0) or 0
    same_dir = sdw.get("direction") == "same_direction"
    ok = bool(named and to_det >= 1000 and same_dir)
    return {"stage": "same_direction_gate",
            "detail": (f"{named} — same-direction (regressive) surround on the T4b/T5b detectors "
                       f"(Egelhaaf's predicted ipsilateral surround)" if named
                       else "no same-direction surround resolved"),
            "name": named, "direction": sdw.get("direction"),
            "to_detectors_syn": to_det, "to_fd3_syn": sdw.get("to_fd3_syn", 0) or 0,
            "label": _label(ok, bool(named and to_det > 0))}


def _stage_identity(k: dict) -> dict:
    """Read Family K's aggregate identity verdict (no recompute)."""
    verdict = None
    for c in k.get("claims", []):
        cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
        if cid == "K.identity_verdict":
            verdict = c.get("verdict") if isinstance(c, dict) else getattr(c, "verdict", None)
    present = verdict in (K.CONFIRMED, K.CONFIRMED_WITH_CAVEAT)
    return {"stage": "identity", "detail": "FD3 == LPT42_Nod4 (Egelhaaf-1985 correlate)",
            "k_identity_verdict": verdict,
            "label": _label(present, verdict == K.CONFIRMED_WITH_CAVEAT and not present)}


def _stage_descending(l: dict) -> dict:
    direct = l.get("direct", {})
    top = direct.get("top_dn")
    n = direct.get("n_dns", 0)
    steering_frac = direct.get("steering_frac")
    ok = bool(top and n)
    return {"stage": "descending", "detail": "FD3 reaches a figure-steering descending neuron",
            "top_dn": top, "n_dns": n, "steering_frac": steering_frac, "label": _label(ok)}


def _stage_motor(l: dict) -> dict:
    motor = l.get("motor", {})
    dom = motor.get("dominant_motor_system") if motor.get("available") else None
    ok = bool(dom)
    return {"stage": "motor", "detail": "FD3's descending output drives wing-steering muscle (MaleCNS)",
            "dominant_motor_system": dom, "available": bool(motor.get("available")),
            "label": _label(ok, motor.get("available") and not dom)}


def run(src, meta, *, p_derived: dict | None = None,
        k_derived: dict | None = None, l_derived: dict | None = None,
        q_derived: dict | None = None, r_derived: dict | None = None) -> dict:
    """Assemble the FD3 circuit spine, now NAMING the intermediate sheet + wide-field inhibitor.

    ``p_derived`` (Family P) is computed here if not supplied. ``k/l/q/r_derived`` are the
    corresponding blocks from ``verification_results.json``; when absent, the stages that read
    them report ABSENT/UNKNOWN rather than recomputing those heavy families.

    Named spine (gated): photoreceptor -> lamina_medulla -> motion_layer_b -> sheet(LPC1) ->
    fd3_afferent -> identity -> descending -> motor. Plus a MODULATORY widefield_inhibitor(LPi14)
    node — reported and counted, but not a hard gate (a missing gate weakens, does not break, the
    feed-forward spine; matching how the paper treats VCH gating an otherwise-complete sheet).
    """
    p = p_derived if p_derived is not None else P.run(src, meta)
    k = k_derived or {}
    l = l_derived or {}
    q = q_derived or {}
    r = r_derived or {}

    # feed-forward spine (gated), in circuit order
    spine = [
        _stage_photoreceptor(p),
        _stage_lamina_medulla(p),
        _stage_motion_layer_b(p),
        _stage_sheet(q),            # NAMED intermediate: T4b/T5b -> LPC1 -> FD3
        _stage_fd3_afferent(p),
        _stage_identity(k),
        _stage_descending(l),
        _stage_motor(l),
    ]
    # modulatory nodes (reported, not gated): the two wide-field gates
    inhibitor = _stage_widefield_inhibitor(r)          # LPi14, opponent, on FD3/sheet
    same_direction = _stage_same_direction_gate(r)     # LPi12, same-direction, on detectors
    stages = spine + [inhibitor, same_direction]

    # Decision gated on the INPUT tiers of the feed-forward spine (photoreceptor..fd3_afferent,
    # i.e. through the named sheet). Identity/descending/motor are the established arms; the
    # inhibitor is modulatory and excluded from the gate.
    gated = spine[:5]
    labels = [s["label"] for s in gated]
    if all(x == PRESENT for x in labels):
        decision = GO
    elif ABSENT in labels:
        decision = NO_GO
    else:
        decision = PARTIAL
    first_break = next((s["stage"] for s in gated if s["label"] == ABSENT), None)

    return {
        "stages": stages,
        "input_decision": decision,
        "first_break": first_break,
        "all_present": all(s["label"] == PRESENT for s in stages),
        "n_present": sum(1 for s in stages if s["label"] == PRESENT),
        "widefield_inhibitor": inhibitor,
        "same_direction_gate": same_direction,
        "track": getattr(src, "track", None),
    }
