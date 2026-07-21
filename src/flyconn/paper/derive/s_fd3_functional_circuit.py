"""Derive family S: the FD3 functional figure-ground circuit, NAMED end to end.

Composes the already-derived families (Q sheet, R inhibitor, K identity, L descending, P input)
into one auditable object that states the isolated functional circuit with measured edge weights:

    T4b/T5b  --(det->sheet)-->  LPC1  --(1120)-->  FD3  --(->DNp26)-->  DNp26  -->  wing
                                 ^                    ^
                      LPi14 --(6631, gate)-- LPi14 --(961, inhibit)--        (LPi14 also -> T4b/T5b 8239)

Pure composition (no heavy pulls): reads the sibling families' derived dicts, like
``fd3_circuit_screen`` reads P/K/L. Emits a named edge list + a single GO/PARTIAL/NO_GO verdict.
"""

from __future__ import annotations

GO, PARTIAL, NO_GO = "GO", "PARTIAL", "NO_GO"


def _g(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _k_identity_verdict(k: dict) -> str | None:
    for c in k.get("claims", []):
        cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
        if cid == "K.identity_verdict":
            return c.get("verdict") if isinstance(c, dict) else getattr(c, "verdict", None)
    return None


def run(src, meta, *, q_derived=None, r_derived=None, k_derived=None,
        l_derived=None, p_derived=None) -> dict:
    """Assemble the named functional circuit from the sibling derived dicts.

    ``src``/``meta`` are accepted for a uniform orchestrator signature but not used (pure
    composition). Each sibling dict is optional; a missing one makes its edges UNKNOWN.
    """
    del src, meta
    q = q_derived or {}
    r = r_derived or {}
    k = k_derived or {}
    l = l_derived or {}
    p = p_derived or {}

    census = _g(p, "census", default={})
    oo = _g(census, "on_off_split", default={})
    named_sheet = _g(q, "named_sheet")
    sheet_prof = _g(q, "profiles", named_sheet, default={}) if named_sheet else {}
    win = _g(r, "winner", default={})
    sdw = _g(r, "same_direction_winner", default={}) or {}
    direct = _g(l, "direct", default={})

    # --- named edges with measured weights ---
    edges = [
        {"src": "T4b/T5b", "dst": named_sheet or "LPC1", "role": "feed-forward (detectors->sheet)",
         "syn": _g(sheet_prof, "layer_b_input_syn"), "measured": _g(sheet_prof, "layer_b_input_syn") is not None},
        {"src": named_sheet or "LPC1", "dst": "FD3", "role": "sheet->figure cell",
         "syn": _g(sheet_prof, "to_fd3_syn"), "measured": _g(sheet_prof, "to_fd3_syn") is not None},
        {"src": "T4b/T5b", "dst": "FD3", "role": "direct motion drive (layer-b)",
         "syn": (oo.get("T4b_ON", 0) + oo.get("T5b_OFF", 0)) or None,
         "measured": bool(oo)},
        {"src": win.get("cell_type") or "LPi14", "dst": "FD3", "role": "wide-field inhibition (opponent)",
         "syn": win.get("to_fd3_syn"), "measured": win.get("to_fd3_syn") is not None},
        {"src": win.get("cell_type") or "LPi14", "dst": named_sheet or "LPC1", "role": "gates the sheet",
         "syn": win.get("to_sheet_syn"), "measured": win.get("to_sheet_syn") is not None},
        {"src": win.get("cell_type") or "LPi14", "dst": "T4b/T5b", "role": "feedback onto detectors",
         "syn": win.get("to_detectors_syn"), "measured": win.get("to_detectors_syn") is not None},
        {"src": "FD3", "dst": _g(direct, "top_dn") or "DNp26", "role": "figure cell->steering DN",
         "syn": None, "measured": _g(direct, "top_dn") is not None},
        {"src": _g(direct, "top_dn") or "DNp26", "dst": "wing", "role": "steering command->wing",
         "syn": None, "measured": _g(l, "motor", "available", default=False)},
    ]
    # the same-direction (LPi12) surround gates the DETECTOR node (a distinct edge, if resolved)
    if sdw.get("cell_type"):
        edges.append({"src": sdw.get("cell_type"), "dst": "T4b/T5b",
                      "role": "wide-field inhibition (same-direction, detector node)",
                      "syn": sdw.get("to_detectors_syn"),
                      "measured": sdw.get("to_detectors_syn") is not None})

    # --- node presence (from the sibling verdicts) ---
    nodes = {
        "detectors": {"present": bool(oo), "detail": "T4b/T5b layer-b motion detectors"},
        "sheet": {"present": _g(q, "feed_forward") is True,
                  "name": named_sheet, "verdict": _named_verdict(q, "Q.sheet_verdict"),
                  "detail": f"{named_sheet} (direction-matched layer-b sheet)"},
        "figure_cell": {"present": _k_identity_verdict(k) in ("CONFIRMED", "CONFIRMED_WITH_CAVEAT"),
                        "name": "FD3 (LPT42_Nod4)", "verdict": _k_identity_verdict(k)},
        "widefield_inhibitor": {"present": bool(win.get("cell_type")),
                                "name": win.get("cell_type"),
                                "verdict": _named_verdict(r, "R.inhibitor_verdict"),
                                "direction": win.get("direction"),
                                "detail": f"{win.get('cell_type')} (VCH-role opponent gate)"},
        "same_direction_inhibitor": {"present": bool(sdw.get("cell_type")),
                                     "name": sdw.get("cell_type"),
                                     "direction": sdw.get("direction"),
                                     "detail": (f"{sdw.get('cell_type')} (same-direction surround on "
                                                f"detectors)" if sdw.get("cell_type") else None)},
        "steering_dn": {"present": _g(direct, "top_dn") is not None, "name": _g(direct, "top_dn")},
        "motor": {"present": _g(l, "motor", "available", default=False),
                  "system": _g(l, "motor", "dominant_motor_system")},
    }

    # --- verdict: the feed-forward spine (detectors->sheet->FD3->DN->motor) must be complete;
    #     the inhibitor is modulatory (reported, weakens if absent but does not break the spine). ---
    spine = ["detectors", "sheet", "figure_cell", "steering_dn", "motor"]
    spine_present = [nodes[n]["present"] for n in spine]
    if all(spine_present):
        decision = GO
    elif not any(spine_present):
        decision = NO_GO
    else:
        decision = PARTIAL
    first_break = next((n for n in spine if not nodes[n]["present"]), None)

    return {
        "circuit": "T4b/T5b -> {LPC1(+LLPC2/3)} -> FD3 -> DNp26 -> wing; LPi14 wide-field opponent gate",
        "nodes": nodes,
        "edges": edges,
        "spine_complete": all(spine_present),
        "widefield_inhibitor_present": nodes["widefield_inhibitor"]["present"],
        "decision": decision,
        "first_break": first_break,
    }


def _named_verdict(fam_derived_or_block: dict, verdict_id: str) -> str | None:
    """Read an aggregate verdict from a family block's claims list, if present."""
    for c in fam_derived_or_block.get("claims", []) if isinstance(fam_derived_or_block, dict) else []:
        cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
        if cid == verdict_id:
            return c.get("verdict") if isinstance(c, dict) else getattr(c, "verdict", None)
    return None
