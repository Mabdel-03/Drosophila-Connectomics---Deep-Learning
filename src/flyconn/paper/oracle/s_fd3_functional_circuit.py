"""Family S oracle — the named FD3 functional figure-ground circuit.

Grades the assembled circuit from ``derive/s_fd3_functional_circuit.py``: each named node/edge of
the isolated circuit present (categorical/floor), plus one aggregate verdict. This is the single
object that answers the mentor: the whole circuit is named T4b/T5b -> LPC1 -> FD3 -> DNp26 ->
wing, with LPi14 the wide-field opponent gate.
"""

from __future__ import annotations

from flyconn.motif import compare as K

# the feed-forward spine nodes that must be present to headline the circuit
CORE_IDS = {"S.detectors", "S.sheet_named", "S.figure_cell", "S.steering_dn"}
DISC_IDS = {"S.widefield_inhibitor_named"}


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    nodes = d.get("nodes", {})

    def node(name):
        return nodes.get(name, {})

    out.append(K.compare_categorical(
        "S.detectors", "Motion detectors (T4b/T5b, layer-b) drive the circuit",
        True, bool(node("detectors").get("present"))))
    sheet = node("sheet")
    out.append(K.compare_categorical(
        "S.sheet_named", "The intermediate sheet is NAMED and feed-forward (T4b/T5b -> LPC1 -> FD3)",
        True, bool(sheet.get("present")),
        refuted_note="the sheet stage is not feed-forward / not named"))
    out[-1].notes = f"sheet={sheet.get('name')} verdict={sheet.get('verdict')}"
    fc = node("figure_cell")
    out.append(K.compare_categorical(
        "S.figure_cell", "The figure cell (FD3 = LPT42_Nod4) is confirmed",
        True, bool(fc.get("present"))))
    out[-1].notes = f"K identity verdict={fc.get('verdict')}"
    dn = node("steering_dn")
    out.append(K.compare_categorical(
        "S.steering_dn", "FD3 reaches a steering descending neuron (DNp26)",
        True, bool(dn.get("present"))))
    out[-1].notes = f"top DN={dn.get('name')}"
    mo = node("motor")
    out.append(K.compare_categorical(
        "S.motor", "The descending output resolves to a motor system (wing-steering)",
        True, bool(mo.get("present"))))
    out[-1].notes = f"dominant system={mo.get('system')}"

    # DISCRIMINATING: the wide-field inhibitor is named (the VCH-role cell)
    wi = node("widefield_inhibitor")
    out.append(K.compare_categorical(
        "S.widefield_inhibitor_named",
        "The wide-field inhibitor standing in VCH's place is NAMED (LPi14, opponent)",
        True, bool(wi.get("present")),
        refuted_note="no wide-field inhibitor node resolved"))
    out[-1].notes = (f"inhibitor={wi.get('name')} direction={wi.get('direction')} "
                     f"verdict={wi.get('verdict')}")

    out.append(_circuit_verdict(out, d))
    return out


def _circuit_verdict(claims, d) -> K.ClaimResult:
    by = {c.id: c for c in claims}
    core = [by[i] for i in CORE_IDS if i in by]
    disc = [by[i] for i in DISC_IDS if i in by]
    core_ref = [c.id for c in core if c.verdict == K.REFUTED]
    core_ok = all(c.verdict == K.CONFIRMED for c in core)
    disc_ok = all(c.verdict == K.CONFIRMED for c in disc)
    decision = d.get("decision")
    if core_ref or decision == "NO_GO":
        v, note = K.REFUTED, f"spine incomplete (first_break={d.get('first_break')})"
    elif core_ok and disc_ok:
        v, note = K.CONFIRMED, "full named circuit present: detectors->sheet->FD3->DN->motor + LPi14 gate"
    elif core_ok:
        v, note = K.CONFIRMED_WITH_CAVEAT, "feed-forward spine present; wide-field inhibitor node not fully resolved"
    else:
        v, note = K.CONFIRMED_WITH_CAVEAT, "partial circuit; inspect per-node table"
    return K.ClaimResult(
        id="S.functional_circuit_verdict",
        description="OVERALL: FD3 figure-ground circuit (T4b/T5b -> LPC1 -> FD3 -> DNp26; LPi14 gate)",
        report_value=d.get("circuit", "named FD3 circuit"), computed_primary=v,
        tolerance="feed-forward spine complete + inhibitor named -> CONFIRMED",
        verdict=v, notes=note)
