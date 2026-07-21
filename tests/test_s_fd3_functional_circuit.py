"""Unit tests for Family S (FD3 functional-circuit assembly) — synthetic sibling dicts, no CAVE.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_s_fd3_functional_circuit.py -q
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.motif import compare as K
from flyconn.paper.derive import s_fd3_functional_circuit as S
from flyconn.paper.oracle import s_fd3_functional_circuit as SO


def _q(named="LPC1", feed_forward=True):
    return {"named_sheet": named, "feed_forward": feed_forward,
            "profiles": {named: {"layer_b_input_syn": 28848, "to_fd3_syn": 1120}} if named else {},
            "claims": [{"id": "Q.sheet_verdict", "verdict": "CONFIRMED_WITH_CAVEAT"}]}


def _r(winner="LPi14"):
    return {"winner": {"cell_type": winner, "direction": "opponent", "to_fd3_syn": 961,
                       "to_sheet_syn": 6631, "to_detectors_syn": 7004} if winner else {},
            "claims": [{"id": "R.inhibitor_verdict", "verdict": "CONFIRMED_WITH_CAVEAT"}]}


def _k(verdict="CONFIRMED_WITH_CAVEAT"):
    return {"claims": [{"id": "K.identity_verdict", "verdict": verdict}]}


def _l(top_dn="DNp26", motor=True):
    return {"direct": {"top_dn": top_dn, "n_dns": 34},
            "motor": {"available": motor, "dominant_motor_system": "wing_steering" if motor else None}}


def _p():
    return {"census": {"on_off_split": {"T4b_ON": 2670, "T5b_OFF": 2183}}}


def _run(**over):
    kw = dict(q_derived=_q(), r_derived=_r(), k_derived=_k(), l_derived=_l(), p_derived=_p())
    kw.update(over)
    return S.run(None, None, **kw)


def test_full_circuit_go():
    d = _run()
    assert d["decision"] == S.GO and d["spine_complete"]
    assert d["widefield_inhibitor_present"]
    by = {c.id: c for c in SO.build_claims(d)}
    assert by["S.functional_circuit_verdict"].verdict == K.CONFIRMED
    for cid in ("S.detectors", "S.sheet_named", "S.figure_cell", "S.steering_dn",
                "S.widefield_inhibitor_named"):
        assert by[cid].verdict == K.CONFIRMED, cid


def test_named_edges_have_weights():
    d = _run()
    edges = {(e["src"], e["dst"]): e for e in d["edges"]}
    assert edges[("LPC1", "FD3")]["syn"] == 1120
    assert edges[("LPi14", "FD3")]["syn"] == 961
    assert edges[("LPi14", "LPC1")]["syn"] == 6631
    assert edges[("LPi14", "T4b/T5b")]["syn"] == 7004


def test_missing_sheet_breaks_spine():
    d = _run(q_derived=_q(named=None, feed_forward=False))
    assert d["first_break"] == "sheet"
    assert d["decision"] in (S.PARTIAL, S.NO_GO)
    by = {c.id: c for c in SO.build_claims(d)}
    assert by["S.sheet_named"].verdict == K.REFUTED
    assert by["S.functional_circuit_verdict"].verdict == K.REFUTED


def test_missing_inhibitor_is_caveat_not_break():
    """The inhibitor is modulatory: absent -> spine still GO, verdict CWC not REFUTED."""
    d = _run(r_derived=_r(winner=None))
    assert d["spine_complete"] and d["decision"] == S.GO
    assert not d["widefield_inhibitor_present"]
    by = {c.id: c for c in SO.build_claims(d)}
    assert by["S.widefield_inhibitor_named"].verdict == K.REFUTED
    # spine complete but the discriminating inhibitor node absent -> caveat
    assert by["S.functional_circuit_verdict"].verdict == K.CONFIRMED_WITH_CAVEAT


def test_missing_motor_partial():
    d = _run(l_derived=_l(motor=False))
    assert d["first_break"] == "motor"
    by = {c.id: c for c in SO.build_claims(d)}
    assert by["S.motor"].verdict == K.REFUTED
