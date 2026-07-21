"""Unit tests for the FD3 full-circuit existence screen — pure stage-labelling logic on synthetic
derived dicts, no live CAVE.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_fd3_circuit_screen.py -q
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.motif import compare as K
from flyconn.paper.derive import fd3_circuit_screen as SC


class _DummySrc:
    track = "offline"


def _p_ok():
    return {
        "census": {"t4t5_syn": 3299, "layer_b_frac_of_t4t5": 98.6,
                   "on_off_split": {"T4b_ON": 1793, "T5b_OFF": 1459}},
        "central_inputs": {"total_central_syn": 5000},
        "contra_inhibition": {"both_present": True},
        "upstream_cascade": {
            "on_limb_t4b": {"n_expected_present": 3}, "off_limb_t5b": {"n_expected_present": 3},
            "lamina_present": ["L1", "L2", "L3"], "photoreceptor_present": ["R1-6", "R7", "R8"]},
    }


def _k_ok():
    return {"claims": [{"id": "K.identity_verdict", "verdict": K.CONFIRMED_WITH_CAVEAT}]}


def _l_ok():
    return {"direct": {"top_dn": "DNp26", "n_dns": 24, "steering_frac": 56.0},
            "motor": {"available": True, "dominant_motor_system": "wing_steering"}}


def _q_ok():
    return {"named_sheet": "LPC1", "feed_forward": True, "sheet_set": ["LPC1", "LLPC3", "LLPC2"],
            "profiles": {"LPC1": {"layer_b_input_syn": 28848, "to_fd3_syn": 1120}}}


def _r_ok():
    return {"winner": {"cell_type": "LPi14", "direction": "opponent",
                       "to_fd3_syn": 961, "to_sheet_syn": 6631, "to_detectors_syn": 7004},
            "same_direction_winner": {"cell_type": "LPi12", "direction": "same_direction",
                                      "to_fd3_syn": 68, "to_sheet_syn": 165, "to_detectors_syn": 18678,
                                      "layer_b_pct": 99.6}}


def _run(**over):
    kw = dict(p_derived=_p_ok(), k_derived=_k_ok(), l_derived=_l_ok(),
              q_derived=_q_ok(), r_derived=_r_ok())
    kw.update(over)
    return SC.run(_DummySrc(), None, **kw)


def test_all_stages_present_go():
    scr = _run()
    labels = {s["stage"]: s["label"] for s in scr["stages"]}
    assert all(v == SC.PRESENT for v in labels.values()), labels
    assert scr["input_decision"] == SC.GO
    assert scr["all_present"] and scr["first_break"] is None
    # 8 spine stages + 2 modulatory gate nodes (opponent LPi14 + same-direction LPi12)
    assert scr["n_present"] == 10
    # the named intermediate + both gates are present
    assert labels["sheet"] == SC.PRESENT and labels["widefield_inhibitor"] == SC.PRESENT
    assert labels["same_direction_gate"] == SC.PRESENT


def test_sheet_stage_named():
    scr = _run()
    sheet = next(s for s in scr["stages"] if s["stage"] == "sheet")
    assert sheet["named_sheet"] == "LPC1" and "LPC1" in sheet["detail"]


def test_missing_sheet_breaks_spine():
    scr = _run(q_derived={})   # no Family Q -> sheet stage absent
    labels = {s["stage"]: s["label"] for s in scr["stages"]}
    assert labels["sheet"] == SC.ABSENT
    assert scr["input_decision"] == SC.NO_GO
    assert scr["first_break"] == "sheet"


def test_missing_inhibitor_is_modulatory_not_gate():
    """No Family R -> the inhibitor node is absent but the feed-forward spine still GOes."""
    scr = _run(r_derived={})
    labels = {s["stage"]: s["label"] for s in scr["stages"]}
    assert labels["widefield_inhibitor"] == SC.ABSENT
    assert scr["input_decision"] == SC.GO       # inhibitor is modulatory, not a gate
    assert not scr["all_present"]


def test_photoreceptor_absent_breaks_no_go():
    p = _p_ok(); p["upstream_cascade"]["photoreceptor_present"] = []
    scr = _run(p_derived=p)
    labels = {s["stage"]: s["label"] for s in scr["stages"]}
    assert labels["photoreceptor"] == SC.ABSENT
    assert scr["input_decision"] == SC.NO_GO
    assert scr["first_break"] == "photoreceptor"


def test_low_layer_b_weak_partial():
    p = _p_ok(); p["census"]["layer_b_frac_of_t4t5"] = 30.0
    scr = _run(p_derived=p)
    labels = {s["stage"]: s["label"] for s in scr["stages"]}
    assert labels["motion_layer_b"] in (SC.WEAK, SC.ABSENT)
    assert scr["input_decision"] in (SC.PARTIAL, SC.NO_GO)


def test_missing_output_families_report_absent_but_input_can_go():
    """The decision gates on the feed-forward spine through fd3_afferent; missing K/L only
    affect the established downstream arms."""
    scr = _run(k_derived={}, l_derived={})
    labels = {s["stage"]: s["label"] for s in scr["stages"]}
    assert scr["input_decision"] == SC.GO
    assert labels["identity"] == SC.ABSENT
    assert labels["descending"] == SC.ABSENT and labels["motor"] == SC.ABSENT
    assert not scr["all_present"]


def test_central_below_floor_weakens_afferent():
    p = _p_ok(); p["central_inputs"]["total_central_syn"] = 100
    scr = _run(p_derived=p)
    labels = {s["stage"]: s["label"] for s in scr["stages"]}
    assert labels["fd3_afferent"] == SC.WEAK
