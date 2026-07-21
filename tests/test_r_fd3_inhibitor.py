"""Unit tests for Family R (FD3 wide-field inhibitor) oracle — synthetic dicts, no live CAVE.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_r_fd3_inhibitor.py -q
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.motif import compare as K
from flyconn.paper.oracle import r_fd3_inhibitor as RO


def _sd_winner(**over):
    """A synthetic same-direction (LPi12-like) winner block: regressive, detector-dominant."""
    w = {
        "cell_type": "LPi12", "nt": "gaba", "super_class": "optic", "is_centrifugal": False,
        "direction": "same_direction", "layer_a_pct": 0.2, "layer_b_pct": 99.6,
        "t4t5_in_frac": 66.5, "to_fd3_syn": 68, "to_detectors_syn": 18678, "to_sheet_syn": 165,
        "t4t5_reciprocal_n": 2769, "frac_of_fd3_inhibition": 2.5, "frac_of_detector_inhibition": 14.0,
        "to_fd3_frac_of_output": 0.16, "to_detectors_frac_of_output": 43.9, "fd3_target_rank": 13,
        "fd3_to_inhibitor_syn": 0,
    }
    w.update(over)
    return w


def _healthy(**over):
    d = {
        "candidate": "LPT42_Nod4", "named_inhibitor": "LPi14",
        "same_direction_inhibitor": "LPi12",
        "pass_set": ["LPi14"], "opponent_gates": ["LPi14"],
        "same_direction_gates": ["LPi12"],
        "winner": {
            "cell_type": "LPi14", "nt": "gaba", "super_class": "optic", "is_centrifugal": False,
            "direction": "opponent", "layer_a_pct": 94.8, "layer_b_pct": 4.5,
            "t4t5_in_frac": 72.6, "to_fd3_syn": 961, "to_detectors_syn": 7004,
            "to_sheet_syn": 6631, "t4t5_reciprocal_n": 300, "fd3_to_inhibitor_syn": 5,
            "frac_of_fd3_inhibition": 35.9, "frac_of_detector_inhibition": 5.3,
            "to_fd3_frac_of_output": 0.94, "to_detectors_frac_of_output": 6.85, "fd3_target_rank": 10,
        },
        "same_direction_winner": _sd_winner(),
        "same_direction_enrichment": {
            "available": True, "cell_type": "LPi12", "obs_to_fd3": 68, "null_mean": 11.1,
            "null_std": 19.3, "z_score": 2.95, "p_enrichment": 0.002, "n_perms": 2000,
            "enriched": True,
        },
        "floor_sweep": [
            {"fd3_floor": 30, "n_pass": 5, "pass": ["Am1", "LPi10", "LPi12", "LPi14", "LPi15"]},
            {"fd3_floor": 68, "n_pass": 3, "pass": ["LPi10", "LPi12", "LPi14"]},
            {"fd3_floor": 100, "n_pass": 2, "pass": ["LPi10", "LPi14"]},
            {"fd3_floor": 300, "n_pass": 1, "pass": ["LPi14"]},
        ],
        "screen": {}, "track": "offline",
    }
    for k, v in over.items():
        d[k] = v
    return d


def _by(claims):
    return {c.id: c for c in claims}


def test_healthy_inhibitor_confirms():
    by = _by(RO.build_claims(_healthy()))
    for cid in ("R.pools_widefield_t4t5", "R.inhibits_fd3", "R.gates_sheet",
                "R.feeds_back_detectors", "R.opponent_tuning", "R.named_opponent_gate"):
        assert by[cid].verdict == K.CONFIRMED, cid
    assert by["R.inhibitor_verdict"].verdict in (K.CONFIRMED, K.CONFIRMED_WITH_CAVEAT)


def test_no_fd3_inhibition_refutes_core():
    d = _healthy(); d["winner"]["to_fd3_syn"] = 10
    by = _by(RO.build_claims(d))
    assert by["R.inhibits_fd3"].verdict == K.REFUTED
    assert by["R.inhibitor_verdict"].verdict == K.REFUTED


def test_does_not_gate_sheet_refutes_core():
    d = _healthy(); d["winner"]["to_sheet_syn"] = 50
    by = _by(RO.build_claims(d))
    assert by["R.gates_sheet"].verdict == K.REFUTED


def test_not_widefield_refutes_core():
    d = _healthy(); d["winner"]["t4t5_in_frac"] = 10.0
    by = _by(RO.build_claims(d))
    assert by["R.pools_widefield_t4t5"].verdict == K.REFUTED


def test_not_opponent_refutes_disc():
    d = _healthy(); d["winner"]["layer_a_pct"] = 5.0   # not layer-a dominant
    by = _by(RO.build_claims(d))
    assert by["R.opponent_tuning"].verdict == K.REFUTED
    assert by["R.inhibitor_verdict"].verdict == K.REFUTED


def test_centrifugal_winner_does_not_break_verdict():
    """The winner being OPTIC (not centrifugal) is the point; a centrifugal winner would REFUTE
    the not_centrifugal corroborating claim but must NOT be required by the CORE/DISC verdict —
    i.e. Family R must never require visual_centrifugal (the departure from Family B)."""
    d = _healthy()
    by = _by(RO.build_claims(d))
    # not_centrifugal is corroborating, CONFIRMED for the optic LPi14
    assert by["R.not_centrifugal"].verdict == K.CONFIRMED
    assert "R.not_centrifugal" not in RO.CORE_IDS and "R.not_centrifugal" not in RO.DISC_IDS


def test_role_homolog_and_nonreciprocal_noted():
    by = _by(RO.build_claims(_healthy()))
    role = by["R.vch_equivalent_role"]
    assert role.verdict == K.CONFIRMED
    assert "role" in role.notes.lower() and "5 syn" in role.notes  # non-reciprocity quantified


# ---- new: the LPi12 same-direction / two-gate story ----

def test_same_direction_surround_resolved_as_lpi12():
    """With LPi12 in the sweep, the same-direction surround is a MEASURED CONFIRMED claim."""
    by = _by(RO.build_claims(_healthy()))
    c = by["R.same_direction_surround_found"]
    assert c.verdict == K.CONFIRMED
    assert "LPi12" in str(c.computed_primary)
    # the old hardcoded-UNVERIFIABLE id is gone
    assert "R.same_direction_surround" not in by


def test_same_direction_is_detector_dominant():
    by = _by(RO.build_claims(_healthy()))
    assert by["R.same_direction_detector_dominant"].verdict == K.CONFIRMED


def test_two_gate_architecture_confirmed_with_caveat():
    by = _by(RO.build_claims(_healthy()))
    v = by["R.two_gate_architecture"].verdict
    assert v == K.CONFIRMED_WITH_CAVEAT


def test_same_direction_does_not_replace_lpi14():
    by = _by(RO.build_claims(_healthy()))
    assert by["R.same_direction_not_replacement"].verdict == K.CONFIRMED
    # if the surround were to gate the sheet + be a major FD3 slice, this would REFUTE (replacement)
    d = _healthy()
    d["same_direction_winner"] = _sd_winner(frac_of_fd3_inhibition=40.0, to_sheet_syn=5000)
    by2 = _by(RO.build_claims(d))
    assert by2["R.same_direction_not_replacement"].verdict == K.REFUTED


def test_fd3_specificity_enriched():
    by = _by(RO.build_claims(_healthy()))
    assert by["R.same_direction_fd3_specificity"].verdict == K.CONFIRMED_WITH_CAVEAT


def test_new_claims_are_corroborating_only():
    """None of the new same-direction claims may enter CORE/DISC — the aggregate LPi14 verdict
    must be unchanged by them."""
    for cid in ("R.same_direction_surround_found", "R.same_direction_detector_dominant",
                "R.two_gate_architecture", "R.same_direction_not_replacement",
                "R.same_direction_fd3_specificity", "R.floor_sensitivity"):
        assert cid not in RO.CORE_IDS and cid not in RO.DISC_IDS


def test_aggregate_verdict_unchanged_by_two_gate_story():
    """The opponent (LPi14) aggregate stands whether or not the same-direction gate is present."""
    with_sd = _by(RO.build_claims(_healthy()))["R.inhibitor_verdict"].verdict
    without_sd = _by(RO.build_claims(_healthy(
        same_direction_inhibitor=None, same_direction_gates=[], same_direction_winner=None,
        same_direction_enrichment={"available": False, "reason": "none"},
    )))["R.inhibitor_verdict"].verdict
    assert with_sd == without_sd


def test_floor_sweep_flips_at_100():
    by = _by(RO.build_claims(_healthy()))
    c = by["R.floor_sensitivity"]
    assert c.verdict == K.CONFIRMED
    assert "LPi12" in str(c.computed_primary)  # passes at some (low) floors
