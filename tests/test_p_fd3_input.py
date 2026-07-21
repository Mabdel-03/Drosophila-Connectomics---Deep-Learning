"""Unit tests for Family P (FD3 afferent pathway) — pure functions on synthetic derived dicts,
no live CAVE and no big files.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_p_fd3_input.py -q
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.motif import compare as K
from flyconn.paper.oracle import p_fd3_input as OP
from flyconn.paper.derive import common as CM
from flyconn.paper.derive import k_fd3_lpt42 as KD


def _healthy_derived(**over):
    d = {
        "census": {"total_input_syn": 13157, "n_partners": 2991, "t4t5_syn": 3299,
                   "t4t5_frac_of_total": 25.1, "layer_b_frac_of_t4t5": 98.6,
                   "t4t5_dominant_direction": "back_to_front",
                   "on_off_split": {"T4b_ON": 1793, "T5b_OFF": 1459, "t4_frac": 55.1},
                   "t4t5_layer_frac": {"a": 0.09, "b": 98.6, "c": 1.2, "d": 0.15},
                   "input_syn_by_super_class": {"optic": 6238, "visual_projection": 4775},
                   "top_input_types": [{"cell_type": "T4b", "syn": 1793, "nt": "acetylcholine"}]},
        "upstream_cascade": {
            "on_limb_t4b": {"n_expected_present": 3, "expected_medulla_present": ["Mi1", "Tm3", "C3"],
                            "expected_frac_of_input": 61.0},
            "off_limb_t5b": {"n_expected_present": 3, "expected_medulla_present": ["Tm1", "Tm2", "Tm9"],
                             "expected_frac_of_input": 70.0},
            "lamina_present": ["L1", "L2", "L3"], "photoreceptor_present": ["R1-6", "R7", "R8"]},
        "central_inputs": {"any_layerb_carrier": True, "n_layerb_carriers": 1, "total_central_syn": 5000,
                           "top_types": [{"cell_type": "LPC1", "dominant_layer": "b"}]},
        "contra_inhibition": {"available": True, "both_present": True, "sufficient": False,
                              "syn_progressive": 7, "syn_regressive": 389, "n_classified": 5},
        "negative_controls": {"no_vch_gate": True, "vch_input_syn": 6, "vch_input_frac": 0.05,
                              "no_layer_a_drive": True, "layer_a_frac_of_t4t5": 0.09},
        "query_truncated": False,
    }
    for k, v in over.items():
        d[k] = {**d.get(k, {}), **v} if isinstance(v, dict) and isinstance(d.get(k), dict) else v
    return d


def _by_id(claims):
    return {c.id: c for c in claims}


def test_healthy_pathway_confirms_with_caveat():
    """All CORE + DISC confirm; the power-capped contra + literature front end make it CWC."""
    claims = OP.build_claims(_healthy_derived())
    by = _by_id(claims)
    for cid in ("P.layer_b_drive", "P.on_off_mix", "P.upstream_cascade_present",
                "P.nc_no_vch_gate", "P.nc_no_layer_a_drive"):
        assert by[cid].verdict == K.CONFIRMED, cid
    assert by["P.input_pathway_verdict"].verdict == K.CONFIRMED_WITH_CAVEAT


def test_layer_b_core_failure_refutes():
    d = _healthy_derived(census={"layer_b_frac_of_t4t5": 40.0})
    by = _by_id(OP.build_claims(d))
    assert by["P.layer_b_drive"].verdict == K.REFUTED
    assert by["P.input_pathway_verdict"].verdict == K.REFUTED


def test_vch_gate_negative_control_refutes():
    """A VCH gate on FD3 (like FD1) must trip the discriminating control -> REFUTED."""
    d = _healthy_derived(negative_controls={"no_vch_gate": False, "vch_input_syn": 900,
                                            "vch_input_frac": 6.8, "no_layer_a_drive": True,
                                            "layer_a_frac_of_t4t5": 0.09})
    by = _by_id(OP.build_claims(d))
    assert by["P.nc_no_vch_gate"].verdict == K.REFUTED
    assert by["P.input_pathway_verdict"].verdict == K.REFUTED


def test_layer_a_drive_negative_control_refutes():
    d = _healthy_derived(negative_controls={"no_vch_gate": True, "vch_input_syn": 6,
                                            "vch_input_frac": 0.05, "no_layer_a_drive": False,
                                            "layer_a_frac_of_t4t5": 45.0})
    by = _by_id(OP.build_claims(d))
    assert by["P.nc_no_layer_a_drive"].verdict == K.REFUTED
    assert by["P.input_pathway_verdict"].verdict == K.REFUTED


def test_missing_on_off_mix_refutes_core():
    d = _healthy_derived(census={"on_off_split": {"T4b_ON": 1793, "T5b_OFF": 0, "t4_frac": 100.0},
                                 "layer_b_frac_of_t4t5": 98.6})
    by = _by_id(OP.build_claims(d))
    assert by["P.on_off_mix"].verdict == K.REFUTED
    assert by["P.input_pathway_verdict"].verdict == K.REFUTED


def test_incomplete_cascade_refutes():
    d = _healthy_derived(upstream_cascade={
        "on_limb_t4b": {"n_expected_present": 0, "expected_medulla_present": [], "expected_frac_of_input": 0.0},
        "off_limb_t5b": {"n_expected_present": 3, "expected_medulla_present": ["Tm1", "Tm2", "Tm9"],
                         "expected_frac_of_input": 70.0},
        "lamina_present": ["L1"], "photoreceptor_present": ["R1-6"]})
    by = _by_id(OP.build_claims(d))
    assert by["P.upstream_cascade_present"].verdict == K.REFUTED


def test_contra_inhibition_power_capped_never_refutes():
    """Bidirectional-but-underpowered -> caveat, not refutation (power must not overturn it)."""
    d = _healthy_derived(contra_inhibition={"available": True, "both_present": True,
                                            "sufficient": False, "syn_progressive": 7,
                                            "syn_regressive": 389, "n_classified": 3,
                                            "dominant_direction": "regressive", "minority_frac": 0.018,
                                            "substantially_bidirectional": False})
    by = _by_id(OP.build_claims(d))
    assert by["P.contra_inhibition_bidirectional"].verdict == K.CONFIRMED_WITH_CAVEAT


def test_contra_inhibition_well_powered_confirms():
    # both channels substantial (minority 0.25) AND well-powered -> CONFIRMED
    d = _healthy_derived(contra_inhibition={"available": True, "both_present": True,
                                            "sufficient": True, "syn_progressive": 40,
                                            "syn_regressive": 120, "n_classified": 8,
                                            "dominant_direction": "regressive", "minority_frac": 0.25,
                                            "substantially_bidirectional": True})
    by = _by_id(OP.build_claims(d))
    assert by["P.contra_inhibition_bidirectional"].verdict == K.CONFIRMED


def test_contra_inhibition_trace_minority_is_caveat_not_confirmed():
    """A powered 'both_present' with only a ~7% minority reads as direction-dominant, not
    bidirectional -> CONFIRMED_WITH_CAVEAT (guards against overclaiming bidirectionality)."""
    d = _healthy_derived(contra_inhibition={"available": True, "both_present": True,
                                            "sufficient": True, "syn_progressive": 387,
                                            "syn_regressive": 30, "n_classified": 6,
                                            "dominant_direction": "progressive", "minority_frac": 0.072,
                                            "substantially_bidirectional": False})
    by = _by_id(OP.build_claims(d))
    assert by["P.contra_inhibition_bidirectional"].verdict == K.CONFIRMED_WITH_CAVEAT


def test_truncation_refutes_guard():
    d = _healthy_derived(query_truncated=True)
    by = _by_id(OP.build_claims(d))
    assert by["P.no_truncation"].verdict == K.REFUTED
    assert by["P.input_pathway_verdict"].verdict == K.REFUTED


def test_t4t5_minority_reported_and_confirmed():
    by = _by_id(OP.build_claims(_healthy_derived()))
    c = by["P.t4t5_minority_of_input"]
    assert c.verdict == K.CONFIRMED and c.computed_primary == 25.1


# --- regression: the lifted contra_inhibition_profile is the exact object K re-exports ---
def test_contra_inhibition_profile_shared_with_family_k():
    assert KD._contra_inhibition is CM.contra_inhibition_profile
    assert KD._INH_LAYER is CM.INH_LAYER
    assert KD.MIN_CLASSIFIED_INH == CM.MIN_CLASSIFIED_INH
    # the layer table is the FD3-relevant progressive/regressive split
    assert CM.INH_LAYER["VCH"] == "a" and CM.INH_LAYER["LPi15"] == "b"
