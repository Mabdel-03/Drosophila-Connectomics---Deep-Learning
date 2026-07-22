"""Unit tests for Family Q (FD3 sheet-set) oracle — synthetic derived dicts, no live CAVE.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_q_fd3_sheet.py -q
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.motif import compare as K
from flyconn.paper.derive import q_fd3_sheet as QD
from flyconn.paper.oracle import q_fd3_sheet as QO


def _prof(layer, direction, to_fd3, layer_b_syn, hard=True):
    return {"present": True, "nt": "acetylcholine", "dominant_layer": layer,
            "dominant_direction": direction, "to_fd3_syn": to_fd3, "layer_b_input_syn": layer_b_syn,
            "dir_margin_pp": 90.0 if hard else 5.0, "dir_boot_stability": 1.0 if hard else 0.5,
            "dir_hard": hard, "reads_layer_b": layer == "b"}


def _healthy(**over):
    profiles = {
        "LPC1": _prof("b", "regressive", 1120, 15126),
        "LLPC2": _prof("c", "upward", 979, 100),
        "LLPC3": _prof("d", "downward", 1071, 60),
        "LLPC1": _prof("a", "progressive", 446, 20),
    }
    d = {
        "candidate": "LPT42_Nod4", "named_sheet": "LPC1",
        "profiles": profiles,
        "n_layerb_sheets": 1,
        "fold_vs_other_layerb": "unique",
        "feed_forward": True,
        "sheet_set": ["LPC1", "LLPC3", "LLPC2"],
        "sheet_set_channels": {"LPC1": "regressive", "LLPC3": "downward", "LLPC2": "upward"},
        "retinotopy_null": {"available": True, "obs_radius_um": 7.0, "null_radius_um": 40.0,
                            "z_score": -55.0, "p_value": 0.002, "n_perms": 500, "local": True},
        "llpc1_control_to_fd3": 446,
        "directional_composition": QD._directional_composition(profiles),
        "entropy_null": {"available": True, "fd3_entropy_bits": 1.9, "null_mean_bits": 1.2,
                         "null_median_bits": 1.1, "n_null_cells": 120, "fd3_percentile": 0.9,
                         "fd3_elevated": True},
        "fd1_reference": {"available": True, "cell_type": "Nod1", "n_cardinal_directions": 2,
                          "matched_frac": 60.0, "context_frac": 40.0, "channel_entropy_bits": 0.97,
                          "sheet_frac_by_channel": {"a": 60.0, "b": 0.0, "c": 40.0, "d": 0.0},
                          "fd3_specific": True},
        "track": "offline",
    }
    for k, v in over.items():
        d[k] = v
    return d


def _by(claims):
    return {c.id: c for c in claims}


def test_healthy_sheet_confirms():
    by = _by(QO.build_claims(_healthy()))
    for cid in ("Q.sheet_cholinergic", "Q.sheet_layer_b", "Q.feed_forward", "Q.retinotopy_local",
                "Q.lpc1_unique_layerb", "Q.nc_not_fd1_sheet"):
        assert by[cid].verdict == K.CONFIRMED, cid
    assert by["Q.sheet_verdict"].verdict in (K.CONFIRMED, K.CONFIRMED_WITH_CAVEAT)


def test_not_layer_b_refutes_core():
    d = _healthy()
    d["profiles"]["LPC1"]["dominant_layer"] = "c"
    by = _by(QO.build_claims(d))
    assert by["Q.sheet_layer_b"].verdict == K.REFUTED
    assert by["Q.sheet_verdict"].verdict == K.REFUTED


def test_multiple_layerb_sheets_refutes_uniqueness():
    by = _by(QO.build_claims(_healthy(n_layerb_sheets=3)))
    assert by["Q.lpc1_unique_layerb"].verdict == K.REFUTED
    assert by["Q.sheet_verdict"].verdict == K.REFUTED


def test_fd1_sheet_dominates_refutes_negative_control():
    # if LLPC1 (FD1's sheet) feeds FD3 as much as LPC1, the "different sheet" control fails
    by = _by(QO.build_claims(_healthy(llpc1_control_to_fd3=1100)))
    assert by["Q.nc_not_fd1_sheet"].verdict == K.REFUTED
    assert by["Q.sheet_verdict"].verdict == K.REFUTED


def test_retinotopy_not_local_refutes():
    d = _healthy()
    d["retinotopy_null"] = {"available": True, "obs_radius_um": 42.0, "null_radius_um": 40.0,
                            "z_score": 1.0, "p_value": 0.5, "n_perms": 500, "local": False}
    by = _by(QO.build_claims(d))
    assert by["Q.retinotopy_local"].verdict == K.REFUTED


def test_retinotopy_unavailable_is_caveat_not_refute():
    d = _healthy()
    d["retinotopy_null"] = {"available": False, "reason": "too few detector loci"}
    by = _by(QO.build_claims(d))
    assert by["Q.retinotopy_local"].verdict == K.UNVERIFIABLE
    # core not all CONFIRMED (retino is UNVERIFIABLE) but no falsifier -> caveat, not refute
    assert by["Q.sheet_verdict"].verdict == K.CONFIRMED_WITH_CAVEAT


def test_sheet_set_framing_corroborating():
    by = _by(QO.build_claims(_healthy()))
    assert by["Q.sheet_set"].verdict in (K.CONFIRMED, K.CONFIRMED_WITH_CAVEAT)


# ---- new: directional composition (the mentor's LLPC1/2/3 -> a/c/d point) ----

def test_directional_composition_pure_function():
    """_directional_composition rolls ->FD3 up by dominant layer; PURE, no connectome."""
    profiles = {
        "LPC1": _prof("b", "regressive", 1120, 15126),
        "LLPC2": _prof("c", "upward", 979, 100),
        "LLPC3": _prof("d", "downward", 1071, 60),
        "LPC2": _prof("c", "upward", 176, 50),
        "LLPC1": _prof("a", "progressive", 446, 20),
    }
    comp = QD._directional_composition(profiles)
    assert comp["n_cardinal_directions"] == 4          # a, b, c, d all represented
    assert comp["syn_by_channel"]["b"] == 1120         # LPC1 (regressive)
    assert comp["syn_by_channel"]["c"] == 979 + 176    # LLPC2 + LPC2 (upward)
    assert comp["channel_members"]["a"] == ["LLPC1"]
    # matched (regressive) share of total sheet input; total = 1120+979+1071+176+446 = 3792
    assert abs(comp["matched_frac"] - round(100.0 * 1120 / 3792, 1)) < 0.05
    assert comp["matched_frac"] < 50.0                  # matched channel is a minority


def test_soft_direction_sheet_excluded_from_channels():
    """A near-tie (not-hard) sheet does not contribute to any cardinal channel."""
    profiles = {
        "LPC1": _prof("b", "regressive", 1120, 15126),
        "LLPC2": _prof("c", "upward", 979, 100, hard=False),   # soft -> excluded
    }
    comp = QD._directional_composition(profiles)
    assert "LLPC2" in comp["soft_direction_sheets"]
    assert comp["syn_by_channel"]["c"] == 0
    assert "LLPC2" not in comp["channel_members"]["c"]


def test_pools_all_cardinal_directions_claim():
    by = _by(QO.build_claims(_healthy()))
    c = by["Q.pools_all_cardinal_directions"]
    assert c.verdict in (K.CONFIRMED, K.CONFIRMED_WITH_CAVEAT)


def test_generic_pooling_downgrades_to_caveat():
    """If the entropy null / FD1 reference show 4-direction pooling is background, downgrade."""
    d = _healthy(fd1_reference={"available": True, "cell_type": "Nod1",
                                "n_cardinal_directions": 4, "matched_frac": 30.0, "context_frac": 70.0,
                                "channel_entropy_bits": 1.95,
                                "sheet_frac_by_channel": {"a": 25, "b": 30, "c": 25, "d": 20},
                                "fd3_specific": False})
    by = _by(QO.build_claims(d))
    assert by["Q.pools_all_cardinal_directions"].verdict == K.CONFIRMED_WITH_CAVEAT


def test_matched_channel_minority_claim():
    by = _by(QO.build_claims(_healthy()))
    assert by["Q.matched_channel_minority"].verdict == K.CONFIRMED


def test_directional_claims_are_corroborating_only():
    for cid in ("Q.pools_all_cardinal_directions", "Q.matched_channel_minority"):
        assert cid not in QO.CORE_IDS and cid not in QO.DISC_IDS


def test_aggregate_sheet_verdict_still_healthy():
    """The new directional claims must not break the CORE/DISC LPC1 verdict."""
    by = _by(QO.build_claims(_healthy()))
    assert by["Q.sheet_verdict"].verdict in (K.CONFIRMED, K.CONFIRMED_WITH_CAVEAT)
