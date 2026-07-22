"""Unit tests for the Family-KD (FD2/FD3 disambiguation) logic.

Pure functions on synthetic derive dicts -- no live CAVE, no big files. These prove the three-way
assignment (FD3=LPT42_Nod4, FD2=LPT21, Nod3=intermediate), the FD2 tie-break (homolateral + frontal
co-location selects LPT21 over the mixed Nod3), the two-sided FD3 branch, and shared-property marking.

Run: /orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python -m pytest tests/test_kd_fd3_disambig.py -q
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.motif import compare as K  # noqa: E402
from flyconn.paper.derive import kd_fd3_disambig as KD  # noqa: E402
from flyconn.paper.oracle import kd_fd3_disambig as KDO  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic candidate blocks (the per-candidate offline shape the oracle reads).
# ---------------------------------------------------------------------------
def _cand(*, layer_b, nt_conf, contra, heterolateral, homolateral, lateral_gap, frontal,
          frontal_coloc, offset, axon_cross, contra_pof, bounded=True):
    return {
        "layer_b_pct": layer_b, "nt_conf": nt_conf, "nt": "acetylcholine",
        "contra_output_pct": contra, "heterolateral": heterolateral, "homolateral": homolateral,
        "rf_more_lateral_both": lateral_gap, "rf_frontal_gap_both": lateral_gap,
        "rf_lateral_gap_both": lateral_gap, "frontal_field": frontal,
        "frontal_colocated": frontal_coloc, "mean_rf_offset": offset,
        "smallfield_bounded": bounded, "axon_crosses_contra_both": axon_cross,
        "projects_contralateral_pof": contra_pof,
        "profile": {"super_class": "visual_projection"},
        "morphology": {"available": True, "source": "skeleton", "cells": []},
        "contra_inhibition": {"available": True, "sufficient": False, "n_classified": 3},
        "rf": {"per_side": {}},
    }


def _fd3():   # LPT42_Nod4 as measured
    return _cand(layer_b=98.6, nt_conf=0.894, contra=90.3, heterolateral=True, homolateral=False,
                 lateral_gap=True, frontal=False, frontal_coloc=False, offset=12.7,
                 axon_cross=True, contra_pof=True)


def _fd2():   # LPT21 as measured (frontal + homolateral)
    return _cand(layer_b=99.2, nt_conf=0.81, contra=1.5, heterolateral=False, homolateral=True,
                 lateral_gap=False, frontal=True, frontal_coloc=True, offset=0.03,
                 axon_cross=True, contra_pof=False)


def _interm():  # Nod3 as measured (mixed + slightly lateral)
    return _cand(layer_b=97.7, nt_conf=0.878, contra=44.7, heterolateral=False, homolateral=False,
                 lateral_gap=False, frontal=False, frontal_coloc=False, offset=3.15,
                 axon_cross=True, contra_pof=False)


def _score(fd, cand, val):
    return val


def _derived(lpt42, nod3, lpt21, *, anchor_best="FD1", live=True):
    # Reproduce the constructive computation with the fine tie-break by calling _constructive on a
    # synthetic score matrix + candidate blocks.
    # Coarse binary screen: FD3 signature {b, lateral_gap, heterolateral}; FD2 {b, not-lateral, not-hetero}.
    def coarse(cand):
        lay = "b"
        latgap = bool(cand["rf_lateral_gap_both"])
        het = bool(cand["heterolateral"])
        # score vs each FD signature
        sig = {"FD1": ("a", False, True), "FD2": ("b", False, False),
               "FD3": ("b", True, True), "FD4": ("a", True, True)}
        return {fd: int(lay == s[0]) + int(latgap == s[1]) + int(het == s[2]) for fd, s in sig.items()}

    cands = {"LPT42_Nod4": lpt42, "Nod3": nod3, "LPT21": lpt21}
    cs = {ct: coarse(c) for ct, c in cands.items()}
    # Nod1 anchor: layer-a frontal heterolateral -> best FD1
    cs["Nod1"] = {"FD1": 3, "FD2": 1, "FD3": 1, "FD4": 2}
    score = {fd: {ct: cs[ct][fd] for ct in cs} for fd in ("FD1", "FD2", "FD3", "FD4")}

    rf_by = {ct: {"more_lateral": c["rf_more_lateral_both"], "has_frontal_gap": c["rf_frontal_gap_both"]}
             for ct, c in cands.items()}
    profiles = {ct: {"n_cells": 2, "dominant_layer": "b", "contra_output_pct": c["contra_output_pct"]}
                for ct, c in cands.items()}
    profiles["Nod1"] = {"n_cells": 4, "dominant_layer": "a", "contra_output_pct": 89.0}
    rf_by["Nod1"] = {"more_lateral": False, "has_frontal_gap": False}

    # Monkeypatch the family screen to our synthetic score matrix by feeding _constructive directly.
    import flyconn.paper.derive.k_fd3_lpt42 as KK
    orig = KK.fd_family_screen
    KK.fd_family_screen = lambda p, r: {"score_matrix": score, "best_match_for_FD3": "LPT42_Nod4",
                                        "best_FD_for_LPT42": "FD3", "fd3_margin": 2,
                                        "reciprocal_best_hit": True, "n_candidates": len(cands)}
    try:
        cons = KD._constructive(profiles, rf_by, cands)
    finally:
        KK.fd_family_screen = orig

    d = {
        "candidates_offline": cands,
        "constructive": cons,
        "contra_cutoff_sweep": {"invariant": True, "first_break_cutoff": None},
        "comparison_rows": KD._comparison_rows({"candidates": cands}, {"available": False}, cons),
        "live": {"available": bool(live), "candidates": cands} if live else {"available": False},
    }
    return d


def _verdict(claims):
    c = next(c for c in claims if c.id == "KD.decision_verdict")
    return c.verdict, c.computed_primary


# ---------------------------------------------------------------------------
# The expected three-way outcome.
# ---------------------------------------------------------------------------
def test_expected_three_way_assignment():
    d = _derived(_fd3(), _interm(), _fd2())
    verdict, decision = _verdict(KDO.build_claims(d))
    assert verdict == K.CONFIRMED
    assert "FD3=LPT42_Nod4" in decision and "FD2=LPT21" in decision and "Nod3=intermediate" in decision


def test_fd2_is_lpt21_not_nod3():
    d = _derived(_fd3(), _interm(), _fd2())
    cons = d["constructive"]
    assert cons["fd2_candidate"] == "LPT21"
    assert cons["fd2_is_lpt21"] is True
    assert cons["lpt21_fd2_fine"] == 2      # homolateral + frontal
    assert cons["nod3_fd2_fine"] < 2        # mixed / not frontal-colocated
    assert cons["nod3_identity"] == "intermediate"


def test_fd2_tie_break_needs_both_homolateral_and_frontal():
    # A candidate that is homolateral but NOT frontal-colocated must not be picked as FD2.
    homo_only = _cand(layer_b=99.0, nt_conf=0.8, contra=2.0, heterolateral=False, homolateral=True,
                      lateral_gap=False, frontal=False, frontal_coloc=False, offset=5.0,
                      axon_cross=True, contra_pof=False)
    d = _derived(_fd3(), homo_only, _interm())   # neither non-FD3 candidate is a clean FD2
    cons = d["constructive"]
    assert cons["fd2_candidate"] is None


def test_two_sided_nod3_can_take_fd3():
    # Give Nod3 the FD3 signature and LPT42 the FD2 signature: the overturning branch fires.
    d = _derived(_fd2(), _fd3(), _interm())  # LPT42 block=FD2-like, Nod3 block=FD3-like
    # Nod3's own-best FD is now FD3; force nod3_identity to FD3 via constructive (screen gives it FD3)
    verdict, decision = _verdict(KDO.build_claims(d))
    # LPT42 fails the FD3 discriminators (it is FD2-like) and Nod3 hits them
    assert "Nod3=FD3" in decision or "partial" in decision  # two-sided rule engaged


def test_shared_rows_do_not_discriminate():
    d = _derived(_fd3(), _interm(), _fd2())
    shared = {"Preferred direction (lobula-plate layer)", "Neurotransmitter",
              "Small-field selectivity (bounded vs null)"}
    rows = {r["property"]: r for r in d["comparison_rows"]}
    for name in shared:
        assert name in rows and rows[name]["discriminates"] is False


def test_lpt42_is_fd3_and_nod3_intermediate_claims():
    d = _derived(_fd3(), _interm(), _fd2())
    claims = KDO.build_claims(d)
    assert next(c for c in claims if c.id == "KD.fd2_is_lpt21").verdict == K.CONFIRMED
    assert next(c for c in claims if c.id == "KD.nod3_intermediate").verdict == K.CONFIRMED
    assert next(c for c in claims if c.id == "KD.lpt42_is_fd3").verdict == K.CONFIRMED


def test_frontal_field_helper_positive_and_negative():
    frontal = {"per_side": {"left": {"ref_frontal_occ": 0.6, "cand_frontal_occ": 0.5,
                                     "more_lateral": False, "has_frontal_gap": False},
                            "right": {"ref_frontal_occ": 0.6, "cand_frontal_occ": 0.55,
                                      "more_lateral": False, "has_frontal_gap": False}}}
    assert KD._is_frontal_field(frontal) is True
    lateral = {"per_side": {"left": {"ref_frontal_occ": 0.6, "cand_frontal_occ": 0.02,
                                     "more_lateral": True, "has_frontal_gap": True}}}
    assert KD._is_frontal_field(lateral) is False
