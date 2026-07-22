"""Unit tests for the Family-K (FD3 == LPT42_Nod4) logic — pure functions on synthetic
inputs, no live CAVE and no big files.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_k_fd3_lpt42.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from flyconn.paper import geometry_hex as GH
from flyconn.paper import azimuth_calibration as AZ
from flyconn.paper.derive import common as CM
from flyconn.motif import compare as K
from flyconn.paper.oracle import k_fd3_lpt42 as KO


def _cloud(p, q=None, w=None) -> GH.RFCloud:
    p = np.asarray(p, dtype=float)
    q = np.zeros_like(p) if q is None else np.asarray(q, dtype=float)
    w = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
    return GH.RFCloud(p=p, q=q, w=w, n_inputs=len(p), n_inputs_total=len(p),
                      total_syn=int(w.sum()), dropout_frac=0.0)


# --- hex cloud statistics ---
def test_pq_centroid_weighted():
    c = _cloud([0, 10], w=[3, 1])  # weighted mean = 2.5
    assert GH.pq_centroid(c)[0] == pytest.approx(2.5)


def test_pq_width_fwhm():
    # symmetric +/-5 cloud: sigma=5 -> FWHM ~ 2.355*5 ~ 11.77
    c = _cloud([-5, 5])
    assert GH.pq_width_p(c) == pytest.approx(2.3548 * 5.0, rel=1e-3)


def test_pq_band_occupancy():
    c = _cloud([-10, -10, 10, 10])  # half frontal, half lateral
    assert GH.pq_band_occupancy(c, -np.inf, 0.0) == pytest.approx(0.5)


# --- differential RF: the FD3 signature (lateral + wider + frontal gap) ---
def test_differential_rf_fd3_like():
    # FD1 anchor: frontal (p~-11), narrow. FD3 candidate: lateral (p~+5), wider, frontal gap.
    ref = _cloud([-13, -12, -11, -10, -9])               # frontal, tight
    cand = _cloud([-2, 0, 5, 10, 15, 16])                # lateral, broad, no frontal mass
    diff = GH.differential_rf(cand, ref)
    assert diff.more_lateral is True
    assert diff.wider is True
    assert diff.has_frontal_gap is True          # cand barely occupies FD1's frontal band
    assert diff.centroid_offset_p > 10


def test_differential_rf_fd1_like_no_gap():
    # A candidate sitting frontal OF the FD1 reference (an even-more-frontal cell) must NOT
    # read as more lateral. (Raw geometry uses the sign of the offset; the derive additionally
    # requires the offset to clear the bootstrap CI before calling K.rf_more_lateral CONFIRMED.)
    ref = _cloud([-11, -9, -7])
    cand = _cloud([-15, -13, -11])               # clearly frontal of the reference
    diff = GH.differential_rf(cand, ref)
    assert diff.more_lateral is False            # negative offset -> not lateral
    assert diff.has_frontal_gap is False         # it sits IN the frontal band, no gap


# --- axis self-test: pins lateral = +p ---
def test_assert_frontal_anchor_passes_when_frontal():
    out = GH.assert_frontal_anchor(_cloud([-12, -10, -8]))
    assert out["nod1_centroid_p"] < 0


def test_assert_frontal_anchor_raises_when_lateral():
    with pytest.raises(AssertionError):
        GH.assert_frontal_anchor(_cloud([5, 10, 15]))  # not frontal -> axis convention wrong


# --- azimuth calibration: monotone, with a stated error budget ---
def test_azimuth_monotone_and_budget():
    p_lo, p_hi = AZ._p_extent()
    assert AZ.pq_to_azimuth(p_lo) < AZ.pq_to_azimuth(p_hi)   # frontal < caudal
    budget = AZ.calibration_error_budget()
    assert budget["combined_deg"] > 0
    assert "CONFIRMED_WITH_CAVEAT" in budget["verdict_ceiling"]


# --- common: layer fractions + dominant layer + BH FDR ---
def test_layer_fraction_and_dominant():
    import pandas as pd
    df = pd.DataFrame({"syn": [80, 20], "t4t5_subtype": ["b", "a"], "is_t4t5": [True, True]})
    assert CM.layer_fraction(df, "b") == pytest.approx(80.0)
    assert CM.layer_c_fraction(df) == pytest.approx(0.0)
    assert CM.dominant_layer(df) == "b"


def test_bh_adjust_monotone_and_bounded():
    raw = [0.001, 0.5, 0.02, float("nan")]
    adj = CM.bh_adjust(raw)
    assert np.isnan(adj[3])                       # NaN passes through
    assert all(0.0 <= a <= 1.0 for a in adj if not np.isnan(a))
    assert adj[0] <= adj[2] <= adj[1]             # rank order preserved after adjustment


# --- aggregation rule: a single CORE refutation -> REFUTED ---
def _mk(cid, verdict):
    return K.ClaimResult(id=cid, description=cid, report_value=True, computed_primary=True,
                         verdict=verdict)


def test_identity_verdict_core_fail_refutes():
    a_core = sorted(KO.CORE_IDS)[0]
    claims = [_mk(i, K.REFUTED if i == a_core else K.CONFIRMED)
              for i in KO.CORE_IDS | KO.DISC_IDS]    # exactly one CORE claim broken
    iv = KO._identity_verdict(claims)
    assert iv.verdict == K.REFUTED


def test_identity_verdict_all_confirmed():
    claims = [_mk(i, K.CONFIRMED) for i in KO.CORE_IDS | KO.DISC_IDS]
    iv = KO._identity_verdict(claims)
    assert iv.verdict == K.CONFIRMED


def test_identity_verdict_caveat_when_corroborating_soft():
    claims = [_mk(i, K.CONFIRMED) for i in KO.CORE_IDS | KO.DISC_IDS]
    claims.append(_mk("K.rf_abs_azimuth", K.CONFIRMED_WITH_CAVEAT))  # corroborating soft
    iv = KO._identity_verdict(claims)
    assert iv.verdict == K.CONFIRMED_WITH_CAVEAT


# --- N5 whole-FD-family uniqueness screen ---
from flyconn.paper.derive import k_fd3_lpt42 as KD


def test_fd_family_screen_reciprocal_best_hit():
    # LPT42_Nod4: layer-b, lateral+gap, heterolateral -> matches FD3 (3/3); the controls match
    # FD3 worse. Reciprocal best hit must hold.
    profiles = {
        "LPT42_Nod4": {"n_cells": 2, "dominant_layer": "b", "contra_output_pct": 90.0},
        "Nod1": {"n_cells": 4, "dominant_layer": "a", "contra_output_pct": 90.0},   # FD1-like
        "Nod5": {"n_cells": 2, "dominant_layer": "c", "contra_output_pct": 45.0},
    }
    rf_by_type = {
        "LPT42_Nod4": {"more_lateral": True, "has_frontal_gap": True},
        "Nod1": {"more_lateral": False, "has_frontal_gap": False},
        "Nod5": {"more_lateral": False, "has_frontal_gap": False},
    }
    s = KD.fd_family_screen(profiles, rf_by_type)
    assert s["best_match_for_FD3"] == "LPT42_Nod4"
    assert s["best_FD_for_LPT42"] == "FD3"
    assert s["reciprocal_best_hit"] is True
    assert s["fd3_margin"] >= 1


def test_fd_family_screen_lpt42_matches_fd3_best():
    # LPT42_Nod4's own best-matching FD signature must be FD3, not FD1/FD2/FD4.
    profiles = {"LPT42_Nod4": {"n_cells": 2, "dominant_layer": "b", "contra_output_pct": 90.0}}
    rf = {"LPT42_Nod4": {"more_lateral": True, "has_frontal_gap": True}}
    s = KD.fd_family_screen(profiles, rf)
    assert s["score_matrix"]["FD3"]["LPT42_Nod4"] == 3   # all three features match FD3
    assert s["score_matrix"]["FD1"]["LPT42_Nod4"] < 3


# --- N2 soma_summary ---
def test_cloud_extent_and_laterality():
    from flyconn.paper import geometry as G
    pts = np.array([[10, 0, 0], [20, 100, 0], [15, 50, 30]], dtype=float)  # x,y,z um
    ext = G.cloud_extent_um(pts)
    assert ext["n"] == 3
    assert ext["span_y"] == pytest.approx(100.0)   # dorso-ventral span
    assert ext["span_x"] == pytest.approx(10.0)     # medio-lateral span
    # axon crossing: fraction with x beyond a threshold
    assert G.fraction_beyond_x_um(pts, 12.0, greater=True) == pytest.approx(2 / 3)
    assert G.fraction_beyond_x_um(pts, 12.0, greater=False) == pytest.approx(1 / 3)


def test_morphology_proxy_claims_are_corroborating_only():
    # A morphology-proxy block must yield corroborating (caveat) claims that are NOT in the
    # CORE/DISC gating sets, so morphology can never flip the identity.
    d = {"morphology": {"available": True, "source": "synapse_cloud_proxy",
                        "skeleton_reason": "no L2 cache",
                        "cells": [
                            {"side": "right", "dv_span_um": 154.7, "axon_crosses_contra": True,
                             "axon_ml_shift_um": -239.0, "axon_nearer_noduli": True,
                             "axon_to_noduli_um": 80.7, "dend_to_noduli_um": 163.5},
                            {"side": "left", "dv_span_um": 147.5, "axon_crosses_contra": True,
                             "axon_ml_shift_um": 236.4, "axon_nearer_noduli": True,
                             "axon_to_noduli_um": 80.6, "dend_to_noduli_um": 169.2}]}}
    claims = KO._morphology_claims(d)
    ids = {c.id for c in claims}
    assert {"K.morph_dv_span", "K.morph_axon_heterolateral", "K.morph_axon_noduli"} <= ids
    assert all(c.verdict == K.CONFIRMED_WITH_CAVEAT for c in claims)   # confirmed, caveat-capped
    assert not (ids & KO.CORE_IDS) and not (ids & KO.DISC_IDS)         # never gating


def test_morphology_skeleton_source_drops_proxy_caveat_wording():
    # A real-skeleton morphology block yields the SAME 3 claims, labelled as skeleton-backed,
    # still corroborating (never gating).
    d = {"morphology": {"available": True, "source": "skeleton",
                        "cells": [
                            {"side": "right", "dv_span_um": 154.7, "axon_crosses_contra": True,
                             "axon_ml_shift_um": -239.0, "axon_nearer_noduli": True,
                             "axon_to_noduli_um": 80.7, "dend_to_noduli_um": 163.5,
                             "n_vertices": 36744},
                            {"side": "left", "dv_span_um": 147.5, "axon_crosses_contra": True,
                             "axon_ml_shift_um": 236.4, "axon_nearer_noduli": True,
                             "axon_to_noduli_um": 80.6, "dend_to_noduli_um": 169.2,
                             "n_vertices": 31000}]}}
    claims = KO._morphology_claims(d)
    ids = {c.id for c in claims}
    assert {"K.morph_dv_span", "K.morph_axon_heterolateral", "K.morph_axon_noduli"} <= ids
    assert all(c.verdict == K.CONFIRMED_WITH_CAVEAT for c in claims)
    # descriptions must say "real skeleton", not "synapse-cloud proxy"
    assert all("real skeleton" in c.description for c in claims)
    assert not (ids & KO.CORE_IDS) and not (ids & KO.DISC_IDS)


def test_morphology_proxy_refutes_axon_when_not_crossing():
    # If neither cell's axon shifts contralaterally, the heterolateral claim must REFUTE
    # (it is the only morphology claim with a hard falsifier branch).
    d = {"morphology": {"available": True, "source": "synapse_cloud_proxy",
                        "cells": [{"side": "right", "dv_span_um": 150, "axon_crosses_contra": False,
                                   "axon_ml_shift_um": 5.0, "axon_nearer_noduli": False,
                                   "axon_to_noduli_um": 200, "dend_to_noduli_um": 100}]}}
    claims = KO._morphology_claims(d)
    het = next(c for c in claims if c.id == "K.morph_axon_heterolateral")
    assert het.verdict == K.REFUTED


def test_soma_summary_bilateral_and_posterior():
    import pandas as pd
    from flyconn.paper import fw_access as FW
    neurons = pd.DataFrame({
        "root_id": [1, 2, 3, 4, 5],
        "cell_type": ["LPT42_Nod4", "LPT42_Nod4", "Nod1", "VCH", "Other"],
        "side": ["right", "left", "right", "left", "right"],
        "super_class": ["visual_projection"] * 5,
        "nt_canonical": ["acetylcholine"] * 5,
        "soma_x": [160000.0, 100000.0, 158000.0, 110000.0, 130000.0],  # midline ~130k
        "soma_y": [70000.0, 62000.0, 69000.0, 60000.0, 65000.0],
        "soma_z": [5500.0, 5300.0, 5280.0, 1100.0, 3000.0],
    })
    meta = FW.NeuronMeta(neurons)
    s = KD.CM.soma_summary(meta, [1, 2], anchor_roots=[3], anterior_roots=[4])
    assert s["available"] is True
    assert s["bilateral_split"] is True            # somata straddle the midline
    assert s["dz_to_anchor"] < 200                 # co-clustered with Nod1
    assert s["dz_to_anterior"] > 0                 # posterior to the centrifugal VCH
