"""Unit tests for the Stage-5 paper-verification logic — pure functions on tiny synthetic
inputs, no live CAVE and no big files.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_paper_verify.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")

from flyconn.paper import geometry as G
from flyconn.paper.derive import common as CM
from flyconn.paper.oracle import consts as C


# --- geometry: unit conversion ---
def test_positions_to_um_is_nm_over_1000():
    pts = np.array([[1000.0, 2000.0, 3000.0]])  # nm
    um = G.positions_to_um(pts)
    assert np.allclose(um, [[1.0, 2.0, 3.0]])


def test_patch_radius_and_frac_within():
    # four points 10 um from origin on the axes -> centroid ~origin; median dist ~10.
    pts = np.array([[10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0]], dtype=float)
    r = G.patch_radius_um(pts, "median")
    assert 9.0 < r < 11.0
    assert G.frac_within(pts, 11.0) == 1.0
    assert G.frac_within(pts, 5.0) == 0.0


def test_median_nearest_vs_min_pairwise():
    # inputs: one synapse at the terminal, the rest 100 um away.
    target = np.array([[0, 0, 0]], dtype=float)
    pts = np.array([[0, 0, 0]] + [[100, 0, 0]] * 9, dtype=float)
    # min-pairwise finds the single 0-distance synapse (0 um) -> misleading.
    assert G.min_pairwise_distance_um(pts, target) == pytest.approx(0.0)
    # median-nearest correctly reports ~100 um (only 1 of 10 is near).
    assert G.median_nearest_distance_um(pts, target) == pytest.approx(100.0)


def test_two_field_separation():
    a = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)      # centroid (1,0,0)
    b = np.array([[20, 0, 0], [22, 0, 0]], dtype=float)    # centroid (21,0,0)
    assert G.two_field_separation_um(a, b) == pytest.approx(20.0)


# --- common: T4/T5 classification + layer fractions ---
def test_is_canonical_t4t5():
    s = pd.Series(["T4a", "T5d", "T4", "LT52", "VCH", None])
    assert CM.is_canonical_t4t5(s).tolist() == [True, True, False, False, False, False]


def test_subtype_letter():
    s = pd.Series(["T4a", "T5b", "T4", "VCH"])
    assert CM.subtype_letter(s).tolist() == ["a", "b", pd.NA, pd.NA]


def test_argmax_nt():
    df = pd.DataFrame({
        "gaba": [0.9, 0.1], "ach": [0.05, 0.8], "glut": [0.05, 0.1],
        "oct": [0, 0], "ser": [0, 0], "da": [0, 0],
    })
    assert CM.argmax_nt(df).tolist() == ["gaba", "ach"]


def test_layer_fractions():
    t45 = pd.DataFrame({
        "root_id": [1, 2, 3, 4],
        "syn": [80, 10, 5, 5],
        "is_t4t5": [True, True, True, True],
        "t4t5_subtype": ["a", "a", "b", "c"],
    })
    assert CM.layer_a_fraction(t45) == pytest.approx(90.0)   # (80+10)/100
    assert CM.layer_b_fraction(t45) == pytest.approx(5.0)


# --- consts: subtype letter helper + layer convention ---
def test_consts_subtype_letter_and_layers():
    assert C.t4t5_subtype_letter("T4a") == "a"
    assert C.t4t5_subtype_letter("LT33") is None
    assert C.T4T5_LAYER["a"] == 1 and C.T4T5_LAYER["b"] == 2
    assert C.LAYER_DIRECTION["a"] == "front_to_back"


# --- MaleCNS motor classification (uses the real annotation feather if present) ---
def test_malecns_motor_classification_if_present():
    from flyconn.paper.malecns import ANNOTATIONS, client as MC
    if not ANNOTATIONS.exists():
        pytest.skip("MaleCNS annotations not downloaded")
    motor = MC.motor_neurons()
    steering = motor[motor["is_wing_steering"]]
    # wing-steering muscles must include the classic hg/i steering MNs and exclude power.
    muscles = set(steering["muscle"].dropna())
    assert any(m.startswith("hg1") for m in muscles)
    assert not any(m.startswith(("DLM", "DVM")) for m in muscles)


# --- oracle build_claims smoke (no data needed for internal-consistency family) ---
def test_internal_consistency_oracle_passes():
    from flyconn.paper.oracle import internal_consistency as IC
    from flyconn.motif import compare as K
    claims = IC.build_claims()
    # the paper's own arithmetic should be internally consistent (no REFUTED).
    verdicts = K.verdict_counts(claims)
    assert verdicts.get("REFUTED", 0) == 0
    assert verdicts.get("CONFIRMED", 0) >= 5
