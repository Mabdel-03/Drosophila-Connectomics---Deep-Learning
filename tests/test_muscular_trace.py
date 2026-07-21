"""Unit tests for the DN->MN->muscle tracer on a synthetic MCNS world - no network.

World (DNs on RHS):
  DNp26 (RHS) -> hg1(LHS,wm), i1(LHS,wm), DLMn c-f(RHS,power)   # contralateral steering
  DNa04 (RHS) -> hg1(RHS,wm)                                    # ipsilateral steering
The tracer must classify systems, split by wing relative to the DN, exclude power muscles
from the steering denominator, and reproduce the S14-style laterality + S15-style table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from flyconn.muscular import muscular_config as C
from flyconn.muscular import trace as T


@pytest.fixture
def world():
    dn_roots = pd.DataFrame({
        "root_id": [1, 2],
        "cell_type": ["DNp26", "DNa04"],
        "somaSide": ["RHS", "RHS"],
    })
    edges = pd.DataFrame({
        "pre_pt_root_id": [1, 1, 1, 2],
        "post_pt_root_id": [10, 11, 12, 13],
        "syn_count": [122, 105, 999, 80],
    })
    mn_ann = pd.DataFrame({
        "root_id": [10, 11, 12, 13],
        "cell_type": ["hg1", "i1", "DLMn c-f", "hg1"],   # -> muscle
        "somaSide": ["LHS", "LHS", "RHS", "RHS"],
        "cell_sub_class": ["wm", "wm", "power", "wm"],
    })
    return dn_roots, edges, mn_ann


def test_classify_motor_system():
    assert T.classify_motor_system("wm") == "wing_steering"
    assert T.classify_motor_system("DLMn c-f") == "wing_power"
    assert T.classify_motor_system("hg1") == "wing_steering"
    assert T.classify_motor_system("MNhm42") == "haltere"
    assert T.classify_motor_system("TTMn") == "jump_ttm"


def test_split_by_wing(world):
    dn_roots, edges, mn_ann = world
    dn_mn = T.dn_to_motor_neurons(edges, dn_roots, mn_ann)
    wing = T.split_by_wing(dn_mn)
    # DNp26 (RHS) -> LHS MNs = contra; DNa04 (RHS) -> RHS MN = ipsi
    p26 = wing[wing["dn"] == "DNp26"]
    assert set(p26.loc[p26["muscle"].isin(["hg1", "i1"]), "wing_rel"]) == {"contra"}
    a04 = wing[wing["dn"] == "DNa04"]
    assert set(a04["wing_rel"]) == {"ipsi"}


def test_dn_wing_laterality_excludes_power(world):
    dn_roots, edges, mn_ann = world
    wing = T.split_by_wing(T.dn_to_motor_neurons(edges, dn_roots, mn_ann))
    lat = T.dn_wing_laterality(wing).set_index("dn")
    # DNp26 steering = hg1(122)+i1(105) contra = 227; the 999 power synapses are EXCLUDED
    assert lat.loc["DNp26", "steering_syn"] == 227
    assert lat.loc["DNp26", "ipsi_frac"] == 0.0
    assert lat.loc["DNp26", "wing"] == "contralateral"
    # DNa04 fully ipsi
    assert lat.loc["DNa04", "ipsi_frac"] == 1.0
    assert lat.loc["DNa04", "wing"] == "ipsilateral"


def test_muscle_table_for_dn(world):
    dn_roots, edges, mn_ann = world
    wing = T.split_by_wing(T.dn_to_motor_neurons(edges, dn_roots, mn_ann))
    tab = T.muscle_table_for_dn(wing, "DNp26").set_index("muscle")
    assert tab.loc["hg1", "syn"] == 122
    assert tab.loc["i1", "syn"] == 105


def test_bilateral_muscle_map_columns(world):
    dn_roots, edges, mn_ann = world
    wing = T.split_by_wing(T.dn_to_motor_neurons(edges, dn_roots, mn_ann))
    bm = T.bilateral_muscle_map(wing, channel_map={"DNp26": "nodtype", "DNa04": "direct"})
    assert {"dn", "channel", "muscle", "motor_system", "wing_rel", "syn_count"} <= set(bm.columns)
    assert set(bm["channel"]) == {"nodtype", "direct"}
