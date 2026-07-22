"""Unit tests for the per-compartment (dendrite vs axon) synapse split — synthetic skeleton +
synapse frame, no network (skeleton fetch is stubbed).

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_compartments.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")

from flyconn.paper.derive import compartments as CP
from flyconn.paper import skeleton_fetch as SK

# anchor FD3 cell = 10; partners with known class
_META = {
    10: ("LPT42_Nod4", "right", "visual_projection", "acetylcholine"),
    20: ("T4b", "right", "optic", "acetylcholine"),
    30: ("LPC1", "right", "visual_projection", "acetylcholine"),
    40: ("LPi14", "left", "optic", "gaba"),       # LEFT soma -> contralateral to right FD3
    50: ("DNp26", "right", "descending", "acetylcholine"),
}
_SYN_COLS = (["pre_pt_root_id", "post_pt_root_id"]
             + [f"{s}_pt_position_{a}" for s in ("pre", "post") for a in ("x", "y", "z")])

# two spatial clusters in um: dendrite near x=100, axon near x=500 (converted to nm in the frame)
_DEND = np.array([100.0, 100.0, 100.0])
_AXON = np.array([500.0, 100.0, 100.0])


class _Meta:
    def __init__(self):
        self.df = pd.DataFrame([{"root_id": r, "cell_type": ct, "side": s,
                                 "super_class": sc, "nt_canonical": nt}
                                for r, (ct, s, sc, nt) in _META.items()])
        self.by_root = self.df.set_index("root_id")

    def root_ids_of_type(self, types, side=None):
        m = self.df["cell_type"].isin(types)
        return self.df.loc[m, "root_id"].tolist()


def _syn_at(pre, post, center_um, n):
    rng = np.random.default_rng(pre * 100 + post)
    rows = []
    for _ in range(n):
        p = (center_um + rng.normal(0, 2, 3)) * 1000.0   # um -> nm
        rows.append([pre, post, *p, *p])
    return rows


class _Src:
    """FD3(10) inputs: T4b + LPC1 land on the DENDRITE cluster; LPi14 lands on the AXON cluster.
    FD3 outputs (for the model's axon-cloud) land on the AXON cluster."""
    track = "offline"

    def __init__(self):
        rows = []
        rows += _syn_at(20, 10, _DEND, 30)   # T4b -> dendrite
        rows += _syn_at(30, 10, _DEND, 40)   # LPC1 -> dendrite
        rows += _syn_at(40, 10, _AXON, 15)   # LPi14 -> axon
        rows += _syn_at(10, 50, _AXON, 25)   # FD3 output -> axon cluster (defines axon centroid)
        self._df = pd.DataFrame(rows, columns=_SYN_COLS)

    def synapses(self, pre_ids=None, post_ids=None):
        df = self._df
        m = pd.Series(False, index=df.index)
        if pre_ids is not None:
            m |= df["pre_pt_root_id"].isin(set(int(x) for x in pre_ids))
        if post_ids is not None:
            m |= df["post_pt_root_id"].isin(set(int(x) for x in post_ids))
        return df[m].copy()


class _Skel:
    """A cable spanning the two clusters (dendrite cluster ... axon cluster)."""
    def __init__(self, root):
        line = np.linspace(_DEND, _AXON, 60)
        self.vertices_um = line
        self.edges = np.column_stack([np.arange(59), np.arange(1, 60)])
        self.source = "synthetic"
        self.root_id = root

    @property
    def ok(self):
        return True


@pytest.fixture(autouse=True)
def _stub_skeleton(monkeypatch):
    monkeypatch.setattr(SK, "fetch_skeleton", lambda src, root: _Skel(root))


def test_model_labels_vertices_by_nearer_cloud():
    m = CP.build_compartment_model(_Src(), _Meta(), 10)
    assert m.ok and m.source == "skeleton"
    # dendrite vertices should be the low-x half, axon the high-x half
    dend_x = m.vertices_um[m.vertex_label == "dendrite"][:, 0]
    axon_x = m.vertices_um[m.vertex_label == "axon"][:, 0]
    assert dend_x.mean() < axon_x.mean()


def test_input_split_dendrite_vs_axon():
    cs = CP.compartment_input_split(_Src(), _Meta(), 10)
    assert cs["available"]
    # motion + sheet land on the dendrite; contra inhibitor on the axon
    cbc = cs["class_by_compartment"]
    assert cbc["T4b"]["dendrite"] > cbc["T4b"]["axon"]
    assert cbc["LPC1"]["dendrite"] > cbc["LPC1"]["axon"]
    assert cbc["LPi14"]["axon"] > cbc["LPi14"]["dendrite"]
    # by-class buckets: dendrite dominated by motion+sheet, axon by contra_inhibitor
    assert cs["dendrite"]["by_class"].get("motion", 0) > 0
    assert cs["dendrite"]["by_class"].get("sheet", 0) > 0
    assert cs["axon"]["by_class"].get("contra_inhibitor", 0) > 0


def test_soma_distance_geodesic_when_edges_present():
    m = CP.build_compartment_model(_Src(), _Meta(), 10)
    src = _Src()
    din = src.synapses(post_ids=[10]); din = din[din["post_pt_root_id"] == 10]
    d, used_geo = CP.soma_distance_um(m, din, "post", geodesic=True)
    assert len(d) == len(din)
    assert used_geo is True   # edges present -> geodesic path used


def test_proxy_fallback_when_no_skeleton(monkeypatch):
    monkeypatch.setattr(SK, "fetch_skeleton", lambda src, root: None)
    m = CP.build_compartment_model(_Src(), _Meta(), 10)
    assert m.source == "synapse_cloud_proxy" and m.ok
    cs = CP.compartment_input_split(_Src(), _Meta(), 10)
    assert cs["available"] and cs["source"] == "synapse_cloud_proxy"
