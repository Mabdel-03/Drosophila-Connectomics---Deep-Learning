"""Smoke tests for the anatomical FD3 figures — render from synthetic skeleton/synapse arrays
with a stubbed skeleton fetcher, no network. Assert each figure writes a non-empty PNG.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_figures_anat.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")

from flyconn.paper import figures_anat as FA
from flyconn.paper import skeleton_fetch as SK
from flyconn.paper import geometry_hex as GH


# --- synthetic connectome fixtures ------------------------------------------------------
FD3_L, FD3_R = 1001, 1002
_TYPES = {
    FD3_L: ("LPT42_Nod4", "left"), FD3_R: ("LPT42_Nod4", "right"),
    2001: ("T4b", "right"), 2002: ("T5b", "right"),
    3001: ("LPC1", "right"), 4001: ("LPi14", "left"), 5001: ("DNp26", "right"),
}


class _Meta:
    def __init__(self):
        rows = [{"root_id": r, "cell_type": ct, "side": s, "super_class": "optic",
                 "nt_canonical": ("gaba" if ct.startswith("LPi") else "acetylcholine")}
                for r, (ct, s) in _TYPES.items()]
        self.df = pd.DataFrame(rows)
        self.by_root = self.df.set_index("root_id")

    def root_ids_of_type(self, types, side=None):
        m = self.df["cell_type"].isin(types)
        if side is not None:
            m &= self.df["side"] == side
        return self.df.loc[m, "root_id"].tolist()


_SYN_COLS = (["pre_pt_root_id", "post_pt_root_id"]
             + [f"{s}_pt_position_{a}" for s in ("pre", "post") for a in ("x", "y", "z")])


class _Src:
    """A tiny synthetic source: a few input synapses onto each FD3 cell from each partner class."""
    track = "offline"

    def synapses(self, pre_ids=None, post_ids=None):
        rng = np.random.default_rng(0)
        recs = []
        posts = post_ids or []
        pres = pre_ids or []
        # inputs onto FD3 (post query): one cluster of synapses per partner class
        for post in posts:
            for pre in (2001, 2002, 3001, 4001):
                for _ in range(20):
                    p = rng.normal(500, 20, 3) * 1000.0  # nm
                    recs.append([pre, post, *p, *p])
        # outputs of a cell (pre query): a cluster
        for pre in pres:
            for _ in range(20):
                p = rng.normal(520, 20, 3) * 1000.0
                recs.append([pre, (posts[0] if posts else 9999), *p, *p])
        return pd.DataFrame(recs, columns=_SYN_COLS) if recs else pd.DataFrame(columns=_SYN_COLS)


class _Skel:
    def __init__(self, root):
        rng = np.random.default_rng(int(root))
        n = 200
        self.vertices_um = rng.normal(500, 15, (n, 3))
        self.edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
        self.source = "synthetic"
        self.root_id = root

    @property
    def ok(self):
        return True


@pytest.fixture(autouse=True)
def _stub_skeletons(monkeypatch):
    monkeypatch.setattr(SK, "fetch_skeleton", lambda src, root: _Skel(root))
    # stub the retinotopy cloud so the hexmap does not need retinotopy_columns.parquet
    def _fake_pq(src, meta, root, syn=None):
        rng = np.random.default_rng(int(root))
        n = 40
        return GH.RFCloud(p=rng.normal(0, 5, n), q=rng.normal(0, 5, n),
                          w=rng.integers(1, 10, n).astype(float),
                          n_inputs=n, n_inputs_total=n, total_syn=int(n * 5), dropout_frac=0.0)
    monkeypatch.setattr(GH, "input_cell_pq", _fake_pq)


def _assert_png(path):
    import os
    assert os.path.exists(path) and os.path.getsize(path) > 1000, f"empty/missing PNG: {path}"


def test_arbor_inputs_renders(tmp_path):
    _assert_png(FA.fd3_arbor_inputs(_Src(), _Meta(), tmp_path / "arbor.png"))


def test_input_hexmap_renders(tmp_path):
    _assert_png(FA.fd3_input_hexmap(_Src(), _Meta(), tmp_path / "hexmap.png"))


def test_circuit_3d_renders(tmp_path):
    _assert_png(FA.fd3_circuit_3d(_Src(), _Meta(), tmp_path / "circuit.png"))


def test_afferent_cascade_renders(tmp_path):
    p = {"census": {"layer_b_frac_of_t4t5": 98.6, "t4t5_frac_of_total": 25.1,
                    "on_off_split": {"T4b_ON": 1793, "T5b_OFF": 1459}},
         "upstream_cascade": {"on_limb_t4b": {"expected_frac_of_input": 61.0},
                              "off_limb_t5b": {"expected_frac_of_input": 70.0}}}
    _assert_png(FA.fd3_afferent_cascade(p, tmp_path / "cascade.png"))


def _p_derived():
    return {"census": {"layer_b_frac_of_t4t5": 98.6, "t4t5_frac_of_total": 25.1,
                       "on_off_split": {"T4b_ON": 1793, "T5b_OFF": 1459}},
            "upstream_cascade": {"on_limb_t4b": {"expected_frac_of_input": 61.0},
                                 "off_limb_t5b": {"expected_frac_of_input": 70.0}}}


def test_functional_circuit_schematic_renders(tmp_path):
    q = {"named_sheet": "LPC1", "profiles": {"LPC1": {"layer_b_input_syn": 28848, "to_fd3_syn": 1120}}}
    r = {"winner": {"cell_type": "LPi14", "direction": "opponent", "layer_a_pct": 94.8,
                    "to_fd3_syn": 961, "to_sheet_syn": 6631, "to_detectors_syn": 7004}}
    _assert_png(FA.fd3_functional_circuit_schematic(q, r, _p_derived(), tmp_path / "schematic.png"))


def test_render_all_produces_every_figure(tmp_path):
    q = {"named_sheet": "LPC1", "profiles": {"LPC1": {"layer_b_input_syn": 28848, "to_fd3_syn": 1120}}}
    r = {"winner": {"cell_type": "LPi14", "direction": "opponent", "layer_a_pct": 94.8,
                    "to_fd3_syn": 961, "to_sheet_syn": 6631, "to_detectors_syn": 7004}}
    out = FA.render_all(_Src(), _Meta(), _p_derived(), tmp_path, q_derived=q, r_derived=r)
    for name in ("fd3_arbor_inputs", "fd3_circuit_3d", "fd3_input_hexmap", "fd3_afferent_cascade",
                 "fd3_functional_circuit_schematic"):
        assert name in out, f"{name} not rendered: {out}"
        _assert_png(out[name])
