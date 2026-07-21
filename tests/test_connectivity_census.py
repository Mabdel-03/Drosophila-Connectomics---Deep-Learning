"""Unit tests for the reusable connectivity census — pure functions on a synthetic synapse frame
+ fake meta, no live CAVE.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_connectivity_census.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from flyconn.paper.derive import connectivity_census as CC

# synthetic neurons: anchor A(10), partners with known class/nt/side/layer
_META = {
    10: ("LPT42_Nod4", "right", "visual_projection", "acetylcholine"),
    20: ("T4b", "right", "optic", "acetylcholine"),
    21: ("T5b", "right", "optic", "acetylcholine"),
    30: ("LPC1", "right", "visual_projection", "acetylcholine"),
    40: ("LPi14", "right", "optic", "gaba"),
    50: ("DNp26", "right", "descending", "acetylcholine"),
}
_SYN_COLS = (["pre_pt_root_id", "post_pt_root_id"]
             + [f"{s}_pt_position_{a}" for s in ("pre", "post") for a in ("x", "y", "z")])


class _Meta:
    def __init__(self):
        self.df = pd.DataFrame([{"root_id": r, "cell_type": ct, "side": s,
                                 "super_class": sc, "nt_canonical": nt}
                                for r, (ct, s, sc, nt) in _META.items()])
        self.by_root = self.df.set_index("root_id")

    def root_ids_of_type(self, types, side=None):
        m = self.df["cell_type"].isin(types)
        if side is not None:
            m &= self.df["side"] == side
        return self.df.loc[m, "root_id"].tolist()


def _syn(pre, post, n):
    rng = np.random.default_rng(pre * 1000 + post)
    rows = []
    for _ in range(n):
        p = rng.normal(500, 5, 3) * 1000.0
        rows.append([pre, post, *p, *p])
    return rows


class _Src:
    """Fixed edges: inputs onto A from T4b/T5b/LPC1/LPi14; outputs A->DNp26; LPi14 reciprocal."""
    track = "offline"

    def __init__(self):
        rows = []
        rows += _syn(20, 10, 30)   # T4b -> A (motion, layer b)
        rows += _syn(21, 10, 25)   # T5b -> A
        rows += _syn(30, 10, 40)   # LPC1 -> A (sheet)
        rows += _syn(40, 10, 15)   # LPi14 -> A (inhibitor)
        rows += _syn(10, 50, 20)   # A -> DNp26 (output)
        rows += _syn(10, 40, 5)    # A -> LPi14 (makes LPi14 reciprocal)
        self._df = pd.DataFrame(rows, columns=_SYN_COLS)

    def synapses(self, pre_ids=None, post_ids=None):
        df = self._df
        m = pd.Series(False, index=df.index)
        if pre_ids is not None:
            m |= df["pre_pt_root_id"].isin(set(int(x) for x in pre_ids))
        if post_ids is not None:
            m |= df["post_pt_root_id"].isin(set(int(x) for x in post_ids))
        return df[m].copy()


def test_input_census_partitions():
    c = CC.partner_census(_Src(), _Meta(), [10], "input")
    assert c.direction == "input"
    assert c.total_syn == 30 + 25 + 40 + 15
    assert c.n_partners == 4
    # by super_class
    assert c.by_super_class["optic"] == 30 + 25 + 15   # T4b+T5b+LPi14
    assert c.by_super_class["visual_projection"] == 40  # LPC1
    # by nt
    assert c.by_nt["gaba"] == 15
    # by layer: T4b+T5b are canonical layer-b -> b fraction 100
    assert c.by_layer["b"] == 100.0 and (c.by_layer["a"] or 0) == 0
    # ranked types sorted, cumulative coverage reaches 100
    assert c.ranked_types[0]["cell_type"] == "LPC1"   # 40 is the largest single type
    assert abs(c.ranked_types[-1]["cum_pct"] - 100.0) < 0.5


def test_output_census_has_no_layer():
    c = CC.partner_census(_Src(), _Meta(), [10], "output")
    assert c.direction == "output"
    assert c.by_layer is None
    assert c.by_super_class.get("descending") == 20   # A->DNp26


def test_reciprocity_detects_lpi14():
    r = CC.reciprocity(_Src(), _Meta(), [10])
    # LPi14 (40) both receives from A (5) and sends to A (15) -> reciprocal
    assert r["n_reciprocal"] == 1
    top = r["top_reciprocal"][0]
    assert top["cell_type"] == "LPi14" and top["in_syn"] == 15 and top["out_syn"] == 5


def test_full_census_structure():
    fc = CC.full_census(_Src(), _Meta(), [10])
    assert set(fc) >= {"input", "output", "reciprocity", "roots", "track"}
    assert fc["input"]["total_syn"] > 0 and fc["output"]["total_syn"] > 0


def test_partner_census_bad_direction():
    import pytest
    with pytest.raises(ValueError):
        CC.partner_census(_Src(), _Meta(), [10], "sideways")
