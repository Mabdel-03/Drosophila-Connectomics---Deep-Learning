"""Regression guard for the Stage-9 inter-hemispheric analysis.

Asserts the two load-bearing facts:
  1. The synapse-space midline self-test passes (the two optic lobes separate cleanly), since the
     whole position-based crossing definition rests on a valid midline.
  2. The readout-crossing positive control reproduces (left Nod1 -> right DNp26 = 159, right Nod1
     -> left DNp26 = 289), and each DNp26's Nod1 input is fully contralateral.

Both require the cached live pulls; they skip cleanly if no live client / cache is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _source():
    from flyconn.paper.fw_access import NeuronMeta, LiveCaveFlyWire, caveclient_available
    if not caveclient_available():
        return None, None
    try:
        src = LiveCaveFlyWire(mat_version=783)
    except Exception:
        return None, None
    return src, NeuronMeta.load("783")


@pytest.mark.skipif(_source()[0] is None, reason="no live CAVE client / cache")
def test_midline_self_test():
    from flyconn.paper.derive import interhemi_common as IH
    src, meta = _source()
    mid = IH.synapse_space_midline(src, meta)
    assert mid["self_test_ok"], mid
    # the two lobes separate, midline between the LLPC1 dendrite bands (~530 um)
    assert 450 < mid["midline_x_um"] < 620, mid["midline_x_um"]
    assert mid["separation_ratio"] >= 2.0, mid["separation_ratio"]


@pytest.mark.skipif(_source()[0] is None, reason="no live CAVE client / cache")
def test_readout_crossing_positive_control():
    from flyconn.paper.derive import n1_readout_crossing as N1
    src, meta = _source()
    d = N1.run(src, meta)
    assert d["left_nod1_to_right_dnp26"] == 159, d["left_nod1_to_right_dnp26"]
    assert d["right_nod1_to_left_dnp26"] == 289, d["right_nod1_to_left_dnp26"]
    # each DNp26 is driven entirely by the contralateral Nod1
    comp = d["dnp26_input_composition"]
    for side, c in comp.items():
        assert c["contra_frac_of_nod1_input"] == 100.0, (side, c)


@pytest.mark.skipif(_source()[0] is None, reason="no live CAVE client / cache")
def test_centrifugal_not_axonal_bridge():
    """VCH/DCH look ~99% contralateral by soma but do not cross by position (the artifact)."""
    from flyconn.paper.derive import n3_centrifugal_gating as N3
    src, meta = _source()
    d = N3.run(src, meta)
    for t in ("VCH", "DCH"):
        g = d["gater_laterality"][t]
        assert g["soma_contra_pct"] >= 90.0, (t, g)        # high by soma
        assert g["position_cross_pct"] <= 10.0, (t, g)     # but does not cross by position
        assert not g["is_axonal_bridge"], (t, g)
