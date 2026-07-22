"""Unit tests for the FD2 = LPT21 confidence and dual-output logic (pure functions, synthetic input)."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.paper.derive import fd2_lpt21 as F2  # noqa: E402


def _identity(**over):
    base = {"layer_b_pct": 99.2, "contra_output_pct": 1.5, "mean_rf_offset": 0.03,
            "frontal_colocated": True, "homolateral": True, "smallfield_bounded": True,
            "nt": "acetylcholine", "nt_conf": 0.81, "smallfield_null": {"p_value": 0.002}}
    base.update(over)
    return base


def _uniqueness(unique=True, n=1, competitors=("LPT23",)):
    return {"unique": unique, "n_fd2_hits": n,
            "competitors": [{"type": c} for c in competitors]}


def _dual(both=True):
    return {"both_bimodal": both, "cells": []}


# ---------------------------------------------------------------------------
# Confidence composition.
# ---------------------------------------------------------------------------
def test_confidence_all_properties_unique_high():
    conf = F2._confidence(_identity(), _uniqueness(), _dual())
    assert conf["n_matched"] == conf["n_properties"] == 6
    assert conf["uniqueness_unique"] is True
    assert conf["uniqueness_factor"] == 1.0
    # point estimate is high but not 1.0 (discounts apply); interval brackets it
    assert 0.75 <= conf["point_estimate"] <= 0.95
    lo, hi = conf["interval"]
    assert lo < conf["point_estimate"] < hi


def test_confidence_drops_when_not_unique():
    solo = F2._confidence(_identity(), _uniqueness(unique=True, n=1), _dual())
    tied = F2._confidence(_identity(), _uniqueness(unique=False, n=3), _dual())
    assert tied["uniqueness_factor"] < solo["uniqueness_factor"]
    assert tied["point_estimate"] < solo["point_estimate"]


def test_confidence_drops_when_property_missing():
    full = F2._confidence(_identity(), _uniqueness(), _dual())
    # homolateral fails (a heterolateral candidate) -> fewer matched properties -> lower point
    part = F2._confidence(_identity(homolateral=False, contra_output_pct=90.0),
                          _uniqueness(), _dual())
    assert part["n_matched"] < full["n_matched"]
    assert part["point_estimate"] < full["point_estimate"]


def test_confidence_discounts_are_explicit_and_reduce_estimate():
    conf = F2._confidence(_identity(), _uniqueness(), _dual())
    assert set(conf["discounts"]) == {"no_independent_anchor", "large_field_unmeasured",
                                      "further_fd_cells_possible", "n_equals_2"}
    assert conf["total_discount"] > 0
    # the statement text reconstructs the estimate transparently
    assert "estimate" in conf["statement"].lower() and "range" in conf["statement"].lower()


def test_dual_output_bonus_property_counts():
    with_dual = F2._confidence(_identity(), _uniqueness(), _dual(both=True))
    without = F2._confidence(_identity(), _uniqueness(), _dual(both=False))
    assert with_dual["n_matched"] == without["n_matched"] + 1
