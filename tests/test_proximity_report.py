from __future__ import annotations

import numpy as np
import pandas as pd

from flyconn.experiments import proximity_report


def test_pair_key_values_sorts_and_fills_unknown():
    assert proximity_report.pair_key_values("optic", "central") == "central | optic"
    assert proximity_report.pair_key_values(None, "central") == "Unknown | central"


def test_wilson_interval_bounds_fraction():
    low, high = proximity_report.wilson_interval(np.array([5]), np.array([10]))
    assert 0 <= low[0] <= 0.5
    assert 0.5 <= high[0] <= 1


def test_distance_adjusted_enrichment_uses_global_bin_rates():
    overall = pd.DataFrame({
        "distance_bin": [0, 1],
        "fraction_connected": [0.5, 0.1],
    })
    curve = pd.DataFrame({
        "partition": ["super_class", "super_class"],
        "pair_key": ["optic | optic", "optic | optic"],
        "distance_bin": [0, 1],
        "near_pair_count": [10, 20],
        "connected_near_pair_count": [8, 2],
        "syn_count_sum": [0.0, 0.0],
        "connected_syn_count_sum": [0.0, 0.0],
    })

    out = proximity_report.add_distance_adjusted_enrichment(curve, overall)

    row = out.iloc[0]
    assert int(row["near_pair_count"]) == 30
    assert int(row["connected_near_pair_count"]) == 10
    assert np.isclose(row["expected_connected"], 7.0)
    assert np.isclose(row["fraction_connected"], 10 / 30)
    assert np.isclose(row["oe_ratio"], 10 / 7)


def test_counts_dict_to_frame_adds_rates_and_labels():
    acc = {("a | b", 0): [10, 2, 5.0, 3.0]}

    out = proximity_report.counts_dict_to_frame(acc, "toy")

    assert out.iloc[0]["distance_bin_nm"] == "0-100"
    assert out.iloc[0]["distance_mid_nm"] == 50
    assert np.isclose(out.iloc[0]["fraction_connected"], 0.2)
    assert "fraction_ci_low" in out.columns


def test_summarize_distance_profile_reports_close_far_fold_change():
    overall = pd.DataFrame({
        "distance_bin": list(range(20)),
        "distance_bin_nm": [f"{i * 100}-{(i + 1) * 100}" for i in range(20)],
        "near_pair_count": [100] * 20,
        "connected_near_pair_count": [50] * 5 + [20] * 10 + [10] * 5,
        "fraction_connected": [0.5] * 5 + [0.2] * 10 + [0.1] * 5,
    })

    out = proximity_report.summarize_distance_profile(overall)

    assert out["first_bin_nm"] == "0-100"
    assert out["last_bin_nm"] == "1900-2000"
    assert np.isclose(out["zero_to_500_nm_fraction_connected"], 0.5)
    assert np.isclose(out["one_point_five_to_two_um_fraction_connected"], 0.1)
    assert np.isclose(out["close_to_far_fold_change"], 5.0)


def test_enrichment_extremes_returns_enriched_and_depleted():
    df = pd.DataFrame({
        "partition": ["nt"] * 4,
        "pair_key": ["a", "b", "c", "d"],
        "near_pair_count": [100, 100, 100, 5],
        "connected_near_pair_count": [10, 10, 10, 1],
        "fraction_connected": [0.1, 0.1, 0.1, 0.2],
        "expected_connected": [10.0, 10.0, 10.0, 1.0],
        "log2_oe": [2.0, -3.0, 0.5, 10.0],
    })

    out = proximity_report.enrichment_extremes(df, n=1, min_near=50)

    assert set(out["pair_key"]) == {"a", "b"}
    assert set(out["direction"]) == {"enriched", "depleted"}
