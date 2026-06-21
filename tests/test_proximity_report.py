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


def test_fit_distance_models_returns_finite_rules():
    overall = pd.DataFrame({
        "distance_bin": list(range(20)),
        "distance_bin_nm": [f"{i * 100}-{(i + 1) * 100}" for i in range(20)],
        "distance_mid_nm": [50 + i * 100 for i in range(20)],
        "near_pair_count": [1000] * 20,
        "connected_near_pair_count": np.linspace(450, 20, 20).astype(int),
        "syn_count_sum": [0.0] * 20,
        "connected_syn_count_sum": [1000.0] * 20,
    })
    overall = proximity_report.add_rate_columns(overall)

    model = proximity_report.fit_distance_models(overall)

    assert np.isfinite(model["logistic_slope_per_um"])
    assert model["logistic_odds_ratio_per_100nm"] < 1
    assert model["log_linear_half_distance_um"] > 0


def test_validate_report_text_rejects_tabs_and_em_dash():
    proximity_report.validate_report_text("plain report text")

    for bad in ["contains\ttab", "contains \u2014 em dash"]:
        try:
            proximity_report.validate_report_text(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid report text to fail validation")


def test_snapshot_report_to_repo_copies_lightweight_outputs(tmp_path):
    report = tmp_path / "report"
    (report / "figures").mkdir(parents=True)
    (report / "tables").mkdir()
    (report / "report.pdf").write_text("pdf")
    (report / "report.tex").write_text("tex")
    (report / "report_manifest.json").write_text("{}")
    (report / "figures" / "figure.pdf").write_text("figure")
    (report / "tables" / "table.csv").write_text("a,b\n1,2\n")
    (report / "tables" / "bulk.parquet").write_text("bulk")

    target = proximity_report.snapshot_report_to_repo(report, tmp_path / "snapshot")

    assert (target / "report.pdf").exists()
    assert (target / "figures" / "figure.pdf").exists()
    assert (target / "tables" / "table.csv").exists()
    assert not (target / "tables" / "bulk.parquet").exists()
    assert (target / "README.md").exists()
