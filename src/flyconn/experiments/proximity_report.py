"""Generate a LaTeX report for whole-connectome mesh proximity results."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import Config, load_config
from ..io import write_json
from .proximity_full import RUN_NAME_DEFAULT


DISTANCE_BIN_NM = 100
MAX_DISTANCE_NM = 2000
DISTANCE_BINS = np.arange(0, MAX_DISTANCE_NM + DISTANCE_BIN_NM, DISTANCE_BIN_NM)
DISTANCE_LABELS = [f"{int(lo)}-{int(hi)}" for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])]

PARTITIONS = {
    "super_class": {
        "summary": "by_super_class_pair.parquet",
        "cols": ("super_class_a", "super_class_b"),
        "labels": ("Broad class 1", "Broad class 2"),
        "min_near": 10_000,
        "top_n": 20,
    },
    "cell_class": {
        "summary": "by_cell_class_pair.parquet",
        "cols": ("cell_class_a", "cell_class_b"),
        "labels": ("Cell class 1", "Cell class 2"),
        "min_near": 5_000,
        "top_n": 30,
    },
    "cell_type": {
        "summary": "by_cell_type_pair.parquet",
        "cols": ("cell_type_a", "cell_type_b"),
        "labels": ("Cell type 1", "Cell type 2"),
        "min_near": 500,
        "top_n": 75,
    },
    "nt": {
        "summary": "by_nt_pair.parquet",
        "cols": ("nt_canonical_a", "nt_canonical_b"),
        "labels": ("NT 1", "NT 2"),
        "min_near": 1_000,
        "top_n": 28,
    },
    "post_neuropil": {
        "summary": "by_dominant_post_neuropil_pair.parquet",
        "cols": ("dominant_post_neuropil_a", "dominant_post_neuropil_b"),
        "labels": ("Dominant postsynaptic neuropil 1", "Dominant postsynaptic neuropil 2"),
        "min_near": 2_000,
        "top_n": 50,
    },
    "pre_neuropil": {
        "summary": "by_dominant_pre_neuropil_pair.parquet",
        "cols": ("dominant_pre_neuropil_a", "dominant_pre_neuropil_b"),
        "labels": ("Dominant presynaptic neuropil 1", "Dominant presynaptic neuropil 2"),
        "min_near": 2_000,
        "top_n": 50,
    },
    "flow": {
        "summary": "by_flow_pair.parquet",
        "cols": ("flow_a", "flow_b"),
        "labels": ("Flow 1", "Flow 2"),
        "min_near": 100,
        "top_n": 10,
    },
    "side": {
        "summary": "by_side_pair.parquet",
        "cols": ("side_a", "side_b"),
        "labels": ("Side 1", "Side 2"),
        "min_near": 100,
        "top_n": 13,
    },
}


@dataclass(frozen=True)
class ReportConfig:
    run_name: str = RUN_NAME_DEFAULT
    batch_size: int = 250_000
    top_type_pairs: int = 75
    force: bool = False
    compile_pdf: bool = False
    install_tectonic: bool = False
    max_batches: int | None = None
    snapshot_repo_report: bool = False
    snapshot_dir: Path = Path("reports/proximity_whole_connectome")


def run_dir(cfg: Config, report_cfg: ReportConfig) -> Path:
    return cfg.paths().root / "experiments" / "proximity" / report_cfg.run_name


def report_dir(run: Path) -> Path:
    return run / "report"


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_int(value: object) -> str:
    return f"{int(value):,}"


def format_float(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def format_pct(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.{digits}f}\\%"


def format_pp(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.{digits}f} percentage points"


def pct_plain(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.{digits}f}%"


def paragraph(text: str) -> str:
    return " ".join(str(text).strip().split())


def sanitize_report_text(text: str) -> str:
    return text.replace("\t", " ").replace("\u2014", "-")


def validate_report_text(text: str) -> None:
    if "\t" in text:
        raise ValueError("Generated report contains tab characters.")
    if "\u2014" in text:
        raise ValueError("Generated report contains em dashes.")


def latex_paragraph(text: str) -> str:
    return sanitize_report_text(paragraph(text))


def pair_key_values(a: object, b: object) -> str:
    left = "Unknown" if pd.isna(a) else str(a)
    right = "Unknown" if pd.isna(b) else str(b)
    return " | ".join(sorted((left, right)))


def pair_key_series(a: pd.Series, b: pd.Series) -> pd.Series:
    left = a.fillna("Unknown").astype(str)
    right = b.fillna("Unknown").astype(str)
    lo = left.where(left <= right, right)
    hi = right.where(left <= right, left)
    return lo + " | " + hi


def split_pair_key(values: pd.Series) -> pd.DataFrame:
    parts = values.str.split(" | ", regex=False, n=1, expand=True)
    if parts.shape[1] == 1:
        parts[1] = ""
    return parts


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    denom = 1 + z**2 / np.maximum(n, 1)
    center = (p + z**2 / (2 * np.maximum(n, 1))) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * np.maximum(n, 1))) / np.maximum(n, 1)) / denom
    return np.maximum(0, center - half), np.minimum(1, center + half)


def distance_bin_index(distance_nm: pd.Series) -> np.ndarray:
    idx = np.floor(distance_nm.to_numpy(np.float64) / DISTANCE_BIN_NM).astype(np.int64)
    return np.clip(idx, 0, len(DISTANCE_LABELS) - 1)


def distance_bin_center_from_label(label: str) -> float:
    left, right = label.split("-")
    return (float(left) + float(right)) / 2


def add_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fraction_connected"] = np.divide(
        out["connected_near_pair_count"],
        out["near_pair_count"],
        out=np.zeros(len(out), dtype=np.float64),
        where=out["near_pair_count"].to_numpy() > 0,
    )
    low, high = wilson_interval(out["connected_near_pair_count"].to_numpy(), out["near_pair_count"].to_numpy())
    out["fraction_ci_low"] = low
    out["fraction_ci_high"] = high
    return out


def update_curve_counts(acc: dict[tuple[str, int], list[int | float]], grouped: pd.DataFrame) -> None:
    for row in grouped.itertuples(index=False):
        key = (str(row.pair_key), int(row.distance_bin))
        current = acc.setdefault(key, [0, 0, 0.0, 0.0])
        current[0] += int(row.near_pair_count)
        current[1] += int(row.connected_near_pair_count)
        current[2] += float(row.syn_count_sum)
        current[3] += float(row.connected_syn_count_sum)


def counts_dict_to_frame(acc: dict[tuple[str, int], list[int | float]], partition: str) -> pd.DataFrame:
    rows = []
    for (pair_key, bin_idx), values in acc.items():
        rows.append({
            "partition": partition,
            "pair_key": pair_key,
            "distance_bin": int(bin_idx),
            "distance_bin_nm": DISTANCE_LABELS[int(bin_idx)],
            "distance_mid_nm": distance_bin_center_from_label(DISTANCE_LABELS[int(bin_idx)]),
            "near_pair_count": int(values[0]),
            "connected_near_pair_count": int(values[1]),
            "syn_count_sum": float(values[2]),
            "connected_syn_count_sum": float(values[3]),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "partition",
            "pair_key",
            "distance_bin",
            "distance_bin_nm",
            "distance_mid_nm",
            "near_pair_count",
            "connected_near_pair_count",
            "syn_count_sum",
            "connected_syn_count_sum",
        ])
    return add_rate_columns(pd.DataFrame(rows))


def load_summary(run: Path, name: str) -> pd.DataFrame:
    path = run / "summaries" / name
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def summary_pair_key(df: pd.DataFrame, partition: str) -> pd.Series:
    info = PARTITIONS[partition]
    left, right = (
        info["cols"][0].removesuffix("_a") + "_1",
        info["cols"][1].removesuffix("_b") + "_2",
    )
    return pair_key_series(df[left], df[right])


def select_pairs(run: Path, overall_fraction: float, report_cfg: ReportConfig) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    for partition, info in PARTITIONS.items():
        df = load_summary(run, info["summary"])
        if df.empty:
            selected[partition] = set()
            continue
        df = df.copy()
        df["pair_key"] = summary_pair_key(df, partition)
        min_near = int(info["min_near"])
        top_n = int(report_cfg.top_type_pairs if partition == "cell_type" else info["top_n"])
        supported = df[df["near_pair_count"] >= min_near].copy()
        if partition == "cell_type":
            connected_supported = supported[supported["connected_near_pair_count"] >= 20]
            candidates = pd.concat([
                supported.sort_values("near_pair_count", ascending=False).head(top_n),
                connected_supported.sort_values("fraction_connected", ascending=False).head(top_n),
                connected_supported.sort_values("fraction_connected", ascending=True).head(top_n),
            ], ignore_index=True)
        else:
            candidates = pd.concat([
                supported.sort_values("near_pair_count", ascending=False).head(top_n),
                supported.sort_values("fraction_connected", ascending=False).head(top_n),
                supported.assign(delta=supported["fraction_connected"] - overall_fraction).sort_values("delta", ascending=True).head(top_n),
            ], ignore_index=True)
        selected[partition] = set(candidates["pair_key"].dropna().astype(str))
    return selected


def stream_distance_curves(run: Path, report_cfg: ReportConfig, selected_pairs: dict[str, set[str]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    parquet_path = run / "near_pairs_enriched.parquet"
    pf = pq.ParquetFile(parquet_path)
    category_columns = sorted({col for info in PARTITIONS.values() for col in info["cols"]})
    columns = [
        "min_distance_nm",
        "connected_any",
        "syn_count_a_to_b",
        "syn_count_b_to_a",
        *category_columns,
    ]
    overall: dict[tuple[str, int], list[int | float]] = {}
    curves = {name: {} for name in PARTITIONS}
    for batch_i, batch in enumerate(pf.iter_batches(batch_size=report_cfg.batch_size, columns=columns)):
        if report_cfg.max_batches is not None and batch_i >= report_cfg.max_batches:
            break
        df = batch.to_pandas()
        df["distance_bin"] = distance_bin_index(df["min_distance_nm"])
        df["connected_i"] = df["connected_any"].astype(bool).astype(np.int64)
        syn_sum = df["syn_count_a_to_b"].fillna(0).astype(np.float64) + df["syn_count_b_to_a"].fillna(0).astype(np.float64)
        df["syn_sum"] = syn_sum
        df["connected_syn_sum"] = syn_sum.where(df["connected_i"].astype(bool), 0.0)

        all_key = pd.Series(["all"] * len(df), index=df.index)
        overall_group = pd.DataFrame({
            "pair_key": all_key,
            "distance_bin": df["distance_bin"],
            "connected": df["connected_i"],
            "syn_sum": df["syn_sum"],
            "connected_syn_sum": df["connected_syn_sum"],
        }).groupby(["pair_key", "distance_bin"], observed=True).agg(
            near_pair_count=("connected", "size"),
            connected_near_pair_count=("connected", "sum"),
            syn_count_sum=("syn_sum", "sum"),
            connected_syn_count_sum=("connected_syn_sum", "sum"),
        ).reset_index()
        update_curve_counts(overall, overall_group)

        for partition, info in PARTITIONS.items():
            keep = selected_pairs.get(partition, set())
            if not keep:
                continue
            a, b = info["cols"]
            pair_key = pair_key_series(df[a], df[b])
            mask = pair_key.isin(keep)
            if not mask.any():
                continue
            part = pd.DataFrame({
                "pair_key": pair_key[mask],
                "distance_bin": df.loc[mask, "distance_bin"],
                "connected": df.loc[mask, "connected_i"],
                "syn_sum": df.loc[mask, "syn_sum"],
                "connected_syn_sum": df.loc[mask, "connected_syn_sum"],
            })
            grouped = part.groupby(["pair_key", "distance_bin"], observed=True).agg(
                near_pair_count=("connected", "size"),
                connected_near_pair_count=("connected", "sum"),
                syn_count_sum=("syn_sum", "sum"),
                connected_syn_count_sum=("connected_syn_sum", "sum"),
            ).reset_index()
            update_curve_counts(curves[partition], grouped)

    overall_df = counts_dict_to_frame(overall, "overall")
    partition_dfs = {name: counts_dict_to_frame(acc, name) for name, acc in curves.items()}
    return overall_df, partition_dfs


def add_distance_adjusted_enrichment(curve: pd.DataFrame, overall_curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return curve
    global_rate = overall_curve.set_index("distance_bin")["fraction_connected"].to_dict()
    tmp = curve.copy()
    tmp["expected_connected"] = tmp.apply(
        lambda r: float(r.near_pair_count) * float(global_rate.get(int(r.distance_bin), 0.0)),
        axis=1,
    )
    grouped = tmp.groupby(["partition", "pair_key"], as_index=False).agg(
        near_pair_count=("near_pair_count", "sum"),
        connected_near_pair_count=("connected_near_pair_count", "sum"),
        expected_connected=("expected_connected", "sum"),
        syn_count_sum=("syn_count_sum", "sum"),
        connected_syn_count_sum=("connected_syn_count_sum", "sum"),
    )
    grouped["fraction_connected"] = grouped["connected_near_pair_count"] / grouped["near_pair_count"]
    grouped["oe_ratio"] = grouped["connected_near_pair_count"] / grouped["expected_connected"].replace(0, np.nan)
    grouped["log2_oe"] = np.log2(
        (grouped["connected_near_pair_count"].astype(float) + 0.5)
        / (grouped["expected_connected"].astype(float) + 0.5)
    )
    grouped["fraction_delta_vs_global_distance"] = (
        grouped["connected_near_pair_count"] - grouped["expected_connected"]
    ) / grouped["near_pair_count"]
    return grouped.sort_values(["near_pair_count"], ascending=False)


def summarize_distance_profile(overall: pd.DataFrame) -> dict[str, float | str]:
    ordered = overall.sort_values("distance_bin").reset_index(drop=True)
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    peak = ordered.loc[ordered["fraction_connected"].idxmax()]
    close = ordered[ordered["distance_bin"] < 5]
    far = ordered[ordered["distance_bin"] >= 15]
    close_fraction = close["connected_near_pair_count"].sum() / close["near_pair_count"].sum()
    far_fraction = far["connected_near_pair_count"].sum() / far["near_pair_count"].sum()
    return {
        "first_bin_nm": str(first["distance_bin_nm"]),
        "first_bin_fraction_connected": float(first["fraction_connected"]),
        "last_bin_nm": str(last["distance_bin_nm"]),
        "last_bin_fraction_connected": float(last["fraction_connected"]),
        "peak_bin_nm": str(peak["distance_bin_nm"]),
        "peak_fraction_connected": float(peak["fraction_connected"]),
        "zero_to_500_nm_fraction_connected": float(close_fraction),
        "one_point_five_to_two_um_fraction_connected": float(far_fraction),
        "close_to_far_fold_change": float(close_fraction / far_fraction) if far_fraction else math.inf,
    }


def summarize_distance_windows(overall: pd.DataFrame) -> pd.DataFrame:
    windows = [
        ("0.0-0.5 um", 0, 5),
        ("0.5-1.0 um", 5, 10),
        ("1.0-1.5 um", 10, 15),
        ("1.5-2.0 um", 15, 20),
    ]
    rows = []
    for label, start, stop in windows:
        part = overall[(overall["distance_bin"] >= start) & (overall["distance_bin"] < stop)]
        near = int(part["near_pair_count"].sum())
        connected = int(part["connected_near_pair_count"].sum())
        fraction = connected / near if near else np.nan
        ci_low, ci_high = wilson_interval(np.array([connected]), np.array([near]))
        rows.append({
            "distance_window": label,
            "near_pair_count": near,
            "connected_near_pair_count": connected,
            "fraction_connected": float(fraction),
            "fraction_ci_low": float(ci_low[0]),
            "fraction_ci_high": float(ci_high[0]),
            "mean_synapses_per_connected_pair": (
                float(part["connected_syn_count_sum"].sum() / connected) if connected else np.nan
            ),
        })
    return pd.DataFrame(rows)


def weighted_r2(observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.average(observed, weights=weights)
    ss_res = np.sum(weights * (observed - predicted) ** 2)
    ss_tot = np.sum(weights * (observed - mean) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def fit_distance_models(overall: pd.DataFrame) -> dict[str, float | str]:
    ordered = overall.sort_values("distance_bin").copy()
    x_um = ordered["distance_mid_nm"].to_numpy(np.float64) / 1000.0
    n = ordered["near_pair_count"].to_numpy(np.float64)
    k = ordered["connected_near_pair_count"].to_numpy(np.float64)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    clipped = np.clip(p, 1e-9, 1 - 1e-9)

    log_rate = np.log(clipped)
    exp_coef = np.polyfit(x_um, log_rate, 1, w=np.sqrt(n))
    exp_slope = float(exp_coef[0])
    exp_intercept = float(exp_coef[1])
    exp_pred = np.exp(exp_intercept + exp_slope * x_um)

    logit_rate = np.log(clipped / (1 - clipped))
    logit_weights = np.sqrt(np.maximum(n * clipped * (1 - clipped), 1.0))
    logit_coef = np.polyfit(x_um, logit_rate, 1, w=logit_weights)
    logit_slope = float(logit_coef[0])
    logit_intercept = float(logit_coef[1])
    logit_pred = 1 / (1 + np.exp(-(logit_intercept + logit_slope * x_um)))

    model_p = np.clip(logit_pred, 1e-12, 1 - 1e-12)
    null_p = np.clip(k.sum() / n.sum(), 1e-12, 1 - 1e-12)
    ll_model = float(np.sum(k * np.log(model_p) + (n - k) * np.log(1 - model_p)))
    ll_null = float(np.sum(k * np.log(null_p) + (n - k) * np.log(1 - null_p)))
    pseudo_r2 = float(1 - ll_model / ll_null) if ll_null else np.nan

    start_rate = float(logit_pred[0])
    half_rate = start_rate / 2
    tenth_rate = start_rate / 10

    def solve_logit_distance(target_rate: float) -> float:
        if not (0 < target_rate < 1) or logit_slope == 0:
            return np.nan
        return float((math.log(target_rate / (1 - target_rate)) - logit_intercept) / logit_slope)

    below_10 = ordered.loc[ordered["fraction_connected"] <= 0.10]
    below_05 = ordered.loc[ordered["fraction_connected"] <= 0.05]
    first_below_10 = str(below_10.iloc[0]["distance_bin_nm"]) if not below_10.empty else ""
    first_below_05 = str(below_05.iloc[0]["distance_bin_nm"]) if not below_05.empty else ""

    return {
        "log_linear_slope_per_um": exp_slope,
        "log_linear_intercept": exp_intercept,
        "log_linear_rate_multiplier_per_um": float(math.exp(exp_slope)),
        "log_linear_half_distance_um": float(math.log(0.5) / exp_slope) if exp_slope < 0 else np.nan,
        "log_linear_weighted_r2": weighted_r2(log_rate, np.log(np.clip(exp_pred, 1e-12, None)), n),
        "logistic_slope_per_um": logit_slope,
        "logistic_intercept": logit_intercept,
        "logistic_odds_ratio_per_um": float(math.exp(logit_slope)),
        "logistic_odds_ratio_per_100nm": float(math.exp(logit_slope * 0.1)),
        "logistic_pseudo_r2": pseudo_r2,
        "logistic_weighted_r2": weighted_r2(p, logit_pred, n),
        "fitted_start_rate": start_rate,
        "fitted_half_rate": half_rate,
        "fitted_half_rate_distance_um": solve_logit_distance(half_rate),
        "fitted_tenth_rate": tenth_rate,
        "fitted_tenth_rate_distance_um": solve_logit_distance(tenth_rate),
        "first_bin_at_or_below_10pct": first_below_10,
        "first_bin_at_or_below_5pct": first_below_05,
    }


def add_model_predictions(overall: pd.DataFrame, model: dict[str, float | str]) -> pd.DataFrame:
    out = overall.sort_values("distance_bin").copy()
    x_um = out["distance_mid_nm"].to_numpy(np.float64) / 1000.0
    logit = float(model["logistic_intercept"]) + float(model["logistic_slope_per_um"]) * x_um
    out["logistic_fit_fraction_connected"] = 1 / (1 + np.exp(-logit))
    out["residual_fraction_connected"] = out["fraction_connected"] - out["logistic_fit_fraction_connected"]
    out["distance_mid_um"] = x_um
    return out


def summarize_partition_effects(enrichments: dict[str, pd.DataFrame]) -> dict[str, dict[str, float | str | int]]:
    out: dict[str, dict[str, float | str | int]] = {}
    for partition, df in enrichments.items():
        if df.empty:
            continue
        supported = df[df["near_pair_count"] >= int(PARTITIONS[partition]["min_near"])].copy()
        if supported.empty:
            continue
        enriched = supported.loc[supported["log2_oe"].idxmax()]
        depleted = supported.loc[supported["log2_oe"].idxmin()]
        out[partition] = {
            "pair_count": int(len(supported)),
            "near_pair_count": int(supported["near_pair_count"].sum()),
            "connected_near_pair_count": int(supported["connected_near_pair_count"].sum()),
            "median_log2_oe": float(supported["log2_oe"].median()),
            "iqr_log2_oe": float(supported["log2_oe"].quantile(0.75) - supported["log2_oe"].quantile(0.25)),
            "top_enriched_pair": str(enriched["pair_key"]),
            "top_enriched_log2_oe": float(enriched["log2_oe"]),
            "top_enriched_fraction_connected": float(enriched["fraction_connected"]),
            "top_enriched_near_pair_count": int(enriched["near_pair_count"]),
            "top_depleted_pair": str(depleted["pair_key"]),
            "top_depleted_log2_oe": float(depleted["log2_oe"]),
            "top_depleted_fraction_connected": float(depleted["fraction_connected"]),
            "top_depleted_near_pair_count": int(depleted["near_pair_count"]),
        }
    return out


def build_partition_overview(enrichments: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for partition, df in enrichments.items():
        if df.empty or "log2_oe" not in df.columns:
            continue
        supported = df[df["near_pair_count"] > 0].copy()
        if supported.empty:
            continue
        rows.append({
            "partition": partition,
            "selected_pair_count": int(len(supported)),
            "near_pair_count": int(supported["near_pair_count"].sum()),
            "connected_near_pair_count": int(supported["connected_near_pair_count"].sum()),
            "median_fraction_connected": float(supported["fraction_connected"].median()),
            "median_log2_oe": float(supported["log2_oe"].median()),
            "max_log2_oe": float(supported["log2_oe"].max()),
            "min_log2_oe": float(supported["log2_oe"].min()),
        })
    return pd.DataFrame(rows).sort_values("near_pair_count", ascending=False) if rows else pd.DataFrame()


def enrichment_extremes(df: pd.DataFrame, n: int = 15, min_near: int = 0) -> pd.DataFrame:
    if df.empty or "log2_oe" not in df.columns:
        return pd.DataFrame()
    supported = df[df["near_pair_count"] >= min_near].copy()
    if supported.empty:
        return pd.DataFrame()
    enriched = supported.nlargest(n, "log2_oe").assign(direction="enriched")
    depleted = supported.nsmallest(n, "log2_oe").assign(direction="depleted")
    return pd.concat([enriched, depleted], ignore_index=True)


def make_figures(
    report_root: Path,
    overall: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
    enrichments: dict[str, pd.DataFrame],
    distance_model: dict[str, float | str] | None = None,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    fig_dir = report_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    overall = overall.sort_values("distance_bin")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(overall["distance_mid_nm"], overall["fraction_connected"], marker="o", color="#1f77b4")
    ax.fill_between(
        overall["distance_mid_nm"].to_numpy(),
        overall["fraction_ci_low"].to_numpy(),
        overall["fraction_ci_high"].to_numpy(),
        color="#1f77b4",
        alpha=0.18,
        linewidth=0,
    )
    ax.set_xlabel("Minimum dendrite mesh distance (nm)")
    ax.set_ylabel("Fraction connected")
    ax.set_title("Connectivity falls sharply with dendrite-mesh distance")
    ax2 = ax.twinx()
    ax2.bar(overall["distance_mid_nm"], overall["near_pair_count"], width=75, color="#bbbbbb", alpha=0.25)
    ax2.set_ylabel("Near-pair count")
    fig.tight_layout()
    path = fig_dir / "overall_distance_curve.pdf"
    fig.savefig(path)
    figures["overall_distance"] = path
    plt.close(fig)

    if distance_model is not None:
        modeled = add_model_predictions(overall, distance_model)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(
            modeled["distance_mid_nm"],
            modeled["fraction_connected"],
            marker="o",
            linewidth=1.4,
            label="Observed 100 nm bins",
            color="#1f77b4",
        )
        ax.plot(
            modeled["distance_mid_nm"],
            modeled["logistic_fit_fraction_connected"],
            linewidth=2.0,
            label="Logistic-binomial fit",
            color="#d62728",
        )
        ax.set_xlabel("Minimum dendrite mesh distance (nm)")
        ax.set_ylabel("Fraction connected")
        ax.set_title("Fitted quantitative rule for distance-dependent connectivity")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / "overall_distance_model_fit.pdf"
        fig.savefig(path)
        figures["overall_distance_model"] = path
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.bar(modeled["distance_mid_nm"], modeled["residual_fraction_connected"], width=75, color="#9467bd", alpha=0.8)
        ax.set_xlabel("Minimum dendrite mesh distance (nm)")
        ax.set_ylabel("Observed minus fitted fraction")
        ax.set_title("Residuals from the global distance rule")
        fig.tight_layout()
        path = fig_dir / "overall_distance_model_residuals.pdf"
        fig.savefig(path)
        figures["overall_distance_residuals"] = path
        plt.close(fig)

    if "super_class" in enrichments and not enrichments["super_class"].empty:
        top = enrichments["super_class"].query("near_pair_count >= 10000").copy()
        parts = split_pair_key(top["pair_key"])
        top["class_1"], top["class_2"] = parts[0], parts[1]
        top = top[top["class_1"].ne("Unknown") & top["class_2"].ne("Unknown")]
        labels = sorted(set(top["class_1"]) | set(top["class_2"]))
        pivot = top.pivot_table(index="class_1", columns="class_2", values="log2_oe", aggfunc="mean").reindex(index=labels, columns=labels)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        sns.heatmap(pivot, center=0, cmap="coolwarm", ax=ax, cbar_kws={"label": "log2 observed/expected"})
        ax.set_title("Distance-adjusted connectivity enrichment by broad class")
        fig.tight_layout()
        path = fig_dir / "super_class_enrichment_heatmap.pdf"
        fig.savefig(path)
        figures["super_class_heatmap"] = path
        plt.close(fig)

    if "nt" in enrichments and not enrichments["nt"].empty:
        nt = enrichments["nt"].query("near_pair_count >= 1000").copy()
        parts = split_pair_key(nt["pair_key"])
        nt["nt_1"], nt["nt_2"] = parts[0], parts[1]
        labels = sorted(set(nt["nt_1"]) | set(nt["nt_2"]))
        pivot = nt.pivot_table(index="nt_1", columns="nt_2", values="log2_oe", aggfunc="mean").reindex(index=labels, columns=labels)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sns.heatmap(pivot, center=0, cmap="coolwarm", ax=ax, cbar_kws={"label": "log2 observed/expected"})
        ax.set_title("Distance-adjusted connectivity enrichment by neurotransmitter")
        fig.tight_layout()
        path = fig_dir / "nt_enrichment_heatmap.pdf"
        fig.savefig(path)
        figures["nt_heatmap"] = path
        plt.close(fig)

    if "post_neuropil" in curves and not curves["post_neuropil"].empty:
        reg_enrich = enrichments["post_neuropil"].copy()
        same = reg_enrich[reg_enrich["pair_key"].map(lambda x: x.split(" | ")[0] == x.split(" | ")[-1])]
        selected = set(same.sort_values("near_pair_count", ascending=False).head(8)["pair_key"])
        reg = curves["post_neuropil"][curves["post_neuropil"]["pair_key"].isin(selected)].copy()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        sns.lineplot(data=reg, x="distance_mid_nm", y="fraction_connected", hue="pair_key", marker="o", ax=ax)
        ax.set_xlabel("Minimum dendrite mesh distance (nm)")
        ax.set_ylabel("Fraction connected")
        ax.set_title("Distance-connectivity curves for dominant postsynaptic neuropils")
        ax.legend(title="Neuropil pair", fontsize=7, title_fontsize=8)
        fig.tight_layout()
        path = fig_dir / "post_neuropil_distance_curves.pdf"
        fig.savefig(path)
        figures["post_neuropil_curves"] = path
        plt.close(fig)

    if "post_neuropil" in enrichments and not enrichments["post_neuropil"].empty:
        region_extremes = enrichment_extremes(enrichments["post_neuropil"], n=8, min_near=2000)
        if not region_extremes.empty:
            region_extremes = region_extremes.sort_values("log2_oe")
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            colors = np.where(region_extremes["log2_oe"] >= 0, "#b2182b", "#2166ac")
            ax.barh(region_extremes["pair_key"], region_extremes["log2_oe"], color=colors, alpha=0.85)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("log2 observed/expected connected")
            ax.set_ylabel("Dominant postsynaptic neuropil pair")
            ax.set_title("Brain-region pairs with strongest distance-adjusted effects")
            ax.tick_params(axis="y", labelsize=6)
            fig.tight_layout()
            path = fig_dir / "post_neuropil_enrichment_extremes.pdf"
            fig.savefig(path)
            figures["post_neuropil_extremes"] = path
            plt.close(fig)

    if "super_class" in curves and not curves["super_class"].empty:
        selected = set(enrichments["super_class"].sort_values("near_pair_count", ascending=False).head(8)["pair_key"])
        sc = curves["super_class"][curves["super_class"]["pair_key"].isin(selected)].copy()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        sns.lineplot(data=sc, x="distance_mid_nm", y="fraction_connected", hue="pair_key", marker="o", ax=ax)
        ax.set_xlabel("Minimum dendrite mesh distance (nm)")
        ax.set_ylabel("Fraction connected")
        ax.set_title("Distance-connectivity curves by broad neuron class")
        ax.legend(title="Class pair", fontsize=7, title_fontsize=8)
        fig.tight_layout()
        path = fig_dir / "super_class_distance_curves.pdf"
        fig.savefig(path)
        figures["super_class_curves"] = path
        plt.close(fig)

    if "cell_type" in enrichments and not enrichments["cell_type"].empty:
        ct = enrichments["cell_type"].copy()
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.scatter(
            np.log10(ct["near_pair_count"]),
            ct["log2_oe"],
            s=np.clip(ct["connected_near_pair_count"] / 10, 8, 80),
            c=ct["fraction_connected"],
            cmap="viridis",
            alpha=0.75,
            edgecolor="none",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("log10 near-pair count")
        ax.set_ylabel("log2 observed/expected connected")
        ax.set_title("Exact neuron-type pair distance-adjusted enrichment")
        top = pd.concat([
            ct.sort_values("log2_oe", ascending=False).head(5),
            ct.sort_values("log2_oe", ascending=True).head(5),
        ])
        for row in top.itertuples(index=False):
            ax.annotate(str(row.pair_key)[:32], (math.log10(row.near_pair_count), row.log2_oe), fontsize=6)
        fig.tight_layout()
        path = fig_dir / "cell_type_enrichment_scatter.pdf"
        fig.savefig(path)
        figures["cell_type_scatter"] = path
        plt.close(fig)

    overview_rows = []
    for partition, df in enrichments.items():
        if df.empty or "log2_oe" not in df.columns:
            continue
        tmp = df[["partition", "pair_key", "near_pair_count", "log2_oe"]].copy()
        tmp = tmp[tmp["near_pair_count"] > 0]
        overview_rows.append(tmp)
    if overview_rows:
        overview = pd.concat(overview_rows, ignore_index=True)
        order = (
            overview.groupby("partition")["near_pair_count"]
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        sns.boxplot(data=overview, x="partition", y="log2_oe", order=order, color="#dddddd", fliersize=2, ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Partition")
        ax.set_ylabel("log2 observed/expected connected")
        ax.set_title("Distance-adjusted enrichment distribution by analysis partition")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = fig_dir / "partition_enrichment_summary.pdf"
        fig.savefig(path)
        figures["partition_summary"] = path
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    connected_mean_syn = np.divide(
        overall["connected_syn_count_sum"],
        overall["connected_near_pair_count"],
        out=np.zeros(len(overall), dtype=np.float64),
        where=overall["connected_near_pair_count"].to_numpy() > 0,
    )
    ax.plot(overall["distance_mid_nm"], connected_mean_syn, marker="o", color="#2ca02c")
    ax.set_xlabel("Minimum dendrite mesh distance (nm)")
    ax.set_ylabel("Mean synapses per connected near pair")
    ax.set_title("Connection strength among connected near pairs")
    fig.tight_layout()
    path = fig_dir / "synapse_strength_by_distance.pdf"
    fig.savefig(path)
    figures["synapse_strength"] = path
    plt.close(fig)

    return figures


def top_table(df: pd.DataFrame, n: int = 10, *, ascending: bool = False) -> pd.DataFrame:
    if df.empty or "log2_oe" not in df.columns:
        return pd.DataFrame(columns=[
            "pair_key",
            "near_pair_count",
            "connected_near_pair_count",
            "fraction_connected",
            "expected_connected",
            "log2_oe",
        ])
    cols = ["pair_key", "near_pair_count", "connected_near_pair_count", "fraction_connected", "expected_connected", "log2_oe"]
    available = [c for c in cols if c in df.columns]
    return df.sort_values("log2_oe", ascending=ascending).head(n)[available]


def dataframe_to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    rows = []
    has_direction = "direction" in df.columns
    col_spec = "llrrrrr" if has_direction else "lrrrrr"
    rows.append(r"\begin{table}[htbp]")
    rows.append(r"\centering")
    rows.append(r"\small")
    rows.append(rf"\caption{{{latex_escape(caption)}}}")
    rows.append(rf"\label{{{label}}}")
    rows.append(r"\resizebox{\linewidth}{!}{%")
    rows.append(rf"\begin{{tabular}}{{{col_spec}}}")
    rows.append(r"\toprule")
    if has_direction:
        rows.append(r"Direction & Pair & Near pairs & Connected & Fraction & Expected & log$_2$(O/E) \\")
    else:
        rows.append(r"Pair & Near pairs & Connected & Fraction & Expected & log$_2$(O/E) \\")
    rows.append(r"\midrule")
    for row in df.itertuples(index=False):
        prefix = f"{latex_escape(row.direction)} & " if has_direction else ""
        rows.append(
            f"{prefix}{latex_escape(row.pair_key)} & {format_int(row.near_pair_count)} & "
            f"{format_int(row.connected_near_pair_count)} & {format_float(row.fraction_connected, 3)} & "
            f"{format_float(row.expected_connected, 1)} & {format_float(row.log2_oe, 2)} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"}")
    rows.append(r"\end{table}")
    return "\n".join(rows)


def distance_windows_to_latex_table(df: pd.DataFrame) -> str:
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Connection probability and connection strength across four prespecified distance windows.}",
        r"\label{tab:distance-windows}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Distance window & Near pairs & Connected & Fraction & 95\% CI & Mean synapses \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        ci = f"{format_float(row.fraction_ci_low, 3)} to {format_float(row.fraction_ci_high, 3)}"
        rows.append(
            f"{latex_escape(row.distance_window)} & {format_int(row.near_pair_count)} & "
            f"{format_int(row.connected_near_pair_count)} & {format_float(row.fraction_connected, 3)} & "
            f"{ci} & {format_float(row.mean_synapses_per_connected_pair, 2)} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(rows)


def figure_block(figures: dict[str, Path], key: str, width: str, caption: str, label: str) -> str:
    path = figures.get(key)
    if path is None:
        return ""
    return "\n".join([
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width={width}\linewidth]{{{path}}}",
        rf"\caption{{{sanitize_report_text(caption)}}}",
        rf"\label{{{label}}}",
        r"\end{figure}",
    ])


def write_tables(
    report_root: Path,
    overall: pd.DataFrame,
    enrichments: dict[str, pd.DataFrame],
    run: Path,
    distance_model: dict[str, float | str] | None = None,
) -> dict[str, Path]:
    table_dir = report_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    overall.to_csv(table_dir / "overall_distance_curve.csv", index=False)
    paths["overall_distance_csv"] = table_dir / "overall_distance_curve.csv"
    distance_windows = summarize_distance_windows(overall)
    distance_windows.to_csv(table_dir / "overall_distance_windows.csv", index=False)
    paths["overall_distance_windows_csv"] = table_dir / "overall_distance_windows.csv"
    if distance_model is not None:
        modeled = add_model_predictions(overall, distance_model)
        modeled.to_csv(table_dir / "overall_distance_model_fit.csv", index=False)
        pd.DataFrame([distance_model]).to_csv(table_dir / "overall_distance_model_summary.csv", index=False)
        paths["overall_distance_model_fit_csv"] = table_dir / "overall_distance_model_fit.csv"
        paths["overall_distance_model_summary_csv"] = table_dir / "overall_distance_model_summary.csv"
    overview = build_partition_overview(enrichments)
    if not overview.empty:
        overview.to_csv(table_dir / "partition_overview.csv", index=False)
        paths["partition_overview_csv"] = table_dir / "partition_overview.csv"
    for name, df in enrichments.items():
        if not df.empty:
            df.to_csv(table_dir / f"{name}_distance_adjusted_enrichment.csv", index=False)
            paths[f"{name}_enrichment_csv"] = table_dir / f"{name}_distance_adjusted_enrichment.csv"
            extremes = enrichment_extremes(df, n=50, min_near=int(PARTITIONS[name]["min_near"]))
            if not extremes.empty:
                extremes.to_csv(table_dir / f"{name}_enrichment_extremes.csv", index=False)
                paths[f"{name}_extremes_csv"] = table_dir / f"{name}_enrichment_extremes.csv"
    for summary in (run / "summaries").glob("by_*_pair.parquet"):
        if summary.name == "by_cell_type_pair.parquet":
            out = table_dir / "appendix_exhaustive_cell_type_pairs.parquet"
            if not out.exists():
                shutil.copy2(summary, out)
            paths["cell_type_appendix"] = out
    return paths


def write_report_tex(
    report_root: Path,
    overall_json: dict,
    overall_curve: pd.DataFrame,
    figures: dict[str, Path],
    enrichments: dict[str, pd.DataFrame],
    distance_model: dict[str, float | str],
    partition_effects: dict[str, dict[str, float | str | int]],
) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    tex_path = report_root / "report.tex"
    rel = lambda p: str(Path(p).relative_to(report_root)).replace(os.sep, "/")
    rel_figures = {k: Path(rel(v)) for k, v in figures.items()}
    headline = overall_json
    ordered = overall_curve.sort_values("distance_bin").reset_index(drop=True)
    first_bin = ordered.iloc[0]
    last_bin = ordered.iloc[-1]
    distance_metrics = summarize_distance_profile(overall_curve)
    distance_windows = summarize_distance_windows(overall_curve)
    modeled = add_model_predictions(overall_curve, distance_model)
    max_abs_residual = float(modeled["residual_fraction_connected"].abs().max())
    first_strength = float(first_bin["connected_syn_count_sum"] / first_bin["connected_near_pair_count"])
    last_strength = float(last_bin["connected_syn_count_sum"] / last_bin["connected_near_pair_count"])
    close_far_delta = (
        float(distance_metrics["zero_to_500_nm_fraction_connected"])
        - float(distance_metrics["one_point_five_to_two_um_fraction_connected"])
    )

    def effect(partition: str, field: str, default: object = np.nan) -> object:
        return partition_effects.get(partition, {}).get(field, default)

    def effect_sentence(partition: str, label: str) -> str:
        if partition not in partition_effects:
            return ""
        e = partition_effects[partition]
        return latex_paragraph(
            f"For {label}, the selected high-support pairs cover {format_int(e['near_pair_count'])} near pairs "
            f"and {format_int(e['connected_near_pair_count'])} connected pairs. The median distance-adjusted effect is "
            f"log2 observed over expected {format_float(e['median_log2_oe'], 2)}, with an interquartile range of "
            f"{format_float(e['iqr_log2_oe'], 2)}. The largest positive effect is "
            f"{latex_escape(e['top_enriched_pair'])}, with log2 observed over expected "
            f"{format_float(e['top_enriched_log2_oe'], 2)} and fraction connected "
            f"{format_pct(e['top_enriched_fraction_connected'], 1)} across "
            f"{format_int(e['top_enriched_near_pair_count'])} near pairs. The largest negative effect is "
            f"{latex_escape(e['top_depleted_pair'])}, with log2 observed over expected "
            f"{format_float(e['top_depleted_log2_oe'], 2)} and fraction connected "
            f"{format_pct(e['top_depleted_fraction_connected'], 1)} across "
            f"{format_int(e['top_depleted_near_pair_count'])} near pairs."
        )

    first_caption = (
        "The blue line shows the observed fraction of neuron pairs with any synaptic connection in each 100 nm "
        "minimum dendrite mesh distance bin. The shaded band is the Wilson 95\\% confidence interval for each bin, "
        "and the gray bars show the number of near pairs contributing to the estimate. The main quantitative "
        f"takeaway is a drop from {format_pct(first_bin['fraction_connected'], 1)} in the {first_bin['distance_bin_nm']} nm bin "
        f"to {format_pct(last_bin['fraction_connected'], 2)} in the {last_bin['distance_bin_nm']} nm bin. When bins are aggregated, "
        f"the 0 to 500 nm window has fraction connected {format_pct(distance_metrics['zero_to_500_nm_fraction_connected'], 1)}, "
        f"whereas the 1.5 to 2.0 micron window has fraction connected "
        f"{format_pct(distance_metrics['one_point_five_to_two_um_fraction_connected'], 2)}, a "
        f"{format_float(distance_metrics['close_to_far_fold_change'], 1)} fold change."
    )
    model_caption = (
        "Points show the observed 100 nm bin fractions and the red curve shows a logistic-binomial fit to the same "
        "binned counts. This fit is used as a compact quantitative rule, not as a mechanistic model. The fitted odds "
        f"ratio per additional 100 nm is {format_float(distance_model['logistic_odds_ratio_per_100nm'], 3)}, "
        f"equivalent to an odds ratio of {format_float(distance_model['logistic_odds_ratio_per_um'], 3)} per micron. "
        f"The fitted half-rate distance is {format_float(distance_model['fitted_half_rate_distance_um'], 2)} microns, "
        f"and the McFadden-style pseudo R squared is {format_float(distance_model['logistic_pseudo_r2'], 3)}."
    )
    residual_caption = (
        "Bars show observed minus fitted connection fraction for each distance bin. Positive residuals indicate bins "
        "where the global distance rule underpredicts connectivity, and negative residuals indicate overprediction. "
        f"The maximum absolute residual across the 20 bins is {format_pp(max_abs_residual, 2)}, which provides a scale "
        "for interpreting where the one-dimensional distance rule is insufficient."
    )
    strength_caption = (
        "The curve shows the mean total synapse count among pairs that are connected, grouped by the same 100 nm "
        "distance bins. This figure separates connection probability from connection strength among successful "
        f"connections. The nearest bin has mean strength {format_float(first_strength, 2)} synapses per connected pair, "
        f"whereas the farthest bin has mean strength {format_float(last_strength, 2)} synapses per connected pair."
    )
    super_heatmap_caption = (
        "Cells show log2 observed over expected connectivity for broad neuron class pairs after adjusting for the "
        "global distance distribution. Red values have more connected pairs than expected from distance alone, blue "
        "values have fewer, and values near zero are close to the distance-only expectation. "
        f"The strongest selected broad-class enrichment is {latex_escape(effect('super_class', 'top_enriched_pair'))} "
        f"with log2 observed over expected {format_float(effect('super_class', 'top_enriched_log2_oe'), 2)}."
    )
    nt_caption = (
        "Cells show the same distance-adjusted observed over expected statistic after grouping each pair by canonical "
        "neurotransmitter labels. The narrower color range relative to some anatomical partitions means that broad "
        "neurotransmitter identity explains less residual variation after distance adjustment. "
        f"The strongest selected neurotransmitter enrichment is {latex_escape(effect('nt', 'top_enriched_pair'))}, "
        f"with log2 observed over expected {format_float(effect('nt', 'top_enriched_log2_oe'), 2)}."
    )
    region_caption = (
        "Lines show distance-connectivity curves for high-support same-region pairs defined by dominant postsynaptic "
        "neuropil. Each point is a 100 nm bin fraction, so differences between curves indicate region-specific "
        "connectivity beyond the global distance trend. The selected postsynaptic neuropil pairs cover "
        f"{format_int(effect('post_neuropil', 'near_pair_count', 0))} near pairs."
    )
    region_extreme_caption = (
        "Bars show the largest positive and negative distance-adjusted effects for dominant postsynaptic neuropil "
        "pairs. Positive bars are pairs with more connected pairs than predicted by distance composition, and negative "
        "bars are pairs with fewer. The top enriched selected region pair is "
        f"{latex_escape(effect('post_neuropil', 'top_enriched_pair'))}, with log2 observed over expected "
        f"{format_float(effect('post_neuropil', 'top_enriched_log2_oe'), 2)}. The strongest depletion is "
        f"{latex_escape(effect('post_neuropil', 'top_depleted_pair'))}, with log2 observed over expected "
        f"{format_float(effect('post_neuropil', 'top_depleted_log2_oe'), 2)}."
    )
    super_curve_caption = (
        "Each line is a distance-connectivity curve for one of the highest-support broad neuron class pairs. The "
        "comparison shows that the distance rule is shared across broad classes, while the vertical separation of "
        "curves quantifies class-specific residual differences. The selected broad-class pairs cover "
        f"{format_int(effect('super_class', 'near_pair_count', 0))} near pairs."
    )
    cell_type_caption = (
        "Each point is a selected exact neuron-type pair. The x-axis is log10 near-pair support, the y-axis is "
        "distance-adjusted log2 observed over expected connectivity, point color is raw fraction connected, and point "
        "area increases with connected-pair count. This representation distinguishes high-confidence effects from "
        "small-support outliers. The strongest selected exact-type enrichment is "
        f"{latex_escape(effect('cell_type', 'top_enriched_pair'))}, with log2 observed over expected "
        f"{format_float(effect('cell_type', 'top_enriched_log2_oe'), 2)}."
    )
    partition_caption = (
        "Boxes summarize the distribution of distance-adjusted log2 observed over expected effects among selected "
        "high-support pairs for each partition. The zero line is the distance-only expectation. Wider boxes and longer "
        "tails identify partitions where metadata labels capture more residual structure after controlling for "
        "distance. In this selected set, exact neuron type spans from "
        f"{format_float(effect('cell_type', 'top_depleted_log2_oe'), 2)} to "
        f"{format_float(effect('cell_type', 'top_enriched_log2_oe'), 2)} in log2 observed over expected."
    )

    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\usepackage{amsmath}",
        r"\title{Whole-Brain Dendrite Mesh Proximity and Synaptic Connectivity}",
        r"\author{FlyWire v783 proximity analysis}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        latex_paragraph(
            "This report quantifies how minimum dendrite mesh distance relates to synaptic connectivity in the "
            f"FlyWire v783 connectome. The analysis includes {format_int(headline['n_neurons'])} neurons and "
            f"{format_int(headline['near_pair_count'])} neuron pairs whose sampled dendrite meshes are within "
            f"2 microns. Of these near pairs, {format_int(headline['connected_near_pair_count'])} have at least one "
            f"synaptic connection in either direction, giving an overall connected fraction of "
            f"{format_pct(headline['fraction_connected'], 2)}. The dominant quantitative rule is distance decay: "
            f"the 0 to 500 nm window has fraction connected "
            f"{format_pct(distance_metrics['zero_to_500_nm_fraction_connected'], 1)}, while the 1.5 to 2.0 micron "
            f"window has fraction connected {format_pct(distance_metrics['one_point_five_to_two_um_fraction_connected'], 2)}."
        ),
        r"\end{abstract}",
        r"\section{Methods}",
        latex_paragraph(
            "The input to the analysis is the completed default whole-connectome proximity run. The manifest records "
            "the exact run name and parameter set. Meshes were fetched from the public "
            "\\url{precomputed://gs://flywire_v141_m783} CloudVolume source, so the analysis does not require a "
            "private FlyWire or CAVE authorization token. The report uses the final enriched pair table generated by "
            "the pipeline, not a sampled subset, unless a smoke-test limit is explicitly supplied on the command line."
        ),
        latex_paragraph(
            "The dendrite mesh proxy is constructed from mesh vertices near postsynaptic sites. Vertices are sampled "
            f"at {format_float(headline['params']['sample_spacing_nm'], 0)} nm spacing after restricting to a "
            f"{format_float(headline['params']['site_radius_nm'], 0)} nm radius around postsynaptic coordinates. "
            "A neuron pair is counted as spatially proximal when the closest sampled dendrite mesh points are within "
            "2 microns. The whole-brain run uses exact pair aggregation across bucketed reduce tasks, with "
            f"{format_int(headline['reduce_buckets'])} reduce buckets in the final reduce stage."
        ),
        latex_paragraph(
            "Connectivity is measured as a binary indicator of whether either direction contains at least one synapse. "
            "For distance curves, pairs are binned into 100 nm distance intervals from 0 to 2 microns. Each bin reports "
            "a connected fraction and a Wilson 95 percent confidence interval. For metadata partitions, the report "
            "compares observed connected counts with expected connected counts obtained by applying the global "
            "distance-bin connection rate to the partition-specific distance composition. This creates a distance-adjusted "
            "observed over expected statistic, reported as log2 observed over expected."
        ),
        latex_paragraph(
            "Two compact distance rules are fit to the binned data. The first is a weighted log-linear model for the "
            "connection fraction, which summarizes multiplicative decay of probability with distance. The second is a "
            "weighted logistic-binomial approximation, which summarizes multiplicative decay of connection odds with "
            "distance. These models are descriptive summaries of the binned observations. They are used to quantify "
            "effect sizes, not to claim that distance alone is a full biological mechanism."
        ),
        r"\section{Results}",
        r"\subsection{Global Distance Rule}",
        latex_paragraph(
            "The global distance curve gives the main quantitative answer. Among all near pairs within 2 microns, "
            f"{format_pct(headline['fraction_connected'], 2)} are connected. The nearest 100 nm bin has fraction "
            f"connected {format_pct(first_bin['fraction_connected'], 1)} across "
            f"{format_int(first_bin['near_pair_count'])} near pairs. The farthest 100 nm bin has fraction connected "
            f"{format_pct(last_bin['fraction_connected'], 2)} across {format_int(last_bin['near_pair_count'])} near "
            f"pairs. The absolute drop from first to last bin is {format_pp(float(first_bin['fraction_connected']) - float(last_bin['fraction_connected']), 1)}."
        ),
        latex_paragraph(
            "Aggregating bins into wider windows makes the rule easier to use. The 0 to 500 nm window has fraction "
            f"connected {format_pct(distance_metrics['zero_to_500_nm_fraction_connected'], 1)}, while the 1.5 to "
            f"2.0 micron window has fraction connected "
            f"{format_pct(distance_metrics['one_point_five_to_two_um_fraction_connected'], 2)}. This is an absolute "
            f"difference of {format_pp(close_far_delta, 1)} and a fold change of "
            f"{format_float(distance_metrics['close_to_far_fold_change'], 1)}. The first 100 nm bin at or below "
            f"10 percent connected is {latex_escape(distance_model['first_bin_at_or_below_10pct'])} nm, and the first "
            f"bin at or below 5 percent connected is {latex_escape(distance_model['first_bin_at_or_below_5pct'])} nm."
        ),
        distance_windows_to_latex_table(distance_windows),
        figure_block(rel_figures, "overall_distance", "0.88", first_caption, "fig:overall-distance"),
        figure_block(rel_figures, "overall_distance_model", "0.88", model_caption, "fig:overall-distance-model"),
        figure_block(rel_figures, "overall_distance_residuals", "0.88", residual_caption, "fig:overall-distance-residuals"),
        latex_paragraph(
            "The fitted logistic-binomial rule estimates that each additional 100 nm multiplies connection odds by "
            f"{format_float(distance_model['logistic_odds_ratio_per_100nm'], 3)}. Equivalently, each additional micron "
            f"multiplies odds by {format_float(distance_model['logistic_odds_ratio_per_um'], 3)}. The weighted "
            f"log-linear model gives a probability multiplier of "
            f"{format_float(distance_model['log_linear_rate_multiplier_per_um'], 3)} per micron and a probability "
            f"half-distance of {format_float(distance_model['log_linear_half_distance_um'], 2)} microns. The logistic "
            f"pseudo R squared is {format_float(distance_model['logistic_pseudo_r2'], 3)}, and the weighted R squared "
            f"on binned fractions is {format_float(distance_model['logistic_weighted_r2'], 3)}."
        ),
        r"\subsection{Connection Strength Among Connected Pairs}",
        latex_paragraph(
            "The binary connection rule asks whether a pair is connected at all. The strength analysis asks a separate "
            "question: among connected near pairs, how many synapses are present? The nearest distance bin has mean "
            f"{format_float(first_strength, 2)} synapses per connected pair, while the farthest bin has mean "
            f"{format_float(last_strength, 2)} synapses per connected pair. This comparison is conditional on being "
            "connected, so it should not be interpreted as the total expected synapse count for arbitrary near pairs."
        ),
        figure_block(rel_figures, "synapse_strength", "0.88", strength_caption, "fig:synapse-strength"),
        r"\subsection{Broad Neuron Classes}",
        effect_sentence("super_class", "broad neuron classes"),
        figure_block(rel_figures, "super_class_heatmap", "0.82", super_heatmap_caption, "fig:super-class-heatmap"),
        figure_block(rel_figures, "super_class_curves", "0.88", super_curve_caption, "fig:super-class-curves"),
        r"\subsection{Cell Classes and Exact Neuron Types}",
        effect_sentence("cell_class", "cell classes"),
        effect_sentence("cell_type", "exact neuron types"),
        figure_block(rel_figures, "cell_type_scatter", "0.88", cell_type_caption, "fig:cell-type-scatter"),
        r"\subsection{Brain Regions}",
        effect_sentence("post_neuropil", "dominant postsynaptic neuropils"),
        effect_sentence("pre_neuropil", "dominant presynaptic neuropils"),
        figure_block(rel_figures, "post_neuropil_curves", "0.9", region_caption, "fig:post-neuropil-curves"),
        figure_block(rel_figures, "post_neuropil_extremes", "0.9", region_extreme_caption, "fig:post-neuropil-extremes"),
        r"\subsection{Neurotransmitter, Flow, and Side}",
        effect_sentence("nt", "canonical neurotransmitter pairs"),
        effect_sentence("flow", "flow labels"),
        effect_sentence("side", "hemisphere side labels"),
        figure_block(rel_figures, "nt_heatmap", "0.78", nt_caption, "fig:nt-heatmap"),
        figure_block(rel_figures, "partition_summary", "0.88", partition_caption, "fig:partition-summary"),
        r"\subsection{Distance-Adjusted Effect Tables}",
    ]
    for partition, caption_name in [
        ("super_class", "Broad neuron class"),
        ("cell_class", "Cell class"),
        ("post_neuropil", "Dominant postsynaptic neuropil"),
        ("pre_neuropil", "Dominant presynaptic neuropil"),
        ("nt", "Neurotransmitter"),
        ("cell_type", "Exact neuron type"),
        ("flow", "Flow"),
        ("side", "Side"),
    ]:
        df = enrichments.get(partition, pd.DataFrame())
        if df.empty:
            continue
        extremes = enrichment_extremes(df, n=5, min_near=int(PARTITIONS[partition]["min_near"]))
        if not extremes.empty:
            parts.append(dataframe_to_latex_table(
                extremes,
                f"{caption_name} pairs with the largest positive and negative distance-adjusted effects among selected supported pairs.",
                f"tab:{partition.replace('_', '-')}-extremes",
            ))
    parts.extend([
        r"\section{Discussion}",
        latex_paragraph(
            "The most defensible quantitative rule from this analysis is that dendrite mesh proximity is a powerful "
            "but incomplete predictor of synaptic connectivity. The evidence for the rule is the monotonic decline "
            f"from {format_pct(first_bin['fraction_connected'], 1)} in the nearest bin to "
            f"{format_pct(last_bin['fraction_connected'], 2)} in the farthest bin, the "
            f"{format_float(distance_metrics['close_to_far_fold_change'], 1)} fold contrast between the close and far "
            "aggregate windows, and the fitted odds multiplier per 100 nm. The residual and partition analyses show "
            "that distance does not explain all structure."
        ),
        latex_paragraph(
            "The distance-adjusted enrichments identify metadata partitions whose labels preserve residual connectivity "
            "structure after controlling for the global distance distribution. Exact neuron type and brain-region labels "
            "show the widest selected effects, which is expected if cell identity and regional circuit architecture impose "
            "specific partner preferences beyond physical opportunity. Neurotransmitter labels show smaller selected "
            "effects, which is consistent with neurotransmitter being a broad physiological class rather than a precise "
            "partner-identity label."
        ),
        latex_paragraph(
            "A practical rule for downstream analysis is to treat 0 to 500 nm as a high-opportunity zone and 1.5 to "
            "2.0 microns as a low-opportunity zone under this mesh sampling definition. The high-opportunity zone is "
            f"connected at {format_pct(distance_metrics['zero_to_500_nm_fraction_connected'], 1)}, while the low-opportunity "
            f"zone is connected at {format_pct(distance_metrics['one_point_five_to_two_um_fraction_connected'], 2)}. "
            "This does not mean distance alone should be used as a classifier. Instead, it provides a baseline expectation "
            "against which region, cell type, neurotransmitter, flow, and side effects can be evaluated."
        ),
        r"\section{Limitations}",
        latex_paragraph(
            "The mesh-distance variable is computed from sampled vertices near postsynaptic sites, not from a complete "
            "continuous surface-to-surface distance between all dendritic compartments. The 250 nm sampling interval and "
            "500 nm postsynaptic-site radius make the computation tractable at whole-brain scale, but they also define "
            "the resolution and biological interpretation of the distance measure."
        ),
        latex_paragraph(
            "The report uses the binary question of whether any synaptic connection exists between a near pair. Direction, "
            "synapse count, and sign are retained in the enriched table and in the strength summaries, but most partition "
            "effects use the binary connected-any outcome. Exact neuron-type plots are selected for support and effect "
            "visibility, so the figure is an interpretable high-support view rather than a complete plot of every type pair."
        ),
        latex_paragraph(
            "The observed over expected adjustment controls for the global distance-bin distribution but does not control "
            "simultaneously for every possible confounder, such as neuron size, synapse count, sampling density, or nested "
            "cell-type structure. Claims in this report should therefore be read as descriptive whole-connectome statistics "
            "and as hypothesis-generating evidence for circuit specificity."
        ),
        r"\section{Reproducibility and Outputs}",
        latex_paragraph(
            "The full enriched near-pair table is stored as \\texttt{near\\_pairs\\_enriched.parquet} in the canonical run "
            "directory. The report directory contains the compiled PDF, LaTeX source, figure PDFs, CSV tables, and a JSON "
            "manifest with all headline metrics. Lightweight CSV tables are suitable for versioned snapshots. Bulk Parquet "
            "tables and mesh caches are intentionally excluded from the repository snapshot."
        ),
        r"\end{document}",
    ])
    text = "\n\n".join(part for part in parts if part)
    validate_report_text(text)
    tex_path.write_text(text)
    return tex_path


def find_compiler() -> str | None:
    for cmd in ("tectonic", "latexmk", "pdflatex"):
        path = shutil.which(cmd)
        if path:
            return cmd
    return None


def install_tectonic_if_requested() -> str | None:
    if find_compiler():
        return find_compiler()
    try:
        subprocess.run(["conda", "install", "-y", "-c", "conda-forge", "tectonic"], check=True)
    except Exception:
        return None
    return find_compiler()


def compile_tex(tex_path: Path, *, install_tectonic: bool = False) -> Path | None:
    compiler = install_tectonic_if_requested() if install_tectonic else find_compiler()
    if compiler is None:
        instructions = tex_path.parent / "compile_instructions.md"
        instructions.write_text(
            "# Compile instructions\n\n"
            "No LaTeX compiler was found on PATH. Install one of `tectonic`, `latexmk`, or `pdflatex`, then run:\n\n"
            "```bash\n"
            f"cd {tex_path.parent}\n"
            "tectonic report.tex\n"
            "```\n"
        )
        return None
    if compiler == "tectonic":
        cmd = ["tectonic", tex_path.name]
    elif compiler == "latexmk":
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name]
    else:
        cmd = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
    subprocess.run(cmd, cwd=tex_path.parent, check=True)
    pdf = tex_path.with_suffix(".pdf")
    return pdf if pdf.exists() else None


def module_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def snapshot_report_to_repo(report_root: Path, snapshot_dir: Path) -> Path:
    target = snapshot_dir if snapshot_dir.is_absolute() else module_repo_root() / snapshot_dir
    if target.exists():
        shutil.rmtree(target)
    (target / "figures").mkdir(parents=True, exist_ok=True)
    (target / "tables").mkdir(parents=True, exist_ok=True)

    for name in ("report.pdf", "report.tex", "report_manifest.json"):
        src = report_root / name
        if src.exists():
            shutil.copy2(src, target / name)

    for src in sorted((report_root / "figures").glob("*.pdf")):
        shutil.copy2(src, target / "figures" / src.name)

    for src in sorted((report_root / "tables").glob("*.csv")):
        shutil.copy2(src, target / "tables" / src.name)

    readme = "\n".join([
        "# Whole-connectome proximity report",
        "",
        "This directory is a lightweight GitHub snapshot of the generated report.",
        "",
        f"Canonical output directory: `{report_root}`",
        "",
        "Included files:",
        "",
        "- `report.pdf` and `report.tex`",
        "- Figure PDFs under `figures/`",
        "- Lightweight CSV support tables under `tables/`",
        "- `report_manifest.json` with run paths and headline metrics",
        "",
        "Excluded files:",
        "",
        "- `near_pairs_enriched.parquet`",
        "- exhaustive Parquet appendices",
        "- mesh cache, tile files, reduce buckets, and Slurm logs",
        "",
        "Regenerate from the repository root with:",
        "",
        "```bash",
        "source slurm/proximity_common.sh",
        "python -m flyconn.experiments.proximity_report \\",
        "  --run-name whole_connectome_lod1_sp250_r500_t2_any \\",
        "  --compile \\",
        "  --snapshot-repo-report",
        "```",
        "",
    ])
    (target / "README.md").write_text(readme)
    return target


def generate_report(cfg: Config, report_cfg: ReportConfig) -> dict:
    run = run_dir(cfg, report_cfg)
    root = report_dir(run)
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    overall_json = json.loads((run / "summaries" / "overall.json").read_text())
    selected_pairs = select_pairs(run, overall_json["fraction_connected"], report_cfg)
    overall_curve, curves = stream_distance_curves(run, report_cfg, selected_pairs)
    overall_curve.to_csv(tables_dir / "overall_distance_curve.csv", index=False)
    distance_model = fit_distance_models(overall_curve)
    enrichments = {}
    for partition, curve in curves.items():
        curve.to_csv(tables_dir / f"{partition}_distance_curve.csv", index=False)
        enrich = add_distance_adjusted_enrichment(curve, overall_curve)
        enrichments[partition] = enrich
        enrich.to_csv(tables_dir / f"{partition}_distance_adjusted_enrichment.csv", index=False)
    partition_effects = summarize_partition_effects(enrichments)
    write_tables(root, overall_curve, enrichments, run, distance_model)
    figures = make_figures(root, overall_curve, curves, enrichments, distance_model)
    tex_path = write_report_tex(root, overall_json, overall_curve, figures, enrichments, distance_model, partition_effects)
    pdf_path = compile_tex(tex_path, install_tectonic=report_cfg.install_tectonic) if report_cfg.compile_pdf else None
    distance_metrics = summarize_distance_profile(overall_curve)
    partition_overview = build_partition_overview(enrichments)
    manifest = {
        "run_dir": str(run),
        "report_dir": str(root),
        "tex_path": str(tex_path),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "snapshot_dir": None,
        "figures": {k: str(v) for k, v in figures.items()},
        "tables_dir": str(tables_dir),
        "overall": overall_json,
        "distance_metrics": distance_metrics,
        "distance_model": distance_model,
        "partition_effects": partition_effects,
        "partition_overview": partition_overview.to_dict(orient="records") if not partition_overview.empty else [],
    }
    write_json(root / "report_manifest.json", manifest)
    if report_cfg.snapshot_repo_report:
        snapshot_path = snapshot_report_to_repo(root, report_cfg.snapshot_dir)
        manifest["snapshot_dir"] = str(snapshot_path)
        write_json(root / "report_manifest.json", manifest)
        shutil.copy2(root / "report_manifest.json", snapshot_path / "report_manifest.json")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="flyconn.experiments.proximity_report")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default=RUN_NAME_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--top-type-pairs", type=int, default=75)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--install-tectonic", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--snapshot-repo-report", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("reports/proximity_whole_connectome"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    report_cfg = ReportConfig(
        run_name=args.run_name,
        batch_size=args.batch_size,
        top_type_pairs=args.top_type_pairs,
        compile_pdf=args.compile,
        install_tectonic=args.install_tectonic,
        force=args.force,
        max_batches=args.max_batches,
        snapshot_repo_report=args.snapshot_repo_report,
        snapshot_dir=args.snapshot_dir,
    )
    result = generate_report(cfg, report_cfg)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
