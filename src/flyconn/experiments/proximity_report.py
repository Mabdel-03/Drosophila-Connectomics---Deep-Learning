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


def make_figures(report_root: Path, overall: pd.DataFrame, curves: dict[str, pd.DataFrame], enrichments: dict[str, pd.DataFrame]) -> dict[str, Path]:
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
    rows.append(r"\begin{table}[htbp]")
    rows.append(r"\centering")
    rows.append(r"\small")
    rows.append(rf"\caption{{{latex_escape(caption)}}}")
    rows.append(rf"\label{{{label}}}")
    rows.append(r"\resizebox{\linewidth}{!}{%")
    rows.append(r"\begin{tabular}{lrrrrr}")
    rows.append(r"\toprule")
    rows.append(r"Pair & Near pairs & Connected & Fraction & Expected & log$_2$(O/E) \\")
    rows.append(r"\midrule")
    for row in df.itertuples(index=False):
        rows.append(
            f"{latex_escape(row.pair_key)} & {format_int(row.near_pair_count)} & "
            f"{format_int(row.connected_near_pair_count)} & {format_float(row.fraction_connected, 3)} & "
            f"{format_float(row.expected_connected, 1)} & {format_float(row.log2_oe, 2)} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"}")
    rows.append(r"\end{table}")
    return "\n".join(rows)


def write_tables(report_root: Path, overall: pd.DataFrame, enrichments: dict[str, pd.DataFrame], run: Path) -> dict[str, Path]:
    table_dir = report_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    overall.to_csv(table_dir / "overall_distance_curve.csv", index=False)
    paths["overall_distance_csv"] = table_dir / "overall_distance_curve.csv"
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
) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    tex_path = report_root / "report.tex"
    rel = lambda p: str(Path(p).relative_to(report_root)).replace(os.sep, "/")
    headline = overall_json
    first_bin = overall_curve.sort_values("distance_bin").iloc[0]
    last_bin = overall_curve.sort_values("distance_bin").iloc[-1]
    super_top = top_table(enrichments.get("super_class", pd.DataFrame()).query("near_pair_count >= 10000"), 8) if "super_class" in enrichments else pd.DataFrame()
    class_top = top_table(enrichments.get("cell_class", pd.DataFrame()).query("near_pair_count >= 5000"), 8) if "cell_class" in enrichments else pd.DataFrame()
    region_top = top_table(enrichments.get("post_neuropil", pd.DataFrame()).query("near_pair_count >= 2000"), 8) if "post_neuropil" in enrichments else pd.DataFrame()
    nt_top = top_table(enrichments.get("nt", pd.DataFrame()).query("near_pair_count >= 1000"), 8) if "nt" in enrichments else pd.DataFrame()
    type_top = top_table(enrichments.get("cell_type", pd.DataFrame()), 8) if "cell_type" in enrichments else pd.DataFrame()
    distance_metrics = summarize_distance_profile(overall_curve)

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
        (
            "We quantified the relationship between minimum dendrite mesh distance and observed synaptic connectivity "
            f"across {format_int(headline['n_neurons'])} FlyWire v783 neurons. Among "
            f"{format_int(headline['near_pair_count'])} neuron pairs with sampled dendrite meshes within 2 $\\mu$m, "
            f"{format_int(headline['connected_near_pair_count'])} were connected, giving an overall connected fraction of "
            f"{100 * headline['fraction_connected']:.2f}\\%."
        ),
        r"\end{abstract}",
        r"\section{Headline Findings}",
        (
            "Connectivity is strongly distance dependent. The nearest 0--100 nm bin has a connected fraction of "
            f"{100 * first_bin['fraction_connected']:.1f}\\%, whereas the 1900--2000 nm bin has a connected fraction of "
            f"{100 * last_bin['fraction_connected']:.2f}\\%. This decay remains visible across broad neuron classes, "
            "neurotransmitter categories, and neuropil-defined brain regions."
        ),
        (
            "Aggregating bins, pairs at 0--500 nm are "
            f"{distance_metrics['close_to_far_fold_change']:.1f}$\\times$ more likely to be connected than pairs at "
            "1.5--2.0 $\\mu$m under the same mesh-based proximity definition."
        ),
        r"\begin{figure}[H]\centering",
        rf"\includegraphics[width=0.88\linewidth]{{{rel(figures['overall_distance'])}}}",
        r"\caption{Overall probability of any synaptic connection as a function of closest dendrite mesh distance. Shaded intervals are Wilson 95\% confidence intervals; bars indicate near-pair support.}",
        r"\label{fig:overall-distance}",
        r"\end{figure}",
        r"\section{Methods Summary}",
        (
            "Meshes were fetched from the public CloudVolume source \\texttt{precomputed://gs://flywire\\_v141\\_m783}. "
            "Dendrite samples were restricted to mesh regions near postsynaptic sites, downsampled at 250 nm spacing, "
            "and neuron pairs were considered spatially proximal if the closest sampled dendrite mesh points were within 2 $\\mu$m. "
            "The final pair aggregation was exact and bucketed across 64 reduce buckets."
        ),
        r"\section{Distance-Connectivity Relationship}",
        rf"\begin{{figure}}[H]\centering\includegraphics[width=0.88\linewidth]{{{rel(figures['synapse_strength'])}}}\caption{{Mean total synapse count among connected near pairs by distance bin.}}\end{{figure}}",
        r"\section{Differences Across Partitions}",
    ]
    if "super_class_heatmap" in figures:
        parts.extend([
            rf"\begin{{figure}}[H]\centering\includegraphics[width=0.82\linewidth]{{{rel(figures['super_class_heatmap'])}}}\caption{{Distance-adjusted observed/expected connectivity by broad neuron class pair.}}\end{{figure}}",
            rf"\begin{{figure}}[H]\centering\includegraphics[width=0.88\linewidth]{{{rel(figures['super_class_curves'])}}}\caption{{Distance-connectivity curves for the highest-support broad class pairs.}}\end{{figure}}",
        ])
    if "nt_heatmap" in figures:
        parts.append(rf"\begin{{figure}}[H]\centering\includegraphics[width=0.78\linewidth]{{{rel(figures['nt_heatmap'])}}}\caption{{Distance-adjusted observed/expected connectivity by neurotransmitter pair.}}\end{{figure}}")
    if "post_neuropil_curves" in figures:
        parts.append(rf"\begin{{figure}}[H]\centering\includegraphics[width=0.9\linewidth]{{{rel(figures['post_neuropil_curves'])}}}\caption{{Distance-connectivity curves for high-support same-neuropil dominant postsynaptic regions.}}\end{{figure}}")
    if "post_neuropil_extremes" in figures:
        parts.append(rf"\begin{{figure}}[H]\centering\includegraphics[width=0.9\linewidth]{{{rel(figures['post_neuropil_extremes'])}}}\caption{{Brain-region pairs with the largest positive and negative distance-adjusted effects.}}\end{{figure}}")
    if "cell_type_scatter" in figures:
        parts.append(rf"\begin{{figure}}[H]\centering\includegraphics[width=0.88\linewidth]{{{rel(figures['cell_type_scatter'])}}}\caption{{High-support exact neuron-type pairs, showing support versus distance-adjusted enrichment.}}\end{{figure}}")
    if "partition_summary" in figures:
        parts.append(rf"\begin{{figure}}[H]\centering\includegraphics[width=0.88\linewidth]{{{rel(figures['partition_summary'])}}}\caption{{Distribution of distance-adjusted effects among selected high-support pairs for each analysis partition.}}\end{{figure}}")

    parts.append(r"\section{Top Distance-Adjusted Enrichments}")
    if not super_top.empty:
        parts.append(dataframe_to_latex_table(super_top, "Top broad neuron class enrichments after distance adjustment.", "tab:super-class-top"))
    if not class_top.empty:
        parts.append(dataframe_to_latex_table(class_top, "Top cell class enrichments after distance adjustment.", "tab:cell-class-top"))
    if not region_top.empty:
        parts.append(dataframe_to_latex_table(region_top, "Top dominant postsynaptic neuropil enrichments after distance adjustment.", "tab:region-top"))
    if not nt_top.empty:
        parts.append(dataframe_to_latex_table(nt_top, "Top neurotransmitter pair enrichments after distance adjustment.", "tab:nt-top"))
    if not type_top.empty:
        parts.append(dataframe_to_latex_table(type_top, "Top exact neuron-type pair enrichments among selected high-support candidates.", "tab:type-top"))
    parts.extend([
        r"\section{Outputs}",
        (
            "The full enriched near-pair table is stored as \\texttt{near\\_pairs\\_enriched.parquet}. "
            "Distance-adjusted enrichment tables and appendix exports are in the report \\texttt{tables/} directory."
        ),
        r"\end{document}",
    ])
    tex_path.write_text("\n\n".join(parts))
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
    enrichments = {}
    for partition, curve in curves.items():
        curve.to_csv(tables_dir / f"{partition}_distance_curve.csv", index=False)
        enrich = add_distance_adjusted_enrichment(curve, overall_curve)
        enrichments[partition] = enrich
        enrich.to_csv(tables_dir / f"{partition}_distance_adjusted_enrichment.csv", index=False)
    write_tables(root, overall_curve, enrichments, run)
    figures = make_figures(root, overall_curve, curves, enrichments)
    tex_path = write_report_tex(root, overall_json, overall_curve, figures, enrichments)
    pdf_path = compile_tex(tex_path, install_tectonic=report_cfg.install_tectonic) if report_cfg.compile_pdf else None
    distance_metrics = summarize_distance_profile(overall_curve)
    partition_overview = build_partition_overview(enrichments)
    manifest = {
        "run_dir": str(run),
        "report_dir": str(root),
        "tex_path": str(tex_path),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "figures": {k: str(v) for k, v in figures.items()},
        "tables_dir": str(tables_dir),
        "overall": overall_json,
        "distance_metrics": distance_metrics,
        "partition_overview": partition_overview.to_dict(orient="records") if not partition_overview.empty else [],
    }
    write_json(root / "report_manifest.json", manifest)
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
    )
    result = generate_report(cfg, report_cfg)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
