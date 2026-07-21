"""Aggregate per-run summaries into one results table.

Scans ``$FLYCONN_DATA_ROOT/v783/models/*/summary.json`` and writes
``$FLYCONN_DATA_ROOT/v783/results/summary.csv`` — one row per run with the headline
metrics, plus a pivot of test accuracy by (subgraph, arch) x init so the
from_data-vs-random contrast is easy to read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..paths import data_root


def collect() -> pd.DataFrame:
    models_dir = data_root() / "v783" / "models"
    rows = []
    for sj in sorted(models_dir.glob("*/summary.json")):
        try:
            d = json.loads(sj.read_text())
        except Exception:
            continue
        rows.append({
            "run_name": d.get("run_name"),
            "subgraph_id": d.get("subgraph_id"),
            "arch": d.get("arch"),
            "init_mode": d.get("init_mode"),
            "variant": d.get("variant", "stage3"),
            "eye": d.get("eye", "learned"),
            "decision": d.get("decision", "linear"),
            # Family 3 (recmul) contrasts: step rule + whether the connectome was trained.
            "dynamics": d.get("info", {}).get("dynamics", d.get("dynamics", "tanh")),
            "learn_core": d.get("info", {}).get("learn_core", d.get("learn_core")),
            # Family 4 (eyeTact): activation axis (= dynamics + nonlinearity) + eye source.
            "nonlinearity": d.get("nonlinearity",
                                  d.get("info", {}).get("net_cfg", {}).get("nonlinearity")),
            "state_norm": d.get("state_norm",
                                d.get("info", {}).get("net_cfg", {}).get("state_norm")),
            "eye_source": d.get("eye_source", d.get("info", {}).get("eye_source")),
            "family": d.get("family", d.get("info", {}).get("family")),
            # Dataset axis (eyeTact_cifar). Legacy MNIST summaries lack these -> default.
            "dataset": d.get("dataset", d.get("info", {}).get("dataset", "mnist")),
            "color": d.get("color", d.get("info", {}).get("color", "luma")),
            # Rigid-eye spectral mode: 'luma' (colorblind) | 'spectral' (per-type R/G/B).
            "eye_color": d.get("info", {}).get("eye_color", "luma"),
            "photoreceptor_sign": d.get("photoreceptor_sign", "inherit"),
            # Family 5 (initablation): the sign knobs that distinguish nomag (force_sign none)
            # from nodirmag (force_sign +1) and the shuffled-sign control.
            "force_sign": d.get("force_sign", d.get("info", {}).get("force_sign", "none")),
            "sign_shuffle": d.get("sign_shuffle",
                                  d.get("info", {}).get("sign_shuffle", "none")),
            "N": d.get("N"), "E": d.get("E"), "T": d.get("T"),
            "n_params": d.get("n_params"),
            "best_val_acc": d.get("best_val_acc"),
            "test_acc": d.get("test_acc"),
            # V3-specific (None for V1/V2/stage3)
            "ncm_acc_raw_euclid": d.get("ncm_acc_raw_euclid"),
            "ncm_acc_cosine": d.get("ncm_acc_cosine"),
            "ncm_acc_lda": d.get("ncm_acc_lda"),
            "shuffle_control_acc": d.get("shuffle_control_acc"),
        })
    return pd.DataFrame(rows)


def write_summary() -> Path:
    df = collect()
    out_dir = data_root() / "v783" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "summary.csv"
    df.sort_values(["subgraph_id", "arch", "init_mode"]).to_csv(csv, index=False)
    # headline pivot: test_acc by (subgraph, arch) rows, init_mode columns
    if not df.empty:
        piv = df.pivot_table(index=["subgraph_id", "arch"], columns="init_mode",
                             values="test_acc")
        if {"from_data", "random"} <= set(piv.columns):
            piv["data_minus_random"] = piv["from_data"] - piv["random"]
        piv.to_csv(out_dir / "test_acc_pivot.csv")
        print(piv.round(4).to_string())
    print(f"\n[aggregate] {len(df)} runs -> {csv}")
    return csv


if __name__ == "__main__":
    write_summary()
