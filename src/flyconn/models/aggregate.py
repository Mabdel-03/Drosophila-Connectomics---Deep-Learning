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
            "N": d.get("N"), "E": d.get("E"), "T": d.get("T"),
            "n_params": d.get("n_params"),
            "best_val_acc": d.get("best_val_acc"),
            "test_acc": d.get("test_acc"),
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
