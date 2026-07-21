"""Build the depth -> pathway -> accuracy table for the eyeTact learned-eye models.

For each activation {linear, tanh, relu} x {untrained, trained}, run the T=12 model and
capture the per-recurrence-step activation wavefront (which cell types light up at each
depth t). Join, per depth t, the TEST ACCURACY of the eyeTact model that was trained at
exactly that T (each T=1..12 is a separate run). The result ties together:
  recurrence depth t  ->  cell types newly active at that hop  ->  test_acc(T=t).

Writes per-(activation,core) CSVs and prints a compact table to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flyconn.models.activity import per_step_activity
from flyconn.paths import data_root

ACTS = ["linear", "tanh", "relu"]
CORES = ["untrained", "trained"]
MODELS = data_root() / "v783" / "models"
OUT = data_root() / "v783" / "activity"
OUT.mkdir(parents=True, exist_ok=True)

# activation threshold (mean |h|) above which a cell type counts as "active" at a step.
ACTIVE_THR = 0.05


def test_acc(act: str, core: str, T: int) -> float | None:
    sj = MODELS / f"eyeTact_whole_learned_{act}_{core}_T{T:02d}" / "summary.json"
    if not sj.exists():
        return None
    return json.load(open(sj)).get("test_acc")


def build(act: str, core: str) -> pd.DataFrame:
    run = f"eyeTact_whole_learned_{act}_{core}_T12"
    ps = per_step_activity(run, max_batches=8)            # rows: (t, cell_type, ...)
    # normalize mean_abs per step so "active" is comparable across steps (signal can shrink)
    ps["frac_of_step_max"] = ps.groupby("t")["mean_abs"].transform(
        lambda s: s / (s.max() + 1e-12))
    # first step each cell type becomes active (its activation wavefront arrival)
    active = ps[ps["mean_abs"] > ACTIVE_THR]
    first_step = active.groupby("cell_type")["t"].min()

    rows = []
    for t in range(1, 13):
        at_t = ps[ps["t"] == t]
        # top cell types by mean_abs at this depth
        top = at_t.sort_values("mean_abs", ascending=False).head(6)
        top_types = ", ".join(top["cell_type"].tolist())
        # newly recruited at this depth (first appear here)
        newly = sorted(first_step[first_step == t].index.tolist())
        # has the readout population activated yet?
        rdt_active = at_t[(at_t["readout_frac"] > 0) & (at_t["mean_abs"] > ACTIVE_THR)]
        n_active_types = int((at_t["mean_abs"] > ACTIVE_THR).sum())
        rows.append({
            "t": t,
            "n_active_types": n_active_types,
            "top_types": top_types,
            "newly_active": ", ".join(newly[:8]),
            "readout_reached": "yes" if len(rdt_active) else "no",
            "test_acc@T=t": test_acc(act, core, t),
        })
    df = pd.DataFrame(rows)
    df.insert(0, "core", core)
    df.insert(0, "activation", act)
    df.to_csv(OUT / f"depth_pathway_{act}_{core}.csv", index=False)
    return df


def main():
    allrows = []
    for act in ACTS:
        for core in CORES:
            print(f"\n{'='*100}\n{act.upper()} | {core.upper()}\n{'='*100}")
            df = build(act, core)
            allrows.append(df)
            show = df.copy()
            show["test_acc@T=t"] = show["test_acc@T=t"].map(
                lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "-")
            print(show.to_string(index=False,
                  columns=["t", "n_active_types", "readout_reached",
                           "test_acc@T=t", "top_types"]))
    pd.concat(allrows, ignore_index=True).to_csv(OUT / "depth_pathway_ALL.csv", index=False)
    print(f"\n[done] full table -> {OUT / 'depth_pathway_ALL.csv'}")


if __name__ == "__main__":
    main()
