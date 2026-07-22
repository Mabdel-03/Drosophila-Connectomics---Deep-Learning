"""Derive family G: LLPC1 sheet output census + Nod1 dominance.

Pulls every output synapse of the 100 sheet LLPC1, aggregates by target cell_type,
and records (drivers/100, total syn, NT) for the paper's named targets. Identifies the
dominant *excitatory* (cholinergic) readout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import sheet as SH
from ..oracle import consts as C


def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    sheet = SH.get_sheet(src, meta, cfg)
    sheet_set = set(int(x) for x in sheet.llpc1_roots)

    # All synapses FROM the 100 sheet LLPC1.
    out = src.synapses(pre_ids=sheet.llpc1_roots.tolist())
    out = out[out["pre_pt_root_id"].isin(sheet_set)]
    output_total_syn = int(len(out))

    # Per target neuron: syn count + number of distinct sheet LLPC1 driving it.
    g = out.groupby("post_pt_root_id")
    per_target = pd.DataFrame({
        "syn": g.size(),
        "drivers": g["pre_pt_root_id"].nunique(),
    }).reset_index().rename(columns={"post_pt_root_id": "root_id"})
    per_target = per_target.join(meta.by_root, on="root_id")

    # Aggregate by cell_type (sum syn; drivers = max distinct LLPC1 across that type's cells).
    byct = per_target.dropna(subset=["cell_type"]).groupby("cell_type").agg(
        syn=("syn", "sum"),
        drivers=("drivers", "max"),
        nt=("nt_canonical", "first"),
    ).reset_index().sort_values("syn", ascending=False)

    targets = {}
    for t in ("PLP249", "Nod1", "PVLP011", "PLP163", "Nod2", "DNbe001"):
        row = byct[byct["cell_type"] == t]
        if len(row):
            r = row.iloc[0]
            targets[t] = {"drivers": int(r["drivers"]), "syn": int(r["syn"]), "nt": str(r["nt"])}
        else:
            targets[t] = {"drivers": 0, "syn": 0, "nt": None}

    # Dominant excitatory (cholinergic) readout among shared targets (drivers>=50),
    # excluding the sheet's own lateral LLPC1 connections and generic feedback.
    cho = byct[(byct["nt"] == "acetylcholine") & (byct["drivers"] >= 50)
               & (~byct["cell_type"].isin(["LLPC1"]))]
    top_exc = str(cho.iloc[0]["cell_type"]) if len(cho) else None

    return {
        "output_total_syn": output_total_syn,
        "targets": targets,
        "top_excitatory": top_exc,
        "nod1_is_top_excitatory": top_exc == "Nod1",
        "_byct_top25": byct.head(25).to_dict(orient="records"),
        "track": src.track,
    }
