"""Stage C: aggregate the downstream targets of the 918 reciprocal T4/T5.

Input is ``recip_downstream_synapses.parquet`` — every synapse whose presynaptic
neuron is one of the reciprocal T4/T5 (Stage A's second pass). The report claims this
population outputs 578,238 synapses onto 144,796 unique targets, with LPi14 the top
sink (Table 4) and various cell-type populations (Section 7.4) / Nodulus neurons
(Table 6) prominent.

Key handling: the report's "(copy N)" rows are distinct neurons (root_ids) that share
a cell_type, so the top-20 is per-root_id and matched on (cell_type, syn, drivers),
never on the cosmetic label. Population rows count distinct root_ids per cell_type.
"""

from __future__ import annotations

import re

import pandas as pd

from . import vch_config as C

_NOD_RE = re.compile(r"^Nod[0-9]$")


def aggregate_targets(recip_syn: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Per downstream target: synapse count + number of distinct T4/T5 drivers."""
    g = recip_syn.groupby("post_pt_root_id")
    targets = pd.DataFrame({
        "syn": g.size(),
        "n_drivers": g["pre_pt_root_id"].nunique(),
    }).reset_index().rename(columns={"post_pt_root_id": "root_id"})
    targets = targets.join(lookup, on="root_id")
    return targets.sort_values("syn", ascending=False).reset_index(drop=True)


def global_stats(recip_syn: pd.DataFrame, targets: pd.DataFrame) -> dict:
    return {
        "total_syn": int(len(recip_syn)),
        "unique_targets": int(targets.shape[0]),
        "n_drivers_used": int(recip_syn["pre_pt_root_id"].nunique()),
    }


def top20_individual(targets: pd.DataFrame) -> pd.DataFrame:
    return targets.head(20)[
        ["root_id", "cell_type", "syn", "n_drivers", "nt_canonical", "super_class"]
    ].reset_index(drop=True)


def celltype_populations(targets: pd.DataFrame) -> pd.DataFrame:
    """copies = distinct neurons (root_ids) per cell_type; total syn & drivers."""
    g = targets.dropna(subset=["cell_type"]).groupby("cell_type")
    pops = pd.DataFrame({
        "copies": g["root_id"].size(),
        "total_syn": g["syn"].sum(),
        "total_drivers": g["n_drivers"].sum(),
    }).reset_index()
    return pops.sort_values("total_syn", ascending=False).reset_index(drop=True)


def nodulus_subtable(targets: pd.DataFrame) -> pd.DataFrame:
    ct = targets["cell_type"].astype("string")
    nod = targets[ct.fillna("").str.match(_NOD_RE)].copy()
    g = nod.groupby("cell_type")
    sub = pd.DataFrame({
        "copies": g["root_id"].size(),
        "total_syn": g["syn"].sum(),
        "total_drivers": g["n_drivers"].sum(),
        "nt": g["nt_canonical"].first(),
    }).reset_index()
    return sub.sort_values("total_syn", ascending=False).reset_index(drop=True)


def annotation_completeness(targets: pd.DataFrame) -> dict:
    """Fraction of downstream targets with a non-null cell_type, split by side.

    The report's "72% left / 97% right tagged" claim has no stated method; this is the
    closest computable proxy. Targets absent from neurons.parquet (non-proofread) have
    NaN side AND NaN cell_type, so they are counted as a separate 'unannotated' bucket.
    """
    have_ct = targets["cell_type"].notna()
    out = {}
    for side in ("left", "right", "center"):
        m = targets["side"] == side
        n = int(m.sum())
        out[side] = {
            "n_targets": n,
            "pct_with_cell_type": float(round(100.0 * (m & have_ct).sum() / n, 1)) if n else None,
        }
    out["unannotated_targets"] = int(targets["side"].isna().sum())
    return out


def vch_self_target(targets: pd.DataFrame, vch_root: int = C.VCH_ROOT) -> dict:
    """Is VCH among its T4/T5's downstream targets, and which VCH neuron(s)?"""
    vch_rows = targets[targets["cell_type"] == "VCH"]
    return {
        "vch_rows": [
            {"root_id": int(r.root_id), "side": str(r.side), "syn": int(r.syn),
             "n_drivers": int(r.n_drivers), "is_left_vch": int(r.root_id) == vch_root}
            for r in vch_rows.itertuples()
        ],
    }
