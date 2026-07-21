"""Reusable, cell-set-agnostic connectivity census.

Factors out the ad-hoc partner aggregation that every family re-implements
(``p_fd3_input._input_census``, ``l_fd3_descending._dn_block``, ``k_fd3_lpt42._type_profile``,
``g_output_census``) into one object, so a *comprehensive* characterization of any cell's
connectivity is a single call. Nothing here is FD3-specific.

Given a set of anchor roots it returns, for each direction (input / output):
  * total synapses + distinct partners,
  * the FULL partition by super_class, by neurotransmitter, by partner laterality (soma side),
  * the T4/T5 lobula-plate layer composition (input side only),
  * the COMPLETE ranked partner and cell-type tables (not a top-N head),
plus a reciprocity block (which partners both send to AND receive from the anchor) and an
optional 2-hop loop-closure detector (anchor -> X -> anchor return paths).

Reuses ``common.partner_counts``/``attach_meta`` for aggregation and
``motif.vch_verify.derive_reciprocal`` for the reciprocal-set core; wraps the source in
``k_fd3_lpt42._CachedSource`` so input, output and reciprocity share one pull each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import common as CM
from .k_fd3_lpt42 import _CachedSource
from ..oracle import consts as C

# Live CAVE truncates a single synapse_query at this many rows; batch multi-hop pulls to stay
# clear (same guard the other families use).
_CAVE_ROW_CAP = 500_000
_SEED_BATCH = 12


@dataclass
class PartnerCensus:
    """One direction (input or output) of a cell set's connectivity, fully partitioned."""

    direction: str                       # "input" | "output"
    total_syn: int
    n_partners: int
    by_super_class: dict                  # syn per super_class, ranked
    by_nt: dict                           # syn per nt_canonical (partner side)
    by_laterality: dict                   # syn per partner soma side (left/right/center/unknown)
    by_layer: dict | None                 # T4/T5 a/b/c/d fractions (input only; None for output)
    ranked_types: list                    # [{cell_type, n_cells, syn, pct, cum_pct, super_class, nt}]
    ranked_partners_head: list            # top individual partner cells (bounded; full table is huge)
    n_unannotated_syn: int                # synapses onto partners with no cell_type

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "total_syn": self.total_syn,
            "n_partners": self.n_partners,
            "by_super_class": self.by_super_class,
            "by_nt": self.by_nt,
            "by_laterality": self.by_laterality,
            "by_layer": self.by_layer,
            "ranked_types": self.ranked_types,
            "ranked_partners_head": self.ranked_partners_head,
            "n_unannotated_syn": self.n_unannotated_syn,
        }


def _anchor_col(direction: str) -> tuple[str, str]:
    """(anchor_id_col, partner_id_col) for a direction. input: partner is presynaptic."""
    if direction == "input":
        return "post_pt_root_id", "pre_pt_root_id"
    if direction == "output":
        return "pre_pt_root_id", "post_pt_root_id"
    raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")


def partner_census(src, meta, roots, direction: str, *, head: int = 25) -> PartnerCensus:
    """Full connectivity census in one direction for a set of anchor ``roots``."""
    roots = [int(x) for x in roots]
    rset = set(roots)
    anchor_col, partner_col = _anchor_col(direction)
    syn = src.synapses(post_ids=roots) if direction == "input" else src.synapses(pre_ids=roots)
    syn = syn[syn[anchor_col].isin(rset)]
    total_syn = int(len(syn))
    counts = CM.attach_meta(CM.partner_counts(syn, partner_col), meta)
    n_partners = int(counts["root_id"].nunique())

    # unannotated (no cell_type) synapse mass — reported so the partition is honest/complete.
    ann = counts.dropna(subset=["cell_type"])
    n_unannotated = int(counts.loc[counts["cell_type"].isna(), "syn"].sum())

    by_sc = {str(k): int(v) for k, v in
             counts.groupby("super_class", dropna=False)["syn"].sum().sort_values(ascending=False).items()
             if pd.notna(k)}
    by_nt = {str(k): int(v) for k, v in
             counts.groupby("nt_canonical", dropna=False)["syn"].sum().sort_values(ascending=False).items()
             if pd.notna(k)}
    lat = counts.copy()
    lat["_side"] = lat["side"].astype(object).where(lat["side"].notna(), "unknown")
    by_lat = {str(k): int(v) for k, v in lat.groupby("_side")["syn"].sum().sort_values(ascending=False).items()}

    by_layer = None
    if direction == "input":
        t45 = counts[counts["is_t4t5"]]
        by_layer = {L: (round(CM.layer_fraction(t45, L), 2) if len(t45) else None) for L in "abcd"}

    # FULL ranked cell-type table with cumulative coverage.
    grp = (ann.groupby("cell_type")
           .agg(n_cells=("root_id", "nunique"), syn=("syn", "sum"),
                super_class=("super_class", "first"), nt=("nt_canonical", "first"))
           .reset_index().sort_values("syn", ascending=False))
    denom = max(total_syn, 1)
    grp["pct"] = (100.0 * grp["syn"] / denom).round(2)
    grp["cum_pct"] = grp["pct"].cumsum().round(2)
    ranked_types = [{"cell_type": str(r.cell_type), "n_cells": int(r.n_cells), "syn": int(r.syn),
                     "pct": float(r.pct), "cum_pct": float(r.cum_pct),
                     "super_class": (None if pd.isna(r.super_class) else str(r.super_class)),
                     "nt": (None if pd.isna(r.nt) else str(r.nt))}
                    for r in grp.itertuples(index=False)]

    top = counts.sort_values("syn", ascending=False).head(head)
    ranked_partners_head = [{"root_id": int(r.root_id), "syn": int(r.syn),
                             "cell_type": (None if pd.isna(r.cell_type) else str(r.cell_type)),
                             "side": (None if pd.isna(r.side) else str(r.side)),
                             "super_class": (None if pd.isna(r.super_class) else str(r.super_class)),
                             "nt": (None if pd.isna(r.nt_canonical) else str(r.nt_canonical))}
                            for r in top.itertuples(index=False)]

    return PartnerCensus(
        direction=direction, total_syn=total_syn, n_partners=n_partners,
        by_super_class=by_sc, by_nt=by_nt, by_laterality=by_lat, by_layer=by_layer,
        ranked_types=ranked_types, ranked_partners_head=ranked_partners_head,
        n_unannotated_syn=n_unannotated)


def reciprocity(src, meta, roots, *, min_syn: int = 1, top: int = 20) -> dict:
    """Which partners both send to AND receive from the anchor set, with synapse weights.

    Uses ``motif.vch_verify.derive_reciprocal`` for the partner-identity intersection, then
    attaches in/out synapse weights + metadata (that helper returns bare sets).
    """
    roots = [int(x) for x in roots]
    rset = set(roots)
    syn_in = src.synapses(post_ids=roots); syn_in = syn_in[syn_in["post_pt_root_id"].isin(rset)]
    syn_out = src.synapses(pre_ids=roots); syn_out = syn_out[syn_out["pre_pt_root_id"].isin(rset)]
    in_counts = CM.partner_counts(syn_in, "pre_pt_root_id").set_index("root_id")["syn"]
    out_counts = CM.partner_counts(syn_out, "post_pt_root_id").set_index("root_id")["syn"]
    in_counts = in_counts[in_counts >= min_syn]
    out_counts = out_counts[out_counts >= min_syn]

    from ...motif.vch_verify import derive_reciprocal
    summary, recip = derive_reciprocal(set(int(x) for x in in_counts.index),
                                       set(int(x) for x in out_counts.index))
    rows = []
    for r in recip:
        rows.append({"root_id": int(r),
                     "in_syn": int(in_counts.get(int(r), 0)),
                     "out_syn": int(out_counts.get(int(r), 0))})
    rows.sort(key=lambda d: d["in_syn"] + d["out_syn"], reverse=True)
    top_rows = CM.attach_meta(pd.DataFrame(rows[:top]) if rows else
                              pd.DataFrame(columns=["root_id", "in_syn", "out_syn"]), meta)
    top_reciprocal = [{"root_id": int(r.root_id), "in_syn": int(r.in_syn), "out_syn": int(r.out_syn),
                       "cell_type": (None if pd.isna(r.cell_type) else str(r.cell_type)),
                       "nt": (None if pd.isna(r.nt_canonical) else str(r.nt_canonical))}
                      for r in top_rows.itertuples(index=False)] if len(top_rows) else []
    return {
        "n_reciprocal": int(summary["n"]),
        "recip_pct_of_inputs": float(summary["pct_of_inputs"]),
        "recip_pct_of_outputs": float(summary["pct_of_outputs"]),
        "n_input_partners": int(len(in_counts)),
        "n_output_partners": int(len(out_counts)),
        "top_reciprocal": top_reciprocal,
    }


def loop_closure(src, meta, roots, *, min_syn: int = 30, top: int = 15) -> dict:
    """2-hop return loops: anchor -> X -> anchor. X = the anchor's strongest output partners;
    for each we check whether it sends back to the anchor. Batched under the 500k row cap.

    Reports the strongest intermediary types on a 2-hop closed loop (e.g. FD3 -> LPi-relay ->
    back onto FD3's drivers). Multi-hop pull batched via the ``_second_hop`` idiom.
    """
    roots = [int(x) for x in roots]
    rset = set(roots)
    out = src.synapses(pre_ids=roots); out = out[out["pre_pt_root_id"].isin(rset)]
    seeds = CM.attach_meta(CM.partner_counts(out, "post_pt_root_id"), meta)
    seeds = seeds[(seeds["syn"] >= min_syn) & (seeds["cell_type"].notna())]
    seed_roots = [int(x) for x in seeds["root_id"]]
    if not seed_roots:
        return {"n_loop_types": 0, "loops": [], "query_truncated": False}
    seed_set = set(seed_roots)
    parts, truncated = [], False
    for i in range(0, len(seed_roots), _SEED_BATCH):
        batch = seed_roots[i:i + _SEED_BATCH]
        hop = src.synapses(pre_ids=batch)
        hop = hop[hop["pre_pt_root_id"].isin(seed_set) & hop["post_pt_root_id"].isin(rset)]
        if len(hop) >= _CAVE_ROW_CAP:
            truncated = True
        parts.append(hop.groupby("pre_pt_root_id").size().rename("back_syn").reset_index())
    back = (pd.concat(parts, ignore_index=True) if parts else
            pd.DataFrame(columns=["pre_pt_root_id", "back_syn"]))
    back = back.rename(columns={"pre_pt_root_id": "root_id"})
    merged = seeds.merge(back, on="root_id", how="inner")
    by = (merged.groupby("cell_type")
          .agg(fwd_syn=("syn", "sum"), back_syn=("back_syn", "sum"), n=("root_id", "nunique"))
          .reset_index().sort_values("back_syn", ascending=False).head(top))
    loops = [{"cell_type": str(r.cell_type), "fwd_syn": int(r.fwd_syn),
              "back_syn": int(r.back_syn), "n_cells": int(r.n)}
             for r in by.itertuples(index=False)]
    return {"n_loop_types": int(len(loops)), "loops": loops, "query_truncated": bool(truncated)}


def full_census(src, meta, roots) -> dict:
    """The complete two-direction census + reciprocity for an anchor set.

    Wraps ``src`` in ``_CachedSource`` so input/output/reciprocity reuse one pull each.
    """
    roots = [int(x) for x in roots]
    src = _CachedSource(src)
    inp = partner_census(src, meta, roots, "input")
    out = partner_census(src, meta, roots, "output")
    recip = reciprocity(src, meta, roots)
    return {
        "roots": roots,
        "input": inp.to_dict(),
        "output": out.to_dict(),
        "reciprocity": recip,
        "track": getattr(src, "track", None),
    }
