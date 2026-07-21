"""Derive family A: VCH-T4/T5 reciprocal loop, from the live CAVE synapse pull.

Primary track = live CAVE (reproduces the paper exactly). The offline secondary track
is computed from the proofread ``edges_full.parquet`` as a cross-check.
"""

from __future__ import annotations

import pandas as pd

from . import common as CM
from ..oracle import consts as C


def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    vch = cfg.vch_root
    vch_in = src.synapses(post_ids=[vch])   # synapses ONTO the gating VCH
    vch_out = src.synapses(pre_ids=[vch])   # synapses FROM the gating VCH

    # Restrict each direction to the rows actually touching VCH on the anchor side
    # (the source may union pre+post pulls if both were requested; here each pull is
    # single-sided, but be defensive).
    vch_in = vch_in[vch_in["post_pt_root_id"] == vch]
    vch_out = vch_out[vch_out["pre_pt_root_id"] == vch]

    in_counts = CM.attach_meta(CM.partner_counts(vch_in, "pre_pt_root_id"), meta)
    out_counts = CM.attach_meta(CM.partner_counts(vch_out, "post_pt_root_id"), meta)

    in_t45 = in_counts[in_counts["is_t4t5"]]
    out_t45 = out_counts[out_counts["is_t4t5"]]

    in_neurons = int(len(in_t45))
    in_syn = int(in_t45["syn"].sum())
    out_neurons = int(len(out_t45))
    out_syn = int(out_t45["syn"].sum())

    recip_roots = set(in_t45["root_id"]) & set(out_t45["root_id"])
    recip_n = len(recip_roots)
    # layer-a reciprocal partners (same-direction front-to-back).
    recip_meta = in_t45[in_t45["root_id"].isin(recip_roots)]
    recip_layer_a_n = int((recip_meta["t4t5_subtype"] == "a").sum())

    total_in_syn = int(len(vch_in))
    total_out_syn = int(len(vch_out))
    vch_nt = str(meta.by_root.loc[vch, "nt_canonical"])

    in_mean = in_syn / in_neurons if in_neurons else float("nan")
    out_mean = out_syn / out_neurons if out_neurons else float("nan")

    d = {
        "vch_nt": "gaba" if vch_nt == "gaba" else vch_nt,
        "vch_input_syn": total_in_syn,
        "vch_output_syn": total_out_syn,
        "vch_upstream_partners": int(vch_in["pre_pt_root_id"].nunique()),
        "t4t5_in_neurons": in_neurons,
        "t4t5_in_syn": in_syn,
        "t4t5_in_pct_of_input": round(100.0 * in_syn / total_in_syn, 1) if total_in_syn else float("nan"),
        "t4t5_out_neurons": out_neurons,
        "t4t5_out_syn": out_syn,
        "reciprocal_n": recip_n,
        "reciprocal_pct_of_inputs": round(100.0 * recip_n / in_neurons, 1) if in_neurons else float("nan"),
        "reciprocal_pct_of_outputs": round(100.0 * recip_n / out_neurons, 1) if out_neurons else float("nan"),
        "reciprocal_layer_a_n": recip_layer_a_n,
        "vch_layer_a_pct": CM.layer_a_fraction(in_t45),
        "exc_inhib_ratio": round(in_mean / out_mean, 2) if out_mean else float("nan"),
        "in_mean_syn": round(in_mean, 2),
        "out_mean_syn": round(out_mean, 2),
        "track": src.track,
        # keep the reciprocal root set for downstream families (G uses it implicitly via LLPC1).
        "_reciprocal_roots": sorted(int(r) for r in recip_roots),
        "_in_subtype_table": _subtype_table(in_t45),
        "_out_subtype_table": _subtype_table(out_t45),
    }
    return d


def _subtype_table(t45: pd.DataFrame) -> list[dict]:
    g = t45.groupby("t4t5_subtype").agg(neurons=("root_id", "nunique"), syn=("syn", "sum"))
    g = g.sort_values("syn", ascending=False).reset_index()
    return g.to_dict(orient="records")
