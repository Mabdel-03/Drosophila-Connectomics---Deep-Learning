"""Stage B: re-derive VCH's claims from the compact VCH-synapse parquet.

All inputs here are small (VCH touches ~37k synapses), so this is plain pandas and
runs in well under a second. The heavy 9.5 GB read happened in Stage A.

Conventions:
  - "primary" track = the raw synapse table (vch_synapses.parquet); the API analog.
  - "secondary" track = the repo's proofread-only derived graph (edges_full.parquet).
  - T4/T5 membership uses a STRICT canonical rule (the 8 subtypes) by default; the
    report's LOOSE substring rule is also computed to quantify its contamination.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import vch_config as C

_CANON_RE = re.compile(r"^T[45][a-d]$")
_UNLETTERED_RE = re.compile(r"^T[45]$")
_LOOSE_RE = re.compile(r"T4|T5")


# ---------------------------------------------------------------------------
# Cell-type classification
# ---------------------------------------------------------------------------
def classify_t4t5(cell_type: pd.Series) -> pd.Series:
    """Three-tier label for each cell_type string:

      one of T4a..T5d  -> the canonical subtype
      'Other/unclear'  -> bare 'T4'/'T5' (subtype-ambiguous true T4/T5)
      'contamination'  -> matches loose 'T4'/'T5' substring but is not the above
                          (e.g. MLt4, LT52, M_lv2PN9t49b)
      'non_t4t5'       -> everything else (incl. NaN)
    """
    s = cell_type.astype("string")

    def label(x: object) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
            return "non_t4t5"
        xs = str(x)
        if _CANON_RE.match(xs):
            return xs
        if _UNLETTERED_RE.match(xs):
            return "Other/unclear"
        if _LOOSE_RE.search(xs):
            return "contamination"
        return "non_t4t5"

    return s.map(label).astype("string")


def is_canonical_t4t5(tier: pd.Series) -> pd.Series:
    return tier.isin(C.CANONICAL_T4T5)


def is_loose_t4t5(tier: pd.Series) -> pd.Series:
    """The report's rule: canonical + Other/unclear + contamination."""
    return tier.isin(list(C.CANONICAL_T4T5) + ["Other/unclear", "contamination"])


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
def load_lookup(neurons: pd.DataFrame) -> pd.DataFrame:
    """root_id-indexed view with the columns we join onto synapse partners."""
    cols = ["root_id", "cell_type", "side", "nt_canonical", "super_class"]
    lk = neurons[cols].copy()
    return lk.set_index("root_id")


def _attach(df: pd.DataFrame, partner_col: str, lookup: pd.DataFrame) -> pd.DataFrame:
    """Join partner cell_type/side/nt/superclass + the three-tier T4/T5 label."""
    joined = df.join(lookup, on=partner_col)
    joined["t4t5_tier"] = classify_t4t5(joined["cell_type"])
    return joined


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------
def split_in_out(vch_syn: pd.DataFrame, vch_root: int = C.VCH_ROOT):
    """Return (inputs, outputs): synapses where VCH is post / pre respectively."""
    ins = vch_syn[vch_syn["post_pt_root_id"] == vch_root].copy()
    outs = vch_syn[vch_syn["pre_pt_root_id"] == vch_root].copy()
    return ins, outs


def derive_overview(ins: pd.DataFrame, outs: pd.DataFrame, vch_nt: str) -> dict:
    return {
        "nt": vch_nt,
        "total_input_syn": int(len(ins)),
        "total_output_syn": int(len(outs)),
        "upstream_partners": int(ins["pre_pt_root_id"].nunique()),
        "downstream_partners": int(outs["post_pt_root_id"].nunique()),
    }


def _subtype_table(df: pd.DataFrame, partner_col: str, total_syn: int) -> pd.DataFrame:
    """neurons / synapses / mean per canonical subtype (+ Other/unclear), % of total."""
    can = df[is_loose_t4t5(df["t4t5_tier"])].copy()
    # Fold contamination into Other/unclear for the report-comparable table, but keep
    # canonical subtypes distinct.
    can["subtype"] = can["t4t5_tier"].where(
        is_canonical_t4t5(can["t4t5_tier"]), other="Other/unclear"
    )
    g = can.groupby("subtype")
    tbl = pd.DataFrame({
        "neurons": g[partner_col].nunique(),
        "syn": g.size(),
    })
    tbl["mean_syn_per_neuron"] = (tbl["syn"] / tbl["neurons"]).round(2)
    tbl["pct_of_vch_total"] = (100.0 * tbl["syn"] / total_syn).round(2)
    return tbl.sort_values("syn", ascending=False)


def derive_t4t5_inputs(ins: pd.DataFrame, lookup: pd.DataFrame, total_input_syn: int):
    j = _attach(ins, "pre_pt_root_id", lookup)
    canon = j[is_canonical_t4t5(j["t4t5_tier"])]
    loose = j[is_loose_t4t5(j["t4t5_tier"])]
    summary = {
        "total_neurons_canonical": int(canon["pre_pt_root_id"].nunique()),
        "total_neurons_loose": int(loose["pre_pt_root_id"].nunique()),
        "total_syn_canonical": int(len(canon)),
        "total_syn_loose": int(len(loose)),
        "pct_of_input": round(100.0 * len(loose) / max(total_input_syn, 1), 2),
        "mean_syn_per_neuron": round(len(loose) / max(loose["pre_pt_root_id"].nunique(), 1), 2),
    }
    table = _subtype_table(j, "pre_pt_root_id", total_input_syn)
    in_roots = set(loose["pre_pt_root_id"].unique())
    return summary, table, in_roots


def derive_t4t5_outputs(outs: pd.DataFrame, lookup: pd.DataFrame, total_output_syn: int):
    j = _attach(outs, "post_pt_root_id", lookup)
    canon = j[is_canonical_t4t5(j["t4t5_tier"])]
    loose = j[is_loose_t4t5(j["t4t5_tier"])]
    summary = {
        "total_neurons_canonical": int(canon["post_pt_root_id"].nunique()),
        "total_neurons_loose": int(loose["post_pt_root_id"].nunique()),
        "total_syn_canonical": int(len(canon)),
        "total_syn_loose": int(len(loose)),
        "pct_of_output": round(100.0 * len(loose) / max(total_output_syn, 1), 2),
        "mean_syn_per_neuron": round(len(loose) / max(loose["post_pt_root_id"].nunique(), 1), 2),
    }
    table = _subtype_table(j, "post_pt_root_id", total_output_syn)
    out_roots = set(loose["post_pt_root_id"].unique())
    return summary, table, out_roots


def derive_reciprocal(in_roots: set[int], out_roots: set[int]) -> tuple[dict, np.ndarray]:
    recip = sorted(in_roots & out_roots)
    n = len(recip)
    summary = {
        "n": n,
        "pct_of_inputs": round(100.0 * n / max(len(in_roots), 1), 2),
        "pct_of_outputs": round(100.0 * n / max(len(out_roots), 1), 2),
    }
    return summary, np.array(recip, dtype=np.int64)


def _hemisphere_of_neuropil(npil: pd.Series) -> pd.Series:
    """Map a neuropil string (e.g. 'LOP_R') to 'R' / 'L' / 'C'."""
    s = npil.astype("string").fillna("")
    out = pd.Series("C", index=s.index, dtype="string")
    out[s.str.endswith("_R")] = "R"
    out[s.str.endswith("_L")] = "L"
    return out


def derive_hemisphere(ins: pd.DataFrame, outs: pd.DataFrame, lookup: pd.DataFrame) -> dict:
    """Two independent laterality tests for the ipsi/contra claim."""
    ji = _attach(ins, "pre_pt_root_id", lookup)
    jo = _attach(outs, "post_pt_root_id", lookup)
    # (a) partner side (canonical T4/T5 only).
    in_side = ji[is_canonical_t4t5(ji["t4t5_tier"])]["side"].value_counts().to_dict()
    out_side = jo[is_canonical_t4t5(jo["t4t5_tier"])]["side"].value_counts().to_dict()
    # (b) per-synapse neuropil hemisphere (all VCH synapses).
    in_npil = _hemisphere_of_neuropil(ins["neuropil"]).value_counts().to_dict()
    out_npil = _hemisphere_of_neuropil(outs["neuropil"]).value_counts().to_dict()
    # (b') medio-lateral coordinate of the synapse ON VCH (input: post side; output: pre side).
    in_x = ins["post_pt_position_x"]
    out_x = outs["pre_pt_position_x"]

    def _dominant(d: dict) -> str:
        d = {k: v for k, v in d.items() if k in ("L", "R", "left", "right")}
        return max(d, key=d.get) if d else "?"

    in_hemi = _dominant(in_npil)
    out_hemi = _dominant(out_npil)
    return {
        "input_partner_side": {str(k): int(v) for k, v in in_side.items()},
        "output_partner_side": {str(k): int(v) for k, v in out_side.items()},
        "input_synapse_neuropil_hemi": {str(k): int(v) for k, v in in_npil.items()},
        "output_synapse_neuropil_hemi": {str(k): int(v) for k, v in out_npil.items()},
        "input_synapse_x_median": float(np.median(in_x)) if len(in_x) else None,
        "output_synapse_x_median": float(np.median(out_x)) if len(out_x) else None,
        "input_dominant_hemi": in_hemi,
        "output_dominant_hemi": out_hemi,
        "input_output_differ": in_hemi != out_hemi,  # the report's falsifiable claim
        "top_input_neuropils": _top_neuropils(ins),
        "top_output_neuropils": _top_neuropils(outs),
    }


def _top_neuropils(df: pd.DataFrame, k: int = 5) -> dict:
    return {str(kk): int(vv) for kk, vv in
            df["neuropil"].value_counts().head(k).items()}


def derive_nt(outs: pd.DataFrame, ins: pd.DataFrame, lookup: pd.DataFrame) -> dict:
    """NT verification at TWO levels: neuron annotation and per-synapse argmax.

    The report's sign claim (VCH=GABA-, T4/T5=ACh+) is fundamentally about neuron
    *identity*. The neuron-level annotation is the primary check; the per-synapse NT
    probabilities are a deeper, noisier corroboration. The sign-determining test is
    whether the expected NT is the *plurality* (most common) prediction.
    """
    nt_cols = list(C.SYN_NT_COLS)

    def argmax_nt(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series([], dtype="string")
        arr = df[nt_cols].to_numpy()
        idx = arr.argmax(axis=1)
        return pd.Series([nt_cols[i] for i in idx], index=df.index, dtype="string")

    out_nt = argmax_nt(outs)
    out_frac_gaba = float((out_nt == "gaba").mean()) if len(out_nt) else None

    ji = _attach(ins, "pre_pt_root_id", lookup)
    t45_in = ji[is_canonical_t4t5(ji["t4t5_tier"])]
    in_nt = argmax_nt(t45_in)
    in_frac_ach = float((in_nt == "ach").mean()) if len(in_nt) else None
    # Neuron-level: fraction of canonical T4/T5 input neurons annotated cholinergic.
    in_neurons = t45_in.drop_duplicates("pre_pt_root_id")
    in_neuron_frac_ach = (
        float((in_neurons["nt_canonical"] == "acetylcholine").mean())
        if len(in_neurons) else None
    )

    def plurality(s: pd.Series) -> str:
        return s.value_counts().idxmax() if len(s) else "?"

    return {
        "vch_output_synapse_nt": {k: int(v) for k, v in out_nt.value_counts().items()},
        "vch_output_frac_gaba": out_frac_gaba,
        "vch_output_plurality_nt": plurality(out_nt),
        "t4t5_input_synapse_nt": {k: int(v) for k, v in in_nt.value_counts().items()},
        "t4t5_input_frac_ach": in_frac_ach,
        "t4t5_input_plurality_nt": plurality(in_nt),
        "t4t5_input_neuron_frac_ach": in_neuron_frac_ach,
    }


def threshold_sweep(ins: pd.DataFrame, outs: pd.DataFrame,
                    floors=(0, 50, 100, 140)) -> dict:
    """VCH in/out synapse totals at various cleft_score floors (attribution).

    If the report's 27,576 / 32,363 are recovered at a lower floor, the gap is a
    threshold difference; if not recovered at any floor, it is genuine snapshot drift.
    """
    out = {}
    for f in floors:
        out[str(f)] = {
            "input_syn": int((ins["cleft_score"] >= f).sum()),
            "output_syn": int((outs["cleft_score"] >= f).sum()),
        }
    out["cleft_score_min_observed"] = {
        "input": int(ins["cleft_score"].min()) if len(ins) else None,
        "output": int(outs["cleft_score"].min()) if len(outs) else None,
    }
    return out


# ---------------------------------------------------------------------------
# Secondary track (proofread-only derived graph)
# ---------------------------------------------------------------------------
def secondary_crosscheck(edges_full: pd.DataFrame, neurons: pd.DataFrame,
                         vch_root: int = C.VCH_ROOT) -> dict:
    """Recompute the headline counts from the repo's proofread edges_full graph."""
    vch_idx = int(neurons.loc[neurons["root_id"] == vch_root, "idx"].iloc[0])
    idx2ct = neurons.set_index("idx")["cell_type"]

    ins = edges_full[edges_full["post_idx"] == vch_idx]
    outs = edges_full[edges_full["pre_idx"] == vch_idx]
    in_tier = classify_t4t5(idx2ct.reindex(ins["pre_idx"]).reset_index(drop=True))
    out_tier = classify_t4t5(idx2ct.reindex(outs["post_idx"]).reset_index(drop=True))
    in_t45 = ins[is_loose_t4t5(in_tier).to_numpy()]
    out_t45 = outs[is_loose_t4t5(out_tier).to_numpy()]
    in_roots = set(neurons.set_index("idx").loc[in_t45["pre_idx"], "root_id"])
    out_roots = set(neurons.set_index("idx").loc[out_t45["post_idx"], "root_id"])
    return {
        "total_input_syn": int(ins["syn_count"].sum()),
        "total_output_syn": int(outs["syn_count"].sum()),
        "upstream_partners": int(ins["pre_idx"].nunique()),
        "downstream_partners": int(outs["post_idx"].nunique()),
        "t4t5_in_neurons": int(in_t45["pre_idx"].nunique()),
        "t4t5_in_syn": int(in_t45["syn_count"].sum()),
        "t4t5_out_neurons": int(out_t45["post_idx"].nunique()),
        "t4t5_out_syn": int(out_t45["syn_count"].sum()),
        "reciprocal": int(len(in_roots & out_roots)),
    }
