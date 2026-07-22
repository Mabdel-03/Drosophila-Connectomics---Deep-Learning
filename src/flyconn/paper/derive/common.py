"""Shared derivation helpers used across families: T4/T5 subtype classification,
per-synapse NT argmax, partner aggregation, and lobula-plate layer fractions.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..oracle import consts as C

_CANON_RE = re.compile(r"^T[45][a-d]$")


def is_canonical_t4t5(cell_type: pd.Series) -> pd.Series:
    return cell_type.astype("string").str.match(_CANON_RE).fillna(False)


def subtype_letter(cell_type: pd.Series) -> pd.Series:
    """'T4a'->'a' for canonical T4/T5, else <NA>."""
    s = cell_type.astype("string")
    ok = s.str.match(_CANON_RE).fillna(False)
    return s.where(ok).str.slice(2, 3)


def argmax_nt(df: pd.DataFrame, cols=C.NT_COLS) -> pd.Series:
    """Per-synapse plurality neurotransmitter (raw-table column name, e.g. 'gaba')."""
    if df.empty:
        return pd.Series([], dtype="string")
    arr = df[list(cols)].to_numpy()
    idx = arr.argmax(axis=1)
    return pd.Series([cols[i] for i in idx], index=df.index, dtype="string")


def partner_counts(syn: pd.DataFrame, partner_col: str) -> pd.DataFrame:
    """Per partner root_id: synapse count. partner_col is the side that is NOT the anchor."""
    g = syn.groupby(partner_col).size().rename("syn").reset_index()
    return g.rename(columns={partner_col: "root_id"})


def attach_meta(counts: pd.DataFrame, meta) -> pd.DataFrame:
    """Join cell_type/side/super_class/nt_canonical onto a per-root_id table."""
    j = counts.join(meta.by_root, on="root_id")
    j["t4t5_subtype"] = subtype_letter(j["cell_type"])
    j["is_t4t5"] = j["t4t5_subtype"].notna()
    return j


def layer_fraction(t4t5_counts: pd.DataFrame, letter: str, weight: str = "syn") -> float:
    """Fraction of (synapse-weighted) T4/T5 drive that is one lobula-plate layer.

    ``letter`` is the subtype letter a/b/c/d (T4T5_LAYER); maps to a motion direction via
    ``consts.LAYER_DIRECTION`` (a=front_to_back, b=back_to_front, c=upward, d=downward).
    """
    t = t4t5_counts[t4t5_counts["is_t4t5"]]
    if t.empty:
        return float("nan")
    total = t[weight].sum()
    x = t.loc[t["t4t5_subtype"] == letter, weight].sum()
    return float(100.0 * x / total) if total else float("nan")


def layer_a_fraction(t4t5_counts: pd.DataFrame, weight: str = "syn") -> float:
    """Fraction of (synapse-weighted) T4/T5 drive that is layer-a (front-to-back)."""
    return layer_fraction(t4t5_counts, "a", weight)


def layer_b_fraction(t4t5_counts: pd.DataFrame, weight: str = "syn") -> float:
    """Fraction of (synapse-weighted) T4/T5 drive that is layer-b (back-to-front)."""
    return layer_fraction(t4t5_counts, "b", weight)


def layer_c_fraction(t4t5_counts: pd.DataFrame, weight: str = "syn") -> float:
    """Fraction of (synapse-weighted) T4/T5 drive that is layer-c (upward)."""
    return layer_fraction(t4t5_counts, "c", weight)


def dominant_layer(t4t5_counts: pd.DataFrame, weight: str = "syn") -> str | None:
    """The single lobula-plate layer letter (a/b/c/d) carrying the most T4/T5 drive."""
    t = t4t5_counts[t4t5_counts["is_t4t5"]]
    if t.empty:
        return None
    by = t.groupby("t4t5_subtype")[weight].sum()
    return str(by.idxmax()) if len(by) else None


def soma_summary(meta, roots, *, anchor_roots=None, anterior_roots=None) -> dict:
    """Soma-location summary for a set of cells (offline; no skeleton needed).

    Maps Egelhaaf's qualitative "cell body in the posterior lateral protocerebrum" onto
    measurable soma coordinates from neurons.parquet. Returns:
      centroid_xyz       mean soma (x,y,z) of ``roots``
      bilateral_split    True iff the cells' soma_x straddle the brain midline (one each side)
      post_z_percentile  percentile of the cells' mean soma_z within all visual_projection
                         somata (higher = more posterior)
      dz_to_anchor       |mean soma_z - anchor mean soma_z| (co-clustering with the FD1=Nod1 anchor)
      dz_to_anterior     mean soma_z - anterior-cells mean soma_z (posterior to the centrifugals)
    ``anchor_roots`` (e.g. Nod1) and ``anterior_roots`` (e.g. VCH/DCH) are optional.
    """
    cols = ["soma_x", "soma_y", "soma_z"]
    if any(c not in meta.df.columns for c in cols):
        return {"available": False, "reason": "soma coordinates not in metadata"}
    sub = meta.by_root.reindex(list(roots)).dropna(subset=cols)
    if len(sub) == 0:
        return {"available": False, "reason": "no soma coordinates for these roots"}
    centroid = sub[cols].mean().to_dict()
    midline = float(meta.df["soma_x"].median())
    sides_of_midline = (sub["soma_x"] > midline)
    bilateral_split = bool(len(sub) >= 2 and sides_of_midline.nunique() == 2)
    vp = meta.df[meta.df["super_class"] == "visual_projection"]
    zmean = float(sub["soma_z"].mean())
    post_z_percentile = float((vp["soma_z"] < zmean).mean()) if len(vp) else float("nan")
    out = {
        "available": True,
        "centroid_xyz": {k: round(float(v), 1) for k, v in centroid.items()},
        "bilateral_split": bilateral_split,
        "post_z_percentile": round(post_z_percentile, 3),
        "midline_x": round(midline, 1),
        "mean_soma_z": round(zmean, 1),
    }
    if anchor_roots is not None:
        a = meta.by_root.reindex(list(anchor_roots)).dropna(subset=["soma_z"])
        out["dz_to_anchor"] = round(abs(zmean - float(a["soma_z"].mean())), 1) if len(a) else None
    if anterior_roots is not None:
        ant = meta.by_root.reindex(list(anterior_roots)).dropna(subset=["soma_z"])
        out["dz_to_anterior"] = round(zmean - float(ant["soma_z"].mean()), 1) if len(ant) else None
    return out


# ---------------------------------------------------------------------------
# Contralateral inhibition stratified by lobula-plate layer (progressive vs regressive).
# ---------------------------------------------------------------------------
# Inhibitory tangential cell types whose preferred lobula-plate layer is known. The opponent
# pair that defines progressive vs regressive wide-field inhibition of the figure sheet
# (paper families B/F): VCH/DCH read layer-a (progressive, front->back); LPi15 reads layer-b
# (regressive, back->front). We classify a contralateral GABA partner by this table.
# Measured (v783): LPi14 is 94.8% layer-a (progressive; FD3's OPPONENT gate) and LPi12 is 99.6%
# layer-b (regressive; FD3's same-direction detector surround) -- see Family R.
INH_LAYER = {"VCH": "a", "DCH": "a", "CT1": "a", "LPi15": "b", "LPi14": "a", "LPi12": "b"}
MIN_CLASSIFIED_INH = 6
# A contra-inhibition channel counts as genuinely PRESENT (not a trace) only if it carries at
# least this fraction of the classified contra-inhibitory synapses. The bare ``syn>0`` test calls
# a cell "bidirectional" on a handful of minority synapses; the measured FD3 (7.1% regressive) and
# FD1 (8.3% progressive) both sit just above zero but are direction-DOMINATED, so the honest
# structure is opposite-dominance, not "FD3 bidirectional vs FD1 unidirectional".
MIN_LAYER_FRAC = 0.15


def contra_inhibition_profile(src, meta, cell_type: str) -> dict:
    """Stratify a cell type's contralateral GABAergic inputs by progressive vs regressive layer.

    FD3 receives BIDIRECTIONAL contralateral inhibition (both progressive- and regressive-tuned),
    whereas FD1=Nod1's contra inhibition is unidirectional (Egelhaaf 1985, FD3-Cell p.203-204).
    ``both_present`` = both a-layer and b-layer contralateral inhibitory channels carry
    non-trivial weight. ``sufficient`` guards against calling the direction from a handful of
    annotated partners (the power limit that keeps this claim from being able to REFUTE).

    Lifted verbatim from ``derive.k_fd3_lpt42._contra_inhibition`` so family K and family P
    (the FD3 input pathway) share one implementation; ``INH_LAYER``/``MIN_CLASSIFIED_INH``
    live here as the single source of truth.
    """
    roots = sorted(int(x) for x in meta.root_ids_of_type([cell_type]))
    if not roots:
        return {"available": False}
    rset = set(roots)
    syn = src.synapses(post_ids=roots)
    syn = syn[syn["post_pt_root_id"].isin(rset)]
    counts = attach_meta(partner_counts(syn, "pre_pt_root_id"), meta)
    # contralateral GABAergic partners
    post_side = meta.by_root.reindex(roots)["side"].iloc[0]
    gaba = counts[(counts["nt_canonical"] == "gaba") & (counts["side"].notna())
                  & (counts["side"] != post_side)].copy()
    gaba["inh_layer"] = gaba["cell_type"].map(INH_LAYER)
    classified = gaba.dropna(subset=["inh_layer"])
    syn_a = float(classified.loc[classified["inh_layer"] == "a", "syn"].sum())
    syn_b = float(classified.loc[classified["inh_layer"] == "b", "syn"].sum())
    tot = syn_a + syn_b
    frac_prog = round(syn_a / tot, 3) if tot else float("nan")
    frac_reg = round(syn_b / tot, 3) if tot else float("nan")
    n_classified = int(classified["root_id"].nunique())
    both = bool(syn_a > 0 and syn_b > 0)
    # honest structure: which direction dominates, and whether BOTH carry a non-trivial share.
    minority_frac = round(min(frac_prog, frac_reg), 3) if tot else float("nan")
    dominant_direction = ("progressive" if syn_a > syn_b else "regressive" if syn_b > syn_a
                          else "balanced") if tot else None
    substantially_bidirectional = bool(tot and minority_frac >= MIN_LAYER_FRAC)
    return {
        "available": True,
        "n_contra_gaba": int(gaba["root_id"].nunique()),
        "n_classified": n_classified,
        "syn_progressive": int(syn_a), "syn_regressive": int(syn_b),
        "frac_progressive": frac_prog, "frac_regressive": frac_reg,
        "both_present": both,
        "dominant_direction": dominant_direction,
        "minority_frac": minority_frac,
        "substantially_bidirectional": substantially_bidirectional,
        "sufficient": bool(n_classified >= MIN_CLASSIFIED_INH),
    }


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values (same order as input).

    NaNs are passed through unchanged and excluded from the ranking. Used to control the
    false-discovery rate across the p-valued claims of a family (the permutation and
    Wilcoxon tests); threshold/effect-size claims are NOT p-valued and are excluded by the
    caller to avoid double-counting.
    """
    arr = np.asarray(pvals, dtype=float)
    ok = np.where(~np.isnan(arr))[0]
    out = np.full(arr.shape, np.nan)
    if len(ok) == 0:
        return out.tolist()
    p = arr[ok]
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    adj = np.clip(adj, 0.0, 1.0)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out.tolist()
