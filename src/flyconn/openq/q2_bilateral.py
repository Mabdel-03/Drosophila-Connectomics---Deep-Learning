"""Q2 — BILATERAL / 4-DIRECTION GENERALIZATION of the figure-ground readout.

The paper analysed only the RIGHT LLPC1 sheet, driven through the front-to-back (layer-a)
T4a exemplar, gated by the LEFT VCH (centrifugal cells cross: a soma on one side gates the
optic lobe of the other). This module derives the structural parallels for the LEFT sheet
too — gated by the RIGHT-soma VCH/DCH — and reports the headline numbers SIDE-BY-SIDE so a
reader can see whether the right sheet (which should match the paper, a positive control)
and the left sheet behave the same way.

Per side it computes the cheap, robust STRUCTURAL parallels that decide generalisation
without a full retinotopy null:
  * VCH/DCH presence + in/out synapse totals (is there a gating centrifugal cell at all);
  * T4/T5 -> VCH input count + synapses, and the reciprocal (VCH -> the same T4/T5) count
    (the "gating loop" the paper builds on);
  * the T4a -> LLPC1 sheet size (how many same-side LLPC1 the gated T4a reach);
  * Nod1's rank/share of that side's LLPC1 output (does the Nod1-dominant readout recur).

The heavy spatial-locality (retinotopic pooling) null is marked UNVERIFIABLE-for-now with
a note: it needs the synapse-position retinotopy machinery (``paper.geometry`` /
``derive.d_retinotopy_null``) run per side, which is out of scope for this cheap structural
parallel. Queries are bounded: id sets come from ``NeuronMeta`` and ``fw.synapses`` is
called per id-set, hitting the source's own parquet cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..motif import compare as K

# Centrifugal-crossing map: which SOMA side gates which LLPC1 SHEET side. The paper's
# left-VCH gates the right sheet; symmetric for the other hemisphere.
GATING_SOMA_FOR_SHEET = {"right": "left", "left": "right"}

# T4a is the layer-a (front-to-back) exemplar channel the paper drove the sheet through.
EXEMPLAR_T4 = "T4a"


def _syn_totals(src, ids: np.ndarray) -> tuple[int, int]:
    """(total input synapses, total output synapses) for an id set, self-edges excluded."""
    id_set = set(int(x) for x in ids)
    if not id_set:
        return 0, 0
    out = src.synapses(pre_ids=ids.tolist())
    out = out[out["pre_pt_root_id"].isin(id_set)]
    inp = src.synapses(post_ids=ids.tolist())
    inp = inp[inp["post_pt_root_id"].isin(id_set)]
    return int(len(inp)), int(len(out))


def _t4t5_to_vch(src, meta, vch_ids: np.ndarray) -> dict:
    """T4/T5 -> VCH input synapses + partner count, and the reciprocal VCH -> same T4/T5.

    The paper's gating-loop headline (1,022 T4/T5 inputs / 12,301 syn, 912 reciprocal) is
    the same structure derived per side here, so each hemisphere's loop is comparable.
    """
    vch_set = set(int(x) for x in vch_ids)
    if not vch_set:
        return {"t4t5_partners": 0, "t4t5_syn": 0, "reciprocal_partners": 0}
    inp = src.synapses(post_ids=vch_ids.tolist())
    inp = inp[inp["post_pt_root_id"].isin(vch_set)]
    pm = meta.by_root.reindex(inp["pre_pt_root_id"].values)
    ct = pm["cell_type"].astype("string").values
    is_t = pd.Series(ct).str.startswith(("T4", "T5"), na=False).to_numpy()
    t4t5_in = inp[is_t]
    t4t5_partners = set(int(x) for x in t4t5_in["pre_pt_root_id"].unique())

    out = src.synapses(pre_ids=vch_ids.tolist())
    out = out[out["pre_pt_root_id"].isin(vch_set)]
    recip = set(int(x) for x in out["post_pt_root_id"].unique()) & t4t5_partners
    return {
        "t4t5_partners": int(len(t4t5_partners)),
        "t4t5_syn": int(len(t4t5_in)),
        "reciprocal_partners": int(len(recip)),
    }


def _sheet_and_nod1(src, meta, sheet_side: str, vch_ids: np.ndarray) -> dict:
    """T4a -> same-side LLPC1 sheet size + Nod1's share/rank of that sheet's output.

    The T4a are restricted to those postsynaptic to the gating (crossing) VCH, exactly as
    the paper defines the right sheet — so the left number is the homologous quantity.
    """
    # 1. VCH-gated same-side T4a (the exemplar channel).
    vch_set = set(int(x) for x in vch_ids)
    vch_out = src.synapses(pre_ids=vch_ids.tolist())
    vch_out = vch_out[vch_out["pre_pt_root_id"].isin(vch_set)]
    om = meta.by_root.reindex(vch_out["post_pt_root_id"].unique())
    t4a = om[(om["cell_type"] == EXEMPLAR_T4) & (om["side"] == sheet_side)].index.to_numpy()

    # 2. those T4a -> same-side LLPC1 (the sheet).
    t4a_set = set(int(x) for x in t4a)
    if not t4a_set:
        return {"n_t4a": 0, "n_llpc1_sheet": 0, "nod1_rank": None, "nod1_share_pct": 0.0,
                "nod1_is_top_excitatory": False}
    t4a_out = src.synapses(pre_ids=t4a.tolist())
    t4a_out = t4a_out[t4a_out["pre_pt_root_id"].isin(t4a_set)]
    pm = meta.by_root.reindex(t4a_out["post_pt_root_id"].values)
    is_sheet = (pm["cell_type"].values == "LLPC1") & (pm["side"].values == sheet_side)
    sheet_roots = sorted(set(int(x) for x in t4a_out["post_pt_root_id"].to_numpy()[is_sheet]))

    # 3. that sheet's output, ranked by target cell type; where does Nod1 sit?
    nod1 = _nod1_rank_of_sheet(src, meta, sheet_roots)
    return {"n_t4a": int(len(t4a_set)), "n_llpc1_sheet": int(len(sheet_roots)), **nod1}


def _nod1_rank_of_sheet(src, meta, sheet_roots: list[int]) -> dict:
    """Nod1's rank + synapse share among a sheet's output cell types (excl. lateral LLPC1)."""
    if not sheet_roots:
        return {"nod1_rank": None, "nod1_share_pct": 0.0, "nod1_is_top_excitatory": False}
    sheet_set = set(sheet_roots)
    out = src.synapses(pre_ids=list(sheet_roots))
    out = out[out["pre_pt_root_id"].isin(sheet_set)]
    pm = meta.by_root.reindex(out["post_pt_root_id"].values).reset_index(drop=True)
    df = pd.DataFrame({"ct": pm["cell_type"].values, "nt": pm["nt_canonical"].values})
    df = df[df["ct"].notna() & (df["ct"] != "LLPC1")]
    total = int(len(df))
    by_ct = df.groupby("ct").size().sort_values(ascending=False).reset_index(name="syn")
    by_ct["rank"] = range(1, len(by_ct) + 1)
    nod1_row = by_ct[by_ct["ct"] == "Nod1"]
    nod1_rank = int(nod1_row["rank"].iloc[0]) if len(nod1_row) else None
    nod1_syn = int(nod1_row["syn"].iloc[0]) if len(nod1_row) else 0
    # Dominant EXCITATORY (cholinergic) readout for the side (matches family-G definition).
    cho = df[df["nt"] == "acetylcholine"].groupby("ct").size().sort_values(ascending=False)
    top_exc = str(cho.index[0]) if len(cho) else None
    return {
        "nod1_rank": nod1_rank,
        "nod1_share_pct": round(100.0 * nod1_syn / total, 2) if total else 0.0,
        "nod1_is_top_excitatory": top_exc == "Nod1",
    }


def _run_side(src, meta, sheet_side: str) -> dict:
    """All structural parallels for one sheet side (the gating VCH is the crossing soma)."""
    gating_soma = GATING_SOMA_FOR_SHEET[sheet_side]
    vch = meta.root_ids_of_type(["VCH"], side=gating_soma)
    dch = meta.root_ids_of_type(["DCH"], side=gating_soma)
    vch_in, vch_out = _syn_totals(src, vch)
    loop = _t4t5_to_vch(src, meta, vch)
    sheet = _sheet_and_nod1(src, meta, sheet_side, vch)
    return {
        "sheet_side": sheet_side,
        "gating_soma_side": gating_soma,
        "gating_present": bool(len(vch) > 0),
        "n_vch": int(len(vch)), "n_dch": int(len(dch)),
        "vch_in_syn": vch_in, "vch_out_syn": vch_out,
        **loop, **sheet,
    }


def run_q2(source_fw, meta=None) -> dict:
    """Derive the figure-ground headline parallels for BOTH the right (paper) and left sheet.

    ``source_fw`` is a ``FlyWireSource`` (live CAVE primary or offline). ``meta`` may be a
    ``NeuronMeta``; if it is None or not a ``NeuronMeta`` (e.g. the CLI routes only the
    source), we load the shared annotation table ourselves -- the same way Q3 does -- so the
    runner does not have to supply it. Returns a per-side dict (right = positive control
    that should match the paper, left = the new generalisation), the ``ClaimResult``s
    comparing left-vs-right structure, and the heavy retinotopy piece flagged UNVERIFIABLE.
    """
    from ..paper.fw_access import NeuronMeta
    if not isinstance(meta, NeuronMeta):
        meta = NeuronMeta.load("783")
    right = _run_side(source_fw, meta, "right")
    left = _run_side(source_fw, meta, "left")

    def _recip_frac(d: dict) -> float:
        return d["reciprocal_partners"] / d["t4t5_partners"] if d["t4t5_partners"] else 0.0

    claims = [
        # Generalisation 1: a gating centrifugal cell exists for both sheets.
        K.compare_categorical(
            "Q2.gating_present", "VCH gating present for left sheet (as for right)",
            True, left["gating_present"],
            refuted_note="no RIGHT-soma VCH found to gate the left sheet"),
        # Generalisation 2: the T4/T5->VCH reciprocal loop fraction is hemisphere-symmetric.
        K.compare_pct(
            "Q2.reciprocal_frac", "T4/T5->VCH reciprocal fraction: left vs right",
            round(100.0 * _recip_frac(right), 1), round(100.0 * _recip_frac(left), 1),
            pp=10.0),
        # Generalisation 3: the left sheet exists at a comparable size to the right.
        K.compare_count(
            "Q2.sheet_size", "LLPC1 sheet size: left vs right (T4a-driven)",
            right["n_llpc1_sheet"], left["n_llpc1_sheet"], rel=0.30, abs_floor=10,
            drift_dir="down"),
        # Generalisation 4: Nod1 dominates the readout on the left too.
        K.compare_categorical(
            "Q2.nod1_dominant_left", "Nod1 is the top excitatory readout of the LEFT sheet",
            True, left["nod1_is_top_excitatory"],
            refuted_note=f"left-sheet Nod1 rank {left['nod1_rank']}, "
                         f"share {left['nod1_share_pct']}%"),
        # Heavy retinotopy / spatial-locality null: deferred.
        K.unverifiable(
            "Q2.retinotopy_left", "Left-sheet pooling is spatially local/retinotopic",
            "local", "needs per-side synapse-position retinotopy null "
                     "(paper.geometry / derive.d_retinotopy_null); deferred as too heavy "
                     "for the cheap structural parallel"),
    ]

    return {
        "question": "Q2_bilateral_generalization",
        "right": right,   # positive control — should match the paper
        "left": left,     # the new generalisation
        "claims": [c.to_dict() for c in claims],
        "track": source_fw.track,
    }


if __name__ == "__main__":  # standalone smoke test
    import json

    from ..paper.fw_access import NeuronMeta, make_source

    src = make_source("auto")
    md = NeuronMeta.load("783")
    print(json.dumps(run_q2(src, md), indent=2, default=str))
