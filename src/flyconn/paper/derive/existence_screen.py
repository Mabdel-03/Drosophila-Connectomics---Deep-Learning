"""Stage-8 Phase 0.5 — LEFT-mirror EXISTENCE SCREEN (cheap go/no-go).

Existence of a left figure-ground mirror is the OPEN QUESTION, not a premise. Before
committing to the full A'-K' + heavy nulls, this screen walks the 7 circuit stages on the
left at low cost and labels each PRESENT / WEAK / ABSENT, so we (a) get an early go/no-go and
(b) can return a true negative ("the mirror breaks at stage X") if a core stage is missing.

The 7 stages (the right circuit's spine):
  1. entry_gate     a crossing (right-soma) VCH/DCH with a real T4/T5 reciprocal loop
  2. gated_sheet    VCH-gated left T4a reach a non-trivial left LLPC1 sheet (floor >= 30)
  3. readout        Nod1 is the top excitatory (cholinergic) readout of the left sheet
  4. direction      the left sheet is layer-a (progressive) driven; LPi15 opponent layer-b
  5. downstream     left Nod1 reaches a DNp26 (the steering-command path exists)
  6. crossing       left Nod1 output is majority contralateral (noduli-group projection)
  7. retinotopy     cheap proxy: per-LLPC1 T4a in-degree is BOUNDED (not field-wide) —
                    the full permutation null is family D' (Phase 2)

Floors guard against the proofreading-completeness confound silently upgrading "absent" to
"present": a near-zero left value or a qualitatively different wiring is ABSENT, not "drift".

Compares the left result to the right (positive control) computed the same way. Reuses
``derive.sheet.get_sheet(cfg=LEFT/RIGHT)``, ``common.layer_fraction`` and the q2 helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as CM
from . import sheet as SH
from ..oracle import consts as C

# Stage floors (the "structure is really there" thresholds).
SHEET_FLOOR_LLPC1 = 30          # left sheet must reach >= this many LLPC1 to be PRESENT
RECIP_FRAC_FLOOR = 0.50         # T4/T5->VCH reciprocal fraction floor for the entry loop
NOD1_SHARE_FLOOR_PCT = 1.0      # Nod1 must carry a non-trivial share of sheet output
CONTRA_FLOOR_PCT = 60.0         # Nod1 majority-contralateral floor
LAYER_A_FLOOR_PCT = 50.0        # sheet drive must be majority layer-a
RETINO_INDEG_MAX = 60          # per-LLPC1 distinct-T4a in-degree below this = bounded (proxy)

PRESENT, WEAK, ABSENT = "PRESENT", "WEAK", "ABSENT"


def _vch_loop(src, meta, cfg: C.SideConfig) -> dict:
    """VCH in/out totals + T4/T5 reciprocal loop for the gating (crossing) VCH of this side."""
    vch = cfg.vch_root
    inp = src.synapses(post_ids=[vch]); inp = inp[inp["post_pt_root_id"] == vch]
    out = src.synapses(pre_ids=[vch]); out = out[out["pre_pt_root_id"] == vch]
    in_counts = CM.attach_meta(CM.partner_counts(inp, "pre_pt_root_id"), meta)
    out_counts = CM.attach_meta(CM.partner_counts(out, "post_pt_root_id"), meta)
    in_t = in_counts[in_counts["is_t4t5"]]
    out_t = out_counts[out_counts["is_t4t5"]]
    recip = set(in_t["root_id"]) & set(out_t["root_id"])
    n_in = int(len(in_t))
    return {
        "vch_root": vch,
        "vch_in_syn": int(len(inp)), "vch_out_syn": int(len(out)),
        "t4t5_partners": n_in, "t4t5_syn": int(in_t["syn"].sum()),
        "reciprocal_partners": int(len(recip)),
        "reciprocal_frac": round(len(recip) / n_in, 3) if n_in else 0.0,
        "vch_nt": str(meta.by_root.loc[vch, "nt_canonical"]),
    }


def _sheet_readout(src, meta, cfg: C.SideConfig) -> dict:
    """Sheet size + Nod1 dominance + layer-a drive fraction for one side."""
    sheet = SH.get_sheet(src, meta, cfg)
    sheet_roots = sheet.llpc1_roots
    # layer composition of the sheet's T4a drive (from the T4a->LLPC1 synapses, pre = T4a).
    t4a_in = CM.attach_meta(CM.partner_counts(sheet.t4a_llpc1_syn, "pre_pt_root_id"), meta)
    layer_a = CM.layer_fraction(t4a_in, "a")
    # sheet output -> top excitatory readout + Nod1 rank/share
    nod = _nod1_of_sheet(src, meta, sheet_roots)
    return {
        "n_t4a": sheet.n_t4a, "n_llpc1": sheet.n_llpc1,
        "n_t4a_llpc1_syn": sheet.n_t4a_llpc1_syn,
        "sheet_layer_a_pct": layer_a,
        **nod,
    }


def _nod1_of_sheet(src, meta, sheet_roots) -> dict:
    """Nod1's rank/share among the sheet's output cell types; the top cholinergic readout."""
    sheet_set = set(int(x) for x in sheet_roots)
    if not sheet_set:
        return {"nod1_rank": None, "nod1_share_pct": 0.0, "nod1_top_excitatory": False,
                "top_excitatory": None}
    out = src.synapses(pre_ids=list(sheet_set))
    out = out[out["pre_pt_root_id"].isin(sheet_set)]
    pm = meta.by_root.reindex(out["post_pt_root_id"].values).reset_index(drop=True)
    df = pd.DataFrame({"ct": pm["cell_type"].values, "nt": pm["nt_canonical"].values})
    df = df[df["ct"].notna() & (df["ct"] != "LLPC1")]
    total = int(len(df))
    by_ct = df.groupby("ct").size().sort_values(ascending=False).reset_index(name="syn")
    by_ct["rank"] = range(1, len(by_ct) + 1)
    nrow = by_ct[by_ct["ct"] == "Nod1"]
    rank = int(nrow["rank"].iloc[0]) if len(nrow) else None
    nsyn = int(nrow["syn"].iloc[0]) if len(nrow) else 0
    cho = df[df["nt"] == "acetylcholine"].groupby("ct").size().sort_values(ascending=False)
    top_exc = str(cho.index[0]) if len(cho) else None
    return {
        "nod1_rank": rank,
        "nod1_share_pct": round(100.0 * nsyn / total, 2) if total else 0.0,
        "nod1_top_excitatory": bool(top_exc == "Nod1"),
        "top_excitatory": top_exc,
    }


def _nod1_downstream_crossing(src, meta, cfg: C.SideConfig) -> dict:
    """Does the side's Nod1 reach a DNp26, and is its output majority contralateral?"""
    nod1 = sorted(int(x) for x in meta.root_ids_of_type(["Nod1"], side=cfg.sheet_side))
    if not nod1:
        return {"n_nod1": 0, "nod1_to_dnp26_syn": 0, "reaches_dnp26": False,
                "contra_output_pct": float("nan")}
    nset = set(nod1)
    out = src.synapses(pre_ids=nod1); out = out[out["pre_pt_root_id"].isin(nset)]
    pm = meta.by_root.reindex(out["post_pt_root_id"].values).reset_index(drop=True)
    dnp26_syn = int((pm["cell_type"].values == "DNp26").sum())
    # contralateral output fraction
    pre_side = meta.by_root.reindex(out["pre_pt_root_id"].values)["side"].to_numpy()
    post_side = meta.by_root.reindex(out["post_pt_root_id"].values)["side"].to_numpy()
    valid = pd.notna(pre_side) & pd.notna(post_side)
    contra = float(100.0 * np.sum((pre_side != post_side) & valid) / max(int(valid.sum()), 1))
    return {
        "n_nod1": len(nod1), "nod1_to_dnp26_syn": dnp26_syn,
        "reaches_dnp26": bool(dnp26_syn > 0),
        "contra_output_pct": round(contra, 2),
    }


def _retinotopy_proxy(src, meta, cfg: C.SideConfig) -> dict:
    """Cheap retinotopy proxy: per-LLPC1 distinct-T4a in-degree is BOUNDED (not field-wide).

    A retinotopic sheet pools a small patch (the paper: ~15 T4a/cell), so the median distinct
    T4a in-degree per LLPC1 is small and well below the size of the VCH-gated T4a pool. A
    non-retinotopic (field-wide) sheet would have each LLPC1 sampling most of the pool. This
    is a proxy only; the permutation null is family D' (Phase 2).
    """
    sheet = SH.get_sheet(src, meta, cfg)
    syn = sheet.t4a_llpc1_syn
    if len(syn) == 0:
        return {"median_indeg": float("nan"), "pool_size": int(sheet.n_t4a), "bounded": False}
    indeg = syn.groupby("post_pt_root_id")["pre_pt_root_id"].nunique()
    med = float(indeg.median())
    pool = int(sheet.n_t4a)
    # bounded iff median in-degree is small absolutely AND a small fraction of the pool
    bounded = bool(med <= RETINO_INDEG_MAX and (pool == 0 or med / pool <= 0.25))
    return {"median_indeg": round(med, 1), "pool_size": pool,
            "indeg_frac_of_pool": round(med / pool, 3) if pool else float("nan"),
            "bounded": bounded}


def _label(value: bool, weak: bool = False) -> str:
    return PRESENT if value else (WEAK if weak else ABSENT)


def _screen_side(src, meta, cfg: C.SideConfig) -> dict:
    loop = _vch_loop(src, meta, cfg)
    sr = _sheet_readout(src, meta, cfg)
    dc = _nod1_downstream_crossing(src, meta, cfg)
    rt = _retinotopy_proxy(src, meta, cfg)

    stages = {
        "1_entry_gate": _label(
            loop["vch_nt"] == "gaba" and loop["reciprocal_frac"] >= RECIP_FRAC_FLOOR
            and loop["t4t5_partners"] > 0),
        "2_gated_sheet": _label(sr["n_llpc1"] >= SHEET_FLOOR_LLPC1,
                                weak=(0 < sr["n_llpc1"] < SHEET_FLOOR_LLPC1)),
        "3_readout": _label(sr["nod1_top_excitatory"] and sr["nod1_share_pct"] >= NOD1_SHARE_FLOOR_PCT,
                            weak=(sr["nod1_rank"] is not None and not sr["nod1_top_excitatory"])),
        "4_direction": _label(
            np.isfinite(sr["sheet_layer_a_pct"]) and sr["sheet_layer_a_pct"] >= LAYER_A_FLOOR_PCT),
        "5_downstream": _label(dc["reaches_dnp26"]),
        "6_crossing": _label(np.isfinite(dc["contra_output_pct"]) and dc["contra_output_pct"] >= CONTRA_FLOOR_PCT,
                             weak=(np.isfinite(dc["contra_output_pct"]) and dc["contra_output_pct"] > 40.0)),
        "7_retinotopy_proxy": _label(rt["bounded"], weak=(np.isfinite(rt["median_indeg"]))),
    }
    core_stages = ("1_entry_gate", "2_gated_sheet", "3_readout")
    core_present = all(stages[s] == PRESENT for s in core_stages)
    any_core_absent = any(stages[s] == ABSENT for s in core_stages)
    return {
        "sheet_side": cfg.sheet_side, "gating_soma_side": cfg.gating_soma_side,
        "loop": loop, "sheet_readout": sr, "downstream_crossing": dc, "retinotopy_proxy": rt,
        "stages": stages,
        "core_present": core_present,
        "any_core_absent": any_core_absent,
        "go": bool(core_present and not any_core_absent),
    }


def run(src, meta) -> dict:
    """Run the existence screen for BOTH sides (right = positive control, left = candidate)."""
    right = _screen_side(src, meta, C.RIGHT)
    left = _screen_side(src, meta, C.LEFT)
    # go/no-go: proceed to the full mirror iff the left core stages are present (and the right
    # positive control is itself sound, else the comparison is unreadable).
    decision = "GO" if (left["go"] and right["core_present"]) else "NO_GO"
    return {
        "question": "left_mirror_existence_screen",
        "right": right, "left": left,
        "decision": decision,
        "track": getattr(src, "track", None),
    }
