"""Derive family C: LLPC1 is the unique figure-output sheet.

Two measurements:
  1. Classify the annotated targets of the 912 reciprocal T4/T5 into functional classes
     and confirm columnar projection sheets are a ~7% minority.
  2. Within the cholinergic columnar projection class, compare total T4/T5 input synapses
     onto LLPC1 vs its direction-sibling sheets (LLPC2/3, LPC1/2) and the looming LPLC2 —
     LLPC1 should dominate ~40x.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import a_vch_loop as DA
from . import common as CM
from ..oracle import consts as C

# Functional-class membership by super_class / cell_type prefix (Table S1 grouping).
COLUMNAR_SHEETS = ("LLPC1", "LLPC2", "LLPC3", "LPC1", "LPC2")  # excitatory columnar projection
LOOMING = ("LPLC2", "LPLC4", "LC4")


def _classify_class(cell_type: str, super_class: str) -> str:
    ct = cell_type if isinstance(cell_type, str) else ""
    super_class = super_class if isinstance(super_class, str) else ""
    if any(ct == s for s in COLUMNAR_SHEETS):
        return "columnar_projection_sheet"
    if ct.startswith(("LPLC", "LC")):
        return "looming_object_vpn"
    if ct.startswith(("TmY", "Tm", "Y", "C3", "Mi", "Tlp")):
        return "optic_intrinsic_feedback"
    if ct.startswith(("LPi", "Li", "LT")) or ct == "CT1":
        return "optic_inhibitory"
    if ct.startswith(("HS", "VCH", "DCH")):
        return "widefield_tangential_centrifugal"
    if super_class in ("visual_projection",):
        return "other_visual_projection"
    return "other"


def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    # 1. reciprocal-T4/T5 broadcast classification.
    da = DA.run(src, meta, cfg)
    recip = da["_reciprocal_roots"]
    bout = src.synapses(pre_ids=recip)
    bout = bout[bout["pre_pt_root_id"].isin(set(recip))]
    pm = meta.by_root.reindex(bout["post_pt_root_id"].values).reset_index(drop=True)
    cls = [
        _classify_class(ct, sc)
        for ct, sc in zip(pm["cell_type"].astype("object").values,
                          pm["super_class"].astype("object").values)
    ]
    bout = bout.reset_index(drop=True)
    bout["fclass"] = cls
    annotated = bout[pm["cell_type"].notna().values]
    class_syn = annotated.groupby("fclass").size()
    total = int(class_syn.sum())
    columnar_pct = round(100.0 * class_syn.get("columnar_projection_sheet", 0) / total, 1) if total else float("nan")

    # 2. RECIPROCAL (VCH-gated) T4/T5 input onto each columnar sheet (right hemisphere).
    #    The paper's "~40x" (p.3) is specifically about the reciprocal-loop T4/T5
    #    population (912 cells), not all T4/T5: LLPC1 gets 17,499 such synapses vs
    #    318-401 for its sibling sheets. Counting all canonical T4/T5 would be ~uniform
    #    across sheets and miss the point.
    recip = set(da["_reciprocal_roots"])
    sheet_t4t5 = {}
    for s in COLUMNAR_SHEETS + ("LPLC2",):
        roots = set(int(r) for r in meta.root_ids_of_type([s], side=cfg.sheet_side))
        if not roots:
            sheet_t4t5[s] = 0
            continue
        ins = src.synapses(post_ids=sorted(roots))
        ins = ins[ins["post_pt_root_id"].isin(roots)]
        sheet_t4t5[s] = int(ins["pre_pt_root_id"].isin(recip).sum())

    llpc1 = sheet_t4t5["LLPC1"]
    siblings = [sheet_t4t5[s] for s in ("LLPC2", "LLPC3", "LPC1", "LPC2")]
    sibling_mean = float(np.mean(siblings)) if siblings else float("nan")
    fold = round(llpc1 / sibling_mean, 1) if sibling_mean else float("nan")
    top_columnar = max(COLUMNAR_SHEETS, key=lambda s: sheet_t4t5[s])

    return {
        "columnar_pct": columnar_pct,
        "class_breakdown": {k: int(v) for k, v in class_syn.items()},
        "sheet_t4t5_input": sheet_t4t5,
        "llpc1_vs_siblings_fold": fold,
        "top_columnar_sheet": top_columnar,
        "lplc2_less_than_llpc1": sheet_t4t5["LPLC2"] < llpc1,
        "track": src.track,
    }
