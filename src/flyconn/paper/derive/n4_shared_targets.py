"""Stage-9 Channel N4: shared downstream convergence (the bilateral integration locus).

Beyond the direct readout crossing of Channel N1, the two figure circuits also meet where their
outputs converge on the same downstream cells. This module finds the cells that receive from BOTH
the left and the right figure readout (Nod1) and, more broadly, from both sheets, classifies them
by role (descending / premotor LAL-WED / central / neuromodulatory), and surfaces the notable
convergence features: the Nod1<->Nod1 cross-talk, the feedback onto the heterolateral bridge H1,
and any neuromodulatory convergence (octopaminergic OA cells).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import interhemi_common as IH


def _targets_of(src, meta, cell_type: str):
    """Distinct downstream target roots of a cell type, both sides separately."""
    res = {}
    for side in ("left", "right"):
        roots = sorted(int(x) for x in meta.root_ids_of_type([cell_type], side=side))
        if not roots:
            res[side] = set()
            continue
        o = src.synapses(pre_ids=roots)
        o = o[o["pre_pt_root_id"].isin(set(roots))]
        res[side] = set(int(x) for x in o["post_pt_root_id"].unique())
    return res


def _classify_role(super_class: str, cell_type: str) -> str:
    sc = str(super_class)
    ct = str(cell_type)
    if sc == "descending":
        return "descending"
    if ct.startswith(("LAL", "WED", "PLP", "PVLP")):
        return "premotor_central"
    if ct.startswith("OA-") or "VUM" in ct:
        return "neuromodulatory"
    if ct in ("Nod1", "Nod2", "Nod3", "Nod5", "LPT42_Nod4"):
        return "readout_crosstalk"
    if ct in ("H1", "H2"):
        return "bridge_feedback"
    if sc == "central":
        return "central"
    return "other"


def run(src, meta, midline: dict | None = None) -> dict:
    if midline is None:
        midline = IH.synapse_space_midline(src, meta)

    nod1 = _targets_of(src, meta, "Nod1")
    shared = nod1["left"] & nod1["right"]
    pm = meta.by_root.reindex(list(shared))
    roles = {}
    detail = []
    for root in shared:
        ct = pm.loc[root, "cell_type"] if root in pm.index else None
        sc = pm.loc[root, "super_class"] if root in pm.index else None
        role = _classify_role(sc, ct)
        roles[role] = roles.get(role, 0) + 1
        detail.append({"root": int(root), "cell_type": str(ct), "super_class": str(sc),
                       "side": str(pm.loc[root, "side"]) if root in pm.index else None, "role": role})

    by_ct = pm["cell_type"].dropna().value_counts().head(20)

    # notable features
    nod1_crosstalk = [x for x in detail if x["role"] == "readout_crosstalk"]
    bridge_fb = [x for x in detail if x["role"] == "bridge_feedback"]
    neuromod = [x for x in detail if x["role"] == "neuromodulatory"]
    descending = [x for x in detail if x["role"] == "descending"]

    return {
        "midline_x_um": midline.get("midline_x_um"),
        "n_left_nod1_targets": len(nod1["left"]),
        "n_right_nod1_targets": len(nod1["right"]),
        "n_shared": len(shared),
        "shared_by_role": roles,
        "shared_target_types": {str(k): int(v) for k, v in by_ct.items()},
        "nod1_crosstalk_cells": nod1_crosstalk,
        "bridge_feedback_cells": bridge_fb,
        "neuromodulatory_convergence": neuromod,
        "descending_convergence": [x["cell_type"] for x in descending],
        "has_nod1_crosstalk": bool(nod1_crosstalk),
        "has_bridge_feedback": bool(bridge_fb),
        "has_neuromod_convergence": bool(neuromod),
        "track": getattr(src, "track", None),
    }
