"""Stage-9 Channel N1: the readout-crossing / command-convergence inter-hemispheric link.

The dominant way the two figure circuits communicate is through their readout. Nod1 is a
noduli-group cell whose axon crosses the midline (position-based output crossing ~87%), and each
side's Nod1 drives the OPPOSITE hemisphere's DNp26 steering command. This module quantifies that
link in both directions, reproduces the per-direction synapse counts as a positive control, and
characterizes its asymmetry and the convergence of the two circuits onto shared steering targets.

Positive control (live v783, measured this session): DNp26_left receives 289 synapses from
right-soma Nod1; DNp26_right receives 159 from left-soma Nod1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import interhemi_common as IH
from ..oracle import consts as C

STEERING_DNS = C.WING_STEERING_DNS  # DNa04, DNbe001, DNge107, DNbe005, DNp26, DNg32, DNge094


def _nod1_to_contra_dn(src, meta, nod1_side: str) -> dict:
    """Synapses from one side's Nod1 onto each side's DNp26, split by DNp26 side."""
    nod1 = sorted(int(x) for x in meta.root_ids_of_type(["Nod1"], side=nod1_side))
    if not nod1:
        return {}
    out = src.synapses(pre_ids=nod1)
    out = out[out["pre_pt_root_id"].isin(set(nod1))]
    pm = meta.by_root.reindex(out["post_pt_root_id"].values)
    out = out.assign(post_ct=pm["cell_type"].values, post_side=pm["side"].values)
    dnp26 = out[out["post_ct"] == "DNp26"]
    by_side = dnp26.groupby("post_side").size().to_dict()
    # all steering DNs reached by this Nod1, split by DN side
    steer = out[out["post_ct"].isin(STEERING_DNS)]
    steer_by_side = steer.groupby("post_side").size().to_dict()
    return {
        "nod1_side": nod1_side,
        "to_dnp26_by_side": {str(k): int(v) for k, v in by_side.items()},
        "to_steering_by_side": {str(k): int(v) for k, v in steer_by_side.items()},
        "n_nod1": len(nod1),
    }


def _dnp26_input_composition(src, meta) -> dict:
    """For each DNp26 (by side), the fraction of its Nod1 input that is contralateral-soma Nod1."""
    res = {}
    for dn_side in ("left", "right"):
        dn = sorted(int(x) for x in meta.root_ids_of_type(["DNp26"], side=dn_side))
        if not dn:
            continue
        inp = src.synapses(post_ids=dn)
        inp = inp[inp["post_pt_root_id"].isin(set(dn))]
        pm = meta.by_root.reindex(inp["pre_pt_root_id"].values)
        inp = inp.assign(pre_ct=pm["cell_type"].values, pre_side=pm["side"].values)
        nod1_in = inp[inp["pre_ct"] == "Nod1"]
        by_side = nod1_in.groupby("pre_side").size().to_dict()
        total_nod1 = int(len(nod1_in))
        contra = int(sum(v for k, v in by_side.items() if k != dn_side))
        res[dn_side] = {
            "nod1_input_by_soma_side": {str(k): int(v) for k, v in by_side.items()},
            "total_nod1_input": total_nod1,
            "contra_nod1_input": contra,
            "contra_frac_of_nod1_input": round(100.0 * contra / total_nod1, 1) if total_nod1 else float("nan"),
            "total_input_syn": int(len(inp)),
        }
    return res


def _shared_steering_targets(src, meta) -> dict:
    """Central / premotor cells that receive from BOTH left and right Nod1 (the convergence locus)."""
    out = {}
    targets = {}
    for side in ("left", "right"):
        nod1 = sorted(int(x) for x in meta.root_ids_of_type(["Nod1"], side=side))
        o = src.synapses(pre_ids=nod1)
        o = o[o["pre_pt_root_id"].isin(set(nod1))]
        targets[side] = set(int(x) for x in o["post_pt_root_id"].unique())
    shared = targets["left"] & targets["right"]
    pm = meta.by_root.reindex(list(shared))
    by_ct = pm["cell_type"].dropna().value_counts().head(15)
    return {
        "n_left_targets": len(targets["left"]),
        "n_right_targets": len(targets["right"]),
        "n_shared": len(shared),
        "shared_target_types": {str(k): int(v) for k, v in by_ct.items()},
    }


def run(src, meta, midline: dict | None = None) -> dict:
    if midline is None:
        midline = IH.synapse_space_midline(src, meta)

    left = _nod1_to_contra_dn(src, meta, "left")
    right = _nod1_to_contra_dn(src, meta, "right")
    dnp26_comp = _dnp26_input_composition(src, meta)
    shared = _shared_steering_targets(src, meta)

    # The crossing counts (the positive-control anchor): left-Nod1 -> right-DNp26, right-Nod1 -> left-DNp26.
    l_to_r = left.get("to_dnp26_by_side", {}).get("right", 0)
    r_to_l = right.get("to_dnp26_by_side", {}).get("left", 0)
    # Nod1 output crossing fraction (position-based), the mechanism behind the link.
    nod1_cross = IH.contra_output_by_position(src, meta,
                                              meta.root_ids_of_type(["Nod1"]), midline)

    return {
        "left_nod1": left,
        "right_nod1": right,
        "left_nod1_to_right_dnp26": int(l_to_r),
        "right_nod1_to_left_dnp26": int(r_to_l),
        "asymmetry_ratio": round(max(l_to_r, r_to_l) / max(min(l_to_r, r_to_l), 1), 2),
        "dnp26_input_composition": dnp26_comp,
        "shared_steering_targets": shared,
        "nod1_output_crossing_pct": nod1_cross["cross_frac"],
        "midline_x_um": midline.get("midline_x_um"),
        "track": getattr(src, "track", None),
    }
