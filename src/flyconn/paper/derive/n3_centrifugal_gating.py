"""Stage-9 Channel N3: the centrifugal gating link and the soma-vs-arbor laterality distinction.

The companion mirror analysis treated the gating VCH as "crossing" because its soma is on the
side opposite the sheet it gates. The position-based analysis sharpens this: VCH/DCH are NOT
axonal inter-hemispheric bridges. Their soma sits on one side but their ENTIRE arbor (dendrite
and axon) lies in the opposite optic lobe, so they do not carry a signal across the midline; they
are gain-control cells local to the lobe they gate, with a displaced soma. This module documents
that distinction (soma-side ~99% contralateral vs position-based ~0% crossing) and characterizes
the gating as the functional consequence of the centrifugal placement: each lobe's figure sheet
is gated by a cell whose cell body is registered to the opposite hemisphere.

This is the inter-hemispheric "gain-control" link in the sense that the two lobes' gating cells
are a bilateral pair whose somata are swapped, but it is not a synaptic bridge between the
circuits. We also confirm the gaters do not couple directly to each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import interhemi_common as IH
from ..oracle import consts as C


def _gater_laterality(src, meta, midline: dict) -> dict:
    """Soma-side vs position-based crossing for VCH and DCH (exposes the artifact)."""
    out = {}
    for t in ("VCH", "DCH"):
        roots = meta.root_ids_of_type([t])
        soma = IH.contra_output_by_soma(src, meta, roots)
        pos = IH.contra_output_by_position(src, meta, roots, midline)
        # per-cell: soma side vs dendrite (arbor) side
        dend = IH.dendrite_side_map(src, meta, roots, midline)
        per = {}
        for r in roots:
            soma_side = str(meta.by_root.loc[int(r), "side"])
            per[int(r)] = {"soma_side": soma_side, "arbor_side": dend.get(int(r)),
                           "soma_arbor_swapped": bool(dend.get(int(r)) is not None
                                                      and dend.get(int(r)) != soma_side)}
        out[t] = {
            "soma_contra_pct": soma["cross_frac"],
            "position_cross_pct": pos["cross_frac"],
            "per_cell": per,
            "is_axonal_bridge": bool(pos["cross_frac"] is not None and pos["cross_frac"] >= 50.0),
            "soma_arbor_swapped": all(v["soma_arbor_swapped"] for v in per.values()),
        }
    return out


def _direct_gater_coupling(src, meta) -> dict:
    """Do the two gating VCHs (and DCHs) synapse on each other directly? (expected: no)."""
    res = {}
    for t in ("VCH", "DCH"):
        l = meta.root_ids_of_type([t], side="left")
        r = meta.root_ids_of_type([t], side="right")
        if len(l) == 0 or len(r) == 0:
            continue
        l, r = int(l[0]), int(r[0])
        o = src.synapses(pre_ids=[l]); o = o[o["pre_pt_root_id"] == l]
        l_to_r = int((o["post_pt_root_id"] == r).sum())
        o2 = src.synapses(pre_ids=[r]); o2 = o2[o2["pre_pt_root_id"] == r]
        r_to_l = int((o2["post_pt_root_id"] == l).sum())
        res[t] = {"left_to_right": l_to_r, "right_to_left": r_to_l,
                  "directly_coupled": bool(l_to_r > 0 or r_to_l > 0)}
    return res


def run(src, meta, midline: dict | None = None) -> dict:
    if midline is None:
        midline = IH.synapse_space_midline(src, meta)
    lat = _gater_laterality(src, meta, midline)
    coupling = _direct_gater_coupling(src, meta)
    return {
        "midline_x_um": midline.get("midline_x_um"),
        "gater_laterality": lat,
        "direct_gater_coupling": coupling,
        "vch_is_axonal_bridge": lat.get("VCH", {}).get("is_axonal_bridge"),
        "vch_soma_arbor_swapped": lat.get("VCH", {}).get("soma_arbor_swapped"),
        "gaters_directly_coupled": any(v.get("directly_coupled") for v in coupling.values()),
        "track": getattr(src, "track", None),
    }
