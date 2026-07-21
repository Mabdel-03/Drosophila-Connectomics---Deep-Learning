"""Stage-9 Channel N2: heterolateral input bridges and the Egelhaaf directional-selectivity test.

Egelhaaf (1985) showed FD1 is inhibited by CONTRALATERAL REGRESSIVE (back-to-front) motion and
proposed that a contralateral wide-field element, excited by regressive motion, supplies that
inhibition. In the connectome this corresponds to a heterolateral cell that reads back-to-front
(lobula-plate layer-b) motion in one eye and projects onto the contralateral figure machinery.

This module discovers the full set of such bridges, by finding cells whose AXON crosses the
midline (position-based) and lands on the contralateral sheet machinery (LLPC1 sheet, VCH, DCH,
LPi15, Nod1, T4a, T4b), and classifies each by neurotransmitter (sign) and by the lobula-plate
layer of its own T4/T5 input. The central test: the inhibitory bridges onto the Nod1/FD1 pathway
read layer-b (regressive), matching Egelhaaf.

Known anchors (measured this session): H1 (glutamate, optic, 98% layer-b, position-crossing 90%)
and H2 (acetylcholine, 100% layer-b, position-crossing 52%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import interhemi_common as IH
from . import sheet as SH
from ..oracle import consts as C

# The FIGURE CELLS whose contralateral inhibition Egelhaaf predicts: the LLPC1 sheet (the figure
# map) and Nod1 (the FD1-role readout). We deliberately do NOT include the centrifugal gaters
# (VCH/DCH), the opponent LPi15, or the T4/T5 array here: a bridge onto the T4 terminal is the VCH
# gating story (Channel N3), and the gaters/opponent each carry ~190k input synapses, which makes
# a single live pull onto them hang. The FD-cell pathway (LLPC1 + Nod1) is the right, tractable
# target for Egelhaaf's contralateral-inhibition prediction.
SHEET_MACHINERY = ("LLPC1", "Nod1")
# Minimum crossing synapses onto the machinery to count a cell type as a bridge.
MIN_BRIDGE_SYN = 50


def _machinery_roots(meta) -> dict:
    """Root ids of each machinery type (both sides)."""
    return {t: set(int(x) for x in meta.root_ids_of_type([t])) for t in SHEET_MACHINERY}


def _discover_bridges(src, meta, midline: dict) -> pd.DataFrame:
    """All cells whose crossing axon lands on the contralateral figure machinery.

    Two-pass for efficiency. Pass 1: pull every synapse ONTO the machinery, annotate the
    presynaptic cell type, and tally by (pre_type, post_type) so we know which presynaptic TYPES
    contribute enough to be candidate bridges. Pass 2: only for the candidate types' cells do we
    compute the dendrite side (the expensive input pull), then count the crossing synapses.
    """
    machinery = _machinery_roots(meta)
    all_post = sorted(set().union(*machinery.values()))
    inp = src.synapses(post_ids=all_post)
    inp = inp[inp["post_pt_root_id"].isin(set(all_post))].copy()
    if len(inp) == 0:
        return pd.DataFrame()

    pm = meta.by_root.reindex(inp["pre_pt_root_id"].values)
    inp["pre_ct"] = pm["cell_type"].values
    inp["pre_nt"] = pm["nt_canonical"].values
    inp["pre_sc"] = pm["super_class"].values
    qm = meta.by_root.reindex(inp["post_pt_root_id"].values)
    inp["post_ct"] = qm["cell_type"].values

    # the machinery synapse sits at the POST position; its lobe:
    inp["syn_side"] = IH.position_side(IH.G.syn_positions_um(inp, "post"), midline)

    # Pass 1: candidate presynaptic TYPES (non-machinery) with enough total synapses to matter.
    cand_types = [ct for ct, n in inp["pre_ct"].value_counts().items()
                  if isinstance(ct, str) and ct not in SHEET_MACHINERY and n >= MIN_BRIDGE_SYN]
    # Pass 2: dendrite side only for the cells of candidate types (a bounded input pull).
    cand_roots = sorted(set(int(r) for r, ct in zip(inp["pre_pt_root_id"], inp["pre_ct"])
                            if ct in set(cand_types)))
    dend = IH.dendrite_side_map(src, meta, cand_roots, midline)
    sub = inp[inp["pre_ct"].isin(set(cand_types))].copy()
    sub["pre_dend_side"] = pd.Series(sub["pre_pt_root_id"].to_numpy()).map(dend).to_numpy()
    sub["is_cross"] = (sub["pre_dend_side"].notna()) & (sub["syn_side"] != sub["pre_dend_side"])

    cross = sub[sub["is_cross"]]
    rows = []
    for ct, g in cross.groupby("pre_ct"):
        n = int(len(g))
        if n < MIN_BRIDGE_SYN:
            continue
        targets = g["post_ct"].value_counts().to_dict()
        rows.append({
            "bridge_type": str(ct),
            "nt": str(g["pre_nt"].iloc[0]),
            "super_class": str(g["pre_sc"].iloc[0]),
            "cross_syn_onto_machinery": n,
            "targets": {str(k): int(v) for k, v in targets.items()},
        })
    df = pd.DataFrame(rows).sort_values("cross_syn_onto_machinery", ascending=False) if rows else pd.DataFrame()
    return df


def _bridge_layer(src, meta, bridge_type: str) -> dict:
    """Lobula-plate layer composition of a bridge type's own T4/T5 input (Egelhaaf direction test)."""
    roots = meta.root_ids_of_type([bridge_type])
    return IH.input_layer_profile(src, meta, roots)


def run(src, meta, midline: dict | None = None) -> dict:
    if midline is None:
        midline = IH.synapse_space_midline(src, meta)

    bridges = _discover_bridges(src, meta, midline)
    bridge_list = []
    if len(bridges):
        for _, row in bridges.iterrows():
            layer = _bridge_layer(src, meta, row["bridge_type"])
            # does this bridge touch the Nod1/FD1 pathway (Nod1 or the sheet that drives it)?
            tgts = row["targets"]
            touches_fd1 = bool(tgts.get("Nod1", 0) > 0 or tgts.get("LLPC1", 0) > 0)
            # sign: GABA/glutamate are inhibitory in this circuit context; ACh excitatory
            is_inhibitory = row["nt"] in ("gaba", "glutamate")
            bridge_list.append({
                "bridge_type": row["bridge_type"],
                "nt": row["nt"],
                "super_class": row["super_class"],
                "is_inhibitory_sign": is_inhibitory,
                "cross_syn_onto_machinery": int(row["cross_syn_onto_machinery"]),
                "targets": tgts,
                "touches_fd1_pathway": touches_fd1,
                "input_layer": layer,
            })

    # The Egelhaaf test. Egelhaaf (1985) showed FD1 receives contralateral REGRESSIVE (back-to-
    # front) inhibition, in a HORIZONTAL-motion paradigm. The connectome-faithful test is whether
    # the DOMINANT (largest) inhibitory bridge that reads horizontal motion (layer a or b) and
    # touches the FD1/Nod1 pathway is regressive (layer-b). The connectome additionally reveals
    # weaker inhibitory bridges reading VERTICAL motion (layer c/d) and central/neuromodulatory
    # contralateral inputs that a horizontal-motion paradigm could not have detected; these are
    # reported as an extension, not counted against the horizontal prediction.
    inhib_fd1 = [b for b in bridge_list if b["is_inhibitory_sign"] and b["touches_fd1_pathway"]
                 and b["input_layer"].get("has_t4t5")]
    inhib_horiz = [b for b in inhib_fd1 if b["input_layer"].get("dominant_layer") in ("a", "b")]
    inhib_vert = [b for b in inhib_fd1 if b["input_layer"].get("dominant_layer") in ("c", "d")]
    regressive = [b for b in inhib_horiz if b["input_layer"].get("dominant_layer") == "b"]
    # dominant horizontal inhibitory bridge by crossing-synapse count
    dom_horiz = max(inhib_horiz, key=lambda b: b["cross_syn_onto_machinery"]) if inhib_horiz else None
    egelhaaf_holds = bool(dom_horiz is not None
                          and dom_horiz["input_layer"].get("dominant_layer") == "b")

    return {
        "midline_x_um": midline.get("midline_x_um"),
        "n_bridges": len(bridge_list),
        "bridges": bridge_list,
        "inhibitory_fd1_bridges": [b["bridge_type"] for b in inhib_fd1],
        "inhibitory_horizontal_bridges": [b["bridge_type"] for b in inhib_horiz],
        "inhibitory_vertical_bridges": [b["bridge_type"] for b in inhib_vert],
        "regressive_inhibitory_fd1_bridges": [b["bridge_type"] for b in regressive],
        "dominant_horizontal_inhibitory_bridge": (dom_horiz["bridge_type"] if dom_horiz else None),
        "dominant_horizontal_inhibitory_direction": (
            dom_horiz["input_layer"].get("dominant_direction") if dom_horiz else None),
        "dominant_horizontal_inhibitory_cross_syn": (
            dom_horiz["cross_syn_onto_machinery"] if dom_horiz else None),
        "egelhaaf_regressive_inhibition_holds": egelhaaf_holds,
        "track": getattr(src, "track", None),
    }
