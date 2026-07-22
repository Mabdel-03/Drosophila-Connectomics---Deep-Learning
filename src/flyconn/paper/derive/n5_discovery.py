"""Stage-9 Channel N5: systematic discovery of all inter-hemispheric edges of the figure circuit.

A completeness guard against missing a coupling channel. Over the union of both circuits' cells,
this enumerates every input synapse, classifies it by the position-based crossing test, and ranks
the presynaptic cell types that bridge into the circuit from the opposite lobe. It explicitly
contrasts the position-based ranking with the soma-side ranking, which is dominated by the T4/T5
artifact (the centrifugal gaters' cross-lobe dendrite makes local T4/T5 appear cross-hemispheric).
The difference between the two rankings is itself a documented result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import interhemi_common as IH

# The figure circuit cells we test for incoming inter-hemispheric edges: the sheet, the Nod
# readouts, and the steering command. We exclude the T4/T5 array (the T4-terminal gating is
# Channel N3, not an external bridge) and the high-input-count machinery VCH/DCH/LPi15 (each ~190k
# input synapses, which makes a single live pull hang); those are characterized in N2/N3 directly.
CIRCUIT_TYPES = ("LLPC1", "Nod1", "Nod2", "LPT42_Nod4", "DNp26")
MIN_TYPE_SYN = 50


def run(src, meta, midline: dict | None = None) -> dict:
    if midline is None:
        midline = IH.synapse_space_midline(src, meta)

    post_roots = sorted(set(int(x) for t in CIRCUIT_TYPES
                            for x in meta.root_ids_of_type([t])))
    inp = src.synapses(post_ids=post_roots)
    inp = inp[inp["post_pt_root_id"].isin(set(post_roots))].copy()
    if len(inp) == 0:
        return {"n_circuit_cells": len(post_roots), "bridges_position": [], "bridges_soma": []}

    pm = meta.by_root.reindex(inp["pre_pt_root_id"].values)
    qm = meta.by_root.reindex(inp["post_pt_root_id"].values)
    inp["pre_ct"] = pm["cell_type"].values
    inp["pre_side"] = pm["side"].values
    inp["post_side"] = qm["side"].values
    inp["syn_side"] = IH.position_side(IH.G.syn_positions_um(inp, "post"), midline)

    # --- soma-side ranking (the naive definition, with the artifact) ---
    soma_cross = inp[(inp["pre_side"].notna()) & (inp["post_side"].notna())
                     & (inp["pre_side"] != inp["post_side"])]
    soma_rank = (soma_cross.groupby("pre_ct").size().sort_values(ascending=False)
                 .head(20).astype(int).to_dict())

    # --- position-based ranking (genuine axonal bridges) ---
    # candidate presynaptic types with enough synapses, then dendrite-side only for those cells.
    cand_types = [ct for ct, n in inp["pre_ct"].value_counts().items()
                  if isinstance(ct, str) and ct not in CIRCUIT_TYPES and n >= MIN_TYPE_SYN]
    cand_roots = sorted(set(int(r) for r, ct in zip(inp["pre_pt_root_id"], inp["pre_ct"])
                            if ct in set(cand_types)))
    dend = IH.dendrite_side_map(src, meta, cand_roots, midline)
    sub = inp[inp["pre_ct"].isin(set(cand_types))].copy()
    sub["pre_dend_side"] = pd.Series(sub["pre_pt_root_id"].to_numpy()).map(dend).to_numpy()
    sub["is_cross"] = (sub["pre_dend_side"].notna()) & (sub["syn_side"] != sub["pre_dend_side"])
    pos_cross = sub[sub["is_cross"]]
    pos_rank = (pos_cross.groupby("pre_ct").size().sort_values(ascending=False)
                .head(20).astype(int).to_dict())

    # cells that appear in the soma ranking but NOT the position ranking = the artifact
    artifact_only = sorted(set(soma_rank) - set(pos_rank))
    genuine = sorted(set(pos_rank))

    total = int(len(inp))
    n_pos_cross = int(len(pos_cross))
    return {
        "midline_x_um": midline.get("midline_x_um"),
        "n_circuit_cells": len(post_roots),
        "total_input_syn": total,
        "position_cross_syn": n_pos_cross,
        "position_cross_frac": round(100.0 * n_pos_cross / total, 1) if total else float("nan"),
        "bridges_position": [{"type": k, "syn": v} for k, v in pos_rank.items()],
        "bridges_soma": [{"type": k, "syn": v} for k, v in soma_rank.items()],
        "artifact_types_soma_only": artifact_only,
        "genuine_bridge_types": genuine,
        "track": getattr(src, "track", None),
    }
