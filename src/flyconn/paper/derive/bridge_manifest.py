"""Stage-9 bridge manifest: every inter-hemispheric bridge cell of the figure circuit, named.

The primary artifact of the inter-hemispheric analysis: which exact neurons carry the coupling,
with root id, neurotransmitter, super_class, soma side, arbor (position-based) side, the
lobula-plate layer they read, and the circuit cells they couple. Built from the channel results so
the named cells are exactly those the analysis identified.
"""

from __future__ import annotations

import numpy as np

from . import interhemi_common as IH

# The cells established as genuine inter-hemispheric bridges (axon crosses the midline onto the
# figure circuit) plus the readout crossers. Refined by the discovery channel at runtime.
CORE_BRIDGE_TYPES = ("H1", "H2", "Nod1", "LPT42_Nod4")


def build(src, meta, midline: dict, extra_types: list[str] | None = None) -> dict:
    types = list(dict.fromkeys(list(CORE_BRIDGE_TYPES) + list(extra_types or [])))
    bridges = {}
    for t in types:
        roots = sorted(int(x) for x in meta.root_ids_of_type([t]))
        if not roots:
            continue
        dend = IH.dendrite_side_map(src, meta, roots, midline)
        layer = IH.input_layer_profile(src, meta, roots)
        cross = IH.contra_output_by_position(src, meta, roots, midline)
        cells = []
        for r in roots:
            row = meta.by_root.loc[r]
            cells.append({
                "root_id": int(r),
                "soma_side": str(row.get("side")),
                "arbor_side": dend.get(r),
                "nt": str(row.get("nt_canonical")),
                "super_class": str(row.get("super_class")),
            })
        bridges[t] = {
            "n_cells": len(roots),
            "nt": str(meta.by_root.loc[roots[0], "nt_canonical"]),
            "super_class": str(meta.by_root.loc[roots[0], "super_class"]),
            "position_output_crossing_pct": cross["cross_frac"],
            "input_layer": layer.get("dominant_layer"),
            "input_direction": layer.get("dominant_direction"),
            "is_inhibitory_sign": str(meta.by_root.loc[roots[0], "nt_canonical"]) in ("gaba", "glutamate"),
            "cells": cells,
        }
    return {"midline_x_um": midline.get("midline_x_um"), "bridge_types": bridges}
