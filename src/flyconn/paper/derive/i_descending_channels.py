"""Derive family I: the three descending routes from the LLPC1 sheet.

  * direct   = sheet LLPC1 -> DN  (super_class == 'descending')
  * nod_relay= sheet-driven Nod-type cells -> DN
  * broadcast= sheet-driven PLP/PVLP core (PVLP011, PLP163, PLP249) -> DN

Counts circuit->DN synapses and the number of distinct DNs per channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import sheet as SH
from ..oracle import consts as C


def _dn_targets(src, meta, source_roots: list[int]) -> tuple[int, int]:
    """(total synapses onto DNs, number of distinct DNs) for a set of source neurons."""
    if not source_roots:
        return 0, 0
    out = src.synapses(pre_ids=source_roots)
    out = out[out["pre_pt_root_id"].isin(set(int(x) for x in source_roots))]
    pm = meta.by_root.reindex(out["post_pt_root_id"].values)
    is_dn = (pm["super_class"].values == "descending")
    dn = out[is_dn]
    return int(len(dn)), int(dn["post_pt_root_id"].nunique())


def _sheet_driven_of_types(src, meta, sheet, cell_types) -> list[int]:
    """Cells of the given types that the sheet drives (right hemisphere)."""
    out = src.synapses(pre_ids=sheet.llpc1_roots.tolist())
    out = out[out["pre_pt_root_id"].isin(set(int(x) for x in sheet.llpc1_roots))]
    pm = meta.by_root.reindex(out["post_pt_root_id"].values)
    mask = np.isin(pm["cell_type"].values, list(cell_types))
    return sorted(set(int(x) for x in out["post_pt_root_id"].to_numpy()[mask]))


def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    sheet = SH.get_sheet(src, meta, cfg)

    # direct: the sheet itself -> DNs.
    direct_syn, direct_dns = _dn_targets(src, meta, sheet.llpc1_roots.tolist())

    # nod relay: sheet-driven Nod-type cells -> DNs.
    nod_cells = _sheet_driven_of_types(src, meta, sheet, ("Nod1", "Nod2", "Nod3", "Nod5"))
    nod_syn, nod_dns = _dn_targets(src, meta, nod_cells)

    # PLP/PVLP broadcast core -> DNs.
    core = _sheet_driven_of_types(src, meta, sheet, ("PVLP011", "PLP163", "PLP249"))
    bcast_syn, bcast_dns = _dn_targets(src, meta, core)

    channels = {
        "direct":    {"syn": direct_syn, "dns": direct_dns},
        "nod_relay": {"syn": nod_syn,    "dns": nod_dns},
        "broadcast": {"syn": bcast_syn,  "dns": bcast_dns},
    }
    strongest = max(channels, key=lambda k: channels[k]["syn"])
    return {"channels": channels, "direct_is_strongest": strongest == "direct", "track": src.track}
