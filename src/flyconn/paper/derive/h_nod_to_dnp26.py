"""Derive family H: Nod1 -> DNp26 relay.

Nod1 cells are the right-hemisphere Nod1 driven by the sheet. We pull their descending
output, count synapses onto descending neurons (super_class == 'descending'), the
fraction onto known steering DNs, and specifically Nod1 -> DNp26. We also measure the
direct sheet -> DNp26 synapses for the "relay exceeds direct" comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import sheet as SH
from ..oracle import consts as C

# Known wing-steering DN cell types (Fig 6 / Table S14) used for the "97% to steering
# DNs" fraction.
STEERING_DN_TYPES = set(C.WING_STEERING_DNS)


def _nod1_roots(src, meta, sheet) -> np.ndarray:
    """The Nod1 cells of the type (all annotated Nod1).

    The paper's "two Nod1 cells make 662 synapses ... DNp26 receives 448 from Nod1"
    aggregates the Nod1 *cell type*: counting all annotated Nod1 cells reproduces 662 DN
    synapses / 448 onto DNp26 exactly, whereas restricting to the 2 right-hemisphere cells
    gives 407 / 289. We therefore score the type.
    """
    return np.array(sorted(int(x) for x in meta.root_ids_of_type(["Nod1"])), dtype=np.int64)


def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    sheet = SH.get_sheet(src, meta, cfg)
    sheet_set = set(int(x) for x in sheet.llpc1_roots)
    nod1_roots = _nod1_roots(src, meta, sheet)

    # Nod1 descending output.
    nod1_out = src.synapses(pre_ids=nod1_roots.tolist())
    nod1_out = nod1_out[nod1_out["pre_pt_root_id"].isin(set(int(x) for x in nod1_roots))]
    pm = meta.by_root.reindex(nod1_out["post_pt_root_id"].values).reset_index(drop=True)
    is_dn = (pm["super_class"].values == "descending")
    nod1_dn = nod1_out[is_dn].copy()
    nod1_dn["post_ct"] = pm.loc[is_dn.nonzero()[0], "cell_type"].values

    nod1_dn_syn = int(len(nod1_dn))
    nod1_n_dns = int(nod1_dn["post_pt_root_id"].nunique())
    steering_syn = int(nod1_dn["post_ct"].isin(STEERING_DN_TYPES).sum())
    nod1_steering_frac = round(100.0 * steering_syn / nod1_dn_syn, 1) if nod1_dn_syn else float("nan")

    # Nod1 -> DNp26 (both DNp26 cells).
    dnp26_roots = set(int(x) for x in meta.root_ids_of_type(["DNp26"]))
    nod1_to_dnp26 = int(nod1_dn["post_pt_root_id"].isin(dnp26_roots).sum())

    # Direct sheet -> DNp26.
    sheet_out = src.synapses(pre_ids=sheet.llpc1_roots.tolist())
    sheet_out = sheet_out[sheet_out["pre_pt_root_id"].isin(sheet_set)]
    direct_to_dnp26 = int(sheet_out["post_pt_root_id"].isin(dnp26_roots).sum())

    # Nod2 NT.
    nod2 = meta.root_ids_of_type(["Nod2"])
    nod2_nt = str(meta.by_root.loc[int(nod2[0]), "nt_canonical"]) if len(nod2) else None

    return {
        "nod1_dn_syn": nod1_dn_syn,
        "nod1_n_dns": nod1_n_dns,
        "nod1_steering_frac": nod1_steering_frac,
        "nod1_to_dnp26": nod1_to_dnp26,
        "direct_to_dnp26": direct_to_dnp26,
        "nod2_nt": nod2_nt,
        "_n_nod1_cells": int(len(nod1_roots)),
        "track": src.track,
    }
