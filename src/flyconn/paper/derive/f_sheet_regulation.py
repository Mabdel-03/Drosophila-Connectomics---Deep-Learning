"""Derive family F: sheet-regulating inhibition (LPi15, PVLP011, VCH-direct).

LPi15 (optic, right exemplar) feed-forward opponent; PVLP011 (central) recurrent gain;
VCH (left soma -> right OL) direct contact onto the sheet.
"""

from __future__ import annotations

import numpy as np

from . import common as CM
from . import sheet as SH
from ..oracle import consts as C


def _to_sheet(src, root: int, sheet_set: set[int]) -> tuple[int, int]:
    """(distinct sheet LLPC1 contacted, total synapses) for a presynaptic neuron."""
    out = src.synapses(pre_ids=[root]); out = out[out["pre_pt_root_id"] == root]
    in_sheet = out[out["post_pt_root_id"].isin(sheet_set)]
    return int(in_sheet["post_pt_root_id"].nunique()), int(len(in_sheet))


def _from_sheet(src, root: int, sheet_set: set[int]) -> int:
    """Distinct sheet LLPC1 that drive this neuron (how many of 100 it reads)."""
    ins = src.synapses(post_ids=[root]); ins = ins[ins["post_pt_root_id"] == root]
    return int(ins[ins["pre_pt_root_id"].isin(sheet_set)]["pre_pt_root_id"].nunique())


def _lpi15_layer_b(src, meta, lpi15_root: int) -> float:
    """Fraction of LPi15's T4/T5 input drive that is layer-b (opposite direction)."""
    ins = src.synapses(post_ids=[lpi15_root]); ins = ins[ins["post_pt_root_id"] == lpi15_root]
    counts = CM.attach_meta(CM.partner_counts(ins, "pre_pt_root_id"), meta)
    t45 = counts[counts["is_t4t5"]]
    return CM.layer_b_fraction(t45)


def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    sheet = SH.get_sheet(src, meta, cfg)
    sheet_set = set(int(x) for x in sheet.llpc1_roots)

    lpi15 = int(meta.root_ids_of_type(["LPi15"], side=cfg.sheet_side)[0])
    pvlp011 = int(meta.root_ids_of_type(["PVLP011"])[0])
    # PVLP011 is a central bilateral pair; pick the one reading the most of the sheet.
    pvlp_cands = [int(x) for x in meta.root_ids_of_type(["PVLP011"])]
    pvlp011 = max(pvlp_cands, key=lambda r: _from_sheet(src, r, sheet_set))

    lpi15_reached, lpi15_syn = _to_sheet(src, lpi15, sheet_set)
    lpi15_layer_b = _lpi15_layer_b(src, meta, lpi15)

    pvlp011_reads = _from_sheet(src, pvlp011, sheet_set)
    pvlp011_feeds_back, _ = _to_sheet(src, pvlp011, sheet_set)

    vch_direct_llpc1, vch_direct_syn = _to_sheet(src, cfg.vch_root, sheet_set)

    return {
        "lpi15_llpc1_reached": lpi15_reached,
        "lpi15_syn": lpi15_syn,
        "lpi15_layer_b_pct": lpi15_layer_b,
        "pvlp011_reads": pvlp011_reads,
        "pvlp011_feeds_back": pvlp011_feeds_back,
        "vch_direct_llpc1": vch_direct_llpc1,
        "vch_direct_syn": vch_direct_syn,
        "track": src.track,
    }
