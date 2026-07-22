"""Derive family B: the 8-inhibitor screen against the 3 figure-circuit criteria.

For each of the 8 largest inhibitory T4/T5 targets we compute:
  (2) LLPC1 sheet contact: how many of the 100 sheet LLPC1 it sends synapses to;
  (3) centrifugal identity: its super_class.
(1) T4/T5-reciprocal counts are large and pan-retinal for some types (CT1 ~6,000); we
report them but the discriminating criteria are (2) and (3), as in the paper.

The pass-all-three set is computed as {centrifugal AND sheet-contacting AND
T4/T5-reciprocal}, expected to be exactly {VCH, DCH}.
"""

from __future__ import annotations

import numpy as np

from . import sheet as SH
from ..oracle import consts as C


def _llpc1_contact(src, meta, roots: list[int], sheet_set: set[int]) -> int:
    """Number of the 100 sheet LLPC1 that this inhibitor TYPE (all its cells) contacts.

    Table 1's "LLPC1 contact /100" is a per-type count: e.g. LT33's single right exemplar
    contacts 0 of the sheet, but the type contacts 7 (verified to match the paper). So we
    aggregate over all cells of the type.
    """
    if not roots:
        return 0
    out = src.synapses(pre_ids=list(roots))
    out = out[out["pre_pt_root_id"].isin(set(int(r) for r in roots))]
    return int(out[out["post_pt_root_id"].isin(sheet_set)]["post_pt_root_id"].nunique())


def _t4t5_reciprocal(src, meta, roots: list[int]) -> int:
    """Right-hemi T4/T5 that both drive AND receive from this inhibitor type (all cells)."""
    from .common import is_canonical_t4t5
    roots = set(int(r) for r in roots)
    ins = src.synapses(post_ids=sorted(roots)); ins = ins[ins["post_pt_root_id"].isin(roots)]
    outs = src.synapses(pre_ids=sorted(roots)); outs = outs[outs["pre_pt_root_id"].isin(roots)]
    im = meta.by_root.reindex(ins["pre_pt_root_id"].unique())
    om = meta.by_root.reindex(outs["post_pt_root_id"].unique())
    in_t45 = set(im[is_canonical_t4t5(im["cell_type"]).values].index)
    out_t45 = set(om[is_canonical_t4t5(om["cell_type"]).values].index)
    return len(in_t45 & out_t45)


def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    sheet = SH.get_sheet(src, meta, cfg)
    sheet_set = set(int(x) for x in sheet.llpc1_roots)

    table1 = {}
    for t in C.INHIBITOR_SCREEN_TYPES:
        # Score the whole TYPE (all its cells, both hemispheres). Table 1's "/100" sheet
        # contact and reciprocal counts are per-type aggregates, not single-exemplar — so
        # e.g. LT33 (right exemplar contacts 0) scores 7 via the type, matching the paper.
        roots = [int(r) for r in meta.root_ids_of_type([t])]
        sc = str(meta.by_root.loc[roots[0], "super_class"])
        llpc1 = _llpc1_contact(src, meta, roots, sheet_set)
        recip = _t4t5_reciprocal(src, meta, roots)
        table1[t] = {
            "roots": roots, "super_class": sc, "llpc1": llpc1,
            "reciprocal": recip, "is_centrifugal": sc == "visual_centrifugal",
        }

    # Pass-all-three: centrifugal AND contacts a meaningful share of the sheet AND
    # is T4/T5-reciprocal. "Sheet-contacting" for the *gate* role means it reaches the
    # sheet at all while being centrifugal; CT1 is reciprocal+centrifugal-adjacent but
    # contacts only ~3/100. Use the paper's logic: centrifugal AND sheet>=50 (majority).
    pass_all = {t for t, r in table1.items()
                if r["is_centrifugal"] and r["llpc1"] >= 50 and r["reciprocal"] > 0}
    return {"table1": table1, "pass_all_three": pass_all, "track": src.track}
