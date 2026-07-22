"""Identify the canonical LLPC1 sheet and its T4a drivers — the spatial substrate that
families B-G all build on.

Definition (Methods / Fig 3 / Table S2): the ``cfg.sheet_side`` LLPC1 cells reached by the
same-side ``cfg.exemplar_t4`` (T4a) that are postsynaptic to the gating VCH
(``cfg.vch_root``, the crossing centrifugal cell — "VCH-gated T4a"). With the default
``cfg=C.RIGHT`` this reproduces the paper exactly: 454 VCH-gated right T4a -> 9,223
synapses -> 100 right LLPC1. With ``cfg=C.LEFT`` it derives the homologous left sheet
(gated by the right-soma VCH), the Stage-8 mirror candidate.

Cached per (source track, sheet side) so right and left runs never collide and every
family on a given side reuses one pull.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..oracle import consts as C


@dataclass
class Sheet:
    llpc1_roots: np.ndarray          # the 100 sheet LLPC1 (right)
    t4a_roots: np.ndarray            # the 454 VCH-gated right T4a
    t4a_llpc1_syn: pd.DataFrame      # T4a->LLPC1 synapses (sheet only)
    n_llpc1: int
    n_t4a: int
    n_t4a_llpc1_syn: int


# Cache key is (source track, sheet side): a left run after a right run must NOT reuse the
# right sheet (the previous bug, when keyed by track alone).
_CACHE: dict[tuple[str, str], Sheet] = {}


def get_sheet(src, meta, cfg: C.SideConfig = C.RIGHT) -> Sheet:
    key = (src.track, cfg.sheet_side)
    if key in _CACHE:
        return _CACHE[key]

    # 1. same-side T4a postsynaptic to the gating VCH (VCH-gated).
    vch_out = src.synapses(pre_ids=[cfg.vch_root])
    vch_out = vch_out[vch_out["pre_pt_root_id"] == cfg.vch_root]
    out_meta = meta.by_root.reindex(vch_out["post_pt_root_id"].unique())
    t4a_roots = out_meta[(out_meta["cell_type"] == cfg.exemplar_t4)
                         & (out_meta["side"] == cfg.sheet_side)].index.to_numpy()

    # 2. those T4a -> same-side LLPC1.
    t4a_out = src.synapses(pre_ids=t4a_roots.tolist())
    t4a_out = t4a_out[t4a_out["pre_pt_root_id"].isin(set(t4a_roots.tolist()))]
    post_meta = meta.by_root.reindex(t4a_out["post_pt_root_id"].values)
    is_sheet = ((post_meta["cell_type"].values == "LLPC1")
                & (post_meta["side"].values == cfg.sheet_side))
    t4a_llpc1 = t4a_out[is_sheet].copy()
    llpc1_roots = np.array(sorted(t4a_llpc1["post_pt_root_id"].unique()), dtype=np.int64)

    sheet = Sheet(
        llpc1_roots=llpc1_roots,
        t4a_roots=np.array(sorted(t4a_roots), dtype=np.int64),
        t4a_llpc1_syn=t4a_llpc1,
        n_llpc1=int(len(llpc1_roots)),
        n_t4a=int(len(t4a_roots)),
        n_t4a_llpc1_syn=int(len(t4a_llpc1)),
    )
    _CACHE[key] = sheet
    return sheet
