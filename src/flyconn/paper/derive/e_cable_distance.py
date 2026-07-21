"""Derive family E: dual dendrite + Euclidean synapse-location analysis.

Primary metric is Euclidean distance between synapse points (the paper states the result
is sign-identical under a Euclidean metric). Three analyses:
  1. LLPC1 input composition: % motion (T4/T5) vs % form (Tm/TmY/Y/Tlp/Li).
  2. Two-field separation: per LLPC1, centroid distance between its T4/T5 motion synapses
     and its Tm/TmY form synapses; median over the sheet.
  3a. T4a terminal gating: per T4a, min distance from its VCH-input synapses to its
      T4a->LLPC1 output terminal, vs the same for its OTHER inputs; fraction where VCH is
      closer, and a Wilcoxon signed-rank test.
  3b. LLPC1 compartments: per LLPC1, min distance from each inhibitory class's synapse to
      the nearest T4a-excitation synapse on that LLPC1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import common as CM
from . import sheet as SH
from .. import geometry as G
from ..oracle import consts as C

FORM_TYPES_PREFIX = ("Tm", "TmY", "Y", "Tlp", "Li")
MAX_T4A_FOR_TERMINAL = 286  # paper's skeleton-resolved subset


def _is_form(ct: pd.Series) -> np.ndarray:
    s = ct.astype("string").fillna("")
    return s.str.startswith(FORM_TYPES_PREFIX).to_numpy()


def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    sheet = SH.get_sheet(src, meta, cfg)
    sheet_roots = sheet.llpc1_roots
    sheet_set = set(int(x) for x in sheet_roots)

    # ---- 1. input composition (all synapses ONTO the sheet) ----
    sin = src.synapses(post_ids=sheet_roots.tolist())
    sin = sin[sin["post_pt_root_id"].isin(sheet_set)]
    pm = meta.by_root.reindex(sin["pre_pt_root_id"].values)
    is_motion = CM.is_canonical_t4t5(pm["cell_type"]).to_numpy()
    is_form = _is_form(pm["cell_type"])
    total = len(sin)
    motion_pct = round(100.0 * is_motion.sum() / total, 1)
    form_pct = round(100.0 * is_form.sum() / total, 1)

    # ---- 2. two-field separation (per LLPC1) ----
    sin = sin.reset_index(drop=True)
    post = sin["post_pt_root_id"].to_numpy()
    pos_on_llpc1 = G.syn_positions_um(sin, "post")
    seps = []
    for llpc1 in sheet_set:
        m = post == llpc1
        mot = pos_on_llpc1[m & is_motion]
        frm = pos_on_llpc1[m & is_form]
        if len(mot) >= 3 and len(frm) >= 3:
            seps.append(G.two_field_separation_um(mot, frm))
    field_sep_um = float(np.median(seps)) if seps else float("nan")

    # ---- 3a. T4a terminal gating ----
    # For each VCH-gated T4a: its VCH-input synapse positions, its OTHER-input positions,
    # and its T4a->LLPC1 output-terminal positions. Min Euclidean distance to the terminal.
    t4a_subset = sheet.t4a_roots[:MAX_T4A_FOR_TERMINAL]
    t4a_set = set(int(x) for x in t4a_subset)
    # output terminals: the T4a->sheet-LLPC1 synapses (pre side = on the T4a axon).
    term = sheet.t4a_llpc1_syn
    term = term[term["pre_pt_root_id"].isin(t4a_set)]
    term_pos = {int(t): G.syn_positions_um(g, "pre")
                for t, g in term.groupby("pre_pt_root_id")}
    # inputs onto these T4a:
    t4a_in = src.synapses(post_ids=sorted(t4a_set))
    t4a_in = t4a_in[t4a_in["post_pt_root_id"].isin(t4a_set)]
    t4a_in_meta = meta.by_root.reindex(t4a_in["pre_pt_root_id"].values).reset_index(drop=True)
    t4a_in = t4a_in.reset_index(drop=True)
    t4a_in["pre_ct"] = t4a_in_meta["cell_type"].values
    t4a_in["pre_root"] = t4a_in["pre_pt_root_id"].to_numpy()
    in_pos = G.syn_positions_um(t4a_in, "post")  # synapse location on the T4a

    # For each T4a: median distance from its VCH-input synapses (and, separately, its
    # OTHER-input synapses) to the NEAREST T4a->LLPC1 output terminal. The per-synapse
    # nearest-terminal median is the paper's metric; a single global min would be
    # dominated by whichever lone "other" synapse sits near the terminal.
    vch_d, other_d = [], []
    for t in t4a_set:
        if t not in term_pos:
            continue
        tp = term_pos[t]
        rows = t4a_in["post_pt_root_id"].to_numpy() == t
        # The GATING VCH specifically (cfg.vch_root): for the left sheet this is the right-soma
        # VCH, so filter by root, not by the "VCH" type (which has a left+right cell).
        is_vch_in = rows & (t4a_in["pre_root"].to_numpy() == cfg.vch_root)
        is_other = rows & (t4a_in["pre_root"].to_numpy() != cfg.vch_root)
        if is_vch_in.sum() == 0 or is_other.sum() == 0:
            continue
        vch_d.append(G.median_nearest_distance_um(in_pos[is_vch_in], tp))
        other_d.append(G.median_nearest_distance_um(in_pos[is_other], tp))
    vch_d = np.array(vch_d); other_d = np.array(other_d)
    n_t4a = int(len(vch_d))
    vch_to_terminal_um = float(np.median(vch_d)) if n_t4a else float("nan")
    other_to_terminal_um = float(np.median(other_d)) if n_t4a else float("nan")
    vch_closest_frac = round(100.0 * float(np.mean(vch_d < other_d)), 1) if n_t4a else float("nan")
    try:
        wilcoxon_p = float(wilcoxon(vch_d, other_d, alternative="less").pvalue) if n_t4a > 10 else float("nan")
    except ValueError:
        wilcoxon_p = float("nan")

    # ---- 3b. LLPC1 compartments ----
    # T4a-excitation synapse positions on each LLPC1 (post side = on the LLPC1 dendrite).
    exc = sheet.t4a_llpc1_syn
    exc_pos_by_llpc1 = {int(l): G.syn_positions_um(g, "post")
                        for l, g in exc.groupby("post_pt_root_id")}

    def _inhib_to_exc(inhib_type: str, want_side: str | None) -> float:
        roots = (meta.root_ids_of_type([inhib_type], side=want_side)
                 if want_side else meta.root_ids_of_type([inhib_type]))
        if inhib_type == "VCH":
            roots = np.array([cfg.vch_root])
        if len(roots) == 0:
            return float("nan")
        ins = src.synapses(pre_ids=[int(r) for r in roots])
        ins = ins[ins["pre_pt_root_id"].isin(set(int(r) for r in roots))
                  & ins["post_pt_root_id"].isin(sheet_set)]
        ds = []
        for llpc1, g in ins.groupby("post_pt_root_id"):
            if int(llpc1) not in exc_pos_by_llpc1:
                continue
            ip = G.syn_positions_um(g, "post")
            # median distance from each inhibitory synapse to its nearest T4a-excitation
            # synapse on the same LLPC1 (the paper's compartment metric).
            ds.append(G.median_nearest_distance_um(ip, exc_pos_by_llpc1[int(llpc1)]))
        return float(np.median(ds)) if ds else float("nan")

    lpi15_to_exc = _inhib_to_exc("LPi15", cfg.sheet_side)
    vch_to_exc = _inhib_to_exc("VCH", None)
    pvlp011_to_exc = _inhib_to_exc("PVLP011", None)

    return {
        "motion_pct": motion_pct,
        "form_pct": form_pct,
        "field_sep_um": field_sep_um,
        "vch_to_terminal_um": vch_to_terminal_um,
        "other_to_terminal_um": other_to_terminal_um,
        "vch_closest_frac": vch_closest_frac,
        "wilcoxon_p": wilcoxon_p,
        "n_t4a": n_t4a,
        "lpi15_to_exc_um": lpi15_to_exc,
        "vch_to_exc_um": vch_to_exc,
        "pvlp011_to_exc_um": pvlp011_to_exc,
        "track": src.track,
    }
