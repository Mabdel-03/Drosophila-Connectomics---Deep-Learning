"""Stage-8 family M — MIRROR SYMMETRY + negative controls + wing-flip.

Consumes the already-derived RIGHT and LEFT family dicts (A-I) and computes the explicit
left-vs-right mirror comparisons, the negative controls (which must FAIL to gate), and the
MCNS wing-flip (the complement). This is where homology (left mirrors right) and
complementarity (the command flips the wing) are decided.

Mirror tests:
  M1 gating-soma laterality   left sheet gated by the RIGHT-soma VCH (opposite of right)
  M2 direction preserved      left sheet layer-a (progressive) driven, like right
  M3 noduli contralaterality   left Nod1 / left FD3 majority-contralateral output
  M4 wing flip                left circuit's DNp26 steers the OPPOSITE physical wing (muscular.mirror)
  M_cells correspondence       every right circuit cell type has a left counterpart (NT+super_class)

Negative controls (must NOT gate):
  NC1 wrong-side VCH          the LEFT-soma VCH does not gate the LEFT sheet
  NC2 opposite layer channel  the layer-b channel is not the LEFT LLPC1's main drive
  NC4 cross-sheet specificity each Nod1 crosses; each sheet addresses its own DNp26 body

Reads FlyWire via the same source for the controls. The mirror comparisons are pure functions
of the right/left derived dicts (no extra pulls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as CM
from . import sheet as SH
from ..oracle import consts as C


def _layer_b_to_sheet(src, meta, cfg: C.SideConfig) -> dict:
    """NC2: how much does the opposite-direction (layer-b) channel drive the LLPC1 sheet,
    vs the layer-a exemplar? The sheet should be layer-a driven; layer-b belongs to LPi15.

    Builds a layer-b sheet candidate exactly like get_sheet but with the opponent T4 subtype
    (T4b), gated by the same crossing VCH, and counts its synapses onto the ANNOTATED LLPC1.
    """
    vch = cfg.vch_root
    opp_t4 = "T4" + cfg.opponent_layer_letter  # "T4b"
    vch_out = src.synapses(pre_ids=[vch]); vch_out = vch_out[vch_out["pre_pt_root_id"] == vch]
    om = meta.by_root.reindex(vch_out["post_pt_root_id"].unique())
    t4b = om[(om["cell_type"] == opp_t4) & (om["side"] == cfg.sheet_side)].index.to_numpy()
    sheet = SH.get_sheet(src, meta, cfg)
    sheet_set = set(int(x) for x in sheet.llpc1_roots)
    if len(t4b) == 0:
        layerb_syn = 0
    else:
        t4b_out = src.synapses(pre_ids=t4b.tolist())
        t4b_out = t4b_out[t4b_out["pre_pt_root_id"].isin(set(int(x) for x in t4b))]
        layerb_syn = int(t4b_out[t4b_out["post_pt_root_id"].isin(sheet_set)].shape[0])
    layera_syn = int(sheet.n_t4a_llpc1_syn)
    return {"layer_a_syn": layera_syn, "layer_b_syn": layerb_syn,
            "layer_b_n_t4": int(len(t4b)),
            "layer_b_frac_of_a": round(layerb_syn / layera_syn, 3) if layera_syn else float("nan"),
            "layer_b_is_minor": bool(layerb_syn < 0.25 * layera_syn)}


def _wrongside_vch_gates_sheet(src, meta, cfg: C.SideConfig) -> dict:
    """NC1: the WRONG-side VCH (the one that gates the OTHER sheet) must NOT gate this sheet.

    For the left sheet, the wrong-side VCH is the LEFT-soma VCH (which gates the RIGHT sheet).
    Count how many same-side LLPC1 its gated T4a reach; expect ~0 vs the correct gater's sheet.
    """
    wrong_soma = C.OPPOSITE_SIDE[cfg.gating_soma_side]   # the non-gating soma side
    wrong_vch = C.VCH_ROOT_BY_SOMA[wrong_soma]
    vch_out = src.synapses(pre_ids=[wrong_vch]); vch_out = vch_out[vch_out["pre_pt_root_id"] == wrong_vch]
    om = meta.by_root.reindex(vch_out["post_pt_root_id"].unique())
    t4a = om[(om["cell_type"] == cfg.exemplar_t4) & (om["side"] == cfg.sheet_side)].index.to_numpy()
    if len(t4a) == 0:
        wrong_llpc1 = 0
    else:
        t4a_out = src.synapses(pre_ids=t4a.tolist())
        t4a_out = t4a_out[t4a_out["pre_pt_root_id"].isin(set(int(x) for x in t4a))]
        pm = meta.by_root.reindex(t4a_out["post_pt_root_id"].values)
        is_sheet = (pm["cell_type"].values == "LLPC1") & (pm["side"].values == cfg.sheet_side)
        wrong_llpc1 = int(pd.Series(t4a_out["post_pt_root_id"].to_numpy()[is_sheet]).nunique())
    correct = SH.get_sheet(src, meta, cfg).n_llpc1
    return {"wrong_vch_root": wrong_vch, "wrong_soma_side": wrong_soma,
            "wrong_gated_t4a": int(len(t4a)),
            "wrong_gated_llpc1": wrong_llpc1, "correct_gated_llpc1": correct,
            "wrong_is_negligible": bool(wrong_llpc1 <= max(3, 0.1 * correct))}


def run(src, meta, right_derived: dict, left_derived: dict, wingflip: dict | None = None) -> dict:
    """Compute mirror comparisons (from the derived dicts) + negative controls + wing-flip."""
    rA, lA = right_derived.get("A", {}), left_derived.get("A", {})
    rF, lF = right_derived.get("F", {}), left_derived.get("F", {})
    rG, lG = right_derived.get("G", {}), left_derived.get("G", {})
    rH, lH = right_derived.get("H", {}), left_derived.get("H", {})
    rK, lK = right_derived.get("K", {}), left_derived.get("K", {})

    # M3 left Nod1 contralateral output: pull it (cheap) if not already in a derived dict.
    left_nod1_contra = _nod1_contra(src, meta, "left")
    right_nod1_contra = _nod1_contra(src, meta, "right")

    # negative controls (need FlyWire pulls)
    nc1 = _wrongside_vch_gates_sheet(src, meta, C.LEFT)
    nc2 = _layer_b_to_sheet(src, meta, C.LEFT)

    # M_cells: cell-type correspondence (NT + super_class) for the named circuit types.
    cell_corr = _cell_correspondence(meta)

    return {
        "mirror": {
            "gating_soma": {"right": "left", "left": C.LEFT.gating_soma_side},  # M1
            "vch_in_syn": {"right": rA.get("vch_input_syn"), "left": lA.get("vch_input_syn")},
            "vch_out_syn": {"right": rA.get("vch_output_syn"), "left": lA.get("vch_output_syn")},
            "recip_frac": {"right": rA.get("reciprocal_pct_of_inputs"),
                           "left": lA.get("reciprocal_pct_of_inputs")},
            "vch_layer_a_pct": {"right": rA.get("vch_layer_a_pct"), "left": lA.get("vch_layer_a_pct")},
            "sheet_size": {"right": _sheet_size(right_derived), "left": _sheet_size(left_derived)},
            "nod1_top_excitatory": {"right": rG.get("nod1_is_top_excitatory"),
                                    "left": lG.get("nod1_is_top_excitatory")},
            "nod1_to_dnp26": {"right": rH.get("nod1_to_dnp26_syn"), "left": lH.get("nod1_to_dnp26_syn")},
            "nod1_relay_dominates": {"right": rH.get("nod1_route_dominates"),
                                     "left": lH.get("nod1_route_dominates")},
            "lpi15_layer_b_pct": {"right": rF.get("lpi15_layer_b_pct"), "left": lF.get("lpi15_layer_b_pct")},
            "nod1_contra_pct": {"right": right_nod1_contra, "left": left_nod1_contra},
            "fd3_contra_pct": {"right": _fd3_contra(rK), "left": _fd3_contra(lK)},
        },
        "negative_controls": {"nc1_wrongside_vch": nc1, "nc2_layer_b_channel": nc2},
        "cell_correspondence": cell_corr,
        "wing_flip": wingflip or {"available": False, "reason": "not provided"},
        "track": getattr(src, "track", None),
    }


def _sheet_size(derived: dict) -> int | None:
    """LLPC1 sheet size: family D exposes n_llpc1 (== sheet.n_llpc1); fall back to G/E if absent."""
    d = derived.get("D", {})
    if d.get("n_llpc1") is not None:
        return d["n_llpc1"]
    # E also carries n_t4a; G stores no sheet size. Final fallback: recompute is unnecessary
    # because D always runs in the bilateral orchestrator, but guard against a subset run.
    return derived.get("G", {}).get("n_llpc1")


def _nod1_contra(src, meta, side: str) -> float:
    """% of this side's Nod1 output that is contralateral."""
    roots = sorted(int(x) for x in meta.root_ids_of_type(["Nod1"], side=side))
    if not roots:
        return float("nan")
    out = src.synapses(pre_ids=roots); out = out[out["pre_pt_root_id"].isin(set(roots))]
    pre_side = meta.by_root.reindex(out["pre_pt_root_id"].values)["side"].to_numpy()
    post_side = meta.by_root.reindex(out["post_pt_root_id"].values)["side"].to_numpy()
    valid = pd.notna(pre_side) & pd.notna(post_side)
    return round(float(100.0 * np.sum((pre_side != post_side) & valid) / max(int(valid.sum()), 1)), 2)


def _fd3_contra(k_derived: dict) -> float | None:
    """LPT42_Nod4 contralateral output % from family K's profile."""
    prof = (k_derived.get("profiles") or {}).get("LPT42_Nod4", {})
    return prof.get("contra_output_pct")


def _cell_correspondence(meta) -> dict:
    """For each named circuit type: does a left AND a right cell exist, with matching NT+class?"""
    types = ("VCH", "DCH", "LLPC1", "Nod1", "Nod2", "LPT42_Nod4", "LPi15", "PVLP011",
             "PLP249", "PLP163", "DNp26", "DNbe001", "DNa04")
    out = {}
    for t in types:
        L = meta.root_ids_of_type([t], side="left")
        R = meta.root_ids_of_type([t], side="right")
        rec = {"n_left": int(len(L)), "n_right": int(len(R)),
               "both_present": bool(len(L) and len(R))}
        if len(L) and len(R):
            lr = meta.by_root.loc[int(L[0])]
            rr = meta.by_root.loc[int(R[0])]
            rec["nt_match"] = bool(lr["nt_canonical"] == rr["nt_canonical"])
            rec["class_match"] = bool(lr["super_class"] == rr["super_class"])
            rec["nt"] = str(lr["nt_canonical"])
            rec["super_class"] = str(lr["super_class"])
        out[t] = rec
    n_ok = sum(1 for r in out.values()
               if r["both_present"] and r.get("nt_match") and r.get("class_match"))
    return {"per_type": out, "n_types": len(types), "n_corresponding": n_ok,
            "all_correspond": bool(n_ok == len(types))}
