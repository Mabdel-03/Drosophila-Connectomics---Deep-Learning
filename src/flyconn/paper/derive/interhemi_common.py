"""Shared helpers for the Stage-9 inter-hemispheric coupling analysis.

The central methodological point: a true inter-hemispheric crossing is defined by where a synapse
physically sits relative to the optic-lobe midline, NOT by the soma-side annotation. Some figure-
circuit cells (the centrifugal VCH/DCH) have their soma on one side and their entire arbor in the
opposite lobe, so the soma-side label (``meta.by_root["side"]``) conflates arbor geometry with
genuine axonal crossing. We therefore work in synapse position space.

Coordinate-frame caveat (verified): ``soma_x`` (neurons.parquet, raw units ~10^5) and synapse
``pt_position_x`` (CAVE, micrometres after /1000, range ~270-780 um) are in DIFFERENT frames, so
the soma_x median is useless as a synapse-space midline. The synapse-x axis itself, however,
separates the two optic lobes cleanly (left-lobe LLPC1 dendrites near 339 um, right-lobe near
730 um), so the midline is derived from the synapse-x distribution of a bilateral reference set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import geometry as G
from . import common as CM

# Reference cell type whose dendritic input cleanly occupies one optic lobe per side, used to
# locate the synapse-space midline. LLPC1 is the figure sheet; its dendrite is strictly same-side.
_MIDLINE_REF_TYPE = "LLPC1"
_MIDLINE_CACHE: dict[str, dict] = {}


def synapse_space_midline(src, meta, ref_type: str = _MIDLINE_REF_TYPE) -> dict:
    """Locate the optic-lobe midline on the synapse position-x axis (micrometres).

    Pulls the dendritic input synapses of the left and right ``ref_type`` cells, takes the median
    synapse-x on each side, and returns their midpoint as the midline. Also returns a self-test:
    the two sides must be well separated (their medians differ by much more than their spread) for
    the midline to be meaningful. Cached per source track.
    """
    key = getattr(src, "track", "default")
    if key in _MIDLINE_CACHE:
        return _MIDLINE_CACHE[key]

    sides = {}
    for side in ("left", "right"):
        roots = [int(x) for x in meta.root_ids_of_type([ref_type], side=side)]
        if not roots:
            continue
        ins = src.synapses(post_ids=roots)
        ins = ins[ins["post_pt_root_id"].isin(set(roots))]
        x = G.syn_positions_um(ins, "post")[:, 0]
        if len(x) == 0:
            continue
        sides[side] = {
            "median_x": float(np.median(x)),
            "p10_x": float(np.percentile(x, 10)),
            "p90_x": float(np.percentile(x, 90)),
            "n": int(len(x)),
        }

    out: dict = {"ref_type": ref_type, "sides": sides}
    if len(sides) == 2:
        lo = min(sides["left"]["median_x"], sides["right"]["median_x"])
        hi = max(sides["left"]["median_x"], sides["right"]["median_x"])
        midline = 0.5 * (lo + hi)
        # which lobe is on which side of the midline (right lobe has the larger synapse-x here)
        right_is_high = sides["right"]["median_x"] > sides["left"]["median_x"]
        # separation self-test: the gap between side medians vs the within-side spread
        spread = 0.5 * ((sides["left"]["p90_x"] - sides["left"]["p10_x"])
                        + (sides["right"]["p90_x"] - sides["right"]["p10_x"]))
        gap = hi - lo
        out.update({
            "midline_x_um": round(midline, 1),
            "right_lobe_is_high_x": bool(right_is_high),
            "side_gap_um": round(gap, 1),
            "within_side_spread_um": round(spread, 1),
            "separation_ratio": round(gap / spread, 2) if spread else float("inf"),
            "self_test_ok": bool(gap > 2.0 * spread),  # sides clearly separated
        })
    else:
        out.update({"midline_x_um": None, "self_test_ok": False,
                    "reason": "need both left and right reference cells"})
    _MIDLINE_CACHE[key] = out
    return out


def position_side(points_um: np.ndarray, midline: dict) -> np.ndarray:
    """Classify each point (N,3 um) as the lobe side 'left'/'right' by its x vs the midline.

    Uses ``right_lobe_is_high_x`` so the labels match the annotation convention regardless of the
    raw axis orientation.
    """
    mid = midline["midline_x_um"]
    high_is_right = midline["right_lobe_is_high_x"]
    x = np.asarray(points_um, dtype=float)[:, 0]
    is_high = x > mid
    if high_is_right:
        return np.where(is_high, "right", "left")
    return np.where(is_high, "left", "right")


def dendrite_side_map(src, meta, roots, midline: dict) -> dict:
    """Per-cell dendrite (input-field) lobe for a set of roots, in ONE batched input pull.

    A synapse is a single physical point, so an inter-hemispheric crossing is NOT a pre-vs-post
    comparison within a synapse; it is that a cell's OUTPUT lands in the lobe OPPOSITE its
    dendritic (input) field. This returns each cell's dendrite side, the reference for its output.
    """
    roots = [int(x) for x in roots]
    if not roots:
        return {}
    ins = src.synapses(post_ids=roots)
    ins = ins[ins["post_pt_root_id"].isin(set(roots))]
    if len(ins) == 0:
        return {r: None for r in roots}
    side = position_side(G.syn_positions_um(ins, "post"), midline)
    df = pd.DataFrame({"root": ins["post_pt_root_id"].to_numpy(), "side": side})
    out = {}
    for r, g in df.groupby("root"):
        vals, counts = np.unique(g["side"].to_numpy(), return_counts=True)
        out[int(r)] = str(vals[int(np.argmax(counts))])
    for r in roots:
        out.setdefault(r, None)
    return out


def contra_output_by_position(src, meta, roots, midline: dict, dend_map: dict | None = None) -> dict:
    """Fraction of a cell set's OUTPUT synapses that land in the lobe OPPOSITE its dendrite.

    Position-based analogue of the soma-side ``_nod1_contra``, immune to the soma-annotation
    artifact. Batched: one input pull (for dendrite sides) and one output pull. ``dend_map`` may
    be supplied to reuse a precomputed dendrite-side map.
    """
    roots = [int(x) for x in roots]
    if not roots:
        return {"n_out": 0, "n_cross": 0, "cross_frac": float("nan")}
    dend = dend_map if dend_map is not None else dendrite_side_map(src, meta, roots, midline)
    out = src.synapses(pre_ids=roots)
    out = out[out["pre_pt_root_id"].isin(set(roots))]
    if len(out) == 0:
        return {"n_out": 0, "n_cross": 0, "cross_frac": float("nan")}
    out_side = position_side(G.syn_positions_um(out, "pre"), midline)
    pre_dend = pd.Series(out["pre_pt_root_id"].to_numpy()).map(dend).to_numpy()
    valid = pd.notna(pre_dend)
    n_cross = int(np.sum((out_side != pre_dend) & valid))
    return {"n_out": int(valid.sum()), "n_cross": n_cross,
            "cross_frac": round(100.0 * n_cross / max(int(valid.sum()), 1), 1)}


def cross_synapses_to_targets(src, meta, pre_roots, midline: dict, dend_map: dict | None = None) -> pd.DataFrame:
    """Output synapses of ``pre_roots`` that land in the lobe opposite each cell's dendrite,
    annotated with post-target cell type / side / NT. Batched (one input + one output pull).
    """
    pre_roots = [int(x) for x in pre_roots]
    if not pre_roots:
        return pd.DataFrame()
    dend = dend_map if dend_map is not None else dendrite_side_map(src, meta, pre_roots, midline)
    out = src.synapses(pre_ids=pre_roots)
    out = out[out["pre_pt_root_id"].isin(set(pre_roots))].copy()
    if len(out) == 0:
        return pd.DataFrame()
    out["out_side"] = position_side(G.syn_positions_um(out, "pre"), midline)
    out["pre_dend_side"] = pd.Series(out["pre_pt_root_id"].to_numpy()).map(dend).to_numpy()
    cross = out[(out["pre_dend_side"].notna()) & (out["out_side"] != out["pre_dend_side"])].copy()
    if len(cross) == 0:
        return cross
    pm = meta.by_root.reindex(cross["post_pt_root_id"].values)
    cross["post_ct"] = pm["cell_type"].values
    cross["post_side"] = pm["side"].values
    cross["post_nt"] = pm["nt_canonical"].values
    return cross


def contra_output_by_soma(src, meta, roots) -> dict:
    """Soma-side contralateral output fraction (the pre_side != post_side annotation pattern).

    Kept so the soma-based and position-based definitions can be reported side by side; the
    difference between them is the arbor-vs-soma laterality artifact.
    """
    roots = [int(x) for x in roots]
    if not roots:
        return {"n_out": 0, "n_cross": 0, "cross_frac": float("nan")}
    out = src.synapses(pre_ids=roots)
    out = out[out["pre_pt_root_id"].isin(set(roots))]
    ps = meta.by_root.reindex(out["pre_pt_root_id"].values)["side"].to_numpy()
    qs = meta.by_root.reindex(out["post_pt_root_id"].values)["side"].to_numpy()
    valid = pd.notna(ps) & pd.notna(qs)
    n_cross = int(np.sum((ps != qs) & valid))
    return {"n_out": int(valid.sum()), "n_cross": n_cross,
            "cross_frac": round(100.0 * n_cross / max(int(valid.sum()), 1), 1)}


def input_layer_profile(src, meta, roots) -> dict:
    """Lobula-plate layer composition of a cell set's T4/T5 input (a/b/c/d fractions + dominant).

    Used to test Egelhaaf's prediction that contralateral inhibitory bridges onto the FD1/Nod1
    pathway are regressive (layer-b) tuned. a = front-to-back (progressive), b = back-to-front
    (regressive), c = upward, d = downward.
    """
    roots = [int(x) for x in roots]
    if not roots:
        return {"has_t4t5": False}
    syn = src.synapses(post_ids=roots)
    syn = syn[syn["post_pt_root_id"].isin(set(roots))]
    counts = CM.attach_meta(CM.partner_counts(syn, "pre_pt_root_id"), meta)
    t45 = counts[counts["is_t4t5"]]
    if len(t45) == 0:
        return {"has_t4t5": False}
    layer = {k: CM.layer_fraction(t45, k) for k in ("a", "b", "c", "d")}
    dom = CM.dominant_layer(t45)
    return {
        "has_t4t5": True,
        "layer_frac": {k: (round(v, 1) if v == v else None) for k, v in layer.items()},
        "dominant_layer": dom,
        "dominant_direction": {"a": "progressive", "b": "regressive",
                               "c": "upward", "d": "downward"}.get(dom),
        "n_t4t5": int(len(t45)),
    }
