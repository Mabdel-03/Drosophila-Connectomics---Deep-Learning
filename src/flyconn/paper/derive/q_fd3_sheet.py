"""Derive family Q: the FD3 intermediate sheet(-set) — the "T4b/T5b -> ??? -> FD3" stage.

The mentor asked us to NAME the columnar cell that sits between the T4/T5 motion detectors and
FD3, the role LLPC1 plays for FD1. This family measures, for each candidate columnar-projection
sheet that feeds FD3, whether it is the direction-matched (layer-b/regressive), cholinergic,
retinotopic, feed-forward sheet — mirroring the FD1 machinery (``sheet.py`` + Family C
uniqueness + Family D retinotopy null).

Measured result (v783, this session): FD3's sheet inputs read DIFFERENT motion channels —
  LPC1  layer-b (regressive 97.2%)  -> FD3 1120 syn   <- the direction-matched horizontal sheet
  LLPC2 layer-c (upward 97.6%)      -> FD3  979 syn   } vertical-motion CONTEXT, not the same role
  LLPC3 layer-d (downward 96.3%)    -> FD3 1071 syn   }
  LPC2  layer-c (upward)            -> FD3  176 syn
  LLPC1 layer-a (progressive)       -> FD3  446 syn   <- FD1's sheet; a NEGATIVE control here
So LPC1 is the plurality AND the unique layer-b member. We name the intermediate as the
SET {LPC1 (horizontal/regressive) + LLPC2/LLPC3 (orthogonal vertical context)}, with LPC1 the
direction-matched horizontal sheet; the uniqueness is reported as a measured OUTCOME, not asserted.

Source: mirrors derive/sheet.py + oracle/c_sheet_elimination.py + derive/d_retinotopy_null.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import common as CM
from . import interhemi_common as IC
from .k_fd3_lpt42 import _CachedSource
from .. import geometry as G
from ..oracle import consts as C

FD3 = "LPT42_Nod4"
# Candidate columnar-projection sheets feeding FD3 (from consts.COLUMNAR_PROJECTION_SHEETS,
# LLPC1 kept as the FD1-sheet negative control).
CANDIDATES = ("LPC1", "LLPC2", "LLPC3", "LPC2", "LLPC1")
NAMED_SHEET = "LPC1"                 # the direction-matched (layer-b) horizontal sheet
DETECTORS = ("T4b", "T5b")          # the layer-b channel FD3 reads
FD3_ANCHOR = "Nod1"                 # = FD1; reference cell for the "is 4-direction pooling generic" test
N_PERMS = 500
SEED = 12345
MIN_SYN_TO_FD3 = 30                 # a candidate must clear this to count as an FD3 sheet input
# a sheet's dominant-layer label is "hard" only if the top layer beats the runner-up by this margin
DIR_MARGIN_PP = 20.0
# and if it survives resampling the sheet's T4/T5 input synapses this fraction of the time
N_DIR_BOOT = 1000
DIR_STABLE_FRAC = 0.95
# random-tangential entropy null: how many LP-tangential cell TYPES to sample for the background
# dist. ~120 gives a stable percentile; each is a feather rescan offline, so keep it bounded.
N_ENTROPY_NULL = 120
_CARDINAL = ("a", "b", "c", "d")


def _fd3_roots(meta, candidate: str = FD3) -> list[int]:
    return sorted(int(x) for x in meta.root_ids_of_type([candidate]))


def _candidate_profile(src, meta, ct: str, fd3_set: set[int]) -> dict:
    """For a candidate sheet: its layer tuning, its ->FD3 syn, its T4b/T5b (layer-b) drive."""
    roots = sorted(int(x) for x in meta.root_ids_of_type([ct]))
    if not roots:
        return {"cell_type": ct, "present": False}
    rset = set(roots)
    # own input layer profile (which motion channel the sheet reads)
    prof = IC.input_layer_profile(src, meta, roots)
    # its output onto FD3
    out = src.synapses(pre_ids=roots); out = out[out["pre_pt_root_id"].isin(rset)]
    to_fd3 = int(len(out[out["post_pt_root_id"].isin(fd3_set)]))
    # its layer-b (T4b/T5b) input synapse count (the direction-matched drive)
    si = src.synapses(post_ids=roots); si = si[si["post_pt_root_id"].isin(rset)]
    ic = CM.attach_meta(CM.partner_counts(si, "pre_pt_root_id"), meta)
    t45 = ic[ic["is_t4t5"]]
    layer_b_syn = int(t45.loc[t45["t4t5_subtype"] == "b", "syn"].sum())
    nt = str(meta.by_root.loc[roots[0], "nt_canonical"])
    sc = str(meta.by_root.loc[roots[0], "super_class"])
    # direction-label confidence: top-minus-runner-up margin + bootstrap stability of the argmax.
    dir_conf = _direction_confidence(t45)
    return {
        "cell_type": ct, "present": True, "n_cells": len(roots),
        "nt": nt, "super_class": sc,
        "dominant_layer": prof.get("dominant_layer"),
        "dominant_direction": prof.get("dominant_direction"),
        "layer_frac": prof.get("layer_frac"),
        "to_fd3_syn": to_fd3,
        "layer_b_input_syn": layer_b_syn,
        "reads_layer_b": bool(prof.get("dominant_layer") == "b"),
        "dominant_layer_letter": prof.get("dominant_layer"),
        "dir_margin_pp": dir_conf["margin_pp"],
        "dir_boot_stability": dir_conf["boot_stability"],
        "dir_hard": bool(dir_conf["margin_pp"] is not None
                         and dir_conf["margin_pp"] >= DIR_MARGIN_PP
                         and (dir_conf["boot_stability"] or 0) >= DIR_STABLE_FRAC),
    }


def _direction_confidence(t45: pd.DataFrame) -> dict:
    """How confident is a sheet's dominant-layer (direction) call?

    A sheet's "direction" is an inferred argmax over its a/b/c/d T4/T5 input, not a certainty. We
    report (a) the top-minus-runner-up percentage-point margin and (b) a bootstrap stability: the
    fraction of resamples of the sheet's T4/T5 input synapses in which the dominant layer is
    unchanged. Only a wide-margin, stable sheet earns a "hard" direction label; a near-tie is
    reported as a tendency. Mirrors the synapse-resampling idea in k_fd3_lpt42._bootstrap_offset_ci.
    """
    if t45.empty:
        return {"margin_pp": None, "boot_stability": None}
    by = t45.groupby("t4t5_subtype")["syn"].sum()
    tot = float(by.sum())
    if tot <= 0:
        return {"margin_pp": None, "boot_stability": None}
    fracs = (by / tot * 100.0).sort_values(ascending=False)
    top = fracs.index[0]
    margin = float(fracs.iloc[0] - (fracs.iloc[1] if len(fracs) > 1 else 0.0))
    # bootstrap over the per-partner synapse counts (weighted resample of the layers)
    letters = by.index.to_numpy()
    weights = by.to_numpy().astype(float)
    p = weights / weights.sum()
    n_draw = int(round(tot))
    rng = np.random.default_rng(SEED)
    same = 0
    for _ in range(N_DIR_BOOT):
        draw = rng.multinomial(n_draw, p)
        if letters[int(np.argmax(draw))] == top:
            same += 1
    return {"margin_pp": round(margin, 1), "boot_stability": round(same / N_DIR_BOOT, 3)}


def _t45_field_positions(src, meta, sheet_roots: list[int]) -> dict[int, np.ndarray]:
    """Mean um position of each layer-b detector's output synapses (its retinotopic locus).

    Pool = the T4b/T5b cells presynaptic to the named sheet (their output-synapse centroids).
    """
    si = src.synapses(post_ids=sheet_roots)
    si = si[si["post_pt_root_id"].isin(set(sheet_roots))]
    ic = CM.attach_meta(CM.partner_counts(si, "pre_pt_root_id"), meta)
    det = ic[ic["cell_type"].isin(DETECTORS)]
    det_roots = [int(x) for x in det["root_id"]]
    if not det_roots:
        return {}
    out = src.synapses(pre_ids=det_roots)
    out = out[out["pre_pt_root_id"].isin(set(det_roots))]
    pos = G.syn_positions_um(out, "pre")
    df = pd.DataFrame({"det": out["pre_pt_root_id"].to_numpy(),
                       "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]})
    return {int(t): g[["x", "y", "z"]].mean().to_numpy() for t, g in df.groupby("det")}


def _retinotopy_null(src, meta, sheet_ct: str) -> dict:
    """Per-sheet-cell layer-b input-patch locality vs an in-degree-preserving permutation null.

    Mirrors d_retinotopy_null but for the named FD3 sheet: each sheet cell keeps its number of
    distinct T4b/T5b drivers but draws which ones at random from the detector pool; recompute the
    hex/field patch radius. Local pooling (obs << null) => a bona-fide retinotopic sheet.
    """
    sheet_roots = sorted(int(x) for x in meta.root_ids_of_type([sheet_ct]))
    if not sheet_roots:
        return {"available": False}
    det_field = _t45_field_positions(src, meta, sheet_roots)
    if len(det_field) < 5:
        return {"available": False, "reason": "too few detector loci"}
    pool = np.vstack([det_field[t] for t in sorted(det_field)])
    n_pool = len(pool)

    # per sheet cell: its distinct detector drivers
    si = src.synapses(post_ids=sheet_roots)
    si = si[si["post_pt_root_id"].isin(set(sheet_roots))]
    sm = meta.by_root.reindex(si["pre_pt_root_id"].values)
    is_det = sm["cell_type"].isin(DETECTORS).to_numpy()
    sub = si[is_det]
    grp = sub.groupby("post_pt_root_id")["pre_pt_root_id"].apply(lambda s: sorted(set(int(x) for x in s)))

    obs_radii, indeg = [], []
    for _cell, drivers in grp.items():
        dpos = np.vstack([det_field[int(t)] for t in drivers if int(t) in det_field])
        if len(dpos) < 2:
            continue
        obs_radii.append(G.patch_radius_um(dpos, "median"))
        indeg.append(len(dpos))
    if not obs_radii:
        return {"available": False, "reason": "no sheet cell with >=2 detector drivers"}
    obs_median = float(np.median(obs_radii))

    rng = np.random.default_rng(SEED)
    null_medians = np.empty(N_PERMS)
    for p in range(N_PERMS):
        radii = []
        for k in indeg:
            sel = rng.choice(n_pool, size=min(k, n_pool), replace=False)
            radii.append(G.patch_radius_um(pool[sel], "median"))
        null_medians[p] = np.median(radii)
    null_mean = float(np.mean(null_medians)); null_std = float(np.std(null_medians))
    z = (obs_median - null_mean) / null_std if null_std else float("nan")
    p_value = max(float((null_medians <= obs_median).mean()), 1.0 / N_PERMS)
    return {"available": True, "sheet": sheet_ct,
            "obs_radius_um": round(obs_median, 3), "null_radius_um": round(null_mean, 3),
            "z_score": round(z, 2), "p_value": p_value, "n_perms": N_PERMS,
            "local": bool(obs_median < null_mean)}


def _directional_composition(profiles: dict) -> dict:
    """Roll each sheet's ->FD3 synapses up by its dominant lobula-plate layer (a/b/c/d) to describe
    the DIRECTIONAL composition of FD3's sheet-relayed input. PURE: consumes the already-computed
    ``profiles`` (no connectome pulls). Only sheets with a "hard" direction label contribute to the
    per-channel roll-up; near-tie sheets are listed separately so a soft call can't inflate a
    channel. ``matched_frac`` is the layer-b (regressive, FD3's own) share; the rest is orthogonal/
    opposite motion CONTEXT. The claim is about FD3's SHEET-RELAYED input (denominator D1), NOT its
    direct T4/T5 tuning (which stays ~99% layer-b)."""
    present = {ct: p for ct, p in profiles.items() if p.get("present") and p.get("to_fd3_syn")}
    total = sum(p["to_fd3_syn"] for p in present.values())
    by_channel = {k: 0 for k in _CARDINAL}
    members = {k: [] for k in _CARDINAL}
    soft = []
    for ct, p in present.items():
        lyr = p.get("dominant_layer")
        if p.get("dir_hard") and lyr in by_channel:
            by_channel[lyr] += p["to_fd3_syn"]
            members[lyr].append(ct)
        else:
            soft.append(ct)
    frac = {k: (round(100.0 * v / total, 1) if total else None) for k, v in by_channel.items()}
    hard_total = sum(by_channel.values())
    n_dirs = sum(1 for k in _CARDINAL if by_channel[k] > 0)
    matched = round(100.0 * by_channel["b"] / total, 1) if total else None   # FD3's own direction
    context = round(100.0 * (hard_total - by_channel["b"]) / total, 1) if total else None
    # Shannon entropy (base-2) over the hard-channel synapse distribution — a scalar "how spread
    # across directions" that the random-tangential null is compared against.
    entropy = _channel_entropy([by_channel[k] for k in _CARDINAL])
    return {
        "sheet_syn_to_fd3": int(total),
        "syn_by_channel": {k: int(v) for k, v in by_channel.items()},
        "sheet_frac_by_channel": frac,
        "channel_members": {k: v for k, v in members.items()},
        "channel_directions": {k: {"a": "progressive", "b": "regressive",
                                   "c": "upward", "d": "downward"}[k] for k in _CARDINAL},
        "n_cardinal_directions": n_dirs,
        "matched_frac": matched,               # layer-b / regressive (FD3's own channel)
        "context_frac": context,               # orthogonal + opposite context
        "soft_direction_sheets": soft,         # sheets whose direction label is a tendency, not hard
        "channel_entropy_bits": entropy,
    }


def _channel_entropy(counts) -> float:
    """Shannon entropy (bits) of a direction-synapse distribution; 0 if all in one channel."""
    arr = np.asarray(counts, dtype=float)
    tot = arr.sum()
    if tot <= 0:
        return 0.0
    p = arr[arr > 0] / tot
    return float(round(-(p * np.log2(p)).sum(), 3))


def _sheet_direction_entropy(src, meta, roots, sheet_roots_by_ct) -> float | None:
    """For an arbitrary cell (roots), the entropy of its sheet-relayed input across a/b/c/d.

    Used to place FD3 in a null of random LP-tangential cells: a cell's input from the candidate
    sheets is bucketed by each sheet's dominant direction and the entropy is computed. High entropy
    = pools many directions. If FD3 is not elevated over the null, four-direction pooling is a
    generic wide-field-tangential property, not FD3-specific."""
    roots = [int(x) for x in roots]
    if not roots:
        return None
    syn = src.synapses(post_ids=roots); syn = syn[syn["post_pt_root_id"].isin(set(roots))]
    counts = {k: 0 for k in _CARDINAL}
    got = False
    for ct, (sroots, lyr) in sheet_roots_by_ct.items():
        if lyr not in counts:
            continue
        n = int(len(syn[syn["pre_pt_root_id"].isin(sroots)]))
        if n:
            counts[lyr] += n; got = True
    if not got:
        return None
    return _channel_entropy([counts[k] for k in _CARDINAL])


def _entropy_null(src, meta, fd3_entropy, sheet_roots_by_ct, exclude) -> dict:
    """Percentile of FD3's sheet-direction entropy within a random LP-tangential-cell null."""
    if fd3_entropy is None:
        return {"available": False}
    # candidate null pool: LPi* + LPT*/Nod* optic/visual_projection tangential cells, minus FD3/FD1
    # and the sheets themselves.
    df = meta.df
    is_tan = df["cell_type"].astype("string").str.match(r"^(LPi|LPT|Nod)").fillna(False)
    is_vis = df["super_class"].isin(["optic", "visual_projection"])
    pool_types = sorted(set(df.loc[is_tan & is_vis, "cell_type"].dropna().astype(str)) - set(exclude))
    rng = np.random.default_rng(SEED)
    if len(pool_types) > N_ENTROPY_NULL:
        pool_types = list(rng.choice(pool_types, size=N_ENTROPY_NULL, replace=False))
    ents = []
    for ct in pool_types:
        try:
            roots = sorted(int(x) for x in meta.root_ids_of_type([ct]))
            e = _sheet_direction_entropy(src, meta, roots, sheet_roots_by_ct)
        except Exception:  # noqa: BLE001 - a transient live-CAVE hiccup on one null type must
            e = None       # not abort the whole entropy null; skip that type and continue.
        if e is not None:
            ents.append(e)
    if len(ents) < 10:
        return {"available": False, "reason": "too few null cells with sheet input"}
    ents = np.asarray(ents)
    pct = float((ents <= fd3_entropy).mean())
    return {"available": True, "fd3_entropy_bits": fd3_entropy,
            "null_mean_bits": round(float(ents.mean()), 3),
            "null_median_bits": round(float(np.median(ents)), 3),
            "n_null_cells": int(len(ents)), "fd3_percentile": round(pct, 3),
            "fd3_elevated": bool(fd3_entropy > np.median(ents))}


def _fd1_reference(src, meta, fd1_ct, fd3_composition) -> dict:
    """Run the identical sheet-direction census on FD1 = Nod1 (the reference cell). If FD1 ALSO
    pools four directions through its sheets, then "FD3 pools four directions" is a generic
    tangential-cell property, not FD3-specific — reported honestly rather than as a discovery."""
    fd1 = sorted(int(x) for x in meta.root_ids_of_type([fd1_ct]))
    if not fd1:
        return {"available": False}
    fd1_set = set(fd1)
    profiles = {ct: _candidate_profile(src, meta, ct, fd1_set) for ct in CANDIDATES}
    comp = _directional_composition(profiles)
    return {"available": True, "cell_type": fd1_ct,
            "n_cardinal_directions": comp["n_cardinal_directions"],
            "matched_frac": comp["matched_frac"], "context_frac": comp["context_frac"],
            "channel_entropy_bits": comp["channel_entropy_bits"],
            "sheet_frac_by_channel": comp["sheet_frac_by_channel"],
            "fd3_specific": bool(comp["n_cardinal_directions"] < fd3_composition["n_cardinal_directions"]
                                 or (comp["channel_entropy_bits"] or 0)
                                 < (fd3_composition["channel_entropy_bits"] or 0))}


def run(src, meta, cfg: C.SideConfig = C.RIGHT, *, candidate: str = FD3,
        anchor: str = FD3_ANCHOR, match_layer: str = "b", candidates: tuple | None = None,
        detectors: tuple = DETECTORS, roots: list[int] | None = None) -> dict:
    # ``match_layer`` is the figure cell's own dominant motion layer; the "direction-matched"
    # sheet is the one reading that same layer ("b" for FD2/FD3's regressive arm, "a" for the
    # FD1/FD4 progressive arm). ``candidates`` overrides the sheet candidate list; ``roots``
    # overrides the figure-cell type lookup (for a sub-set of a type). ``detectors`` selects the
    # ON/OFF detectors used by the retinotopy probe.
    del cfg  # bilateral by construction (figure-cell pair scored together)
    del detectors  # retinotopy probe uses module DETECTORS; kept for signature uniformity
    src = _CachedSource(src)
    cand_list = tuple(candidates) if candidates is not None else CANDIDATES
    fd3 = [int(x) for x in roots] if roots is not None else _fd3_roots(meta, candidate)
    fd3_set = set(fd3)

    profiles = {ct: _candidate_profile(src, meta, ct, fd3_set) for ct in cand_list}
    present = {ct: p for ct, p in profiles.items() if p.get("present")}

    # direction-matched sheets feeding the figure cell above the floor (same motion layer)
    layerb_sheets = {ct: p for ct, p in present.items()
                     if p.get("dominant_layer_letter") == match_layer
                     and p["to_fd3_syn"] >= MIN_SYN_TO_FD3}
    # the named sheet is the top direction-matched sheet by ->figure-cell synapses
    named = max(layerb_sheets, key=lambda ct: layerb_sheets[ct]["to_fd3_syn"]) if layerb_sheets else None

    # uniqueness fold: named sheet's ->FD3 layer-b syn vs the OTHER layer-b sheets (should be the
    # only one -> fold reported as "unique layer-b"). Also fold vs sibling sheets overall.
    other_layerb = [p["to_fd3_syn"] for ct, p in layerb_sheets.items() if ct != named]
    fold_vs_other_layerb = (layerb_sheets[named]["to_fd3_syn"] / (sum(other_layerb) / len(other_layerb))
                            if named and other_layerb else float("inf"))
    n_layerb_sheets = len(layerb_sheets)

    # feed-forward chain check: detectors -> named sheet -> FD3 both non-zero
    named_prof = present.get(named, {})
    feed_forward = bool(named and named_prof.get("layer_b_input_syn", 0) > 0
                        and named_prof.get("to_fd3_syn", 0) > 0)

    retino = _retinotopy_null(src, meta, named) if named else {"available": False}

    # the sheet-SET: the direction-matched named sheet + the orthogonal-channel siblings that
    # also feed the figure cell (exclude the opposite-direction reference sheet, whichever it is)
    ref_sheet = "LLPC1" if match_layer == "b" else "LPC1"
    sheet_set = sorted([ct for ct, p in present.items()
                        if p["to_fd3_syn"] >= MIN_SYN_TO_FD3 and ct != ref_sheet],
                       key=lambda ct: present[ct]["to_fd3_syn"], reverse=True)

    # DIRECTIONAL COMPOSITION: FD3 pools sheets carrying all four cardinal directions (mentor's
    # LLPC1/2/3 -> a/c/d point). Computed over ALL present sheets (LLPC1 included), by dominant
    # layer, from the already-computed profiles.
    composition = _directional_composition(profiles)
    # is 4-direction pooling FD3-specific? two controls: the FD1=Nod1 reference + a random-tangential
    # entropy null. Both compare FD3's sheet-direction spread against a baseline.
    sheet_roots_by_ct = {ct: (set(sorted(int(x) for x in meta.root_ids_of_type([ct]))),
                              present[ct].get("dominant_layer"))
                         for ct in present if present[ct].get("dir_hard")}
    fd3_entropy = _sheet_direction_entropy(src, meta, fd3, sheet_roots_by_ct)
    exclude = set(CANDIDATES) | {candidate, anchor}
    entropy_null = _entropy_null(src, meta, fd3_entropy, sheet_roots_by_ct, exclude)
    fd1_reference = _fd1_reference(src, meta, anchor, composition)

    return {
        "candidate": candidate, "named_sheet": named,
        "profiles": {ct: {k: v for k, v in p.items()} for ct, p in profiles.items()},
        "n_layerb_sheets": n_layerb_sheets,
        "fold_vs_other_layerb": (round(fold_vs_other_layerb, 2)
                                 if np.isfinite(fold_vs_other_layerb) else "unique"),
        "feed_forward": feed_forward,
        "sheet_set": sheet_set,
        "sheet_set_channels": {ct: present[ct]["dominant_direction"] for ct in sheet_set},
        "retinotopy_null": retino,
        "llpc1_control_to_fd3": present.get("LLPC1", {}).get("to_fd3_syn"),  # FD1 sheet ~446 (weak)
        "directional_composition": composition,
        "entropy_null": entropy_null,
        "fd1_reference": fd1_reference,
        "track": src.track,
    }
