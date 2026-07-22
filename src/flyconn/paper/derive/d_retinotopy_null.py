"""Derive family D: T4a->LLPC1 retinotopy null model.

Observed: for each of the 100 LLPC1, the input-patch radius is the median distance from
its T4a-input synapses to their centroid (um). We use the per-T4a centroid (mean of that
T4a's synapses onto the LLPC1) as the 'input location', matching the paper's notion of a
columnar input patch.

Null (in-degree-preserving): each LLPC1 keeps its observed number of distinct T4a inputs
but draws WHICH T4a at random from the full VCH-gated T4a pool (the 454), and the patch
radius is recomputed from those T4a's field positions. 500 permutations -> z, p.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import sheet as SH
from .. import geometry as G
from ..oracle import consts as C

N_PERMS = 500
SEED = 12345


def _t4a_field_positions(sheet, src) -> dict[int, np.ndarray]:
    """Mean um position of each VCH-gated T4a's output synapses (its retinotopic locus)."""
    out = src.synapses(pre_ids=sheet.t4a_roots.tolist())
    out = out[out["pre_pt_root_id"].isin(set(int(x) for x in sheet.t4a_roots))]
    pos = G.syn_positions_um(out, "pre")
    df = pd.DataFrame({"t4a": out["pre_pt_root_id"].to_numpy(),
                       "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]})
    return {int(t): g[["x", "y", "z"]].mean().to_numpy()
            for t, g in df.groupby("t4a")}


def run(src, meta, cfg: C.SideConfig = C.RIGHT) -> dict:
    sheet = SH.get_sheet(src, meta, cfg)
    syn = sheet.t4a_llpc1_syn  # T4a -> sheet-LLPC1 synapses

    # Per-LLPC1 list of its distinct T4a drivers, and the synapse positions (post side =
    # the synapse location on the LLPC1 dendrite; we use the T4a presynaptic centroid for
    # the retinotopic input locus, consistent with "input-patch").
    t4a_field = _t4a_field_positions(sheet, src)
    t4a_pool = np.array(sorted(t4a_field.keys()), dtype=np.int64)
    pool_pos = np.vstack([t4a_field[int(t)] for t in t4a_pool])

    # observed: for each LLPC1, the centroid-spread of its drivers' field positions.
    grp = syn.groupby("post_pt_root_id")["pre_pt_root_id"].apply(lambda s: sorted(set(s)))
    obs_radii, obs_within10, indeg = [], [], []
    for llpc1, drivers in grp.items():
        dpos = np.vstack([t4a_field[int(t)] for t in drivers if int(t) in t4a_field])
        if len(dpos) < 2:
            continue
        obs_radii.append(G.patch_radius_um(dpos, "median"))
        obs_within10.append(G.frac_within(dpos, 10.0))
        indeg.append(len(dpos))
    obs_radii = np.array(obs_radii)
    obs_median = float(np.median(obs_radii))
    frac_within_10_obs = float(np.mean(obs_within10))

    # null: preserve each LLPC1's in-degree, draw random T4a from the pool.
    rng = np.random.default_rng(SEED)
    null_medians = np.empty(N_PERMS)
    null_within10 = np.empty(N_PERMS)
    n_pool = len(pool_pos)
    for p in range(N_PERMS):
        radii, within = [], []
        for k in indeg:
            sel = rng.choice(n_pool, size=k, replace=False)
            dpos = pool_pos[sel]
            radii.append(G.patch_radius_um(dpos, "median"))
            within.append(G.frac_within(dpos, 10.0))
        null_medians[p] = np.median(radii)
        null_within10[p] = np.mean(within)
    null_median = float(np.mean(null_medians))
    null_std = float(np.std(null_medians))
    z = (obs_median - null_median) / null_std if null_std else float("nan")
    # empirical p: fraction of perms with median <= observed.
    p_value = float((null_medians <= obs_median).mean())
    p_value = max(p_value, 1.0 / N_PERMS)  # floor at resolution

    return {
        "n_t4a": sheet.n_t4a,
        "n_llpc1": sheet.n_llpc1,
        "n_t4a_llpc1_syn": sheet.n_t4a_llpc1_syn,
        "obs_radius_um": obs_median,
        "null_radius_um": null_median,
        "z_score": z,
        "p_value": p_value,
        "n_perms": N_PERMS,
        "frac_within_10um_obs": frac_within_10_obs,
        "frac_within_10um_null": float(np.mean(null_within10)),
        "track": src.track,
    }
