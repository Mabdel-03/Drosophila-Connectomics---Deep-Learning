"""Prepare the FlyWire visual-columns retinotopy into ``visual_columns_raw.csv``.

The public Codex ``column_assignment.csv.gz`` (FAFB v783, no auth) gives a per-root_id
optic-lobe column + hex (p, q) for the COLUMNAR cell types and R7/R8 — but NOT for the
outer photoreceptors R1-6, which are the bulk of the model's visual input nodes. A raw
join therefore leaves R1-6 unplaced and the >=80% R1-6 coverage gate in
``retinotopy.load_retinotopy(source='auto')`` falls back to ``pos_grid``.

This module fills R1-6 (and any other unplaced input neuron) with a column/(p, q) by:
  1. **Connectivity propagation** (primary): each R1-6 inherits the column of the
     columnar neuron it is most strongly connected to (either direction), summed over the
     FULL synapse table. R1-6 -> L1/L2/L3 in the same lamina cartridge, which are columnar
     and carry a column, so this is retinotopically faithful.
  2. **Spatial nearest-neighbor** (fallback): any still-unplaced R1-6 inherits the column
     of the nearest already-placed neuron on the SAME side, in 3D (pos_x, pos_y, pos_z)
     space — retinotopically adjacent receptors are physically adjacent.

It writes ``processed/visual_columns_raw.csv`` (root_id, column_id, p, q) which
``retinotopy.fetch_visual_columns()`` then normalizes into ``retinotopy_columns.parquet``.

Run once after the column file is downloaded:
    curl -L -o $PROC/column_assignment.csv.gz \
      "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz"
    python -m flyconn.data_prep.prepare_visual_columns
    python -c "from flyconn.data_prep.retinotopy import fetch_visual_columns; fetch_visual_columns(force=True)"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..io import read_parquet
from ..models.subgraphs import PHOTORECEPTOR_TYPES

COLUMN_ASSIGNMENT = "column_assignment.csv.gz"
RAW_OUT = "visual_columns_raw.csv"
# The model's outer-photoreceptor input type, absent from the column file and propagated here.
PROPAGATE_TYPE = "R1-6"


def prepare(*, chunk: int = 512) -> str:
    """Build ``visual_columns_raw.csv`` with R1-6 columns propagated. Returns its path."""
    proc = load_config().paths().processed
    src = proc / COLUMN_ASSIGNMENT
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Download it first (public, no auth):\n"
            "  curl -L -o "
            f"{src} \"https://storage.googleapis.com/flywire-data/codex/data/fafb/783/"
            "column_assignment.csv.gz\""
        )

    n = read_parquet(load_config().paths().neurons)
    root = n["root_id"].to_numpy()
    side = n["side"].to_numpy()

    # Direct columns from the file (columnar types + R7/R8), aligned to neuron-table order.
    col = (pd.read_csv(src).drop_duplicates("root_id")
           .set_index("root_id")[["column_id", "p", "q"]])
    joined = pd.DataFrame({"root_id": root}, index=n.index).join(col, on="root_id")
    colid = joined["column_id"].to_numpy(dtype=float)
    p = joined["p"].to_numpy(dtype=float)
    q = joined["q"].to_numpy(dtype=float)
    has = np.isfinite(p) & np.isfinite(q)

    prop_type = (n["cell_type"] == PROPAGATE_TYPE).to_numpy()
    targets = np.flatnonzero(prop_type)

    # (1) Connectivity propagation over the FULL synapse table, both directions.
    e = read_parquet(load_config().paths().edges_full)[["pre_idx", "post_idx", "syn_count"]]
    out_e = (e[e["pre_idx"].isin(targets) & has[e["post_idx"].to_numpy()]]
             .rename(columns={"pre_idx": "r", "post_idx": "c"}))
    in_e = (e[e["post_idx"].isin(targets) & has[e["pre_idx"].to_numpy()]]
            .rename(columns={"post_idx": "r", "pre_idx": "c"}))
    both = (pd.concat([out_e[["r", "c", "syn_count"]], in_e[["r", "c", "syn_count"]]])
            .groupby(["r", "c"], as_index=False)["syn_count"].sum()
            .sort_values("syn_count", ascending=False).drop_duplicates("r", keep="first"))
    r = both["r"].to_numpy()
    c = both["c"].to_numpy()
    colid[r], p[r], q[r] = colid[c], p[c], q[c]

    # (2) Spatial nearest-neighbor fallback (3D pos), per side, for any still-unplaced.
    px = n["pos_x"].to_numpy(dtype=float)
    py = n["pos_y"].to_numpy(dtype=float)
    pz = (n["pos_z"].to_numpy(dtype=float) if "pos_z" in n.columns
          else np.zeros(len(n)))
    placed = np.isfinite(p) & np.isfinite(q)
    for s in np.unique(side[prop_type]):
        donor = np.flatnonzero(placed & (side == s) & np.isfinite(px) & np.isfinite(py))
        need = np.flatnonzero(prop_type & (side == s) & ~placed
                              & np.isfinite(px) & np.isfinite(py))
        if donor.size == 0 or need.size == 0:
            continue
        D = np.stack([px[donor], py[donor], pz[donor]], 1)
        Nd = np.stack([px[need], py[need], pz[need]], 1)
        for i in range(0, len(need), chunk):
            block = Nd[i:i + chunk]
            j = ((block[:, None, :] - D[None, :, :]) ** 2).sum(2).argmin(1)
            dst, srcj = need[i:i + chunk], donor[j]
            colid[dst], p[dst], q[dst] = colid[srcj], p[srcj], q[srcj]
        placed = np.isfinite(p) & np.isfinite(q)

    # Write the raw CSV (root_id, column_id, p, q) for every placed neuron. Drop side==NA
    # rows defensively so the downstream eye/eye-map never sees a NaN eye (only ~a handful
    # of non-photoreceptor center/unknown-side neurons).
    keep = placed & np.isin(side, ["left", "right"])
    out = pd.DataFrame({"root_id": root[keep], "column_id": colid[keep],
                        "p": p[keep], "q": q[keep]})
    out["column_id"] = out["column_id"].astype("Int64")
    out_path = proc / RAW_OUT
    out.to_csv(out_path, index=False)

    # Report photoreceptor coverage (the metric the >=80% gate checks).
    placed_roots = set(out["root_id"])
    for ct in PHOTORECEPTOR_TYPES:
        ids = n.loc[n["cell_type"] == ct, "root_id"]
        frac = ids.isin(placed_roots).mean() if len(ids) else float("nan")
        print(f"[prepare_visual_columns] {ct}: "
              f"{int(ids.isin(placed_roots).sum())}/{len(ids)} ({frac:.1%})")
    print(f"[prepare_visual_columns] wrote {len(out)} rows -> {out_path}")
    return str(out_path)


if __name__ == "__main__":
    prepare()
