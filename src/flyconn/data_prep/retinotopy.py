"""Retinotopic (columnar / hex) coordinates for optic-lobe neurons.

The rigid "eye" needs to place each photoreceptor on a 2D retinal lattice so an image
can be rendered onto it the way the fly's eye delivers it. neurons.parquet has NO
column/hex assignment, so this module provides two sources, keyed by ``root_id``:

  1. ``columns``  — the FlyWire-native per-root_id map (Codex "Visual Columns": root_id
     -> 1 of 796 columns; hsseung/OpticLobe.jl id2pq: root_id -> (p, q) hex). Fetched
     ONCE by ``fetch_visual_columns`` (needs a Google login / Julia DataDeps, so it is a
     manual step, never run inside training) and cached to
     ``processed/retinotopy_columns.parquet`` with columns [root_id, column_id, p, q, eye].
     This is the most faithful source, BUT it covers only ~23,452 columnar OL neurons of
     31 types; R1-6 coverage is unverified (measure with ``coverage_report``).

  2. ``pos_grid`` — derived from the ``pos_x/pos_y`` anchor coordinates already in
     neurons.parquet (per eye: PCA to the eye's principal plane, then bin onto a hex
     lattice sized to ~the column count). Self-contained, no external fetch. Less faithful
     (raw EM-anchor positions, not projected retinal hex) but the DEFAULT until the
     ``columns`` coverage is proven.

``load_retinotopy(neurons, source=...)`` returns a DataFrame aligned to ``neurons`` (one
row per neuron, NaN column_id/p/q for non-assigned neurons), so the eye module can join it
to any subgraph by local order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..io import read_parquet, write_parquet

# Photoreceptor + columnar optic types we try to place on the lattice. Photoreceptors are
# the input cells; the rest are carried so the same map can later drive weight-sharing.
PHOTORECEPTOR_TYPES = ("R1-6", "R7", "R8")

RETINOTOPY_PARQUET = "retinotopy_columns.parquet"


def _retinotopy_path():
    return load_config().paths().processed / RETINOTOPY_PARQUET


# --------------------------------------------------------------------------- columns
def fetch_visual_columns(*, force: bool = False) -> str:
    """ONE-TIME, MANUAL fetch of the FlyWire visual-columns map -> processed parquet.

    Needs an authenticated Codex download (Google login) or a Julia run of
    hsseung/OpticLobe.jl (DataDeps auto-download). Because that auth cannot happen on a
    headless SLURM node, this is a deliberate manual step. It expects ONE of:

      * an already-downloaded CSV at ``processed/visual_columns_raw.csv`` with at least a
        ``root_id`` column and a ``column_id`` (and optionally ``p``/``q``/``hex1``/``hex2``);
      * or the OpticLobe.jl export at ``processed/id2pq_raw.csv`` (root_id, p, q).

    It normalizes whichever is present into ``processed/retinotopy_columns.parquet``
    [root_id, column_id, p, q, eye] and returns the written path. Raises a clear
    instruction if no raw file is found, rather than failing silently.
    """
    proc = load_config().paths().processed
    out = proc / RETINOTOPY_PARQUET
    if out.exists() and not force:
        return str(out)

    raw_candidates = {
        "visual_columns_raw.csv": ("column_id",),
        "id2pq_raw.csv": ("p", "q"),
    }
    found = None
    for name in raw_candidates:
        p = proc / name
        if p.exists():
            found = p
            break
    if found is None:
        raise FileNotFoundError(
            "No raw visual-columns file found. This is a MANUAL one-time step:\n"
            "  1. Download the Codex 'Visual Columns' CSV (root_id -> column) from\n"
            "     https://codex.flywire.ai/app/visual_columns_challenge  (Google login),\n"
            f"     save it to {proc/'visual_columns_raw.csv'} (must have root_id, column_id);\n"
            "     OR export OpticLobe.jl id2pq to "
            f"{proc/'id2pq_raw.csv'} (root_id, p, q).\n"
            "  2. Re-run fetch_visual_columns(). Until then the eye uses source='pos_grid'."
        )

    df = pd.read_csv(found)
    df.columns = [c.strip().lower() for c in df.columns]
    if "root_id" not in df.columns:
        raise ValueError(f"{found.name} lacks a 'root_id' column; got {list(df.columns)}")
    # Harmonize coordinate column names.
    rename = {"hex1": "p", "hex2": "q", "col": "column_id", "column": "column_id"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ("column_id", "p", "q"):
        if col not in df.columns:
            df[col] = np.nan
    # Attach eye/side from the neuron table (the raw map may not carry it).
    neurons = read_parquet(load_config().paths().neurons)
    side = neurons.set_index("root_id")["side"]
    df["eye"] = df["root_id"].map(side).map({"left": 0, "right": 1}).astype("Int8")
    keep = ["root_id", "column_id", "p", "q", "eye"]
    write_parquet(df[keep].drop_duplicates("root_id"), out)
    return str(out)


def _load_columns_map() -> pd.DataFrame | None:
    """Return the cached columns parquet, or None if it has not been fetched."""
    p = _retinotopy_path()
    if not p.exists():
        return None
    return read_parquet(p)


# -------------------------------------------------------------------------- pos grid
def derive_grid_from_pos(neurons: pd.DataFrame) -> pd.DataFrame:
    """Derive a per-neuron (p, q) hex lattice from pos_x/pos_y, per eye.

    For each eye, project the photoreceptors' (pos_x, pos_y) onto their 2D principal
    axes (PCA), rescale to a unit square, and read off integer hex-lattice indices.
    Returns a DataFrame aligned to ``neurons`` index with [column_id, p, q, eye]; non
    photoreceptors get NaN. This is the self-contained fallback (no external fetch).
    """
    out = pd.DataFrame(
        {"column_id": np.nan, "p": np.nan, "q": np.nan, "eye": pd.array([pd.NA] * len(neurons), dtype="Int8")},
        index=neurons.index,
    )
    is_photo = neurons["cell_type"].isin(PHOTORECEPTOR_TYPES).to_numpy()
    has_pos = neurons[["pos_x", "pos_y"]].notna().all(axis=1).to_numpy()
    for side, eye_code in (("left", 0), ("right", 1)):
        # Receptors with a missing pos_x/pos_y stay unassigned (NaN row) -> dark.
        sel = is_photo & has_pos & (neurons["side"] == side).to_numpy()
        if sel.sum() < 8:
            continue
        xy = neurons.loc[sel, ["pos_x", "pos_y"]].to_numpy(dtype=np.float64)
        xy = xy - xy.mean(0)
        # PCA: rotate onto principal axes so the lattice axes align with the eye's spread.
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        proj = xy @ vt.T                                  # [n, 2] principal coords
        # Normalize each axis to [0, 1] then to an approximately hex-sized integer grid.
        span = proj.max(0) - proj.min(0)
        span[span == 0] = 1.0
        unit = (proj - proj.min(0)) / span
        # ~sqrt(n) cells per axis gives ~n columns; hexish via row-offset on p.
        n_side = max(2, int(round(np.sqrt(sel.sum()))))
        q = np.clip(np.round(unit[:, 1] * (n_side - 1)).astype(int), 0, n_side - 1)
        p_off = (q % 2) * 0.5                              # hex row offset
        p = np.clip(np.round(unit[:, 0] * (n_side - 1) - p_off).astype(int), 0, n_side - 1)
        col = p * n_side + q                               # dense column id within the eye
        idx = neurons.index[sel]
        out.loc[idx, "p"] = p
        out.loc[idx, "q"] = q
        out.loc[idx, "column_id"] = col
        out.loc[idx, "eye"] = eye_code
    return out


# --------------------------------------------------------------------------- loader
def load_retinotopy(neurons: pd.DataFrame, *, source: str = "auto") -> pd.DataFrame:
    """Per-neuron retinotopy aligned to ``neurons`` (rows: column_id, p, q, eye).

    source:
      * "columns"  — require the fetched FlyWire columns parquet (raises if absent).
      * "pos_grid" — always derive from pos_x/pos_y.
      * "auto"     — use "columns" if the parquet exists AND covers >=80% of R1-6,
                     else fall back to "pos_grid".
    """
    if source == "pos_grid":
        return derive_grid_from_pos(neurons)

    cmap = _load_columns_map()
    if source == "columns":
        if cmap is None:
            raise FileNotFoundError(
                "source='columns' but retinotopy_columns.parquet is missing; "
                "run flyconn.data_prep.retinotopy.fetch_visual_columns() first."
            )
        return _align_columns(neurons, cmap)

    # auto
    if cmap is not None:
        aligned = _align_columns(neurons, cmap)
        cov = _r16_coverage(neurons, aligned)
        if cov >= 0.80:
            return aligned
    return derive_grid_from_pos(neurons)


def _align_columns(neurons: pd.DataFrame, cmap: pd.DataFrame) -> pd.DataFrame:
    """Join the root_id-keyed columns map onto ``neurons`` order."""
    m = cmap.set_index("root_id")
    joined = neurons[["root_id"]].join(m, on="root_id")
    return joined[["column_id", "p", "q", "eye"]].set_index(neurons.index)


def _r16_coverage(neurons: pd.DataFrame, aligned: pd.DataFrame) -> float:
    is_r16 = (neurons["cell_type"] == "R1-6").to_numpy()
    if is_r16.sum() == 0:
        return 0.0
    has_col = aligned["column_id"].notna().to_numpy()
    return float((is_r16 & has_col).sum() / is_r16.sum())


def coverage_report(source: str = "auto") -> dict:
    """Per-receptor-type assignment coverage and per-eye column counts (Phase 0)."""
    neurons = read_parquet(load_config().paths().neurons)
    aligned = load_retinotopy(neurons, source=source)
    rep: dict = {"source": source, "by_type": {}, "columns_per_eye": {}}
    for t in PHOTORECEPTOR_TYPES:
        sel = (neurons["cell_type"] == t).to_numpy()
        if sel.sum() == 0:
            continue
        assigned = int(aligned.loc[sel, "column_id"].notna().sum())
        rep["by_type"][t] = {"total": int(sel.sum()), "assigned": assigned,
                             "frac": round(assigned / sel.sum(), 4)}
    is_photo = neurons["cell_type"].isin(PHOTORECEPTOR_TYPES).to_numpy()
    # eye is a nullable Int8 (neurons with side not in {left,right} are <NA>); compare with
    # fill_value so the mask is a plain bool array (a nullable-bool mask breaks .loc).
    eye_eq = aligned["eye"].eq
    for eye_code, name in ((0, "left"), (1, "right")):
        sel = is_photo & eye_eq(eye_code).fillna(False).to_numpy()
        cols = aligned.loc[sel, "column_id"].dropna().unique()
        rep["columns_per_eye"][name] = int(len(cols))
    return rep
