"""Build the canonical neuron (node) table.

The proofread_root_ids_783.npy list is the AUTHORITATIVE node set (~139,255 neurons).
We left-join the Schlegel annotation TSV onto it (so unannotated proofread neurons
are kept with null labels), sort by root_id for a fully reproducible ordering, and
assign each neuron a contiguous integer ``idx`` 0..N-1 — the node id used by every
adjacency matrix and downstream model.

Outputs:
  processed/neurons.parquet        (full node table; row i == node idx i)
  processed/node_index_map.parquet (just idx <-> root_id, the universal join key)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..io import write_parquet
from . import schemas


def _load_root_ids(npy_path: Path) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=False)
    arr = np.asarray(arr).reshape(-1).astype(np.int64)
    if arr.size == 0:
        raise ValueError(f"No root_ids loaded from {npy_path}")
    if np.unique(arr).size != arr.size:
        # de-dupe defensively but warn loudly
        print(f"[build_neurons] WARNING: {arr.size - np.unique(arr).size} duplicate "
              f"root_ids in {npy_path.name}; de-duplicating.")
        arr = np.unique(arr)
    return arr


def _load_annotations(tsv_path: Path) -> pd.DataFrame:
    # root_id can be large; read as Int64 (nullable) then coerce after.
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    if "root_id" not in df.columns:
        raise KeyError(
            f"{tsv_path.name} has no 'root_id' column. Columns: {sorted(df.columns)}"
        )
    df["root_id"] = pd.to_numeric(df["root_id"], errors="coerce").astype("Int64")
    keep = [c for c in schemas.ANNOTATION_KEEP if c in df.columns]
    missing = [c for c in schemas.ANNOTATION_KEEP if c not in df.columns]
    if missing:
        print(f"[build_neurons] note: annotation columns absent (kept as null): {missing}")
    df = df[keep].copy()
    # collapse duplicate annotation rows per root_id (keep first), if any.
    df = df.drop_duplicates(subset="root_id", keep="first")
    return df


def build_neuron_table(cfg: Config) -> pd.DataFrame:
    paths = cfg.paths()
    root_ids = _load_root_ids(paths.raw_file("proofread_root_ids_783.npy"))

    ann_spec = next(f for f in cfg.github_files if f.role == "annotations")
    ann = _load_annotations(paths.raw_file(ann_spec.key))

    nodes = pd.DataFrame({"root_id": root_ids})
    df = nodes.merge(ann, on="root_id", how="left")
    df = df.sort_values("root_id", kind="stable").reset_index(drop=True)
    df.insert(0, "idx", np.arange(len(df), dtype=np.int64))

    # Normalize neurotransmitter to the canonical vocabulary used by the sign policies.
    if "top_nt" in df.columns:
        df["nt_canonical"] = df["top_nt"].map(schemas.canonical_nt)
    else:
        df["nt_canonical"] = pd.Series([None] * len(df), dtype="object")

    # Order columns: idx, root_id, then the schema order, then nt_canonical.
    ordered = ["idx", "root_id"]
    for c in schemas.NEURONS_SCHEMA:
        if c in df.columns and c not in ordered:
            ordered.append(c)
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]
    return df


def run(cfg: Config) -> pd.DataFrame:
    paths = cfg.paths().ensure()
    df = build_neuron_table(cfg)

    n_annot = int(df["super_class"].notna().sum()) if "super_class" in df else 0
    print(f"[build_neurons] N={len(df)} nodes; "
          f"{n_annot} have a super_class annotation "
          f"({len(df) - n_annot} unannotated).")

    write_parquet(df, paths.neurons)
    write_parquet(df[["idx", "root_id"]], paths.node_index_map)
    print(f"[build_neurons] wrote {paths.neurons.name} and {paths.node_index_map.name}")
    return df
