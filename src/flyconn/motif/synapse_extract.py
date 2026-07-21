"""Chunked extraction from the 9.5 GB raw synapse table (flywire_synapses_783.feather).

The feather is an Arrow-IPC *file* of ~1,985 record batches. We memory-map it and
pull one batch at a time, keep only rows touching the roots of interest, and write
survivors incrementally to a compact parquet. Peak memory is ~one batch; a full scan
filtering to a single neuron completes in a few seconds.

This raw table is the closest offline analog to the live CAVE ``synapse_query`` the
report used: it is per-synapse, includes non-proofread partners, and carries the
per-synapse NT probabilities, neuropil, and 3D positions every claim needs.

The per-batch filter is factored into the pure ``_filter_batch`` so it can be unit
tested on a tiny in-memory IPC file without the real 9.5 GB input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

# Columns we keep from the raw synapse table (everything the verification needs).
SYN_COLS = [
    "pre_pt_root_id",
    "post_pt_root_id",
    "neuropil",
    "connection_score",
    "cleft_score",
    "gaba",
    "ach",
    "glut",
    "oct",
    "ser",
    "da",
    "pre_pt_position_x",
    "pre_pt_position_y",
    "pre_pt_position_z",
    "post_pt_position_x",
    "post_pt_position_y",
    "post_pt_position_z",
]


def _roots_array(roots: Iterable[int] | None) -> pa.Array | None:
    if roots is None:
        return None
    return pa.array(sorted(int(r) for r in roots), type=pa.int64())


def _filter_batch(
    batch: pa.RecordBatch,
    pre_arr: pa.Array | None,
    post_arr: pa.Array | None,
) -> pa.RecordBatch:
    """Keep rows where pre_pt_root_id in pre_arr OR post_pt_root_id in post_arr.

    A None side imposes no constraint on that side. If both are None, nothing is kept
    (returns an empty batch) — callers always pass at least one constraint.
    """
    mask = None
    if pre_arr is not None:
        mask = pc.is_in(batch.column("pre_pt_root_id"), value_set=pre_arr)
    if post_arr is not None:
        m2 = pc.is_in(batch.column("post_pt_root_id"), value_set=post_arr)
        mask = m2 if mask is None else pc.or_(mask, m2)
    if mask is None:
        # No constraints -> keep nothing (avoid accidentally dumping the whole file).
        mask = pc.equal(batch.column("pre_pt_root_id"), batch.column("pre_pt_root_id"))
        mask = pc.and_(mask, pc.scalar(False))
    return batch.filter(mask)


def extract_synapses_for_roots(
    feather_path: str | Path,
    out_path: str | Path,
    pre_roots: Iterable[int] | None,
    post_roots: Iterable[int] | None,
    *,
    columns: list[str] = SYN_COLS,
    log_every: int = 400,
) -> dict:
    """Stream the synapse feather, keep matching rows, write a single parquet.

    Returns stats: {n_batches, n_in, n_kept, out_path}.
    """
    feather_path = Path(feather_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pre_arr = _roots_array(pre_roots)
    post_arr = _roots_array(post_roots)

    src = pa.memory_map(str(feather_path), "r")
    reader = ipc.open_file(src)
    n_batches = reader.num_record_batches

    writer: pq.ParquetWriter | None = None
    n_in = n_kept = 0
    try:
        for b in range(n_batches):
            batch = reader.get_batch(b)
            n_in += batch.num_rows
            kept = _filter_batch(batch, pre_arr, post_arr)
            if kept.num_rows:
                tbl = pa.Table.from_batches([kept]).select(columns)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), tbl.schema)
                writer.write_table(tbl)
                n_kept += kept.num_rows
            if log_every and b % log_every == 0:
                print(f"[extract] batch {b}/{n_batches} kept={n_kept}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # Nothing matched: write an empty parquet with the right schema so downstream
        # reads don't crash on a missing file.
        empty = pa.table({c: pa.array([], type=pa.int64() if "id" in c or "position" in c
                                      or "score" in c else pa.float64())
                          for c in columns})
        pq.write_table(empty, str(out_path))

    return {
        "n_batches": n_batches,
        "n_in": n_in,
        "n_kept": n_kept,
        "out_path": str(out_path),
    }
