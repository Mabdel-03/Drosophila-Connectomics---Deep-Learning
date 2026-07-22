"""Query cache: turn every live CAVE query into a reproducible offline parquet.

This is what makes the closed loop honest. The extract stage runs *with* network and
populates the cache; the verify stage runs *without* network and reads the same parquet
back. A cache hit is byte-for-byte the data the report was checked against, and every
file carries a ``.meta.json`` sidecar recording the dataset, materialization, table,
exact query params, row count, and when/with-what it was fetched.

Layout (composes with ``DataPaths``; lives beside raw/processed/reports/motif):

    <FLYCONN_DATA_ROOT>/v<version>/cave_cache/<dataset>/<kind>/<query_hash>.parquet
    <FLYCONN_DATA_ROOT>/v<version>/cave_cache/<dataset>/<kind>/<query_hash>.meta.json

The cache key (``query_hash``) is an md5 over the canonicalised (dataset, materialization,
table, kind, params), so it is stable across runs and order-insensitive in the params
dict. Bumping a dataset's materialization changes the hash -> a clean re-fetch with no
stale reads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..io import read_parquet, write_json, write_parquet
from ..paths import cache_root
from .datasets import CaveDataset

CACHE_DIRNAME = "cave_cache"


def _canonical(obj: Any) -> Any:
    """Recursively coerce params into a JSON-canonical, hashable structure.

    Sorts dict keys, turns sets/tuples into sorted/ordered lists, stringifies
    non-JSON scalars (e.g. numpy ints) so the hash is identical run-to-run.
    """
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, set):
        return [_canonical(v) for v in sorted(obj, key=str)]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def query_hash(dataset_key: str, materialization: object, table: str, kind: str,
               params: dict) -> str:
    """Deterministic, order-insensitive cache key for a query."""
    payload = _canonical(
        {
            "dataset": dataset_key,
            "materialization": str(materialization),
            "table": table,
            "kind": kind,
            "params": params or {},
        }
    )
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def cache_dir(ds: CaveDataset, kind: str, *, version: str = "783") -> Path:
    d = cache_root() / f"v{version}" / CACHE_DIRNAME / ds.cache_subdir / kind
    return d


def cache_path(ds: CaveDataset, kind: str, qhash: str, *, version: str = "783") -> Path:
    return cache_dir(ds, kind, version=version) / f"{qhash}.parquet"


def meta_path(parquet: Path) -> Path:
    return parquet.with_suffix(".meta.json")


def cached_or_call(
    ds: CaveDataset,
    kind: str,
    params: dict,
    fn: Callable[[], pd.DataFrame],
    *,
    version: str = "783",
    refresh: bool = False,
    allow_network: bool = True,
    caveclient_version: str | None = None,
    queried_at: str | None = None,
) -> pd.DataFrame:
    """Return a query result, reading the cache if present else calling ``fn``.

    ``fn`` is the (network) thunk that actually hits CAVE and returns a DataFrame; it is
    only invoked on a miss (or ``refresh``). On a miss with ``allow_network=False`` we
    raise a precise error instead of silently going to the network -- the verify stage
    runs cache-only and must fail loudly if an expected extract wasn't run first.
    """
    table = ds.synapse_table if kind == "synapses" else ",".join(ds.annotation_tables)
    qhash = query_hash(ds.key, ds.materialization, table, kind, params)
    path = cache_path(ds, kind, qhash, version=version)

    if path.is_file() and not refresh:
        return read_parquet(path)

    if not allow_network:
        raise RuntimeError(
            f"Cache miss for {ds.key}/{kind} (hash {qhash}) and allow_network=False.\n"
            f"  expected: {path}\n"
            f"  params: {params}\n"
            "Run the network-enabled extract stage first to populate the cache "
            "(scripts/muscular_extract.py), then re-run this cache-only step."
        )

    df = fn()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"query fn for {ds.key}/{kind} returned {type(df)!r}, expected DataFrame")

    write_parquet(df, path)
    write_json(
        meta_path(path),
        {
            "dataset": ds.key,
            "datastack": ds.datastack,
            "materialization": ds.materialization,
            "table": table,
            "kind": kind,
            "params": _canonical(params or {}),
            "query_hash": qhash,
            "n_rows": int(len(df)),
            "columns": list(df.columns),
            "queried_at": queried_at,
            "caveclient_version": caveclient_version,
        },
    )
    return df
