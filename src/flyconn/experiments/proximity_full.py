"""Whole-connectome batch pipeline for dendrite mesh proximity analysis.

This module scales ``flyconn.experiments.proximity`` from deterministic pilots to
the full FlyWire v783 neuron set. It is intentionally organized as restartable
batch phases so Slurm arrays can cache mesh/sample work and retry failed shards.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from ..config import Config, load_config
from ..io import read_json, read_parquet, write_json, write_parquet
from . import proximity

RUN_NAME_DEFAULT = "whole_connectome_lod1_sp250_r500_t2_any"
TILE_KEY_STRIDE_Y = 1_000_000
TILE_KEY_STRIDE_X = 1_000_000_000_000

NEURON_NAME_COLUMNS = [
    "idx",
    "root_id",
    "cell_type",
    "hemibrain_type",
    "synonyms",
    "supertype",
    "vfb_id",
    "fbbt_id",
    "nerve",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "side",
    "flow",
    "top_nt",
    "top_nt_conf",
    "nt_canonical",
    "known_nt",
    "known_nt_source",
    "ito_lee_hemilineage",
    "hartenstein_hemilineage",
    "nucleus_id",
    "soma_x",
    "soma_y",
    "soma_z",
    "pos_x",
    "pos_y",
    "pos_z",
]


@dataclass(frozen=True)
class FullRunParams:
    run_name: str = RUN_NAME_DEFAULT
    shard_size: int = 500
    max_neurons: int | None = None
    threshold_um: float = 2.0
    connection: str = "any"
    mesh_source: str = proximity.MESH_SOURCE_CLOUDVOLUME_PUBLIC
    mesh_path: str = proximity.PUBLIC_FLYWIRE_MESH_PATH
    dataset: str = "public"
    lod: int = 1
    lod_fallback: int | None = 0
    site_radius_nm: float = 500.0
    sample_spacing_nm: float = 250.0
    mesh_units: str = "nm"
    synapse_coordinate_units: str = "nm"
    tile_nm: float = 50_000.0
    pair_core_chunk_size: int = 512
    pair_spill_rows: int = 500_000
    reduce_buckets: int = 64
    force: bool = False

    @property
    def threshold_nm(self) -> float:
        return float(self.threshold_um) * 1000.0

    def proximity_params(self) -> proximity.ProximityParams:
        return proximity.ProximityParams(
            subgraph="whole",
            n=0,
            seed=0,
            threshold_um=self.threshold_um,
            connection=self.connection,
            mesh_source=self.mesh_source,
            mesh_path=self.mesh_path,
            dataset=self.dataset,
            lod=self.lod,
            lod_fallback=self.lod_fallback,
            site_radius_nm=self.site_radius_nm,
            sample_spacing_nm=self.sample_spacing_nm,
            mesh_units=self.mesh_units,
            synapse_coordinate_units=self.synapse_coordinate_units,
        )


def run_dir(cfg: Config, params: FullRunParams) -> Path:
    return cfg.paths().root / "experiments" / "proximity" / params.run_name


def mesh_cache_dir(cfg: Config, params: FullRunParams) -> Path:
    mesh_cache_name = params.mesh_path.rstrip("/").split("/")[-1].replace(":", "_")
    return cfg.paths().root / "mesh_cache" / params.mesh_source / mesh_cache_name / f"lod{params.lod}"


def load_params(path: Path) -> FullRunParams:
    data = read_json(path / "run_config.json")["params"]
    return FullRunParams(**data)


def write_empty_parquet(path: Path, columns: dict[str, str]) -> None:
    df = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})
    write_parquet(df, path)


def load_annotation_extras(raw_dir: Path, root_ids: Iterable[int]) -> pd.DataFrame:
    path = raw_dir / "Supplemental_file1_neuron_annotations.tsv"
    roots = pd.Series(np.asarray(list(root_ids), dtype=np.int64), name="root_id")
    if not path.exists():
        return roots.to_frame()

    cols = [
        "root_id",
        "supertype",
        "known_nt",
        "known_nt_source",
        "nerve",
        "vfb_id",
        "status",
        "dimorphism",
        "fru_dsx",
        "synonyms",
    ]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in cols)
    df = df[df["root_id"].isin(set(roots.astype(int)))]
    return roots.to_frame().merge(df, on="root_id", how="left")


def _top_neuropils(group: pd.DataFrame, top_k: int) -> str:
    parts = []
    for row in group.sort_values("count", ascending=False).head(top_k).itertuples(index=False):
        parts.append(f"{row.neuropil}:{int(row.count)}")
    return ";".join(parts)


def compute_neuropil_stats(
    counts: pd.DataFrame,
    *,
    root_col: str,
    prefix: str,
    root_ids: Iterable[int],
    top_k: int = 3,
) -> pd.DataFrame:
    roots = pd.DataFrame({"root_id": np.asarray(list(root_ids), dtype=np.int64)})
    if counts.empty:
        out = roots.copy()
        out[f"{prefix}_synapse_count"] = 0
        out[f"dominant_{prefix}_neuropil"] = pd.NA
        out[f"dominant_{prefix}_neuropil_fraction"] = np.nan
        out[f"top_{prefix}_neuropils"] = ""
        return out

    df = counts[[root_col, "neuropil", "count"]].rename(columns={root_col: "root_id"}).copy()
    df = df[df["root_id"].isin(set(roots["root_id"].astype(int)))]
    if df.empty:
        return compute_neuropil_stats(df, root_col="root_id", prefix=prefix, root_ids=root_ids, top_k=top_k)

    totals = df.groupby("root_id", as_index=False)["count"].sum().rename(columns={"count": f"{prefix}_synapse_count"})
    dominant_idx = df.groupby("root_id")["count"].idxmax()
    dominant = df.loc[dominant_idx, ["root_id", "neuropil", "count"]].rename(
        columns={"neuropil": f"dominant_{prefix}_neuropil", "count": f"dominant_{prefix}_neuropil_count"}
    )
    top = (
        df.groupby("root_id", sort=False)
        .apply(lambda g: _top_neuropils(g, top_k), include_groups=False)
        .reset_index(name=f"top_{prefix}_neuropils")
    )

    out = roots.merge(totals, on="root_id", how="left").merge(dominant, on="root_id", how="left").merge(top, on="root_id", how="left")
    out[f"{prefix}_synapse_count"] = out[f"{prefix}_synapse_count"].fillna(0).astype("int64")
    out[f"dominant_{prefix}_neuropil_fraction"] = (
        out[f"dominant_{prefix}_neuropil_count"] / out[f"{prefix}_synapse_count"].replace(0, np.nan)
    )
    out = out.drop(columns=[f"dominant_{prefix}_neuropil_count"])
    out[f"top_{prefix}_neuropils"] = out[f"top_{prefix}_neuropils"].fillna("")
    return out


def build_neuron_region_stats(cfg: Config, root_ids: Iterable[int]) -> pd.DataFrame:
    paths = cfg.paths()
    roots = np.asarray(list(root_ids), dtype=np.int64)
    post = pd.read_feather(paths.raw_file("per_neuron_neuropil_count_post_783.feather"))
    pre = pd.read_feather(paths.raw_file("per_neuron_neuropil_count_pre_783.feather"))
    post_stats = compute_neuropil_stats(post, root_col="post_pt_root_id", prefix="post", root_ids=roots)
    pre_stats = compute_neuropil_stats(pre, root_col="pre_pt_root_id", prefix="pre", root_ids=roots)
    return post_stats.merge(pre_stats, on="root_id", how="outer")


def build_shard_manifest(neurons: pd.DataFrame, shard_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    neurons = neurons.sort_values("idx", kind="stable").reset_index(drop=True).copy()
    neurons["shard_id"] = (np.arange(len(neurons)) // int(shard_size)).astype(np.int32)
    rows = []
    for shard_id, group in neurons.groupby("shard_id", sort=True):
        rows.append({
            "shard_id": int(shard_id),
            "start_row": int(group.index.min()),
            "stop_row": int(group.index.max() + 1),
            "n_neurons": int(len(group)),
            "min_idx": int(group["idx"].min()),
            "max_idx": int(group["idx"].max()),
            "postsites_path": f"postsites/shard_{int(shard_id):05d}.parquet",
            "sample_path": f"samples/shard_{int(shard_id):05d}.npz",
        })
    return neurons, pd.DataFrame(rows)


def _postsites_schema() -> pa.Schema:
    return pa.schema([
        ("shard_id", pa.int32()),
        ("idx", pa.int64()),
        ("root_id", pa.int64()),
        ("x_nm", pa.float64()),
        ("y_nm", pa.float64()),
        ("z_nm", pa.float64()),
        ("neuropil", pa.string()),
    ])


def partition_postsynaptic_sites(
    synapse_path: Path,
    neurons: pd.DataFrame,
    out_dir: Path,
    *,
    coordinate_units: str,
    force: bool = False,
) -> None:
    success = out_dir / "_SUCCESS.json"
    if success.exists() and not force:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    post, x, y, z = proximity._synapse_columns(synapse_path)
    cols = proximity._feather_column_names(synapse_path)
    neuropil = proximity.resolve_column(cols, ("neuropil",), what="neuropil")
    wanted_cols = [post, x, y, z, neuropil]

    root_to_idx = {int(r.root_id): int(r.idx) for r in neurons[["root_id", "idx"]].itertuples(index=False)}
    root_to_shard = {int(r.root_id): int(r.shard_id) for r in neurons[["root_id", "shard_id"]].itertuples(index=False)}
    wanted_roots = pa.array(np.asarray(list(root_to_idx), dtype=np.int64))
    writers: dict[int, pq.ParquetWriter] = {}
    n_rows = 0

    try:
        with pa.memory_map(str(synapse_path), "r") as source:
            reader = ipc.open_file(source)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i).select(wanted_cols)
                table = pa.Table.from_batches([batch])
                mask = pc.is_in(table[post], value_set=wanted_roots)
                if int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0) == 0:
                    continue
                df = table.filter(mask).to_pandas().rename(
                    columns={post: "root_id", x: "x_nm", y: "y_nm", z: "z_nm", neuropil: "neuropil"}
                )
                df["idx"] = df["root_id"].map(root_to_idx)
                df["shard_id"] = df["root_id"].map(root_to_shard)
                df = df.dropna(subset=["idx", "shard_id"])
                if df.empty:
                    continue
                if coordinate_units == "voxel":
                    df[["x_nm", "y_nm", "z_nm"]] = df[["x_nm", "y_nm", "z_nm"]].to_numpy(np.float64) * proximity.VOXEL_NM
                elif coordinate_units != "nm":
                    raise ValueError("coordinate_units must be 'nm' or 'voxel'")
                df = df[["shard_id", "idx", "root_id", "x_nm", "y_nm", "z_nm", "neuropil"]]
                df["shard_id"] = df["shard_id"].astype(np.int32)
                df["idx"] = df["idx"].astype(np.int64)
                df["root_id"] = df["root_id"].astype(np.int64)
                for shard_id, group in df.groupby("shard_id", sort=False):
                    shard_id = int(shard_id)
                    writer = writers.get(shard_id)
                    if writer is None:
                        path = out_dir / f"shard_{shard_id:05d}.parquet"
                        writer = pq.ParquetWriter(path, _postsites_schema())
                        writers[shard_id] = writer
                    writer.write_table(pa.Table.from_pandas(group, schema=_postsites_schema(), preserve_index=False))
                    n_rows += len(group)
    finally:
        for writer in writers.values():
            writer.close()

    write_json(success, {"n_rows": int(n_rows), "n_shards_written": int(len(writers))})


def prepare_run(cfg: Config, params: FullRunParams) -> dict:
    paths = cfg.paths().ensure()
    out = run_dir(cfg, params)
    out.mkdir(parents=True, exist_ok=True)

    neurons = read_parquet(paths.neurons).sort_values("idx", kind="stable").reset_index(drop=True)
    if params.max_neurons is not None:
        neurons = neurons.head(int(params.max_neurons)).copy()

    extras = load_annotation_extras(paths.raw, neurons["root_id"])
    manifest = neurons.merge(extras, on="root_id", how="left", suffixes=("", "_annotation"))
    region_stats = build_neuron_region_stats(cfg, manifest["root_id"])
    manifest = manifest.merge(region_stats, on="root_id", how="left")
    manifest, shards = build_shard_manifest(manifest, params.shard_size)

    write_parquet(region_stats, out / "neuron_region_stats.parquet")
    write_parquet(manifest, out / "neurons_manifest.parquet")
    write_parquet(shards, out / "shard_manifest.parquet")
    write_json(out / "run_config.json", {
        "params": asdict(params),
        "n_neurons": int(len(manifest)),
        "n_shards": int(len(shards)),
        "data_root": str(paths.root),
    })

    syn_spec = next(f for f in cfg.zenodo_files if f.role == "synapses")
    partition_postsynaptic_sites(
        paths.raw_file(syn_spec.key),
        manifest[["root_id", "idx", "shard_id"]],
        out / "postsites",
        coordinate_units=params.synapse_coordinate_units,
        force=params.force,
    )
    return {"run_dir": str(out), "n_neurons": int(len(manifest)), "n_shards": int(len(shards))}


def _sample_cache_expected(params: FullRunParams) -> dict[str, object]:
    return {
        "mesh_source": params.mesh_source,
        "mesh_path": params.mesh_path if params.mesh_source == proximity.MESH_SOURCE_CLOUDVOLUME_PUBLIC else f"fafbseg:{params.dataset}",
        "lod_requested": params.lod,
        "site_radius_nm": float(params.site_radius_nm),
        "sample_spacing_nm": float(params.sample_spacing_nm),
        "synapse_coordinate_units": params.synapse_coordinate_units,
    }


def open_cloudvolume_with_retries(mesh_path: str, *, attempts: int | None = None, base_sleep_sec: float | None = None):
    """Open public CloudVolume metadata with retries for transient cluster DNS issues."""
    n_attempts = attempts if attempts is not None else int(os.environ.get("FLYCONN_CLOUDVOLUME_OPEN_ATTEMPTS", "6"))
    sleep_sec = base_sleep_sec if base_sleep_sec is not None else float(os.environ.get("FLYCONN_CLOUDVOLUME_OPEN_SLEEP_SEC", "20"))
    last_error: Exception | None = None
    for attempt in range(1, max(1, n_attempts) + 1):
        try:
            return proximity.open_cloudvolume(mesh_path)
        except Exception as e:
            last_error = e
            if attempt >= max(1, n_attempts):
                break
            delay = min(float(sleep_sec) * (2 ** (attempt - 1)), 300.0)
            print(
                f"CloudVolume open failed on attempt {attempt}/{n_attempts}: {e}; retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(f"failed to open public CloudVolume mesh after {max(1, n_attempts)} attempts: {last_error}") from last_error


def sample_shard(cfg: Config, params: FullRunParams, shard_id: int) -> dict:
    proximity.require_mesh_stack(params.mesh_source)
    out = run_dir(cfg, params)
    shards = read_parquet(out / "shard_manifest.parquet")
    row = shards[shards["shard_id"] == int(shard_id)]
    if row.empty:
        return {"shard_id": int(shard_id), "status": "missing-shard"}

    manifest = read_parquet(out / "neurons_manifest.parquet")
    shard_neurons = manifest[manifest["shard_id"] == int(shard_id)].sort_values("idx", kind="stable")
    post_path = out / str(row.iloc[0]["postsites_path"])
    post = read_parquet(post_path) if post_path.exists() else pd.DataFrame(columns=["root_id", "x_nm", "y_nm", "z_nm"])

    sample_cache = out / "sample_neuron_cache"
    sample_cache.mkdir(parents=True, exist_ok=True)
    samples_dir = out / "samples"
    stats_dir = out / "sample_stats"
    failures_dir = out / "mesh_failures"
    for d in (samples_dir, stats_dir, failures_dir):
        d.mkdir(parents=True, exist_ok=True)

    volume = open_cloudvolume_with_retries(params.mesh_path) if params.mesh_source == proximity.MESH_SOURCE_CLOUDVOLUME_PUBLIC else None
    mesh_cache = mesh_cache_dir(cfg, params)
    prox_params = params.proximity_params()
    expected = _sample_cache_expected(params)

    def fetcher(root_id: int) -> proximity.MeshFetchResult:
        if params.mesh_source == proximity.MESH_SOURCE_CLOUDVOLUME_PUBLIC:
            return proximity.fetch_cloudvolume_mesh_arrays(
                root_id,
                volume=volume,
                mesh_path=params.mesh_path,
                lod=params.lod,
                lod_fallback=params.lod_fallback,
                mesh_units=params.mesh_units,
            )
        return proximity.fetch_fafbseg_mesh_arrays(
            root_id,
            dataset=params.dataset,
            lod=params.lod,
            threads=1,
            mesh_units=params.mesh_units,
        )

    grouped_post = {int(k): g[["x_nm", "y_nm", "z_nm"]].to_numpy(np.float64) for k, g in post.groupby("root_id", sort=False)}
    points: list[np.ndarray] = []
    idx_labels: list[np.ndarray] = []
    root_labels: list[np.ndarray] = []
    stats: list[dict] = []
    failures: list[dict] = []

    for neuron in shard_neurons[["idx", "root_id"]].itertuples(index=False):
        t0 = time.time()
        idx = int(neuron.idx)
        root_id = int(neuron.root_id)
        post_sites = grouped_post.get(root_id, np.empty((0, 3), dtype=np.float64))
        stat = {
            "shard_id": int(shard_id),
            "idx": idx,
            "root_id": root_id,
            "n_postsites": int(len(post_sites)),
            "n_samples": 0,
            "lod_requested": int(params.lod),
            "lod_used": pd.NA,
            "mesh_cache_status": pd.NA,
            "error": pd.NA,
        }
        try:
            neuron_cache = sample_cache / f"{root_id}.npz"
            cached = proximity.load_samples_cache(neuron_cache, expected=expected)
            if cached is None:
                mesh_result = proximity.load_or_fetch_mesh_arrays(mesh_cache, root_id, fetcher)
                samples = proximity.dendrite_samples_from_mesh(
                    mesh_result.vertices_nm,
                    mesh_result.faces,
                    post_sites,
                    site_radius_nm=params.site_radius_nm,
                    sample_spacing_nm=params.sample_spacing_nm,
                )
                if samples.size == 0:
                    raise ValueError("no dendrite samples after postsynaptic-site restriction")
                proximity.write_samples_cache(neuron_cache, root_id, samples, mesh_result=mesh_result, params=prox_params)
                stat["lod_used"] = mesh_result.lod_used
                stat["mesh_cache_status"] = mesh_result.cache_status
            else:
                samples = cached
                stat["mesh_cache_status"] = "sample-cache"
            stat["n_samples"] = int(len(samples))
            points.append(samples.astype(np.float32))
            idx_labels.append(np.full(len(samples), idx, dtype=np.int64))
            root_labels.append(np.full(len(samples), root_id, dtype=np.int64))
        except Exception as e:
            stat["error"] = str(e)
            failures.append({"shard_id": int(shard_id), "idx": idx, "root_id": root_id, "error": str(e)})
        finally:
            stat["runtime_sec"] = float(time.time() - t0)
            stats.append(stat)

    if points:
        all_points = np.vstack(points).astype(np.float32)
        all_idx = np.concatenate(idx_labels).astype(np.int64)
        all_roots = np.concatenate(root_labels).astype(np.int64)
    else:
        all_points = np.empty((0, 3), dtype=np.float32)
        all_idx = np.empty((0,), dtype=np.int64)
        all_roots = np.empty((0,), dtype=np.int64)

    np.savez_compressed(samples_dir / f"shard_{int(shard_id):05d}.npz", points_nm=all_points, idx=all_idx, root_id=all_roots)
    write_parquet(pd.DataFrame(stats), stats_dir / f"shard_{int(shard_id):05d}.parquet")
    write_parquet(pd.DataFrame(failures, columns=["shard_id", "idx", "root_id", "error"]), failures_dir / f"shard_{int(shard_id):05d}.parquet")
    return {"shard_id": int(shard_id), "n_neurons": int(len(shard_neurons)), "n_points": int(len(all_points)), "n_failures": int(len(failures))}


def encode_tile_key(tx: np.ndarray, ty: np.ndarray, tz: np.ndarray) -> np.ndarray:
    return tx.astype(np.int64) * TILE_KEY_STRIDE_X + ty.astype(np.int64) * TILE_KEY_STRIDE_Y + tz.astype(np.int64)


def decode_tile_key(key: int) -> tuple[int, int, int]:
    tx = int(key // TILE_KEY_STRIDE_X)
    rem = int(key % TILE_KEY_STRIDE_X)
    ty = int(rem // TILE_KEY_STRIDE_Y)
    tz = int(rem % TILE_KEY_STRIDE_Y)
    return tx, ty, tz


def tile_keys_for_points(points_nm: np.ndarray, tile_nm: float) -> np.ndarray:
    coords = np.floor(points_nm / float(tile_nm)).astype(np.int64)
    return encode_tile_key(coords[:, 0], coords[:, 1], coords[:, 2])


def missing_sample_shards(out: Path) -> list[int]:
    shard_manifest = out / "shard_manifest.parquet"
    if not shard_manifest.exists():
        return []
    expected = set(read_parquet(shard_manifest)["shard_id"].astype(int).tolist())
    samples_dir = out / "samples"
    existing = {
        int(path.stem.split("_")[1])
        for path in samples_dir.glob("shard_*.npz")
        if path.stem.split("_")[-1].isdigit()
    } if samples_dir.exists() else set()
    return sorted(expected - existing)


def build_tiles(cfg: Config, params: FullRunParams) -> dict:
    out = run_dir(cfg, params)
    missing = missing_sample_shards(out)
    if missing:
        preview = ",".join(str(i) for i in missing[:20])
        suffix = "..." if len(missing) > 20 else ""
        raise RuntimeError(f"{len(missing)} sample shard(s) are missing before tile build: {preview}{suffix}")

    parts_dir = out / "tiles" / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[int, int] = {}
    n_parts: dict[int, int] = {}
    total_points = 0
    samples_dir = out / "samples"
    shard_paths = sorted(samples_dir.glob("shard_*.npz"))
    for shard_path in shard_paths:
        shard_id = int(shard_path.stem.split("_")[1])
        z = np.load(shard_path)
        points = z["points_nm"].astype(np.float32)
        idx = z["idx"].astype(np.int64)
        roots = z["root_id"].astype(np.int64)
        n = len(points)
        if n == 0:
            continue
        point_ids = np.arange(total_points, total_points + n, dtype=np.int64)
        total_points += n
        keys = tile_keys_for_points(points, params.tile_nm)
        for key in np.unique(keys):
            mask = keys == key
            key_int = int(key)
            tile_dir = parts_dir / f"tile_{key_int}"
            tile_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                tile_dir / f"shard_{shard_id:05d}.npz",
                point_id=point_ids[mask],
                idx=idx[mask],
                root_id=roots[mask],
                points_nm=points[mask],
            )
            counts[key_int] = counts.get(key_int, 0) + int(mask.sum())
            n_parts[key_int] = n_parts.get(key_int, 0) + 1

    rows = []
    for tile_id, key in enumerate(sorted(counts)):
        tx, ty, tz = decode_tile_key(key)
        rows.append({
            "tile_id": int(tile_id),
            "tile_key": int(key),
            "tx": int(tx),
            "ty": int(ty),
            "tz": int(tz),
            "n_points": int(counts[key]),
            "n_parts": int(n_parts[key]),
        })
    manifest = pd.DataFrame(rows)
    write_parquet(manifest, out / "tiles" / "tile_manifest.parquet")
    summary = {"n_tiles": int(len(manifest)), "n_points": int(total_points), "tile_nm": float(params.tile_nm)}
    write_json(out / "tiles" / "build_tiles_summary.json", summary)
    return summary


def neighbor_tile_keys(tile_row: pd.Series, *, radius_nm: float, tile_nm: float, existing_keys: set[int]) -> list[int]:
    halo = int(math.ceil(float(radius_nm) / float(tile_nm)))
    keys = []
    for dx in range(-halo, halo + 1):
        for dy in range(-halo, halo + 1):
            for dz in range(-halo, halo + 1):
                key = int(encode_tile_key(
                    np.array([int(tile_row.tx) + dx]),
                    np.array([int(tile_row.ty) + dy]),
                    np.array([int(tile_row.tz) + dz]),
                )[0])
                if key in existing_keys:
                    keys.append(key)
    return keys


def load_tile_parts(parts_dir: Path, tile_key: int) -> dict[str, np.ndarray]:
    paths = sorted((parts_dir / f"tile_{int(tile_key)}").glob("*.npz"))
    if not paths:
        return {
            "point_id": np.empty((0,), dtype=np.int64),
            "idx": np.empty((0,), dtype=np.int64),
            "root_id": np.empty((0,), dtype=np.int64),
            "points_nm": np.empty((0, 3), dtype=np.float32),
        }
    point_ids = []
    idxs = []
    roots = []
    points = []
    for path in paths:
        z = np.load(path)
        point_ids.append(z["point_id"].astype(np.int64))
        idxs.append(z["idx"].astype(np.int64))
        roots.append(z["root_id"].astype(np.int64))
        points.append(z["points_nm"].astype(np.float32))
    return {
        "point_id": np.concatenate(point_ids),
        "idx": np.concatenate(idxs),
        "root_id": np.concatenate(roots),
        "points_nm": np.vstack(points),
    }


def _empty_tile_pairs() -> pd.DataFrame:
    return pd.DataFrame({
        "idx_a": pd.Series(dtype="int64"),
        "idx_b": pd.Series(dtype="int64"),
        "root_id_a": pd.Series(dtype="int64"),
        "root_id_b": pd.Series(dtype="int64"),
        "min_distance_nm": pd.Series(dtype="float64"),
        "n_close_sample_pairs": pd.Series(dtype="int64"),
        "closest_a_x_nm": pd.Series(dtype="float64"),
        "closest_a_y_nm": pd.Series(dtype="float64"),
        "closest_a_z_nm": pd.Series(dtype="float64"),
        "closest_b_x_nm": pd.Series(dtype="float64"),
        "closest_b_y_nm": pd.Series(dtype="float64"),
        "closest_b_z_nm": pd.Series(dtype="float64"),
        "mid_x_nm": pd.Series(dtype="float64"),
        "mid_y_nm": pd.Series(dtype="float64"),
        "mid_z_nm": pd.Series(dtype="float64"),
    })


def aggregate_point_pairs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_tile_pairs()
    min_idx = df.groupby(["idx_a", "idx_b"])["min_distance_nm"].idxmin()
    closest = df.loc[min_idx].copy()
    counts = (
        df.groupby(["idx_a", "idx_b"], as_index=False)["n_close_sample_pairs"]
        .sum()
        .rename(columns={"n_close_sample_pairs": "n_close_sample_pairs_total"})
    )
    out = closest.merge(counts, on=["idx_a", "idx_b"], how="left")
    out["n_close_sample_pairs"] = out["n_close_sample_pairs_total"].astype(np.int64)
    out = out.drop(columns=["n_close_sample_pairs_total"])
    return out[_empty_tile_pairs().columns]


def update_pair_accumulator(acc: dict[tuple[int, int], list], pairs: pd.DataFrame) -> None:
    for row in pairs.itertuples(index=False):
        key = (int(row.idx_a), int(row.idx_b))
        count = int(row.n_close_sample_pairs)
        values = [
            int(row.root_id_a),
            int(row.root_id_b),
            float(row.min_distance_nm),
            count,
            float(row.closest_a_x_nm),
            float(row.closest_a_y_nm),
            float(row.closest_a_z_nm),
            float(row.closest_b_x_nm),
            float(row.closest_b_y_nm),
            float(row.closest_b_z_nm),
            float(row.mid_x_nm),
            float(row.mid_y_nm),
            float(row.mid_z_nm),
        ]
        current = acc.get(key)
        if current is None:
            acc[key] = values
            continue
        current[3] += count
        if values[2] < current[2]:
            current[0] = values[0]
            current[1] = values[1]
            current[2] = values[2]
            current[4:] = values[4:]


def pair_accumulator_to_frame(acc: dict[tuple[int, int], list]) -> pd.DataFrame:
    if not acc:
        return _empty_tile_pairs()
    rows = []
    for (idx_a, idx_b), values in acc.items():
        rows.append((
            idx_a,
            idx_b,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            values[9],
            values[10],
            values[11],
            values[12],
        ))
    return pd.DataFrame.from_records(rows, columns=_empty_tile_pairs().columns)


def flush_pair_accumulator(acc: dict[tuple[int, int], list], spill_dir: Path, part_id: int) -> Path | None:
    if not acc:
        return None
    spill_dir.mkdir(parents=True, exist_ok=True)
    path = spill_dir / f"part_{int(part_id):05d}.parquet"
    write_parquet(pair_accumulator_to_frame(acc), path)
    acc.clear()
    return path


def aggregate_chunk_pairs(
    core: dict[str, np.ndarray],
    cand: dict[str, np.ndarray],
    cand_tree: cKDTree,
    *,
    start: int,
    stop: int,
    threshold_nm: float,
) -> tuple[pd.DataFrame, int]:
    neighbors = cand_tree.query_ball_point(core["points_nm"][start:stop], threshold_nm)
    lengths = np.fromiter((len(x) for x in neighbors), dtype=np.int64, count=stop - start)
    n_raw = int(lengths.sum())
    if n_raw == 0:
        return _empty_tile_pairs(), 0

    chunk_rows = np.repeat(np.arange(start, stop, dtype=np.int64), lengths)
    cand_cols = np.concatenate([np.asarray(x, dtype=np.int64) for x in neighbors if len(x)])
    core_ids = core["point_id"][chunk_rows]
    cand_ids = cand["point_id"][cand_cols]
    core_idx = core["idx"][chunk_rows]
    cand_idx = cand["idx"][cand_cols]
    mask = (core_ids < cand_ids) & (core_idx != cand_idx)
    if not np.any(mask):
        return _empty_tile_pairs(), 0

    chunk_rows = chunk_rows[mask]
    cand_cols = cand_cols[mask]
    core_idx = core["idx"][chunk_rows]
    cand_idx = cand["idx"][cand_cols]
    core_root = core["root_id"][chunk_rows]
    cand_root = cand["root_id"][cand_cols]
    core_pts = core["points_nm"][chunk_rows].astype(np.float64)
    cand_pts = cand["points_nm"][cand_cols].astype(np.float64)
    dist = np.linalg.norm(core_pts - cand_pts, axis=1)
    order = core_idx < cand_idx
    a_pts = np.where(order[:, None], core_pts, cand_pts)
    b_pts = np.where(order[:, None], cand_pts, core_pts)
    raw = pd.DataFrame({
        "idx_a": np.where(order, core_idx, cand_idx).astype(np.int64),
        "idx_b": np.where(order, cand_idx, core_idx).astype(np.int64),
        "root_id_a": np.where(order, core_root, cand_root).astype(np.int64),
        "root_id_b": np.where(order, cand_root, core_root).astype(np.int64),
        "min_distance_nm": dist.astype(np.float64),
        "n_close_sample_pairs": np.ones(len(dist), dtype=np.int64),
        "closest_a_x_nm": a_pts[:, 0],
        "closest_a_y_nm": a_pts[:, 1],
        "closest_a_z_nm": a_pts[:, 2],
        "closest_b_x_nm": b_pts[:, 0],
        "closest_b_y_nm": b_pts[:, 1],
        "closest_b_z_nm": b_pts[:, 2],
    })
    raw["mid_x_nm"] = (raw["closest_a_x_nm"] + raw["closest_b_x_nm"]) / 2
    raw["mid_y_nm"] = (raw["closest_a_y_nm"] + raw["closest_b_y_nm"]) / 2
    raw["mid_z_nm"] = (raw["closest_a_z_nm"] + raw["closest_b_z_nm"]) / 2
    return aggregate_point_pairs(raw), int(len(raw))


def pair_tile(cfg: Config, params: FullRunParams, tile_id: int) -> dict:
    out = run_dir(cfg, params)
    manifest = read_parquet(out / "tiles" / "tile_manifest.parquet")
    row = manifest[manifest["tile_id"] == int(tile_id)]
    pairs_dir = out / "near_pairs_tiles"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    out_path = pairs_dir / f"tile_{int(tile_id):06d}.parquet"
    if out_path.exists() and not params.force:
        return {"tile_id": int(tile_id), "status": "exists", "n_pairs": int(pq.ParquetFile(out_path).metadata.num_rows)}

    tmp_path = pairs_dir / f".tile_{int(tile_id):06d}.tmp.parquet"
    spill_dir = out / "near_pairs_tile_spills" / f"tile_{int(tile_id):06d}"
    if spill_dir.exists():
        shutil.rmtree(spill_dir)

    if row.empty:
        write_parquet(_empty_tile_pairs(), tmp_path)
        os.replace(tmp_path, out_path)
        return {"tile_id": int(tile_id), "status": "missing-tile", "n_pairs": 0}

    tile_row = row.iloc[0]
    parts_dir = out / "tiles" / "parts"
    existing_keys = set(manifest["tile_key"].astype(int))
    core = load_tile_parts(parts_dir, int(tile_row.tile_key))
    if len(core["point_id"]) == 0:
        write_parquet(_empty_tile_pairs(), tmp_path)
        os.replace(tmp_path, out_path)
        return {"tile_id": int(tile_id), "n_pairs": 0}

    candidates = [load_tile_parts(parts_dir, key) for key in neighbor_tile_keys(tile_row, radius_nm=params.threshold_nm, tile_nm=params.tile_nm, existing_keys=existing_keys)]
    cand = {
        "point_id": np.concatenate([c["point_id"] for c in candidates]),
        "idx": np.concatenate([c["idx"] for c in candidates]),
        "root_id": np.concatenate([c["root_id"] for c in candidates]),
        "points_nm": np.vstack([c["points_nm"] for c in candidates]),
    }

    cand_tree = cKDTree(cand["points_nm"])
    chunk_size = max(1, int(params.pair_core_chunk_size))
    spill_rows = max(1, int(params.pair_spill_rows))
    acc: dict[tuple[int, int], list] = {}
    spill_paths: list[Path] = []
    n_point_pairs = 0
    n_chunks = 0
    start = 0
    spill_id = 0
    while start < len(core["point_id"]):
        stop = min(start + chunk_size, len(core["point_id"]))
        try:
            chunk_pairs, chunk_point_pairs = aggregate_chunk_pairs(
                core,
                cand,
                cand_tree,
                start=start,
                stop=stop,
                threshold_nm=params.threshold_nm,
            )
        except MemoryError:
            if chunk_size == 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            print(
                f"tile {int(tile_id)} chunk {start}:{stop} exceeded memory; retrying with chunk_size={chunk_size}",
                file=sys.stderr,
                flush=True,
            )
            continue
        n_chunks += 1
        n_point_pairs += int(chunk_point_pairs)
        update_pair_accumulator(acc, chunk_pairs)
        if len(acc) >= spill_rows:
            path = flush_pair_accumulator(acc, spill_dir, spill_id)
            if path is not None:
                spill_paths.append(path)
                spill_id += 1
        start = stop

    if spill_paths:
        path = flush_pair_accumulator(acc, spill_dir, spill_id)
        if path is not None:
            spill_paths.append(path)
        tile_pairs = reduce_near_pair_tiles(spill_paths)
    else:
        tile_pairs = pair_accumulator_to_frame(acc)
    write_parquet(tile_pairs, tmp_path)
    os.replace(tmp_path, out_path)
    if spill_dir.exists():
        shutil.rmtree(spill_dir)
    return {
        "tile_id": int(tile_id),
        "n_pairs": int(len(tile_pairs)),
        "n_close_sample_pairs": int(n_point_pairs),
        "n_chunks": int(n_chunks),
        "n_spills": int(len(spill_paths)),
    }


def reduce_near_pair_tiles(tile_paths: Iterable[Path | pd.DataFrame]) -> pd.DataFrame:
    frames = []
    path_items = []
    for item in tile_paths:
        if isinstance(item, pd.DataFrame):
            frames.append(item)
        else:
            path_items.append(Path(item))
    for path in sorted(path_items):
        if path.exists():
            frames.append(read_parquet(path))
    frames = [df for df in frames if not df.empty]
    if not frames:
        return _empty_tile_pairs()
    return aggregate_point_pairs(pd.concat(frames, ignore_index=True))


def missing_pair_tiles(out: Path) -> list[int]:
    tile_manifest = out / "tiles" / "tile_manifest.parquet"
    if not tile_manifest.exists():
        return []
    expected = set(read_parquet(tile_manifest)["tile_id"].astype(int).tolist())
    pairs_dir = out / "near_pairs_tiles"
    existing = {
        int(path.stem.split("_")[1])
        for path in pairs_dir.glob("tile_*.parquet")
        if path.stem.split("_")[-1].isdigit()
    } if pairs_dir.exists() else set()
    return sorted(expected - existing)


def pair_bucket_ids(df: pd.DataFrame, n_buckets: int) -> np.ndarray:
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")
    a = df["idx_a"].to_numpy(np.uint64, copy=False)
    b = df["idx_b"].to_numpy(np.uint64, copy=False)
    return ((a * np.uint64(1_000_003) + b) % np.uint64(n_buckets)).astype(np.int64)


def reduce_bucket(cfg: Config, params: FullRunParams, bucket_id: int) -> dict:
    out = run_dir(cfg, params)
    missing_tiles = missing_pair_tiles(out)
    if missing_tiles:
        preview = ",".join(str(i) for i in missing_tiles[:20])
        suffix = "..." if len(missing_tiles) > 20 else ""
        raise RuntimeError(f"{len(missing_tiles)} tile-pair output(s) are missing before reduce-bucket: {preview}{suffix}")

    n_buckets = max(1, int(params.reduce_buckets))
    bucket = int(bucket_id)
    if bucket < 0 or bucket >= n_buckets:
        raise RuntimeError(f"bucket_id {bucket} is outside 0..{n_buckets - 1}")

    bucket_dir = out / "near_pairs_reduce_buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    out_path = bucket_dir / f"bucket_{bucket:05d}.parquet"
    if out_path.exists() and not params.force:
        return {"bucket_id": bucket, "status": "exists", "n_pairs": int(pq.ParquetFile(out_path).metadata.num_rows)}

    frames = []
    n_input_rows = 0
    for path in sorted((out / "near_pairs_tiles").glob("tile_*.parquet")):
        df = read_parquet(path)
        if df.empty:
            continue
        mask = pair_bucket_ids(df, n_buckets) == bucket
        if np.any(mask):
            part = df.loc[mask].copy()
            n_input_rows += int(len(part))
            frames.append(part)

    bucket_pairs = reduce_near_pair_tiles(frames)
    tmp_path = bucket_dir / f".bucket_{bucket:05d}.tmp.parquet"
    write_parquet(bucket_pairs, tmp_path)
    os.replace(tmp_path, out_path)
    return {"bucket_id": bucket, "n_input_rows": int(n_input_rows), "n_pairs": int(len(bucket_pairs))}


def missing_reduce_buckets(out: Path, n_buckets: int) -> list[int]:
    bucket_dir = out / "near_pairs_reduce_buckets"
    existing = {
        int(path.stem.split("_")[1])
        for path in bucket_dir.glob("bucket_*.parquet")
        if path.stem.split("_")[-1].isdigit()
    } if bucket_dir.exists() else set()
    return sorted(set(range(max(1, int(n_buckets)))) - existing)


def _summary_row_from_counts(group_values: tuple, group_cols: list[str], near_count: int, connected_count: int) -> dict:
    row = {col: val for col, val in zip(group_cols, group_values)}
    row.update({
        "near_pair_count": int(near_count),
        "connected_near_pair_count": int(connected_count),
        "fraction_connected": None if near_count == 0 else float(connected_count / near_count),
        "median_min_distance_nm": np.nan,
    })
    return row


def _accumulate_group_counts(acc: dict[tuple, list[int]], df: pd.DataFrame, group_cols: list[str]) -> None:
    if df.empty:
        return
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        current = acc.setdefault(key, [0, 0])
        current[0] += int(len(g))
        current[1] += int(g["connected_any"].sum())


def finalize_reduce_buckets(cfg: Config, params: FullRunParams) -> dict:
    paths = cfg.paths()
    out = run_dir(cfg, params)
    n_buckets = max(1, int(params.reduce_buckets))
    missing_buckets = missing_reduce_buckets(out, n_buckets)
    if missing_buckets:
        preview = ",".join(str(i) for i in missing_buckets[:20])
        suffix = "..." if len(missing_buckets) > 20 else ""
        raise RuntimeError(f"{len(missing_buckets)} reduce bucket(s) are missing before finalize: {preview}{suffix}")

    edges_full = read_parquet(paths.edges_full)
    edges = read_parquet(paths.edges)
    manifest = read_parquet(out / "neurons_manifest.parquet")
    max_idx = int(manifest["idx"].max()) if not manifest.empty else 0
    per_neuron_near = np.zeros(max_idx + 1, dtype=np.int64)
    per_neuron_connected = np.zeros(max_idx + 1, dtype=np.int64)
    distance_counts: dict[tuple, list[int]] = {}
    category_counts: dict[str, dict[tuple, list[int]]] = {}
    category_paths = {
        "super_class": "by_super_class_pair.parquet",
        "cell_class": "by_cell_class_pair.parquet",
        "cell_type": "by_cell_type_pair.parquet",
        "nt_canonical": "by_nt_pair.parquet",
        "side": "by_side_pair.parquet",
        "flow": "by_flow_pair.parquet",
        "dominant_post_neuropil": "by_dominant_post_neuropil_pair.parquet",
        "dominant_pre_neuropil": "by_dominant_pre_neuropil_pair.parquet",
    }
    summaries = out / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)

    enriched_path = out / "near_pairs_enriched.parquet"
    tmp_enriched = out / ".near_pairs_enriched.tmp.parquet"
    if tmp_enriched.exists():
        tmp_enriched.unlink()
    writer: pq.ParquetWriter | None = None
    near_pair_count = 0
    connected_near_pair_count = 0
    try:
        for path in sorted((out / "near_pairs_reduce_buckets").glob("bucket_*.parquet")):
            near = read_parquet(path)
            if near.empty:
                continue
            near = annotate_connection_counts(near, edges_full, prefix="any")
            near = annotate_connection_counts(near, edges, prefix="canonical")
            near["connected_any"] = near["any_connected_any"]
            near["syn_count_a_to_b"] = near["any_syn_count_a_to_b"]
            near["syn_count_b_to_a"] = near["any_syn_count_b_to_a"]
            enriched = enrich_pairs(near, manifest)
            table = pa.Table.from_pandas(enriched, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_enriched, table.schema)
            writer.write_table(table)

            connected = enriched["connected_any"].to_numpy(bool, copy=False)
            idx_a = enriched["idx_a"].to_numpy(np.int64, copy=False)
            idx_b = enriched["idx_b"].to_numpy(np.int64, copy=False)
            np.add.at(per_neuron_near, idx_a, 1)
            np.add.at(per_neuron_near, idx_b, 1)
            np.add.at(per_neuron_connected, idx_a, connected.astype(np.int64))
            np.add.at(per_neuron_connected, idx_b, connected.astype(np.int64))
            near_pair_count += int(len(enriched))
            connected_near_pair_count += int(connected.sum())

            bins = np.arange(0, params.threshold_nm + 100, 100)
            dist = enriched[["connected_any", "min_distance_nm"]].copy()
            dist["distance_bin_nm"] = pd.cut(dist["min_distance_nm"], bins=bins, right=False, include_lowest=True).astype(str)
            _accumulate_group_counts(distance_counts, dist, ["distance_bin_nm"])

            for column in category_paths:
                if f"{column}_a" not in enriched.columns or f"{column}_b" not in enriched.columns:
                    continue
                a = enriched[f"{column}_a"].fillna("None").astype(str)
                b = enriched[f"{column}_b"].fillna("None").astype(str)
                tmp = enriched[["connected_any"]].copy()
                tmp[f"{column}_1"] = np.where(a <= b, a, b)
                tmp[f"{column}_2"] = np.where(a <= b, b, a)
                _accumulate_group_counts(category_counts.setdefault(column, {}), tmp, [f"{column}_1", f"{column}_2"])
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        write_parquet(_empty_tile_pairs(), tmp_enriched)
    os.replace(tmp_enriched, enriched_path)

    overall = {
        "params": asdict(params),
        "n_neurons": int(len(manifest)),
        "near_pair_count": int(near_pair_count),
        "connected_near_pair_count": int(connected_near_pair_count),
        "fraction_connected": None if near_pair_count == 0 else float(connected_near_pair_count / near_pair_count),
        "reduce_mode": "bucketed",
        "reduce_buckets": int(n_buckets),
    }
    write_json(summaries / "overall.json", overall)

    per = pd.DataFrame({
        "idx": np.arange(max_idx + 1, dtype=np.int64),
        "near_pair_count": per_neuron_near,
        "connected_near_pair_count": per_neuron_connected,
    })
    per["fraction_connected"] = np.where(per["near_pair_count"] == 0, np.nan, per["connected_near_pair_count"] / per["near_pair_count"])
    per_neuron = manifest.merge(per, on="idx", how="left")
    per_neuron[["near_pair_count", "connected_near_pair_count"]] = per_neuron[["near_pair_count", "connected_near_pair_count"]].fillna(0).astype("int64")
    write_parquet(per_neuron, summaries / "by_neuron.parquet")

    distance_rows = [
        _summary_row_from_counts(key, ["distance_bin_nm"], values[0], values[1])
        for key, values in distance_counts.items()
    ]
    write_parquet(pd.DataFrame(distance_rows), summaries / "by_distance_bin.parquet")
    for column, path_name in category_paths.items():
        rows = [
            _summary_row_from_counts(key, [f"{column}_1", f"{column}_2"], values[0], values[1])
            for key, values in category_counts.get(column, {}).items()
        ]
        if rows:
            write_parquet(pd.DataFrame(rows), summaries / path_name)
    return overall


def annotate_connection_counts(near: pd.DataFrame, edges: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    out = near.copy()
    a_col = f"{prefix}_syn_count_a_to_b"
    b_col = f"{prefix}_syn_count_b_to_a"
    connected_col = f"{prefix}_connected_any"
    if out.empty:
        out[a_col] = pd.Series(dtype="int64")
        out[b_col] = pd.Series(dtype="int64")
        out[connected_col] = pd.Series(dtype="bool")
        return out
    idxs = set(out["idx_a"].astype(int)) | set(out["idx_b"].astype(int))
    sub = edges[edges["pre_idx"].isin(idxs) & edges["post_idx"].isin(idxs)]
    counts = {(int(r.pre_idx), int(r.post_idx)): int(r.syn_count) for r in sub[["pre_idx", "post_idx", "syn_count"]].itertuples(index=False)}
    out[a_col] = [counts.get((int(a), int(b)), 0) for a, b in out[["idx_a", "idx_b"]].itertuples(index=False)]
    out[b_col] = [counts.get((int(b), int(a)), 0) for a, b in out[["idx_a", "idx_b"]].itertuples(index=False)]
    out[connected_col] = (out[a_col] > 0) | (out[b_col] > 0)
    return out


def enrich_pairs(near: pd.DataFrame, neurons_manifest: pd.DataFrame) -> pd.DataFrame:
    manifest_cols = [c for c in NEURON_NAME_COLUMNS + [
        "pre_synapse_count",
        "dominant_pre_neuropil",
        "dominant_pre_neuropil_fraction",
        "top_pre_neuropils",
        "post_synapse_count",
        "dominant_post_neuropil",
        "dominant_post_neuropil_fraction",
        "top_post_neuropils",
    ] if c in neurons_manifest.columns]
    base = neurons_manifest[manifest_cols].copy()
    a = base.add_suffix("_a").rename(columns={"idx_a": "idx_a"})
    b = base.add_suffix("_b").rename(columns={"idx_b": "idx_b"})
    out = near.merge(a, on="idx_a", how="left").merge(b, on="idx_b", how="left")
    return out


def summarize_boolean(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        cols = (group_cols or []) + ["near_pair_count", "connected_near_pair_count", "fraction_connected", "median_min_distance_nm"]
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        grouped = [((), df)]
    rows = []
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols or [], key)}
        connected = int(g["connected_any"].sum())
        n = int(len(g))
        row.update({
            "near_pair_count": n,
            "connected_near_pair_count": connected,
            "fraction_connected": None if n == 0 else connected / n,
            "median_min_distance_nm": float(g["min_distance_nm"].median()) if n else None,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def category_pair_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    a = df[f"{column}_a"].fillna("None").astype(str)
    b = df[f"{column}_b"].fillna("None").astype(str)
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    tmp = df[["connected_any", "min_distance_nm"]].copy()
    tmp[f"{column}_1"] = lo
    tmp[f"{column}_2"] = hi
    return summarize_boolean(tmp, [f"{column}_1", f"{column}_2"])


def write_summaries(out_dir: Path, near: pd.DataFrame, neurons_manifest: pd.DataFrame, params: FullRunParams) -> dict:
    summaries = out_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)

    overall = {
        "params": asdict(params),
        "n_neurons": int(len(neurons_manifest)),
        "near_pair_count": int(len(near)),
        "connected_near_pair_count": int(near["connected_any"].sum()) if "connected_any" in near else 0,
        "fraction_connected": None if len(near) == 0 else float(near["connected_any"].sum() / len(near)),
    }
    write_json(summaries / "overall.json", overall)

    endpoint = pd.concat([
        near[["idx_a", "connected_any"]].rename(columns={"idx_a": "idx"}),
        near[["idx_b", "connected_any"]].rename(columns={"idx_b": "idx"}),
    ], ignore_index=True) if not near.empty else pd.DataFrame({"idx": pd.Series(dtype="int64"), "connected_any": pd.Series(dtype="bool")})
    per_neuron = endpoint.groupby("idx", as_index=False).agg(
        near_pair_count=("connected_any", "size"),
        connected_near_pair_count=("connected_any", "sum"),
    )
    per_neuron["fraction_connected"] = per_neuron["connected_near_pair_count"] / per_neuron["near_pair_count"]
    per_neuron = neurons_manifest.merge(per_neuron, on="idx", how="left")
    per_neuron[["near_pair_count", "connected_near_pair_count"]] = per_neuron[["near_pair_count", "connected_near_pair_count"]].fillna(0).astype("int64")
    write_parquet(per_neuron, summaries / "by_neuron.parquet")

    if near.empty:
        write_parquet(pd.DataFrame(), summaries / "by_distance_bin.parquet")
    else:
        bins = np.arange(0, params.threshold_nm + 100, 100)
        tmp = near[["connected_any", "min_distance_nm"]].copy()
        tmp["distance_bin_nm"] = pd.cut(tmp["min_distance_nm"], bins=bins, right=False, include_lowest=True).astype(str)
        write_parquet(summarize_boolean(tmp, ["distance_bin_nm"]), summaries / "by_distance_bin.parquet")

    for column, path_name in [
        ("super_class", "by_super_class_pair.parquet"),
        ("cell_class", "by_cell_class_pair.parquet"),
        ("cell_type", "by_cell_type_pair.parquet"),
        ("nt_canonical", "by_nt_pair.parquet"),
        ("side", "by_side_pair.parquet"),
        ("flow", "by_flow_pair.parquet"),
        ("dominant_post_neuropil", "by_dominant_post_neuropil_pair.parquet"),
        ("dominant_pre_neuropil", "by_dominant_pre_neuropil_pair.parquet"),
    ]:
        if f"{column}_a" in near.columns and f"{column}_b" in near.columns:
            write_parquet(category_pair_summary(near, column), summaries / path_name)
    return overall


def reduce_run(cfg: Config, params: FullRunParams) -> dict:
    paths = cfg.paths()
    out = run_dir(cfg, params)
    missing_tiles = missing_pair_tiles(out)
    if missing_tiles:
        preview = ",".join(str(i) for i in missing_tiles[:20])
        suffix = "..." if len(missing_tiles) > 20 else ""
        raise RuntimeError(f"{len(missing_tiles)} tile-pair output(s) are missing before reduce: {preview}{suffix}")
    near = reduce_near_pair_tiles(Path(p) for p in glob.glob(str(out / "near_pairs_tiles" / "tile_*.parquet")))
    edges_full = read_parquet(paths.edges_full)
    edges = read_parquet(paths.edges)
    near = annotate_connection_counts(near, edges_full, prefix="any")
    near = annotate_connection_counts(near, edges, prefix="canonical")
    near["connected_any"] = near["any_connected_any"]
    near["syn_count_a_to_b"] = near["any_syn_count_a_to_b"]
    near["syn_count_b_to_a"] = near["any_syn_count_b_to_a"]
    manifest = read_parquet(out / "neurons_manifest.parquet")
    near = enrich_pairs(near, manifest)
    write_parquet(near, out / "near_pairs_enriched.parquet")
    return write_summaries(out, near, manifest, params)


def status_run(cfg: Config, params: FullRunParams) -> dict:
    out = run_dir(cfg, params)
    status = {"run_dir": str(out), "exists": out.exists()}
    for rel, key in [
        ("shard_manifest.parquet", "n_shards"),
        ("tiles/tile_manifest.parquet", "n_tiles"),
    ]:
        path = out / rel
        status[key] = int(len(read_parquet(path))) if path.exists() else 0
    sample_shards_done = len(list((out / "samples").glob("shard_*.npz"))) if (out / "samples").exists() else 0
    missing_samples = missing_sample_shards(out)
    shard_manifest = out / "shard_manifest.parquet"
    status["sample_shards_done"] = sample_shards_done
    status["sample_shards_expected"] = int(len(read_parquet(shard_manifest))) if shard_manifest.exists() else 0
    status["sample_shards_missing"] = int(len(missing_samples))
    status["sample_shards_missing_preview"] = ",".join(str(i) for i in missing_samples[:50])
    tile_pair_files_done = len(list((out / "near_pairs_tiles").glob("tile_*.parquet"))) if (out / "near_pairs_tiles").exists() else 0
    missing_tiles = missing_pair_tiles(out)
    status["tile_pair_files_done"] = tile_pair_files_done
    status["tile_pair_files_expected"] = status["n_tiles"]
    status["tile_pair_files_missing"] = int(len(missing_tiles))
    status["tile_pair_files_missing_preview"] = ",".join(str(i) for i in missing_tiles[:50])
    status["summary_exists"] = (out / "summaries" / "overall.json").exists()
    return status


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-name", default=RUN_NAME_DEFAULT)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--max-neurons", type=int, default=None)
    ap.add_argument("--threshold-um", type=float, default=2.0)
    ap.add_argument("--connection", choices=["any", "canonical"], default="any")
    ap.add_argument("--mesh-source", choices=[proximity.MESH_SOURCE_CLOUDVOLUME_PUBLIC, proximity.MESH_SOURCE_FAFBSEG], default=proximity.MESH_SOURCE_CLOUDVOLUME_PUBLIC)
    ap.add_argument("--mesh-path", default=proximity.PUBLIC_FLYWIRE_MESH_PATH)
    ap.add_argument("--dataset", default="public")
    ap.add_argument("--lod", type=int, default=1)
    ap.add_argument("--lod-fallback", type=int, default=0)
    ap.add_argument("--site-radius-nm", type=float, default=500.0)
    ap.add_argument("--sample-spacing-nm", type=float, default=250.0)
    ap.add_argument("--mesh-units", choices=["nm", "voxel"], default="nm")
    ap.add_argument("--synapse-coordinate-units", choices=["nm", "voxel"], default="nm")
    ap.add_argument("--tile-nm", type=float, default=50_000.0)
    ap.add_argument("--pair-core-chunk-size", type=int, default=512)
    ap.add_argument("--pair-spill-rows", type=int, default=500_000)
    ap.add_argument("--reduce-buckets", type=int, default=64)
    ap.add_argument("--force", action="store_true")


def params_from_args(args: argparse.Namespace) -> FullRunParams:
    return FullRunParams(
        run_name=args.run_name,
        shard_size=args.shard_size,
        max_neurons=args.max_neurons,
        threshold_um=args.threshold_um,
        connection=args.connection,
        mesh_source=args.mesh_source,
        mesh_path=args.mesh_path,
        dataset=args.dataset,
        lod=args.lod,
        lod_fallback=args.lod_fallback,
        site_radius_nm=args.site_radius_nm,
        sample_spacing_nm=args.sample_spacing_nm,
        mesh_units=args.mesh_units,
        synapse_coordinate_units=args.synapse_coordinate_units,
        tile_nm=args.tile_nm,
        pair_core_chunk_size=args.pair_core_chunk_size,
        pair_spill_rows=args.pair_spill_rows,
        reduce_buckets=args.reduce_buckets,
        force=args.force,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="flyconn.experiments.proximity_full")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "build-tiles", "reduce", "status"]:
        ap = sub.add_parser(name)
        add_common_args(ap)
    ap = sub.add_parser("sample-shard")
    add_common_args(ap)
    ap.add_argument("--shard-id", type=int, required=True)
    ap = sub.add_parser("pair-tile")
    add_common_args(ap)
    ap.add_argument("--tile-id", type=int, required=True)
    ap = sub.add_parser("reduce-bucket")
    add_common_args(ap)
    ap.add_argument("--bucket-id", type=int, required=True)
    ap = sub.add_parser("finalize-reduce")
    add_common_args(ap)
    ap = sub.add_parser("status-field")
    add_common_args(ap)
    ap.add_argument("--field", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    params = params_from_args(args)
    try:
        if args.command == "prepare":
            result = prepare_run(cfg, params)
        elif args.command == "sample-shard":
            result = sample_shard(cfg, params, args.shard_id)
        elif args.command == "build-tiles":
            result = build_tiles(cfg, params)
        elif args.command == "pair-tile":
            result = pair_tile(cfg, params, args.tile_id)
        elif args.command == "reduce-bucket":
            result = reduce_bucket(cfg, params, args.bucket_id)
        elif args.command == "finalize-reduce":
            result = finalize_reduce_buckets(cfg, params)
        elif args.command == "reduce":
            result = reduce_run(cfg, params)
        elif args.command == "status":
            result = status_run(cfg, params)
        elif args.command == "status-field":
            result = status_run(cfg, params).get(args.field, 0)
            print(result)
            return 0
        else:
            raise ValueError(f"unknown command {args.command!r}")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
