"""Dendrite mesh proximity vs. real connectivity experiment.

Pilot question:
  Among neuron pairs whose synapse-derived dendrite mesh samples are within a
  spatial threshold, what fraction have any real synaptic connection?

The mesh stack is intentionally optional and imported lazily. Pure helpers in
this module are unit-testable without FlyWire credentials or network access.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from scipy.spatial import cKDTree

from ..config import Config, load_config
from ..io import read_json, read_parquet, write_json, write_parquet
from ..models.subgraphs import SUBGRAPHS

VOXEL_NM = np.array([4.0, 4.0, 40.0], dtype=np.float64)
PUBLIC_FLYWIRE_MESH_PATH = "precomputed://gs://flywire_v141_m783"
MESH_SOURCE_CLOUDVOLUME_PUBLIC = "cloudvolume-public"
MESH_SOURCE_FAFBSEG = "fafbseg"

POST_POSITION_ALIASES = {
    "x": ("post_pt_position_x", "post_position_x", "post_x", "x"),
    "y": ("post_pt_position_y", "post_position_y", "post_y", "y"),
    "z": ("post_pt_position_z", "post_position_z", "post_z", "z"),
}


@dataclass(frozen=True)
class ProximityParams:
    subgraph: str = "optic_left"
    n: int = 1000
    seed: int = 0
    threshold_um: float = 2.0
    connection: str = "any"
    mesh_source: str = MESH_SOURCE_CLOUDVOLUME_PUBLIC
    mesh_path: str = PUBLIC_FLYWIRE_MESH_PATH
    dataset: str = "public"
    lod: int = 1
    lod_fallback: int | None = 0
    site_radius_nm: float = 500.0
    sample_spacing_nm: float = 250.0
    mesh_units: str = "nm"
    synapse_coordinate_units: str = "nm"
    threads: int = 5

    @property
    def threshold_nm(self) -> float:
        return float(self.threshold_um) * 1000.0


@dataclass(frozen=True)
class MeshFetchResult:
    vertices_nm: np.ndarray
    faces: np.ndarray | None
    mesh_source: str
    mesh_path: str | None = None
    lod_requested: int | None = None
    lod_used: int | None = None
    cache_status: str = "fetched"


def experiment_dir(cfg: Config, p: ProximityParams) -> Path:
    name = f"{p.subgraph}_n{p.n}_seed{p.seed}"
    return cfg.paths().root / "experiments" / "proximity" / name


def select_neuron_sample(neurons: pd.DataFrame, subgraph: str, n: int, seed: int) -> pd.DataFrame:
    """Return a deterministic sample from an existing stage-3 subgraph selector."""
    if subgraph not in SUBGRAPHS:
        raise KeyError(f"unknown subgraph {subgraph!r}; choices: {sorted(SUBGRAPHS)}")
    if n <= 0:
        raise ValueError("--n must be positive")

    pool = neurons.loc[SUBGRAPHS[subgraph](neurons)].sort_values("idx", kind="stable")
    if pool.empty:
        raise ValueError(f"subgraph {subgraph!r} selected 0 neurons")
    if n > len(pool):
        raise ValueError(f"requested n={n:,} but subgraph {subgraph!r} has {len(pool):,} neurons")

    out = pool.sample(n=n, random_state=seed, replace=False)
    return out.sort_values("idx", kind="stable").reset_index(drop=True)


def resolve_column(columns: Iterable[str], aliases: Iterable[str], *, what: str) -> str:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise KeyError(f"Could not find {what}; tried {tuple(aliases)} in {sorted(columns)}")


def _feather_column_names(path: Path) -> list[str]:
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_file(source).schema.names


def _synapse_columns(path: Path) -> tuple[str, str, str, str]:
    cols = _feather_column_names(path)
    post = resolve_column(
        cols,
        ("post_pt_root_id", "post_root_id", "post", "postsynaptic_root_id"),
        what="postsynaptic root id",
    )
    x = resolve_column(cols, POST_POSITION_ALIASES["x"], what="postsynaptic x coordinate")
    y = resolve_column(cols, POST_POSITION_ALIASES["y"], what="postsynaptic y coordinate")
    z = resolve_column(cols, POST_POSITION_ALIASES["z"], what="postsynaptic z coordinate")
    return post, x, y, z


def load_postsynaptic_sites_nm(
    synapse_path: str | Path,
    root_ids: Iterable[int],
    *,
    coordinate_units: str = "nm",
    voxel_nm: np.ndarray = VOXEL_NM,
) -> dict[int, np.ndarray]:
    """Load postsynaptic coordinates for selected roots in nanometers.

    The 9.5 GB synapse feather is scanned in Arrow record batches so only selected
    rows and four columns are materialized in pandas.
    """
    if coordinate_units not in {"nm", "voxel"}:
        raise ValueError("coordinate_units must be 'nm' or 'voxel'")
    path = Path(synapse_path)
    roots = np.asarray(list(root_ids), dtype=np.int64)
    if roots.size == 0:
        return {}

    post, x, y, z = _synapse_columns(path)
    wanted_cols = [post, x, y, z]
    wanted_roots = pa.array(roots)
    pieces: list[pd.DataFrame] = []

    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i).select(wanted_cols)
            table = pa.Table.from_batches([batch])
            mask = pc.is_in(table[post], value_set=wanted_roots)
            if int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0) == 0:
                continue
            filtered = table.filter(mask).to_pandas()
            filtered = filtered.rename(columns={post: "root_id", x: "x", y: "y", z: "z"})
            pieces.append(filtered)

    out = {int(r): np.empty((0, 3), dtype=np.float64) for r in roots}
    if not pieces:
        return out

    df = pd.concat(pieces, ignore_index=True)
    coords = df[["x", "y", "z"]].to_numpy(np.float64)
    if coordinate_units == "voxel":
        coords = coords * voxel_nm
    df = pd.DataFrame({"root_id": df["root_id"].to_numpy(np.int64)})
    df[["x_nm", "y_nm", "z_nm"]] = coords
    for root_id, group in df.groupby("root_id", sort=False):
        out[int(root_id)] = group[["x_nm", "y_nm", "z_nm"]].to_numpy(np.float64)
    return out


def _flywire_secret_path() -> Path:
    return Path.home() / ".cloudvolume" / "secrets" / "global.daf-apis.com-cave-secret.json"


def _flywire_env_token() -> str | None:
    token = os.environ.get("CAVE_TOKEN") or os.environ.get("FLYWIRE_TOKEN")
    token = token.strip() if token else ""
    return token or None


def has_saved_flywire_auth() -> bool:
    """Best-effort saved-token check used before expensive mesh fetches."""
    secret = _flywire_secret_path()
    if not secret.exists():
        return False
    try:
        data = read_json(secret)
    except Exception:
        return False
    return bool(data.get("token"))


def has_flywire_auth() -> bool:
    return bool(_flywire_env_token()) or has_saved_flywire_auth()


def configure_flywire_auth_from_env() -> bool:
    """Persist an env-provided token where fafbseg/cloud-volume expect it."""
    token = _flywire_env_token()
    if not token:
        return False
    if has_saved_flywire_auth():
        return True

    from fafbseg import flywire

    with contextlib.redirect_stdout(io.StringIO()):
        flywire.set_chunkedgraph_secret(token, overwrite=True)
    return has_saved_flywire_auth()


def require_mesh_stack(mesh_source: str) -> None:
    if mesh_source == MESH_SOURCE_CLOUDVOLUME_PUBLIC:
        try:
            import cloudvolume  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Mesh experiment dependencies are missing. Install with "
                "`pip install -e '.[mesh]'` inside the project environment."
            ) from e
        return

    if mesh_source == MESH_SOURCE_FAFBSEG:
        try:
            import fafbseg  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "fafbseg mesh dependencies are missing. Install with "
                "`pip install -e '.[mesh-auth]'` inside the project environment."
            ) from e
        if not (configure_flywire_auth_from_env() or has_saved_flywire_auth()):
            raise RuntimeError(
                "FlyWire/CAVE auth token not found for --mesh-source fafbseg. Set "
                "CAVE_TOKEN or FLYWIRE_TOKEN, create one with "
                "`from fafbseg import flywire; flywire.set_chunkedgraph_secret('<token>')` "
                "or place it in ~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json."
            )
        return

    raise ValueError(f"unknown mesh_source {mesh_source!r}")


def _coerce_single_mesh(obj):
    """Return one MeshNeuron-like object from fafbseg's possible return shapes."""
    if hasattr(obj, "vertices"):
        return obj
    if hasattr(obj, "__len__") and len(obj) > 0:
        first = obj[0]
        if hasattr(first, "vertices"):
            return first
    return obj


def _mesh_arrays_from_object(
    mesh,
    root_id: int,
    *,
    mesh_units: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    vertices = np.asarray(getattr(mesh, "vertices"), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.size == 0:
        raise ValueError(f"mesh for root_id={root_id} has no Nx3 vertices")
    if mesh_units == "voxel":
        vertices = vertices * VOXEL_NM
    elif mesh_units != "nm":
        raise ValueError(f"unknown mesh_units {mesh_units!r}")

    faces_obj = getattr(mesh, "faces", None)
    faces = None
    if faces_obj is not None:
        faces = np.asarray(faces_obj, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
            faces = None
    return vertices, faces


def open_cloudvolume(mesh_path: str):
    from cloudvolume import CloudVolume

    return CloudVolume(mesh_path, use_https=True, fill_missing=True, cache=False)


def fetch_cloudvolume_mesh_arrays(
    root_id: int,
    *,
    volume,
    mesh_path: str,
    lod: int,
    lod_fallback: int | None,
    mesh_units: str,
) -> MeshFetchResult:
    """Fetch one public CloudVolume mesh without FlyWire/CAVE credentials."""
    lods = [int(lod)]
    if lod_fallback is not None and int(lod_fallback) not in lods:
        lods.append(int(lod_fallback))

    last_error: Exception | None = None
    for lod_candidate in lods:
        try:
            meshes = volume.mesh.get(int(root_id), lod=lod_candidate, allow_missing=True)
            mesh = meshes.get(int(root_id)) if hasattr(meshes, "get") else _coerce_single_mesh(meshes)
            if mesh is None:
                raise ValueError(f"mesh for root_id={root_id} is missing")
            vertices, faces = _mesh_arrays_from_object(mesh, root_id, mesh_units=mesh_units)
            return MeshFetchResult(
                vertices_nm=vertices,
                faces=faces,
                mesh_source=MESH_SOURCE_CLOUDVOLUME_PUBLIC,
                mesh_path=mesh_path,
                lod_requested=int(lod),
                lod_used=lod_candidate,
            )
        except Exception as e:
            last_error = e

    assert last_error is not None
    raise RuntimeError(f"failed to fetch public mesh for root_id={root_id}: {last_error}") from last_error


def fetch_fafbseg_mesh_arrays(
    root_id: int,
    *,
    dataset: str,
    lod: int,
    threads: int,
    mesh_units: str,
) -> MeshFetchResult:
    """Fetch one FlyWire mesh through fafbseg; this path requires FlyWire auth."""
    from fafbseg import flywire

    mesh = flywire.get_mesh_neuron(
        int(root_id),
        omit_failures=False,
        threads=threads,
        lod=lod,
        progress=False,
        dataset=dataset,
    )
    mesh = _coerce_single_mesh(mesh)
    vertices, faces = _mesh_arrays_from_object(mesh, root_id, mesh_units=mesh_units)
    return MeshFetchResult(
        vertices_nm=vertices,
        faces=faces,
        mesh_source=MESH_SOURCE_FAFBSEG,
        mesh_path=f"fafbseg:{dataset}",
        lod_requested=int(lod),
        lod_used=int(lod),
    )


def load_or_fetch_mesh_arrays(
    cache_dir: Path,
    root_id: int,
    fetcher: Callable[[int], MeshFetchResult],
) -> MeshFetchResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{int(root_id)}.npz"
    if path.exists():
        z = np.load(path)
        faces = z["faces"] if "faces" in z.files and z["faces"].size else None
        return MeshFetchResult(
            vertices_nm=z["vertices_nm"],
            faces=faces,
            mesh_source=str(z["mesh_source"][0]) if "mesh_source" in z.files else "unknown",
            mesh_path=str(z["mesh_path"][0]) if "mesh_path" in z.files else None,
            lod_requested=int(z["lod_requested"][0]) if "lod_requested" in z.files else None,
            lod_used=int(z["lod_used"][0]) if "lod_used" in z.files else None,
            cache_status="cached",
        )

    result = fetcher(int(root_id))
    payload = {
        "vertices_nm": result.vertices_nm.astype(np.float32),
        "mesh_source": np.array([result.mesh_source]),
        "mesh_path": np.array([result.mesh_path or ""]),
    }
    if result.lod_requested is not None:
        payload["lod_requested"] = np.array([int(result.lod_requested)], dtype=np.int64)
    if result.lod_used is not None:
        payload["lod_used"] = np.array([int(result.lod_used)], dtype=np.int64)
    payload["faces"] = np.empty((0, 3), dtype=np.int64) if result.faces is None else result.faces.astype(np.int64)
    np.savez_compressed(path, **payload)
    return result


def face_centroids(vertices: np.ndarray, faces: np.ndarray | None) -> np.ndarray:
    if faces is None or faces.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    return vertices[faces[valid]].mean(axis=1)


def downsample_points_nm(points: np.ndarray, spacing_nm: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if spacing_nm <= 0:
        raise ValueError("sample spacing must be positive")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    vox = np.floor(points / float(spacing_nm)).astype(np.int64)
    _, first = np.unique(vox, axis=0, return_index=True)
    return points[np.sort(first)]


def dendrite_samples_from_mesh(
    vertices_nm: np.ndarray,
    faces: np.ndarray | None,
    post_sites_nm: np.ndarray,
    *,
    site_radius_nm: float,
    sample_spacing_nm: float,
) -> np.ndarray:
    """Keep mesh samples near postsynaptic sites and voxel-downsample them."""
    if site_radius_nm <= 0:
        raise ValueError("site radius must be positive")
    post_sites_nm = np.asarray(post_sites_nm, dtype=np.float64)
    if post_sites_nm.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    candidates = np.vstack([np.asarray(vertices_nm, dtype=np.float64), face_centroids(vertices_nm, faces)])
    if candidates.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    site_tree = cKDTree(post_sites_nm)
    keep = site_tree.query(candidates, distance_upper_bound=float(site_radius_nm))[0] <= site_radius_nm
    return downsample_points_nm(candidates[keep], sample_spacing_nm)


def _metadata_matches(z, expected: dict[str, object]) -> bool:
    for key, expected_value in expected.items():
        if key not in z.files:
            return False
        observed = z[key][0]
        if isinstance(expected_value, float):
            if not np.isclose(float(observed), expected_value):
                return False
        else:
            if str(observed) != str(expected_value):
                return False
    return True


def write_samples_cache(
    path: Path,
    root_id: int,
    samples_nm: np.ndarray,
    *,
    mesh_result: MeshFetchResult,
    params: ProximityParams,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        root_id=np.array([int(root_id)], dtype=np.int64),
        samples_nm=samples_nm.astype(np.float32),
        mesh_source=np.array([mesh_result.mesh_source]),
        mesh_path=np.array([mesh_result.mesh_path or ""]),
        cache_status=np.array([mesh_result.cache_status]),
        lod_requested=np.array([params.lod], dtype=np.int64),
        lod_used=np.array([-1 if mesh_result.lod_used is None else mesh_result.lod_used], dtype=np.int64),
        site_radius_nm=np.array([params.site_radius_nm], dtype=np.float64),
        sample_spacing_nm=np.array([params.sample_spacing_nm], dtype=np.float64),
        synapse_coordinate_units=np.array([params.synapse_coordinate_units]),
    )


def load_samples_cache(path: Path, *, expected: dict[str, object] | None = None) -> np.ndarray | None:
    if not path.exists():
        return None
    z = np.load(path)
    if expected is not None and not _metadata_matches(z, expected):
        return None
    return z["samples_nm"].astype(np.float64)


def find_near_pairs(
    sample_points_by_idx: dict[int, np.ndarray],
    idx_to_root: dict[int, int],
    threshold_nm: float,
) -> pd.DataFrame:
    """Find unordered neuron pairs with any sample points within threshold."""
    if threshold_nm <= 0:
        raise ValueError("distance threshold must be positive")

    labels: list[np.ndarray] = []
    points: list[np.ndarray] = []
    for idx, pts in sample_points_by_idx.items():
        pts = np.asarray(pts, dtype=np.float64)
        if pts.size == 0:
            continue
        points.append(pts)
        labels.append(np.full(len(pts), int(idx), dtype=np.int64))

    if not points:
        return _empty_near_pairs()

    all_points = np.vstack(points)
    owners = np.concatenate(labels)
    tree = cKDTree(all_points)
    coo = tree.sparse_distance_matrix(tree, threshold_nm, output_type="coo_matrix")
    mask = (coo.row < coo.col) & (owners[coo.row] != owners[coo.col])
    if not np.any(mask):
        return _empty_near_pairs()

    a = owners[coo.row[mask]]
    b = owners[coo.col[mask]]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    pairs = pd.DataFrame({
        "idx_a": lo.astype(np.int64),
        "idx_b": hi.astype(np.int64),
        "distance_nm": coo.data[mask].astype(np.float64),
    })
    grouped = (
        pairs.groupby(["idx_a", "idx_b"], as_index=False)
        .agg(min_distance_nm=("distance_nm", "min"), n_close_sample_pairs=("distance_nm", "size"))
    )
    grouped["root_id_a"] = grouped["idx_a"].map(idx_to_root).astype("int64")
    grouped["root_id_b"] = grouped["idx_b"].map(idx_to_root).astype("int64")
    return grouped[[
        "idx_a",
        "idx_b",
        "root_id_a",
        "root_id_b",
        "min_distance_nm",
        "n_close_sample_pairs",
    ]]


def _empty_near_pairs() -> pd.DataFrame:
    return pd.DataFrame({
        "idx_a": pd.Series(dtype="int64"),
        "idx_b": pd.Series(dtype="int64"),
        "root_id_a": pd.Series(dtype="int64"),
        "root_id_b": pd.Series(dtype="int64"),
        "min_distance_nm": pd.Series(dtype="float64"),
        "n_close_sample_pairs": pd.Series(dtype="int64"),
    })


def annotate_connections(near_pairs: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Annotate unordered pairs with directional and either-direction synapse counts."""
    out = near_pairs.copy()
    if out.empty:
        out["syn_count_a_to_b"] = pd.Series(dtype="int64")
        out["syn_count_b_to_a"] = pd.Series(dtype="int64")
        out["connected_any"] = pd.Series(dtype="bool")
        return out

    idxs = set(out["idx_a"].astype(int)) | set(out["idx_b"].astype(int))
    sub = edges[edges["pre_idx"].isin(idxs) & edges["post_idx"].isin(idxs)]
    counts = {
        (int(r.pre_idx), int(r.post_idx)): int(r.syn_count)
        for r in sub[["pre_idx", "post_idx", "syn_count"]].itertuples(index=False)
    }

    ab = []
    ba = []
    for a, b in out[["idx_a", "idx_b"]].itertuples(index=False):
        ab.append(counts.get((int(a), int(b)), 0))
        ba.append(counts.get((int(b), int(a)), 0))
    out["syn_count_a_to_b"] = np.asarray(ab, dtype=np.int64)
    out["syn_count_b_to_a"] = np.asarray(ba, dtype=np.int64)
    out["connected_any"] = (out["syn_count_a_to_b"] > 0) | (out["syn_count_b_to_a"] > 0)
    return out


def summarize(
    params: ProximityParams,
    near_pairs: pd.DataFrame,
    sample_neurons: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict:
    near_pair_count = int(len(near_pairs))
    connected = int(near_pairs["connected_any"].sum()) if "connected_any" in near_pairs else 0
    return {
        "params": asdict(params),
        "threshold_nm": params.threshold_nm,
        "n_requested": int(params.n),
        "n_sampled": int(len(sample_neurons)),
        "n_processed": int(sample_neurons["idx"].nunique() - len(failures)) if "idx" in failures else int(len(sample_neurons)),
        "n_mesh_failures": int(len(failures)),
        "near_pair_count": near_pair_count,
        "connected_near_pair_count": connected,
        "fraction_connected": None if near_pair_count == 0 else connected / near_pair_count,
    }


def write_report(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frac = summary["fraction_connected"]
    frac_txt = "n/a" if frac is None else f"{frac:.6f}"
    lines = [
        "# Mesh Proximity Connectivity Report",
        "",
        "## Summary",
        f"- Sampled neurons: {summary['n_sampled']:,}",
        f"- Processed neurons: {summary['n_processed']:,}",
        f"- Mesh/sample failures: {summary['n_mesh_failures']:,}",
        f"- Near unordered pairs: {summary['near_pair_count']:,}",
        f"- Connected near pairs: {summary['connected_near_pair_count']:,}",
        f"- Fraction connected: {frac_txt}",
        "",
        "## Parameters",
    ]
    for key, value in summary["params"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines))


def run(params: ProximityParams, *, cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    require_mesh_stack(params.mesh_source)

    paths = cfg.paths().ensure()
    out_dir = experiment_dir(cfg, params)
    mesh_cache_name = params.mesh_path.rstrip("/").split("/")[-1].replace(":", "_")
    mesh_cache = paths.root / "mesh_cache" / params.mesh_source / mesh_cache_name / f"lod{params.lod}"
    sample_cache = out_dir / "dendrite_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    neurons = read_parquet(paths.neurons)
    sample = select_neuron_sample(neurons, params.subgraph, params.n, params.seed)
    write_parquet(sample, out_dir / "sample_neurons.parquet")

    roots = sample["root_id"].astype("int64").to_numpy()
    syn_spec = next(f for f in cfg.zenodo_files if f.role == "synapses")
    post_sites = load_postsynaptic_sites_nm(
        paths.raw_file(syn_spec.key),
        roots,
        coordinate_units=params.synapse_coordinate_units,
    )

    failures: list[dict] = []
    samples_by_idx: dict[int, np.ndarray] = {}

    volume = open_cloudvolume(params.mesh_path) if params.mesh_source == MESH_SOURCE_CLOUDVOLUME_PUBLIC else None

    def fetcher(root_id: int) -> MeshFetchResult:
        if params.mesh_source == MESH_SOURCE_CLOUDVOLUME_PUBLIC:
            return fetch_cloudvolume_mesh_arrays(
                root_id,
                volume=volume,
                mesh_path=params.mesh_path,
                lod=params.lod,
                lod_fallback=params.lod_fallback,
                mesh_units=params.mesh_units,
            )
        return fetch_fafbseg_mesh_arrays(
            root_id,
            dataset=params.dataset,
            lod=params.lod,
            threads=params.threads,
            mesh_units=params.mesh_units,
        )

    sample_cache_expected = {
        "mesh_source": params.mesh_source,
        "mesh_path": params.mesh_path if params.mesh_source == MESH_SOURCE_CLOUDVOLUME_PUBLIC else f"fafbseg:{params.dataset}",
        "lod_requested": params.lod,
        "site_radius_nm": float(params.site_radius_nm),
        "sample_spacing_nm": float(params.sample_spacing_nm),
        "synapse_coordinate_units": params.synapse_coordinate_units,
    }

    for row in sample[["idx", "root_id"]].itertuples(index=False):
        idx = int(row.idx)
        root_id = int(row.root_id)
        sample_path = sample_cache / f"{root_id}.npz"
        cached_samples = load_samples_cache(sample_path, expected=sample_cache_expected)
        if cached_samples is not None:
            samples_by_idx[idx] = cached_samples
            continue
        try:
            mesh_result = load_or_fetch_mesh_arrays(mesh_cache, root_id, fetcher)
            samples = dendrite_samples_from_mesh(
                mesh_result.vertices_nm,
                mesh_result.faces,
                post_sites.get(root_id, np.empty((0, 3), dtype=np.float64)),
                site_radius_nm=params.site_radius_nm,
                sample_spacing_nm=params.sample_spacing_nm,
            )
            if samples.size == 0:
                raise ValueError("no dendrite samples after postsynaptic-site restriction")
            write_samples_cache(sample_path, root_id, samples, mesh_result=mesh_result, params=params)
            samples_by_idx[idx] = samples
        except Exception as e:
            failures.append({"idx": idx, "root_id": root_id, "error": str(e)})

    failure_df = pd.DataFrame(failures, columns=["idx", "root_id", "error"])
    write_parquet(failure_df, out_dir / "mesh_failures.parquet")

    idx_to_root = {int(r.idx): int(r.root_id) for r in sample[["idx", "root_id"]].itertuples(index=False)}
    near = find_near_pairs(samples_by_idx, idx_to_root, params.threshold_nm)

    edge_path = paths.edges_full if params.connection == "any" else paths.edges
    edges = read_parquet(edge_path)
    near = annotate_connections(near, edges)
    write_parquet(near, out_dir / "near_pairs.parquet")

    summary = summarize(params, near, sample, failure_df)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "proximity_report.md", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> ProximityParams:
    ap = argparse.ArgumentParser(prog="flyconn.experiments.proximity")
    ap.add_argument("--config", default=None)
    ap.add_argument("--subgraph", default="optic_left", choices=sorted(SUBGRAPHS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold-um", type=float, default=2.0)
    ap.add_argument("--connection", choices=["any", "canonical"], default="any")
    ap.add_argument("--mesh-source", choices=[MESH_SOURCE_CLOUDVOLUME_PUBLIC, MESH_SOURCE_FAFBSEG], default=MESH_SOURCE_CLOUDVOLUME_PUBLIC)
    ap.add_argument("--mesh-path", default=PUBLIC_FLYWIRE_MESH_PATH)
    ap.add_argument("--dataset", default="public")
    ap.add_argument("--lod", type=int, default=1)
    ap.add_argument("--lod-fallback", type=int, default=0)
    ap.add_argument("--site-radius-nm", type=float, default=500.0)
    ap.add_argument("--sample-spacing-nm", type=float, default=250.0)
    ap.add_argument("--mesh-units", choices=["nm", "voxel"], default="nm")
    ap.add_argument("--synapse-coordinate-units", choices=["nm", "voxel"], default="nm")
    ap.add_argument("--threads", type=int, default=5)
    args = ap.parse_args(argv)

    return ProximityParams(
        subgraph=args.subgraph,
        n=args.n,
        seed=args.seed,
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
        threads=args.threads,
    )


def main(argv: list[str] | None = None) -> int:
    params = parse_args(argv)
    # argparse owns --config, but ProximityParams deliberately contains only
    # experiment parameters written to summary.json.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", default=None)
    known, _ = ap.parse_known_args(argv)
    cfg = load_config(known.config)
    try:
        run(params, cfg=cfg)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
