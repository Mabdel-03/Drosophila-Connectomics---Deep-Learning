"""Live-only skeleton/morphology retrieval for the FD3 anatomy claims (Family K).

Egelhaaf's FD3 anatomy (dendrite in the MEDIAL lobula plate sparing the lateral border and
proximal part; full dorso-ventral span; an extra ipsilateral protocerebral dendrite; an axon
crossing the midline POSTERIOR to the noduli and terminating in the contralateral posterior
optic foci) is morphological, not readable from the synapse *count* tables alone. We fetch
the precomputed skeleton from the CAVE skeleton service (``caveclient.skeletonservice``),
falling back to the L2 cache (``caveclient.l2cache``) for representative vertices. Neither
needs cloudvolume.

This is **live-only**: it requires a CAVE token + network. The offline track has no
skeletons, so the caller (``derive.k_fd3_lpt42``) downgrades all morphology claims to
UNVERIFIABLE when ``src.track != 'live'``. Skeletons are cached to parquet so a credentialed
run leaves a permanent, re-auditable artifact and later runs need no token.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import fw_access as FW
from ..io import read_parquet, write_parquet
from ..paths import cache_root


@dataclass
class Skeleton:
    """A neuron skeleton: vertices in micrometres + edges + the root it came from."""

    root_id: int
    vertices_um: np.ndarray        # (N, 3) micrometres
    edges: np.ndarray              # (M, 2) vertex-index pairs (empty for L2 fallback)
    source: str                    # "skeleton_service" | "l2cache"

    @property
    def ok(self) -> bool:
        return len(self.vertices_um) >= 2


def _cache_path(root_id: int) -> Path:
    d = cache_root() / f"v{FW.VERSION}" / "paper" / "live_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"skel_{root_id}.parquet"


def _edges_path(root_id: int) -> Path:
    return _cache_path(root_id).with_name(f"skel_{root_id}_edges.parquet")


def _from_cache(root_id: int) -> Skeleton | None:
    p = _cache_path(root_id)
    if not p.exists():
        return None
    df = read_parquet(p)
    verts = df[["x", "y", "z"]].to_numpy(dtype=float)
    src = str(df["source"].iloc[0]) if "source" in df.columns and len(df) else "cache"
    ep = _edges_path(root_id)
    edges = (read_parquet(ep)[["a", "b"]].to_numpy(dtype=int)
             if ep.exists() else np.empty((0, 2), int))
    return Skeleton(root_id=int(root_id), vertices_um=verts, edges=edges, source=src)


def _to_cache(skel: Skeleton) -> None:
    df = pd.DataFrame(skel.vertices_um, columns=["x", "y", "z"])
    df["source"] = skel.source
    write_parquet(df, _cache_path(skel.root_id))
    if len(skel.edges):
        write_parquet(pd.DataFrame(skel.edges, columns=["a", "b"]), _edges_path(skel.root_id))


def _from_fafbseg(root_id: int) -> Skeleton | None:
    """Skeletonize via fafbseg.flywire.skeletonize_neuron (meshes via CloudVolume +
    skeletonizes locally with skeletor). Independent of the CAVE skeleton service / L2 cache,
    so this is the ONLY path that yields a real skeleton on the public datastack.

    Auth: CloudVolume reads ~/.cloudvolume/secrets/chunkedgraph-secret.json (the FlyWire token).
    Returns None (never raises) if fafbseg is absent, the token is missing, or the fetch fails.
    """
    try:
        from fafbseg import flywire
    except Exception:
        fetch_skeleton.last_skip_reason = ("fafbseg not installed; "
                                           "pip install fafbseg in the flyconn_cave env")
        return None
    try:
        flywire.set_default_dataset("public")
        n = flywire.skeletonize_neuron(int(root_id), dataset="public")
        verts = n.nodes[["x", "y", "z"]].to_numpy(dtype=float) / 1000.0  # nm -> um
        # navis node/parent connectivity -> vertex-index edge list.
        nid = {int(v): i for i, v in enumerate(n.nodes["node_id"].to_numpy())}
        edges = np.array(
            [[nid[int(a)], nid[int(b)]]
             for a, b in zip(n.nodes["node_id"], n.nodes["parent_id"]) if int(b) in nid],
            dtype=int) if len(n.nodes) else np.empty((0, 2), int)
        skel = Skeleton(int(root_id), verts, edges, "fafbseg")
        if skel.ok:
            _to_cache(skel)
            return skel
    except Exception as e:  # noqa: BLE001 — degrade, never crash the family
        fetch_skeleton.last_skip_reason = f"fafbseg skeletonize: {type(e).__name__}: {e}"
        print(f"[skeleton_fetch] fafbseg skeletonize failed for {root_id}: {e}")
    return None


def fetch_skeleton(src, root_id: int) -> Skeleton | None:
    """Fetch one real skeleton (never raises; returns None on any failure).

    Order: project-tree cache -> fafbseg local skeletonization (the working path on the public
    datastack) -> CAVE skeleton service -> L2 cache. The fafbseg path needs no live ``src`` and
    works on either track; the CAVE-service / L2 paths are kept as fallbacks but are dead on
    ``flywire_fafb_public`` (no L2 cache -> ``NoL2CacheException``). ``last_skip_reason`` carries
    the reason when nothing is returned, so the caller falls back to the synapse-cloud proxy.
    """
    cached = _from_cache(root_id)
    if cached is not None:
        return cached

    # 1. fafbseg local skeletonization — the only path yielding a real skeleton here.
    skel = _from_fafbseg(root_id)
    if skel is not None:
        return skel

    # The remaining CAVE paths need a live client; if absent, the proxy takes over.
    client = getattr(src, "client", None)
    if client is None:
        if not fetch_skeleton.last_skip_reason:
            fetch_skeleton.last_skip_reason = "no live client and fafbseg unavailable"
        return None

    # 2. Precomputed skeleton service (attribute is `.skeleton` on the instantiated client).
    svc = getattr(client, "skeleton", None) or getattr(client, "skeletonservice", None)
    try:
        sk = svc.get_skeleton(int(root_id), output_format="dict")
        verts = np.asarray(sk["vertices"], dtype=float) / 1000.0  # nm -> um
        edges = np.asarray(sk.get("edges", np.empty((0, 2))), dtype=int)
        skel = Skeleton(int(root_id), verts, edges, "skeleton_service")
        if skel.ok:
            _to_cache(skel)
            return skel
    except Exception as e:  # noqa: BLE001 — degrade to L2, never crash the family
        fetch_skeleton.last_skip_reason = f"skeleton service: {type(e).__name__}: {e}"
        print(f"[skeleton_fetch] skeleton service failed for {root_id}: {e}")

    # 2. L2-cache fallback: representative points of the L2 chunks of the root.
    try:
        l2_ids = client.chunkedgraph.get_leaves(int(root_id), stop_layer=2)
        data = client.l2cache.get_l2data(list(map(int, l2_ids)), attributes=["rep_coord_nm"])
        pts = [v["rep_coord_nm"] for v in data.values() if v.get("rep_coord_nm")]
        if len(pts) >= 2:
            verts = np.asarray(pts, dtype=float) / 1000.0
            skel = Skeleton(int(root_id), verts, np.empty((0, 2), int), "l2cache")
            _to_cache(skel)
            return skel
    except Exception as e:  # noqa: BLE001
        fetch_skeleton.last_skip_reason = f"l2cache: {type(e).__name__}: {e}"
        print(f"[skeleton_fetch] l2cache fallback failed for {root_id}: {e}")
    return None


fetch_skeleton.last_skip_reason = ""  # type: ignore[attr-defined]
