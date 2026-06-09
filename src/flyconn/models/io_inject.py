"""Input-node and readout-node selection + a BFS reachability guard.

Biological I/O sets (so the model is a connectome classifier, not a disguised MLP):
  * INPUT nodes  = photoreceptors (cell_type R1-6/R7/R8). Present in whole-brain and in
    the optic* subgraphs (which use the visual_mask union). MNIST is injected here only.
  * READOUT nodes = a biological output subset of the FINAL activations:
        optic*      -> visual_projection neurons (VPN)
        whole*      -> descending + visual_projection
    A linear head maps these to 10 logits.

BFS guard: the unroll length T must be >= the hop distance from the input set to the
readout set on the directed sub-adjacency, or the readout sees no signal. We compute
that distance from the subgraph's own edge buffers (no extra data load).
"""

from __future__ import annotations

import collections

import numpy as np

from .subgraphs import PHOTORECEPTOR_TYPES, Subgraph


def input_local_ids(sub: Subgraph) -> np.ndarray:
    """Local ids of photoreceptor input nodes within the subgraph."""
    n = sub.neurons
    if "cell_type" not in n.columns:
        raise KeyError("subgraph neuron table lacks 'cell_type'")
    mask = n["cell_type"].isin(PHOTORECEPTOR_TYPES).to_numpy()
    ids = np.flatnonzero(mask)
    return ids.astype(np.int64)


def readout_local_ids(sub: Subgraph) -> np.ndarray:
    """Local ids of the biological readout nodes for this subgraph."""
    n = sub.neurons
    sc = n["super_class"]
    if sub.subgraph_id.startswith("optic"):
        mask = (sc == "visual_projection").to_numpy()
    else:  # whole*
        mask = sc.isin(["descending", "visual_projection"]).to_numpy()
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        # Fallback: if no biological readout node is present (e.g. a hemisphere with no
        # VPN/descending after the side filter), use the highest-out-degree nodes.
        deg = np.bincount(sub.col_idx, minlength=sub.N)
        ids = np.argsort(deg)[-min(256, sub.N):]
    return ids.astype(np.int64)


def bfs_hops(sub: Subgraph, sources: np.ndarray, targets: np.ndarray) -> dict:
    """Min hop distance from any source to the target set on the directed graph.

    Edges flow presyn (col_idx) -> postsyn (row_idx). Returns dict with reachable count
    and min/median/max hop distance to the target set.
    """
    adj = collections.defaultdict(list)
    for post, pre in zip(sub.row_idx.tolist(), sub.col_idx.tolist()):
        adj[pre].append(post)
    dist = {int(s): 0 for s in sources.tolist()}
    q = collections.deque(dist)
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):  # noqa: B905
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    tset = set(int(t) for t in targets.tolist())
    reached = [dist[t] for t in tset if t in dist]
    return {
        "n_targets": len(tset),
        "n_reached": len(reached),
        "min_hops": int(min(reached)) if reached else None,
        "median_hops": int(np.median(reached)) if reached else None,
        "max_hops": int(max(reached)) if reached else None,
    }


def recommend_T(sub: Subgraph, input_ids: np.ndarray, readout_ids: np.ndarray,
                floor: int = 10) -> tuple[int, dict]:
    """Recommended unroll T = max(floor, max_hops to readout). Returns (T, bfs_report)."""
    rep = bfs_hops(sub, input_ids, readout_ids)
    depth = rep["max_hops"] or floor
    return max(floor, depth), rep
