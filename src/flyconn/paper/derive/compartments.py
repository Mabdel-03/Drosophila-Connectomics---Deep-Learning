"""Per-compartment (dendrite vs axon) localization of a cell's synapses.

``k_fd3_lpt42._skeleton_morphology`` already splits skeleton VERTICES into dendrite/axon (by
whether a vertex is nearer the input-synapse cloud or the output-synapse cloud), but nothing
assigns each *input synapse* to a compartment. This module adds that: it builds a compartment
model of a neuron's skeleton, then labels arbitrary synapses (e.g. FD3's inputs) dendrite/axon by
nearest skeleton vertex, and reports the input mass per compartment × partner class.

Answers "WHICH inputs land on the dendrite vs the axon" — e.g. does the motion drive (T4b/T5b) +
the LPC1 sheet arrive on the lobula-plate dendritic tuft, while the contralateral inhibition
arrives on the crossing axon (as Egelhaaf's FD3 anatomy predicts).

Real skeletons come from ``skeleton_fetch.fetch_skeleton`` (live/fafbseg; cached to the project
tree, then readable in any env). When no skeleton is available the model falls back to a
synapse-cloud proxy (vertices = the cell's own input+output synapse points) and callers should
downgrade the corresponding claims to CONFIRMED_WITH_CAVEAT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import skeleton_fetch as SK
from .. import geometry as G

_MOTION_TYPES = ("T4b", "T5b", "T4a", "T5a", "T4c", "T5c", "T4d", "T5d")
_SHEET_PREFIXES = ("LPC", "LLPC", "Tlp", "MeLp", "LPTe")
_DENDRITE, _AXON = "dendrite", "axon"


@dataclass
class CompartmentModel:
    root_id: int
    vertices_um: np.ndarray          # (N,3)
    vertex_label: np.ndarray         # (N,) "dendrite"|"axon"
    soma_proxy: np.ndarray | None    # (3,) dendrite centroid (soma/primary-neurite proxy)
    edges: np.ndarray                # (M,2) vertex-index pairs (empty if none)
    source: str                      # "skeleton" | "synapse_cloud_proxy"

    @property
    def ok(self) -> bool:
        return len(self.vertices_um) >= 2


def build_compartment_model(src, meta, root: int) -> CompartmentModel:
    """Label a neuron's skeleton vertices dendrite/axon by nearer input vs output synapse cloud.

    Lifts the exact rule from ``k_fd3_lpt42._skeleton_morphology`` (a vertex is dendrite if it is
    nearer the input-synapse-cloud centroid than the output-cloud centroid). Falls back to a
    synapse-cloud proxy (input points = dendrite, output points = axon) when no skeleton exists.
    """
    root = int(root)
    din = src.synapses(post_ids=[root]); din = din[din["post_pt_root_id"] == root]
    dout = src.synapses(pre_ids=[root]); dout = dout[dout["pre_pt_root_id"] == root]
    in_pts = G.syn_positions_um(din, "post") if len(din) else np.empty((0, 3))
    out_pts = G.syn_positions_um(dout, "pre") if len(dout) else np.empty((0, 3))
    in_c = G.centroid(in_pts) if len(in_pts) else None
    out_c = G.centroid(out_pts) if len(out_pts) else None

    skel = SK.fetch_skeleton(src, root)
    if skel is not None and skel.ok:
        v = skel.vertices_um
        if in_c is not None and out_c is not None:
            d_in = np.linalg.norm(v - in_c, axis=1)
            d_out = np.linalg.norm(v - out_c, axis=1)
            label = np.where(d_in <= d_out, _DENDRITE, _AXON)
        else:
            label = np.full(len(v), _DENDRITE)
        soma = G.centroid(v[label == _DENDRITE]) if (label == _DENDRITE).any() else None
        return CompartmentModel(root, v, label, soma, np.asarray(skel.edges), "skeleton")

    # Proxy: the cell's own synapse points ARE the model; input->dendrite, output->axon.
    v = np.vstack([p for p in (in_pts, out_pts) if len(p)]) if (len(in_pts) or len(out_pts)) \
        else np.empty((0, 3))
    label = np.concatenate([np.full(len(in_pts), _DENDRITE), np.full(len(out_pts), _AXON)])
    soma = in_c
    return CompartmentModel(root, v, label, soma, np.empty((0, 2), dtype=int), "synapse_cloud_proxy")


def assign_synapses_to_compartments(model: CompartmentModel, syn_df: pd.DataFrame,
                                    on_side: str) -> np.ndarray:
    """Label each synapse dendrite/axon by its nearest skeleton vertex (cKDTree).

    ``on_side`` = "post" for synapses landing ON this cell (its inputs), "pre" for its outputs.
    Returns a (len(syn_df),) array of "dendrite"/"axon"/"" (empty when the model has no vertices).
    """
    if not model.ok or syn_df is None or len(syn_df) == 0:
        return np.array([""] * (0 if syn_df is None else len(syn_df)), dtype=object)
    from scipy.spatial import cKDTree
    pts = G.syn_positions_um(syn_df, on_side)
    _, idx = cKDTree(model.vertices_um).query(pts)
    return model.vertex_label[idx]


def soma_distance_um(model: CompartmentModel, syn_df: pd.DataFrame, on_side: str,
                     *, geodesic: bool = True) -> tuple[np.ndarray, bool]:
    """Distance of each synapse to the soma proxy. Euclidean by default; geodesic (on-cable,
    ``scipy.sparse.csgraph.dijkstra`` over the edge graph) when edges are present and requested.

    Returns (distances_um, used_geodesic).
    """
    if not model.ok or syn_df is None or len(syn_df) == 0 or model.soma_proxy is None:
        return np.array([]), False
    from scipy.spatial import cKDTree
    pts = G.syn_positions_um(syn_df, on_side)
    _, vidx = cKDTree(model.vertices_um).query(pts)          # nearest vertex per synapse
    if geodesic and len(model.edges) > 0:
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra
            v = model.vertices_um
            a, b = model.edges[:, 0], model.edges[:, 1]
            w = np.linalg.norm(v[a] - v[b], axis=1)
            n = len(v)
            g = csr_matrix((np.concatenate([w, w]),
                           (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n, n))
            soma_v = int(cKDTree(v).query(model.soma_proxy[None, :])[1][0])
            dist = dijkstra(g, indices=soma_v, directed=False)
            d = dist[vidx]
            if np.isfinite(d).all():          # disconnected components -> fall back to euclidean
                return d, True
        except Exception:  # noqa: BLE001 — never fail the census on a geodesic hiccup
            pass
    return np.linalg.norm(pts - model.soma_proxy, axis=1), False


def _class_of(cell_type, nt, side, anchor_side) -> str:
    ct = "" if (cell_type is None or (isinstance(cell_type, float) and pd.isna(cell_type))) else str(cell_type)
    if ct in _MOTION_TYPES:
        return "motion"
    if ct.startswith(_SHEET_PREFIXES):
        return "sheet"
    ntv = "" if nt is None or (isinstance(nt, float) and pd.isna(nt)) else str(nt)
    sv = "" if side is None or (isinstance(side, float) and pd.isna(side)) else str(side)
    if ntv == "gaba" and sv and sv != anchor_side:
        return "contra_inhibitor"
    if ntv in ("gaba", "glutamate"):
        return "inhibitor"
    return "other"


def compartment_input_split(src, meta, root: int) -> dict:
    """Split a cell's INPUT synapses by compartment (dendrite/axon) AND by partner class.

    Returns per-compartment totals + by-class breakdown, and a `class_by_compartment` cross-tab
    ({T4b: {dendrite, axon}, LPC1: {...}, LPi14: {...}}). The load-bearing FD3 expectation:
    motion (T4b/T5b) + the LPC-sheet on the dendrite; contralateral GABA on the crossing axon.
    LPi12 (the same-direction detector-gate) is tracked alongside LPi14 (the opponent FD3-gate)
    so their arbor targeting can be compared directly.
    """
    root = int(root)
    model = build_compartment_model(src, meta, root)
    anchor_side = str(meta.by_root.loc[root, "side"]) if root in meta.by_root.index else ""
    din = src.synapses(post_ids=[root]); din = din[din["post_pt_root_id"] == root]
    if len(din) == 0 or not model.ok:
        return {"available": False, "source": model.source, "root_id": root}

    comp = assign_synapses_to_compartments(model, din, "post")
    pre = meta.by_root.reindex(din["pre_pt_root_id"].values)
    ct = pre["cell_type"].to_numpy(); nt = pre["nt_canonical"].to_numpy(); sd = pre["side"].to_numpy()
    klass = np.array([_class_of(c, n, s, anchor_side) for c, n, s in zip(ct, nt, sd)])
    df = pd.DataFrame({"compartment": comp, "cls": klass, "cell_type": ct})
    df = df[df["compartment"] != ""]

    def _block(sub):
        return {"total_syn": int(len(sub)),
                "by_class": {k: int(v) for k, v in sub["cls"].value_counts().items()}}
    out = {c: _block(df[df["compartment"] == c]) for c in (_DENDRITE, _AXON)}

    # cross-tab for the named circuit partners of interest
    interest = ("T4b", "T5b", "LPC1", "LLPC1", "LLPC2", "LLPC3", "LPi14", "LPi12", "LPi02", "LPi10")
    cbc = {}
    for t in interest:
        sub = df[df["cell_type"] == t]
        if len(sub):
            cbc[t] = {c: int((sub["compartment"] == c).sum()) for c in (_DENDRITE, _AXON)}

    total = int(len(df))
    return {
        "available": True,
        "root_id": root, "side": anchor_side, "source": model.source,
        "n_input_syn_localized": total,
        "dendrite": out[_DENDRITE], "axon": out[_AXON],
        "dendrite_frac": round(out[_DENDRITE]["total_syn"] / total, 3) if total else None,
        "class_by_compartment": cbc,
    }
