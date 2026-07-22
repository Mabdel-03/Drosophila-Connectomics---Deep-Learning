"""Anatomical connectome figures for the FD4 search: REAL neuron shapes + synapse positions.

Renders actual FlyWire anatomy for the FD4 report, which documents a negative result:

  fd4_nod1_quartet      the 4 Nod1 cells rendered together (real skeletons): all four share the
                        same frontal, full-dorso-ventral, single-arbor morphology, the visual proof
                        that Nod1 is one homogeneous population (= FD1), not FD1 + FD4.
  fd4_input_hexmap      the retinotopic (p,q) columns feeding each Nod1 cell: all frontal, none
                        carrying the whole-eye lateral field FD4 would have. Contrasted with the
                        FD3 = LPT42_Nod4 lateral field.
  fd4_arbor_inputs      the Nod1/FD1 skeleton with input synapses coloured by class (the progressive
                        arm any FD4 correlate would use: T4a/T5a motion, LLPC1 sheet, VCH gate).
  fd4_circuit_3d        the progressive arm as real neurons: T4a + T5a + LLPC1 + Nod1 + DNp26.

Reuses the FD3/FD2 anatomical helpers. Cache-first skeletons via ``skeleton_fetch``; falls back to
synapse clouds when a skeleton is absent, so a figure is always produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from . import geometry as G
from . import geometry_hex as GH
from . import skeleton_fetch as SK
from .figures_anat import (COL, _save, _skeleton_lines, _cloud, _autoscale_3d,
                           _exemplar_root, _MOTION_TYPES, _SHEET_PREFIXES)

PROG = "Nod1"          # = FD1; the progressive figure-output population
FD3 = "LPT42_Nod4"     # = FD3; the regressive, lateral-RF cell (contrast)
_SIDE_COL = {"left": "#1f77b4", "right": "#C0392B"}


def _prog_roots(meta) -> list[int]:
    return sorted(int(x) for x in meta.root_ids_of_type([PROG]))


# ---------------------------------------------------------------------------
# 1. The 4 Nod1 cells side by side: all the same shape (the homogeneity proof).
# ---------------------------------------------------------------------------
def fd4_nod1_quartet(src, meta, out: Path) -> Path:
    """Render all 4 Nod1 cells as real skeletons, one panel each, to show they share one frontal,
    full-dorso-ventral, single-arbor morphology. If the four cells were FD1 + FD4, one pair would
    have a restricted dorso-ventral dendrite and a distinct field; they do not."""
    roots = _prog_roots(meta)
    n = max(len(roots), 1)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(13, 6 * nrow))
    for i, r in enumerate(roots):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        side = str(meta.by_root.loc[r, "side"])
        pts = []
        skel = SK.fetch_skeleton(src, r)
        drew = _skeleton_lines(ax, skel, _SIDE_COL.get(side, "#333"), lw=0.8, alpha=0.75)
        if drew:
            pts.append(skel.vertices_um)
        din = src.synapses(post_ids=[r]); din = din[din["post_pt_root_id"] == r]
        if len(din):
            pos = G.syn_positions_um(din, "post"); pts.append(pos)
            if not drew:
                _cloud(ax, pos, _SIDE_COL.get(side, "#333"), s=3, alpha=0.2)
            dv = float(np.percentile(pos[:, 1], 95) - np.percentile(pos[:, 1], 5))
            ax.set_title(f"Nod1 {side} cell {r}\nD-V dendrite span {dv:.0f} um "
                         f"({'skeleton' if drew else 'synapse cloud'})", fontsize=10)
        if pts:
            _autoscale_3d(ax, np.vstack(pts), cubic=True, pad=0.02)
        ax.view_init(elev=18, azim=-60)
    fig.suptitle("The four Nod1 cells share one frontal, full-dorso-ventral, single-arbor "
                 "morphology: one homogeneous population (FD1), not a separable FD1 + FD4 split",
                 fontsize=13, y=0.99)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 2. Retinotopic hexmap: all Nod1 cells frontal; FD3 lateral (the contrast).
# ---------------------------------------------------------------------------
def fd4_input_hexmap(src, meta, out: Path) -> Path:
    """The T4/T5 input columns of each Nod1 cell (all frontal) and, for contrast, FD3 = LPT42_Nod4
    (lateral). None of the Nod1 cells carries the whole-eye lateral field FD4 would have."""
    roots = _prog_roots(meta)
    fd3 = sorted(int(x) for x in meta.root_ids_of_type([FD3]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # left panel: each Nod1 cell's cloud, coloured by side
    ax = axes[0]
    for r in roots:
        side = str(meta.by_root.loc[r, "side"])
        cloud = GH.input_cell_pq(src, meta, r)
        if not cloud.ok:
            continue
        ax.scatter(cloud.p, cloud.q, s=6, color=_SIDE_COL.get(side, "#333"), alpha=0.35)
        cp = float(np.average(cloud.p, weights=cloud.w)); cq = float(np.average(cloud.q, weights=cloud.w))
        ax.scatter([cp], [cq], marker="+", s=220, color=_SIDE_COL.get(side, "#333"), linewidths=2.5)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.axvspan(5, ax.get_xlim()[1], color="#C97A0A", alpha=0.08)
    ax.set_xlabel("hex p (azimuth; frontal < 0 < lateral)")
    ax.set_ylabel("hex q (elevation)")
    ax.set_title("All 4 Nod1 cells: frontal input columns (no lateral FD4 field)", fontsize=10)

    # right panel: FD3 (lateral) for contrast
    ax = axes[1]
    allp, allq, allw = [], [], []
    for r in fd3:
        cloud = GH.input_cell_pq(src, meta, r)
        if cloud.ok:
            allp.append(cloud.p); allq.append(cloud.q); allw.append(cloud.w)
    if allp:
        p = np.concatenate(allp); q = np.concatenate(allq); w = np.concatenate(allw)
        ax.scatter(p, q, s=6, color="#C0392B", alpha=0.4)
        cp = float(np.average(p, weights=w))
        ax.scatter([cp], [float(np.average(q, weights=w))], marker="+", s=220, color="#C0392B",
                   linewidths=2.5, label=f"FD3 centroid p={cp:.1f}")
        ax.legend(fontsize=8)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("hex p (azimuth; frontal < 0 < lateral)")
    ax.set_title("FD3 = LPT42_Nod4 for contrast: a lateral field", fontsize=10)
    fig.suptitle("The Nod1/FD1 cells are frontal; no Nod1 cell has FD4's whole-eye lateral field "
                 "(real T4/T5 (p,q) positions)", fontsize=11)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 3. Nod1/FD1 arbor + inputs (the progressive arm any FD4 correlate would use).
# ---------------------------------------------------------------------------
def fd4_arbor_inputs(src, meta, out: Path) -> Path:
    roots = _prog_roots(meta)[:2]   # one exemplar per side
    n = max(len(roots), 1)
    fig = plt.figure(figsize=(12, 9 * n))
    for i, r in enumerate(roots):
        ax = fig.add_subplot(n, 1, i + 1, projection="3d")
        side = str(meta.by_root.loc[r, "side"])
        pts = []
        skel = SK.fetch_skeleton(src, r)
        drew = _skeleton_lines(ax, skel, "#333333", lw=0.7, alpha=0.55)
        if drew:
            pts.append(skel.vertices_um)
        din = src.synapses(post_ids=[r]); din = din[din["post_pt_root_id"] == r]
        if len(din) == 0:
            continue
        pm = meta.by_root.reindex(din["pre_pt_root_id"].values)
        ct = pm["cell_type"].astype(object).where(pm["cell_type"].notna(), "").to_numpy()
        nt = pm["nt_canonical"].astype(object).where(pm["nt_canonical"].notna(), "").to_numpy()
        pos = G.syn_positions_um(din, "post"); pts.append(pos)
        if not drew:
            _cloud(ax, pos, "#333333", s=3, alpha=0.15)
        is_motion = np.array([c in _MOTION_TYPES for c in ct])
        is_sheet = np.array([isinstance(c, str) and c.startswith(_SHEET_PREFIXES) for c in ct])
        is_inhib = (nt == "gaba")
        other = ~(is_motion | is_sheet | is_inhib)
        _cloud(ax, pos[other], "#cfcfcf", s=8, alpha=0.35, label="other")
        _cloud(ax, pos[is_sheet], COL["sheet"], s=20, alpha=0.85, label="LPC/LLPC sheets")
        _cloud(ax, pos[is_inhib], COL["inhibitor"], s=26, alpha=0.95, label="GABA inhibitors")
        _cloud(ax, pos[is_motion], COL["motion"], s=28, alpha=0.95, label="T4/T5 motion")
        ax.set_title(f"Nod1 (= FD1) {side} cell: {'skeleton' if drew else 'synapse-cloud arbor'}",
                     fontsize=13, pad=8)
        _autoscale_3d(ax, np.vstack(pts), cubic=True, pad=0.02)
        ax.view_init(elev=18, azim=-60)
        if i == 0:
            ax.legend(fontsize=11, loc="upper left", markerscale=1.6, framealpha=0.9)
    fig.suptitle("Where inputs land on the progressive figure cell Nod1 (= FD1), the arm any FD4 "
                 "correlate would use (real synapse positions, um)", fontsize=15, y=0.99)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 4. The progressive arm as real neurons.
# ---------------------------------------------------------------------------
def fd4_circuit_3d(src, meta, out: Path, top_dn: str = "DNp26", sheet: str = "LLPC1") -> Path:
    prog = _prog_roots(meta)
    picks = [
        ("T4a (motion, ON)", _exemplar_root(src, meta, "T4a", prog), COL.get("T4b", "#2ca02c")),
        ("T5a (motion, OFF)", _exemplar_root(src, meta, "T5a", prog), COL.get("T5b", "#17becf")),
        (f"{sheet} (layer-a sheet)", _exemplar_root(src, meta, sheet, prog), COL["sheet"]),
        (f"{top_dn} (descending)", _exemplar_root(src, meta, top_dn), COL.get("dnp26", "#9467bd")),
    ]
    fig = plt.figure(figsize=(15, 13))
    ax = fig.add_subplot(111, projection="3d")
    all_pts, legend_handles = [], []

    def _add(root, color, label, lw=1.1):
        skel = SK.fetch_skeleton(src, root) if root is not None else None
        drew = _skeleton_lines(ax, skel, color, lw=lw, alpha=0.85)
        if drew:
            all_pts.append(skel.vertices_um)
        elif root is not None:
            si = src.synapses(pre_ids=[root]); si = si[si["pre_pt_root_id"] == root]
            if len(si):
                p = G.syn_positions_um(si, "pre"); _cloud(ax, p, color, s=6, alpha=0.5); all_pts.append(p)
        if label:
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=3.5, label=label))

    for label, root, color in picks:
        _add(root, color, label)
    for j, r in enumerate(prog[:2]):
        _add(r, "#1f77b4", "Nod1 (= FD1)" if j == 0 else None, lw=1.4)
    if all_pts:
        _autoscale_3d(ax, np.vstack(all_pts), cubic=True, pad=0.02)
    ax.view_init(elev=20, azim=-70)
    ax.legend(handles=legend_handles, fontsize=13, loc="upper left", framealpha=0.9)
    ax.set_title(f"The progressive figure arm in anatomical space: T4a/T5a -> {sheet} -> Nod1 -> {top_dn}",
                 fontsize=15, pad=12)
    return _save(fig, out)


def render_all(src, meta, fig_dir: Path) -> dict:
    fig_dir = Path(fig_dir)
    made = {}
    for name, fn in [("fd4_nod1_quartet", fd4_nod1_quartet),
                     ("fd4_input_hexmap", fd4_input_hexmap),
                     ("fd4_arbor_inputs", fd4_arbor_inputs),
                     ("fd4_circuit_3d", fd4_circuit_3d)]:
        try:
            made[name] = str(fn(src, meta, fig_dir / f"{name}.png"))
        except Exception as e:  # noqa: BLE001
            made[name] = f"error: {type(e).__name__}: {e}"
    return made
