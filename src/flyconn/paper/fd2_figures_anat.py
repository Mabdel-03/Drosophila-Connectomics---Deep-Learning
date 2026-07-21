"""Anatomical connectome figures for the FD2 = LPT21 circuit: REAL neuron shapes + synapse positions.

Renders actual FlyWire anatomy in micrometre brain space for the FD2 report:

  fd2_arbor_inputs      the LPT21 skeleton (cable) with its INPUT synapses coloured by partner class.
  fd2_circuit_3d        the chain as real neurons: exemplar T4b + T5b + the sheet, both LPT21 cells,
                        and the top descending neuron.
  fd2_input_hexmap      the retinotopic (p,q) columns feeding LPT21 vs the FD1=Nod1 anchor (FD2's
                        field sits frontally, at the FD1 position).
  fd2_dual_output       each LPT21 cell's OUTPUT synapses, coloured by the two terminal fields, in
                        anatomical space: the connectome correlate of FD2's main + frontal branches.
  fd2_afferent_cascade  the tiered pathway R1-6 -> lamina -> medulla -> T4b/T5b -> LPT21 -> DN -> wing.
  fd2_functional_circuit the named circuit schematic (detectors -> sheet -> FD2 -> DN, with the gate).

Reuses the FD3 anatomical helpers (skeleton line drawing, synapse clouds, 3D autoscale, exemplar
picking). Cache-first skeletons via ``skeleton_fetch``; falls back to synapse clouds when a skeleton
is absent, so a figure is always produced.
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

FD2 = "LPT21"
ANCHOR = "Nod1"


def _fd2_roots(meta) -> list[int]:
    return sorted(int(x) for x in meta.root_ids_of_type([FD2]))


# ---------------------------------------------------------------------------
# 1. LPT21 arbor + input synapses by class.
# ---------------------------------------------------------------------------
def fd2_arbor_inputs(src, meta, out: Path) -> Path:
    roots = _fd2_roots(meta)
    n = max(len(roots), 1)
    fig = plt.figure(figsize=(12, 9 * n))
    for i, r in enumerate(roots):
        ax = fig.add_subplot(n, 1, i + 1, projection="3d")
        side = str(meta.by_root.loc[r, "side"])
        panel_pts = []
        skel = SK.fetch_skeleton(src, r)
        drew = _skeleton_lines(ax, skel, "#333333", lw=0.7, alpha=0.55)
        if drew:
            panel_pts.append(skel.vertices_um)
        din = src.synapses(post_ids=[r]); din = din[din["post_pt_root_id"] == r]
        if len(din) == 0:
            continue
        pm = meta.by_root.reindex(din["pre_pt_root_id"].values)
        ct = pm["cell_type"].astype(object).where(pm["cell_type"].notna(), "").to_numpy()
        nt = pm["nt_canonical"].astype(object).where(pm["nt_canonical"].notna(), "").to_numpy()
        pos = G.syn_positions_um(din, "post")
        panel_pts.append(pos)
        if not drew:
            _cloud(ax, pos, "#333333", s=3, alpha=0.15)
        is_motion = np.array([c in _MOTION_TYPES for c in ct])
        is_sheet = np.array([isinstance(c, str) and c.startswith(_SHEET_PREFIXES) for c in ct])
        is_inhib = (nt == "gaba")
        other = ~(is_motion | is_sheet | is_inhib)
        _cloud(ax, pos[other], "#cfcfcf", s=8, alpha=0.35, label="other")
        _cloud(ax, pos[is_sheet], COL["sheet"], s=20, alpha=0.85, label="LPC/LLPC sheets")
        _cloud(ax, pos[is_inhib], COL["inhibitor"], s=26, alpha=0.95, label="LPi inhibitors")
        _cloud(ax, pos[is_motion], COL["motion"], s=28, alpha=0.95, label="T4b/T5b motion")
        ax.set_title(f"LPT21 {side} cell: {'reconstructed skeleton' if drew else 'synapse-cloud arbor'}",
                     fontsize=13, pad=8)
        _autoscale_3d(ax, np.vstack(panel_pts), cubic=True, pad=0.02)
        ax.view_init(elev=18, azim=-60)
        if i == 0:
            ax.legend(fontsize=11, loc="upper left", markerscale=1.6, framealpha=0.9)
    fig.suptitle("Where FD2's inputs land on its arbor (real synapse positions, um)", fontsize=16, y=0.99)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 2. The chain as real neurons in one view.
# ---------------------------------------------------------------------------
def fd2_circuit_3d(src, meta, out: Path, top_dn: str = "DNp26", sheet: str = "LPC1") -> Path:
    fd2 = _fd2_roots(meta)
    picks = [
        ("T4b (motion, ON)", _exemplar_root(src, meta, "T4b", fd2), COL["T4b"]),
        ("T5b (motion, OFF)", _exemplar_root(src, meta, "T5b", fd2), COL["T5b"]),
        (f"{sheet} (layer-b sheet)", _exemplar_root(src, meta, sheet, fd2), COL["sheet"]),
        (f"{top_dn} (descending)", _exemplar_root(src, meta, top_dn), COL["dnp26"]),
    ]
    fig = plt.figure(figsize=(15, 13))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []; legend_handles = []

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
    for j, r in enumerate(fd2):
        _add(r, "#1B7837", "FD2 (LPT21)" if j == 0 else None, lw=1.4)
    if all_pts:
        _autoscale_3d(ax, np.vstack(all_pts), cubic=True, pad=0.02)
    ax.view_init(elev=20, azim=-70)
    ax.legend(handles=legend_handles, fontsize=13, loc="upper left", framealpha=0.9)
    ax.set_title(f"FD2 circuit in anatomical space: T4b/T5b -> LPT21 -> {top_dn}", fontsize=16, pad=12)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 3. Retinotopic hex map: LPT21 columns vs the FD1 anchor (both frontal).
# ---------------------------------------------------------------------------
def fd2_input_hexmap(src, meta, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fd2 = _fd2_roots(meta)
    nod1 = sorted(int(x) for x in meta.root_ids_of_type([ANCHOR]))
    panels = [("FD2 = LPT21", fd2, "#1B7837"), ("FD1 = Nod1 (anchor)", nod1, COL["fd1"])]
    for ax, (title, roots, color) in zip(axes, panels):
        allp, allq, allw = [], [], []
        for r in roots:
            cloud = GH.input_cell_pq(src, meta, r)
            if cloud.ok:
                allp.append(cloud.p); allq.append(cloud.q); allw.append(cloud.w)
        if allp:
            p = np.concatenate(allp); q = np.concatenate(allq); w = np.concatenate(allw)
            sc = ax.scatter(p, q, s=8 + 40 * (w / w.max()), c=w, cmap="viridis", alpha=0.7)
            fig.colorbar(sc, ax=ax, shrink=0.7, label="synapses (weight)")
            cp = float(np.average(p, weights=w)); cq = float(np.average(q, weights=w))
            ax.scatter([cp], [cq], marker="+", s=200, c=color, linewidths=2.5,
                       label=f"centroid p={cp:.1f}")
            ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("hex p (azimuth proxy; + = lateral)")
        ax.set_ylabel("hex q (elevation proxy)")
        ax.set_title(title, fontsize=10)
    fig.suptitle("FD2's retinotopic input columns sit frontally, at the FD1 position "
                 "(real T4/T5 (p,q) positions)", fontsize=11)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 4. Dual output: the two output terminal fields in anatomical space (unique to FD2).
# ---------------------------------------------------------------------------
def fd2_dual_output_3d(src, meta, out: Path) -> Path:
    """Each LPT21 cell's OUTPUT synapses, coloured by which of the two terminal fields they fall in
    (split at the largest medio-lateral gap). The two separated fields are the connectome correlate of
    Egelhaaf's main (ipsilateral posterior optic foci) and second (frontal, anterior optic foci)
    axonal branches."""
    roots = _fd2_roots(meta)
    n = max(len(roots), 1)
    fig = plt.figure(figsize=(12, 8 * n))
    for i, r in enumerate(roots):
        ax = fig.add_subplot(n, 1, i + 1, projection="3d")
        side = str(meta.by_root.loc[r, "side"])
        skel = SK.fetch_skeleton(src, r)
        drew = _skeleton_lines(ax, skel, "#333333", lw=0.6, alpha=0.4)
        out_syn = src.synapses(pre_ids=[r]); out_syn = out_syn[out_syn["pre_pt_root_id"] == r]
        pos = G.syn_positions_um(out_syn, "pre")
        if len(pos) < 50:
            continue
        x = np.sort(pos[:, 0]); gi = int(np.argmax(np.diff(x))); split = (x[gi] + x[gi + 1]) / 2.0
        lo = pos[pos[:, 0] < split]; hi = pos[pos[:, 0] >= split]
        main, second = (lo, hi) if len(lo) >= len(hi) else (hi, lo)
        _cloud(ax, main, "#1B7837", s=14, alpha=0.7, label=f"main terminal field ({len(main)})")
        _cloud(ax, second, "#C97A0A", s=18, alpha=0.9, label=f"second terminal field ({len(second)})")
        pts = pos if not drew else np.vstack([pos, skel.vertices_um])
        _autoscale_3d(ax, pts, cubic=True, pad=0.03)
        ax.view_init(elev=18, azim=-60)
        ax.set_title(f"LPT21 {side} cell: two separated output terminal fields", fontsize=13, pad=8)
        if i == 0:
            ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    fig.suptitle("FD2's dual axonal output in anatomical space (real output-synapse positions, um)",
                 fontsize=15, y=0.99)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 5. Afferent cascade (data-driven schematic).
# ---------------------------------------------------------------------------
def fd2_afferent_cascade(fd2_derived: dict, out: Path, top_dn: str = "DNp26") -> Path:
    census = (fd2_derived or {}).get("afferent", {}).get("census", {})
    oo = census.get("on_off_split", {})
    fig, ax = plt.subplots(figsize=(12, 6)); ax.axis("off")
    nodes = {
        "R1-6": (0, 2.5), "L1/L2/L3": (1, 2.5),
        "Mi/Tm3\n(ON)": (2, 3.5), "Tm1/2/4/9\n(OFF)": (2, 1.5),
        "T4b\n(ON, layer-b)": (3, 3.5), "T5b\n(OFF, layer-b)": (3, 1.5),
        "FD2\n(LPT21)": (4.2, 2.5), f"{top_dn}\n(desc.)": (5.4, 2.5), "wing": (6.4, 2.5),
    }
    ncol = {"R1-6": COL["fd1"], "L1/L2/L3": COL["fd1"], "Mi/Tm3\n(ON)": COL["T4b"],
            "Tm1/2/4/9\n(OFF)": COL["T5b"], "T4b\n(ON, layer-b)": COL["T4b"],
            "T5b\n(OFF, layer-b)": COL["T5b"], "FD2\n(LPT21)": "#1B7837",
            f"{top_dn}\n(desc.)": COL["dnp26"], "wing": "#555555"}

    def edge(a, b, style, color, label=None):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.annotate("", xy=(x2 - 0.32, y2), xytext=(x1 + 0.32, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, ls=style))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, fontsize=7, ha="center", color=color)

    edge("R1-6", "L1/L2/L3", (0, (2, 2)), "#999999")
    edge("L1/L2/L3", "Mi/Tm3\n(ON)", "dashed", COL["T4b"])
    edge("L1/L2/L3", "Tm1/2/4/9\n(OFF)", "dashed", COL["T5b"])
    edge("Mi/Tm3\n(ON)", "T4b\n(ON, layer-b)", "dashed", COL["T4b"])
    edge("Tm1/2/4/9\n(OFF)", "T5b\n(OFF, layer-b)", "dashed", COL["T5b"])
    edge("T4b\n(ON, layer-b)", "FD2\n(LPT21)", "solid", COL["T4b"], f"{census.get('t4b_syn')} syn")
    edge("T5b\n(OFF, layer-b)", "FD2\n(LPT21)", "solid", COL["T5b"], f"{census.get('t5b_syn')} syn")
    edge("FD2\n(LPT21)", f"{top_dn}\n(desc.)", "solid", "#1B7837")
    edge(f"{top_dn}\n(desc.)", "wing", "solid", COL["dnp26"])
    for name, (x, y) in nodes.items():
        ax.add_patch(plt.Circle((x, y), 0.30, color=ncol[name], alpha=0.22))
        ax.text(x, y, name, fontsize=7.5, ha="center", va="center")
    ax.set_xlim(-0.5, 7); ax.set_ylim(0.5, 4.5)
    ax.set_title("The afferent cascade to FD2 (solid = measured on FD2's detectors; dashed = column wiring)",
                 fontsize=11)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 6. Functional circuit schematic (named end-to-end).
# ---------------------------------------------------------------------------
def fd2_functional_circuit(fd2_derived: dict, out: Path) -> Path:
    sheet = (fd2_derived or {}).get("sheet", {}).get("named_sheet") or "LPC1"
    gate = (fd2_derived or {}).get("gate", {}).get("named_inhibitor") or "LPi14"
    eff = (fd2_derived or {}).get("efferent", {}).get("direct", {})
    top_dn = eff.get("ranking", [{}])[0].get("cell_type") if eff.get("ranking") else "DNp26"
    fig, ax = plt.subplots(figsize=(11, 5)); ax.axis("off")
    nodes = {"T4b/T5b\n(layer-b)": (0, 1.5), f"{sheet}\n(sheet)": (1.5, 1.5),
             "FD2\n(LPT21)": (3, 1.5), f"{top_dn}\n(desc.)": (4.5, 1.5), "wing": (5.7, 1.5),
             f"{gate}\n(wide-field gate)": (2.25, 3.0)}
    ncol = {"T4b/T5b\n(layer-b)": "#d62728", f"{sheet}\n(sheet)": COL["sheet"],
            "FD2\n(LPT21)": "#1B7837", f"{top_dn}\n(desc.)": COL["dnp26"], "wing": "#555555",
            f"{gate}\n(wide-field gate)": COL["inhibitor"]}

    def exc(a, b):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.annotate("", xy=(x2 - 0.42, y2), xytext=(x1 + 0.42, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.8))

    def inh(a, b):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.annotate("", xy=(x2, y2 + 0.35), xytext=(x1, y1 - 0.30),
                    arrowprops=dict(arrowstyle="-[", color=COL["inhibitor"], lw=1.8))

    exc("T4b/T5b\n(layer-b)", f"{sheet}\n(sheet)")
    exc(f"{sheet}\n(sheet)", "FD2\n(LPT21)")
    exc("FD2\n(LPT21)", f"{top_dn}\n(desc.)")
    exc(f"{top_dn}\n(desc.)", "wing")
    inh(f"{gate}\n(wide-field gate)", "FD2\n(LPT21)")
    inh(f"{gate}\n(wide-field gate)", f"{sheet}\n(sheet)")
    for name, (x, y) in nodes.items():
        ax.add_patch(plt.Circle((x, y), 0.42, color=ncol[name], alpha=0.22))
        ax.text(x, y, name, fontsize=8, ha="center", va="center")
    ax.set_xlim(-0.7, 6.4); ax.set_ylim(0.6, 3.7)
    ax.set_title(f"FD2's named figure-ground circuit: T4b/T5b -> {sheet} -> FD2 -> {top_dn} -> wing "
                 f"(gated by {gate})", fontsize=11)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# dispatcher.
# ---------------------------------------------------------------------------
def render_all(src, meta, fd2_derived: dict, figdir: Path) -> None:
    sheet = (fd2_derived or {}).get("sheet", {}).get("named_sheet") or "LPC1"
    eff = (fd2_derived or {}).get("efferent", {}).get("direct", {})
    top_dn = eff.get("ranking", [{}])[0].get("cell_type") if eff.get("ranking") else "DNp26"
    figdir = Path(figdir)
    for name, fn in (
        ("fd2_arbor_inputs.png", lambda o: fd2_arbor_inputs(src, meta, o)),
        ("fd2_circuit_3d.png", lambda o: fd2_circuit_3d(src, meta, o, top_dn=top_dn, sheet=sheet)),
        ("fd2_input_hexmap.png", lambda o: fd2_input_hexmap(src, meta, o)),
        ("fd2_dual_output_3d.png", lambda o: fd2_dual_output_3d(src, meta, o)),
        ("fd2_afferent_cascade.png", lambda o: fd2_afferent_cascade(fd2_derived, o, top_dn=top_dn)),
        ("fd2_functional_circuit.png", lambda o: fd2_functional_circuit(fd2_derived, o)),
    ):
        try:
            fn(figdir / name)
        except Exception as exc:  # noqa: BLE001 - one failure must not abort the rest
            print(f"[fd2-anat] {name} skipped: {type(exc).__name__}: {exc}")
