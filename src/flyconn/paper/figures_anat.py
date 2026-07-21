"""Anatomical connectome figures for the FD3 circuit: REAL neuron shapes + synapse locations.

Unlike ``figures.py`` (matplotlib abstractions of derived scalars), these render actual FlyWire
anatomy in micrometre brain space:

  fd3_arbor_inputs      the FD3 skeleton (grey cable) with its INPUT synapses coloured by
                        partner class (T4b/T5b motion, LPC/LLPC sheets, LPi/contra inhibitors).
                        shows WHERE on the dendrite each input tier lands.
  fd3_circuit_3d        the whole chain as real neurons in one view: exemplar T4b + T5b, FD3
                        (both L/R), and DNp26, colour-coded per class.
  fd3_input_hexmap      a true retinotopic hex scatter of FD3's input columns (RFCloud p,q,w)
                        vs the FD1=Nod1 anchor, showing the lateral offset + frontal gap in eye space.
  fd3_afferent_cascade  the tiered pathway (R1-6 -> lamina -> medulla -> T4b/T5b -> FD3 -> DNp26
                        -> wing), edges styled solid (measured on FD3) / dashed (per-type wiring)
                        / grey (literature) per the honesty tiers.

Data is real and already available: skeletons via ``skeleton_fetch.fetch_skeleton`` (cache-first,
works offline once prefetched with ``scripts/fetch_fd3_skeletons.py``), synapse coordinates via
``geometry.syn_positions_um``, retinotopy via ``geometry_hex.input_cell_pq``. When a partner
skeleton is absent, the renderer falls back to that partner's synapse cloud (still real anatomy)
so a figure is always produced. All renders are matplotlib (Agg); a ``navis`` cable render is an
optional upgrade but not required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from . import geometry as G
from . import geometry_hex as GH
from . import skeleton_fetch as SK

# Cell-class colours (consistent with figures.py conventions).
COL = {
    "fd3": "#333333",          # FD3 cable, neutral dark
    "T4b": "#d62728",          # layer-b motion (red), ON
    "T5b": "#ff7f0e",          # layer-b motion (orange), OFF
    "motion": "#d62728",
    "sheet": "#1f77b4",        # LPC/LLPC columnar sheets (blue)
    "inhibitor": "#9467bd",    # LPi/contra inhibitors (purple)
    "dnp26": "#2ca02c",        # descending output (green)
    "fd1": "#7f7f7f",          # FD1=Nod1 anchor (grey)
}

_MOTION_TYPES = ("T4b", "T5b")
_SHEET_PREFIXES = ("LPC", "LLPC", "Tlp", "MeLp", "LPTe")


# High-resolution defaults so the fine neuron cables read clearly in print.
_DPI = 320


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out


def _fd3_roots(meta) -> list[int]:
    return sorted(int(x) for x in meta.root_ids_of_type(["LPT42_Nod4"]))


def _skeleton_lines(ax, skel, color, lw=0.6, alpha=0.75):
    """Draw a Skeleton's edges as a 3D line collection (real neuron cable)."""
    if skel is None or not skel.ok or len(skel.edges) == 0:
        return False
    v = skel.vertices_um
    segs = v[skel.edges]  # (M, 2, 3)
    lc = Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(lc)
    return True


def _cloud(ax, pts, color, s=2, alpha=0.5, label=None):
    if pts is None or len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=color, alpha=alpha,
               edgecolors="none", label=label, depthshade=False)


def _autoscale_3d(ax, all_pts, *, cubic=True, pad=0.05):
    """Scale the 3D box to the plotted anatomy with a small padding.

    ``cubic`` keeps equal aspect (true proportions); set False to fill the frame per axis when a
    neuron is very anisotropic. ``pad`` is the fraction of the span added as margin so the cable
    is not clipped at the box edge.
    """
    if not len(all_pts):
        return
    lo, hi = all_pts.min(0), all_pts.max(0)
    ctr = (lo + hi) / 2
    span = hi - lo
    if cubic:
        r = float(span.max()) / 2 or 1.0
        r *= (1 + pad)
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
    else:
        m = span * (pad + 0.02) + 1.0
        ax.set_xlim(lo[0] - m[0], hi[0] + m[0])
        ax.set_ylim(lo[1] - m[1], hi[1] + m[1])
        ax.set_zlim(lo[2] - m[2], hi[2] + m[2])
    try:
        ax.set_box_aspect(span if not cubic else (1, 1, 1))  # fill the panel
    except Exception:  # noqa: BLE001 - older mpl without set_box_aspect on 3d
        pass
    ax.set_xlabel("medio-lateral (um)", fontsize=8, labelpad=2)
    ax.set_ylabel("dorso-ventral (um)", fontsize=8, labelpad=2)
    ax.set_zlabel("antero-posterior (um)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6, pad=0)


# ---------------------------------------------------------------------------
# 1. FD3 dendrite with input synapses coloured by partner tier.
# ---------------------------------------------------------------------------
def fd3_arbor_inputs(src, meta, out: Path) -> Path:
    """FD3 skeleton (if cached) + its input synapse cloud coloured by partner class.

    One large panel per FD3 cell, each scaled to that cell's own arbor (not the pooled L+R
    extent, which shrank each neuron into a corner), at high DPI so the cable and the coloured
    input synapses are clearly resolved.
    """
    roots = _fd3_roots(meta)
    n = max(len(roots), 1)
    # Stack the per-cell panels VERTICALLY (portrait) so each neuron gets a big square panel and
    # the figure fills a page top-to-bottom rather than being a short wide strip.
    fig = plt.figure(figsize=(12, 9 * n))
    for i, r in enumerate(roots):
        ax = fig.add_subplot(n, 1, i + 1, projection="3d")
        side = str(meta.by_root.loc[r, "side"])
        panel_pts = []
        # FD3 cable (real skeleton) or, as fallback, its own input-synapse cloud as a proxy arbor.
        skel = SK.fetch_skeleton(src, r)
        drew_cable = _skeleton_lines(ax, skel, COL["fd3"], lw=0.7, alpha=0.55)
        if drew_cable:
            panel_pts.append(skel.vertices_um)

        din = src.synapses(post_ids=[r]); din = din[din["post_pt_root_id"] == r]
        if len(din) == 0:
            continue
        pre_meta = meta.by_root.reindex(din["pre_pt_root_id"].values)
        # Plain python-str arrays (NA -> "") so np boolean ops don't choke on pandas <NA>.
        ct = pre_meta["cell_type"].astype(object).where(pre_meta["cell_type"].notna(), "").to_numpy()
        nt = pre_meta["nt_canonical"].astype(object).where(pre_meta["nt_canonical"].notna(), "").to_numpy()
        pside = pre_meta["side"].astype(object).where(pre_meta["side"].notna(), "").to_numpy()
        pos = G.syn_positions_um(din, "post")  # where inputs land ON fd3
        panel_pts.append(pos)
        if not drew_cable:  # proxy arbor = the input cloud itself
            _cloud(ax, pos, COL["fd3"], s=3, alpha=0.15)

        is_motion = np.array([c in _MOTION_TYPES for c in ct])
        is_sheet = np.array([isinstance(c, str) and c.startswith(_SHEET_PREFIXES) for c in ct])
        is_inhib = (nt == "gaba") & (pside != "") & (pside != side)
        other = ~(is_motion | is_sheet | is_inhib)

        # Larger, opaque markers so the input classes stand out against the cable.
        _cloud(ax, pos[other], "#cfcfcf", s=8, alpha=0.35, label="other")
        _cloud(ax, pos[is_sheet], COL["sheet"], s=20, alpha=0.85, label="LPC/LLPC sheets")
        _cloud(ax, pos[is_inhib], COL["inhibitor"], s=26, alpha=0.95, label="contra inhibitors")
        _cloud(ax, pos[is_motion], COL["motion"], s=28, alpha=0.95, label="T4b/T5b motion")
        ax.set_title(f"FD3 {side} cell: {'reconstructed skeleton' if drew_cable else 'synapse-cloud arbor'}",
                     fontsize=13, pad=8)
        _autoscale_3d(ax, np.vstack(panel_pts), cubic=True, pad=0.02)
        ax.view_init(elev=18, azim=-60)
        if i == 0:
            ax.legend(fontsize=11, loc="upper left", markerscale=1.6, framealpha=0.9)
    fig.suptitle("Where FD3's inputs land on its arbor (real synapse positions, µm)",
                 fontsize=16, y=0.98)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 2. The whole chain as real neurons in one anatomical view.
# ---------------------------------------------------------------------------
def _exemplar_root(src, meta, cell_type: str, near_roots: list[int] | None = None) -> int | None:
    """Pick one exemplar cell of a type: the one with the most synapses onto ``near_roots``
    (so the drawn T4b/T5b actually contacts FD3), else the first."""
    roots = sorted(int(x) for x in meta.root_ids_of_type([cell_type]))
    if not roots:
        return None
    if near_roots:
        near = set(int(x) for x in near_roots)
        syn = src.synapses(post_ids=list(near))
        syn = syn[syn["post_pt_root_id"].isin(near)]
        counts = syn.groupby("pre_pt_root_id").size()
        cand = [r for r in roots if r in counts.index]
        if cand:
            return int(max(cand, key=lambda r: counts.get(r, 0)))
    return roots[0]


def fd3_circuit_3d(src, meta, out: Path) -> Path:
    """One large 3D view: exemplar T4b + T5b -> FD3 (both) -> DNp26, real skeletons where available.

    Rendered big and at high DPI with thick cables so the individual neurons and how they connect
    are clearly legible; the box is scaled to the neurons with only a little padding.
    """
    fd3 = _fd3_roots(meta)
    picks = [
        ("T4b (motion, ON)", _exemplar_root(src, meta, "T4b", fd3), COL["T4b"]),
        ("T5b (motion, OFF)", _exemplar_root(src, meta, "T5b", fd3), COL["T5b"]),
        ("LPC1 (layer-b sheet)", _exemplar_root(src, meta, "LPC1", fd3), COL["sheet"]),
        ("LPi14 (wide-field opponent gate)", _exemplar_root(src, meta, "LPi14", fd3), COL["inhibitor"]),
        ("DNp26 (descending)", _exemplar_root(src, meta, "DNp26"), COL["dnp26"]),
    ]
    fig = plt.figure(figsize=(15, 13))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    legend_handles = []

    def _add(root, color, label, lw=1.1):
        skel = SK.fetch_skeleton(src, root) if root is not None else None
        drew = _skeleton_lines(ax, skel, color, lw=lw, alpha=0.85)
        if drew:
            all_pts.append(skel.vertices_um)
        elif root is not None:
            # fallback: the cell's synapse cloud (inputs+outputs) as its footprint
            si = src.synapses(pre_ids=[root]); si = si[si["pre_pt_root_id"] == root]
            if len(si):
                p = G.syn_positions_um(si, "pre"); _cloud(ax, p, color, s=6, alpha=0.5); all_pts.append(p)
        if label:
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=3.5, label=label))

    for label, root, color in picks:
        _add(root, color, label)
    # FD3 pair drawn a touch thicker (the cell of interest); label once.
    for j, r in enumerate(fd3):
        side = str(meta.by_root.loc[r, "side"])
        _add(r, COL["fd3"], "FD3 (LPT42_Nod4)" if j == 0 else None, lw=1.4)

    if all_pts:
        _autoscale_3d(ax, np.vstack(all_pts), cubic=True, pad=0.02)
    ax.view_init(elev=20, azim=-70)
    ax.legend(handles=legend_handles, fontsize=13, loc="upper left", framealpha=0.9)
    ax.set_title("FD3 circuit in anatomical space: T4b/T5b (motion) -> FD3 -> DNp26 (descending)",
                 fontsize=16, pad=12)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 3. Retinotopic hex map of FD3's input columns vs the FD1 anchor.
# ---------------------------------------------------------------------------
def fd3_input_hexmap(src, meta, out: Path) -> Path:
    """Real retinotopic (p,q) scatter of the T4/T5 columns feeding FD3 vs FD1=Nod1."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fd3 = _fd3_roots(meta)
    nod1 = sorted(int(x) for x in meta.root_ids_of_type(["Nod1"]))
    panels = [("FD3 = LPT42_Nod4", fd3, COL["motion"]), ("FD1 = Nod1 (anchor)", nod1, COL["fd1"])]
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
    fig.suptitle("FD3's retinotopic input columns are more lateral than FD1's "
                 "(real T4/T5 (p,q) positions)", fontsize=11)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 4. Tiered afferent cascade diagram (data-driven, honesty-tiered).
# ---------------------------------------------------------------------------
def fd3_afferent_cascade(p_derived: dict, out: Path) -> Path:
    """Schematic-but-data-driven pathway: R1-6 -> lamina -> medulla -> T4b/T5b -> FD3 -> DNp26.

    Edge STYLE encodes evidence tier: solid = measured ON FD3 (T4/T5->FD3), dashed = per-type
    canonical wiring (the upstream cascade, measured per type), grey = literature (photoreceptor
    front end). Reads the Family-P derived dict for the measured synapse counts.
    """
    census = (p_derived or {}).get("census", {})
    cascade = (p_derived or {}).get("upstream_cascade", {})
    oo = census.get("on_off_split", {})

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    # node columns (x) and rows (y)
    nodes = {
        "R1-6": (0, 2.5), "L1/L2/L3": (1, 2.5),
        "Mi/Tm3\n(ON)": (2, 3.5), "Tm1/2/4/9\n(OFF)": (2, 1.5),
        "T4b\n(ON, layer-b)": (3, 3.5), "T5b\n(OFF, layer-b)": (3, 1.5),
        "FD3\n(LPT42_Nod4)": (4.2, 2.5), "DNp26\n(desc.)": (5.4, 2.5), "wing": (6.4, 2.5),
    }
    ncol = {"R1-6": COL["fd1"], "L1/L2/L3": COL["fd1"],
            "Mi/Tm3\n(ON)": COL["T4b"], "Tm1/2/4/9\n(OFF)": COL["T5b"],
            "T4b\n(ON, layer-b)": COL["T4b"], "T5b\n(OFF, layer-b)": COL["T5b"],
            "FD3\n(LPT42_Nod4)": COL["fd3"], "DNp26\n(desc.)": COL["dnp26"], "wing": "#8c564b"}
    for name, (x, y) in nodes.items():
        ax.add_patch(plt.Circle((x, y), 0.33, color=ncol[name], alpha=0.85, zorder=3))
        ax.text(x, y - 0.55, name, ha="center", va="top", fontsize=8)

    def edge(a, b, style, color, label=None):
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6, ls=style,
                                    shrinkA=14, shrinkB=14), zorder=2)
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, ha="center", fontsize=7, color=color)

    # Tier 3 (literature / grey): photoreceptor -> lamina
    edge("R1-6", "L1/L2/L3", (0, (3, 3)), "#999999", "histaminergic (lit.)")
    # Tier 2 (dashed / per-type canonical wiring): lamina -> medulla -> T4/T5
    edge("L1/L2/L3", "Mi/Tm3\n(ON)", "--", COL["T4b"])
    edge("L1/L2/L3", "Tm1/2/4/9\n(OFF)", "--", COL["T5b"])
    on_frac = cascade.get("on_limb_t4b", {}).get("expected_frac_of_input")
    off_frac = cascade.get("off_limb_t5b", {}).get("expected_frac_of_input")
    edge("Mi/Tm3\n(ON)", "T4b\n(ON, layer-b)", "--", COL["T4b"],
         f"{on_frac}% of T4b input" if on_frac is not None else None)
    edge("Tm1/2/4/9\n(OFF)", "T5b\n(OFF, layer-b)", "--", COL["T5b"],
         f"{off_frac}% of T5b input" if off_frac is not None else None)
    # Tier 1 (solid / measured on FD3): T4b/T5b -> FD3
    edge("T4b\n(ON, layer-b)", "FD3\n(LPT42_Nod4)", "-", COL["T4b"],
         f"{oo.get('T4b_ON')} syn" if oo.get("T4b_ON") is not None else None)
    edge("T5b\n(OFF, layer-b)", "FD3\n(LPT42_Nod4)", "-", COL["T5b"],
         f"{oo.get('T5b_OFF')} syn" if oo.get("T5b_OFF") is not None else None)
    # Output arm (established, Family L)
    edge("FD3\n(LPT42_Nod4)", "DNp26\n(desc.)", "-", COL["dnp26"])
    edge("DNp26\n(desc.)", "wing", "-", "#8c564b")

    ax.set_xlim(-0.6, 7.1); ax.set_ylim(0.4, 4.6)
    # legend for the tiers
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#333", lw=1.8, ls="-", label="Tier 1: measured on FD3"),
        Line2D([0], [0], color="#333", lw=1.8, ls="--", label="Tier 2: per-type wiring"),
        Line2D([0], [0], color="#999", lw=1.8, ls=(0, (3, 3)), label="Tier 3: literature"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower center", ncol=3, frameon=False)
    lb = census.get("layer_b_frac_of_t4t5"); t45f = census.get("t4t5_frac_of_total")
    ax.set_title(f"FD3 afferent cascade: photoreceptor -> layer-b T4b/T5b -> FD3 -> descending route\n"
                 f"(motion drive {lb}% layer-b, but only {t45f}% of FD3's total input)", fontsize=11)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 5. The ISOLATED functional figure-ground circuit, NAMED (data-driven schematic).
# ---------------------------------------------------------------------------
def fd3_functional_circuit_schematic(q_derived: dict, r_derived: dict, p_derived: dict,
                                     out: Path) -> Path:
    """The named functional circuit with measured synapse weights, LPi14 drawn as an inhibitory
    (flat-head) opponent gate: T4b/T5b -> LPC1 -> FD3 -> DNp26 -> wing, LPi14 gating sheet+FD3+
    detectors. This is THE figure that isolates the circuit the mentor asked to name."""
    census = (p_derived or {}).get("census", {})
    oo = census.get("on_off_split", {})
    named = (q_derived or {}).get("named_sheet") or "LPC1"
    sheet_prof = ((q_derived or {}).get("profiles", {}) or {}).get(named, {})
    win = (r_derived or {}).get("winner", {}) or {}
    inh = win.get("cell_type") or "LPi14"
    sdw = (r_derived or {}).get("same_direction_winner") or {}
    sd_inh = sdw.get("cell_type")   # LPi12 (same-direction detector gate); None if not resolved

    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.axis("off")
    nodes = {
        "T4b\n(ON, layer-b)": (0.5, 4.2), "T5b\n(OFF, layer-b)": (0.5, 2.8),
        f"{named}\n(layer-b sheet)": (2.2, 3.5), "FD3\n(LPT42_Nod4)": (4.0, 3.5),
        "DNp26\n(steering)": (5.6, 3.5), "wing": (6.8, 3.5),
        f"{inh}\n(wide-field\nopponent gate)": (2.6, 1.0),
    }
    ncol = {"T4b\n(ON, layer-b)": COL["T4b"], "T5b\n(OFF, layer-b)": COL["T5b"],
            f"{named}\n(layer-b sheet)": COL["sheet"], "FD3\n(LPT42_Nod4)": COL["fd3"],
            "DNp26\n(steering)": COL["dnp26"], "wing": "#8c564b",
            f"{inh}\n(wide-field\nopponent gate)": COL["inhibitor"]}
    sd_node = None
    if sd_inh:
        sd_node = f"{sd_inh}\n(same-direction\ndetector gate)"
        nodes[sd_node] = (0.5, 0.9)          # sits under the detectors it gates
        ncol[sd_node] = "#7b3294"            # distinct purple for the same-direction gate
    for name, (x, y) in nodes.items():
        ax.add_patch(plt.Circle((x, y), 0.34, color=ncol[name], alpha=0.9, zorder=3))
        ax.text(x, y - 0.55, name, ha="center", va="top", fontsize=8)

    def _exc(a, b, label=None):
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=2.0, shrinkA=16, shrinkB=16), zorder=2)
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.14, label, ha="center", fontsize=8, color="#333")

    def _inh_edge(a, b, label=None, color=COL["inhibitor"]):
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),  # flat bar = inhibition
                    arrowprops=dict(arrowstyle="-[", color=color, lw=2.0,
                                    shrinkA=16, shrinkB=16), zorder=2)
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 0.18, label, ha="center", fontsize=8,
                    color=color)

    det_sheet = sheet_prof.get("layer_b_input_syn")
    sheet_fd3 = sheet_prof.get("to_fd3_syn")
    on, off = oo.get("T4b_ON"), oo.get("T5b_OFF")
    _exc("T4b\n(ON, layer-b)", f"{named}\n(layer-b sheet)", f"{det_sheet} syn" if det_sheet else None)
    _exc("T5b\n(OFF, layer-b)", f"{named}\n(layer-b sheet)")
    _exc(f"{named}\n(layer-b sheet)", "FD3\n(LPT42_Nod4)", f"{sheet_fd3} syn" if sheet_fd3 else None)
    # direct motion drive onto FD3 (curved, lighter)
    _exc("T5b\n(OFF, layer-b)", "FD3\n(LPT42_Nod4)",
         f"direct {(on or 0)+(off or 0)} syn" if oo else None)
    _exc("FD3\n(LPT42_Nod4)", "DNp26\n(steering)")
    _exc("DNp26\n(steering)", "wing")
    # LPi14 inhibitory (opponent) edges
    _inh_edge(f"{inh}\n(wide-field\nopponent gate)", "FD3\n(LPT42_Nod4)",
              f"{win.get('to_fd3_syn')} syn" if win.get("to_fd3_syn") else None)
    _inh_edge(f"{inh}\n(wide-field\nopponent gate)", f"{named}\n(layer-b sheet)",
              f"gate {win.get('to_sheet_syn')} syn" if win.get("to_sheet_syn") else None)
    _inh_edge(f"{inh}\n(wide-field\nopponent gate)", "T5b\n(OFF, layer-b)",
              f"feedback {win.get('to_detectors_syn')} syn" if win.get("to_detectors_syn") else None)
    # LPi12 same-direction gate: dominates detector-level inhibition (distinct color)
    sd_col = "#7b3294"
    if sd_node:
        _inh_edge(sd_node, "T5b\n(OFF, layer-b)",
                  f"{sdw.get('to_detectors_syn')} syn" if sdw.get("to_detectors_syn") else None,
                  color=sd_col)
        _inh_edge(sd_node, "T4b\n(ON, layer-b)", None, color=sd_col)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#333", lw=2, marker=">", label="excitatory (measured syn)"),
               Line2D([0], [0], color=COL["inhibitor"], lw=2,
                      label=f"opponent gate ({inh}, on FD3/sheet)")]
    if sd_node:
        handles.append(Line2D([0], [0], color=sd_col, lw=2,
                              label=f"same-direction gate ({sd_inh}, on detectors)"))
    ax.legend(handles=handles, fontsize=9, loc="lower right", frameon=False)
    ax.set_xlim(-0.2, 7.4); ax.set_ylim(0.0, 5.0)
    la = win.get("layer_a_pct")
    subtitle = (f"with {inh} the wide-field opponent gate (layer-a {la}%, VCH functional-role homolog)"
                + (f" and {sd_inh} the same-direction surround on the detectors"
                   f" (layer-b {sdw.get('layer_b_pct')}%)" if sd_node else ""))
    ax.set_title("The FD3 figure-ground circuit, named: "
                 f"T4b/T5b -> {named} -> FD3 -> DNp26 -> wing\n" + subtitle,
                 fontsize=11)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Render-all convenience (used by the report builder + tests).
# ---------------------------------------------------------------------------
def render_all(src, meta, p_derived: dict, figdir: Path,
               q_derived: dict | None = None, r_derived: dict | None = None) -> dict:
    """Render every anatomical figure into ``figdir``; return {name: path}. Each figure is
    guarded so one failure (e.g. a missing skeleton) does not abort the rest."""
    figdir = Path(figdir)
    out = {}
    jobs = [
        ("fd3_arbor_inputs", lambda o: fd3_arbor_inputs(src, meta, o)),
        ("fd3_circuit_3d", lambda o: fd3_circuit_3d(src, meta, o)),
        ("fd3_input_hexmap", lambda o: fd3_input_hexmap(src, meta, o)),
        ("fd3_afferent_cascade", lambda o: fd3_afferent_cascade(p_derived, o)),
        ("fd3_functional_circuit_schematic",
         lambda o: fd3_functional_circuit_schematic(q_derived or {}, r_derived or {}, p_derived, o)),
    ]
    for name, fn in jobs:
        try:
            out[name] = str(fn(figdir / f"{name}.png"))
        except Exception as e:  # noqa: BLE001 - never let one figure abort the report
            print(f"[figures_anat] {name} failed: {type(e).__name__}: {e}")
    return out
