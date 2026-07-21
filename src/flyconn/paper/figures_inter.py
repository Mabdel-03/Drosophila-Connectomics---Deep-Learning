"""Stage-9 figures: the inter-hemispheric coupling architecture.

Matplotlib Agg (headless). Each function is defensive: missing data -> a placeholder panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def readout_crossing(run: dict, out: Path) -> Path:
    """Nod1 -> contralateral DNp26, both directions, with the asymmetry."""
    n1 = run["channels"].get("N1", {}).get("derived", {})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if not n1:
        ax.text(0.5, 0.5, "no N1 data", ha="center"); return _save(fig, out)
    l2r = n1.get("left_nod1_to_right_dnp26", 0)
    r2l = n1.get("right_nod1_to_left_dnp26", 0)
    ax.bar(["left Nod1\n-> right DNp26", "right Nod1\n-> left DNp26"], [l2r, r2l],
           color=["#2E86C1", "#C0392B"])
    for i, v in enumerate([l2r, r2l]):
        ax.text(i, v + 4, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("synapses")
    ax.set_title(f"Readout crossing: each Nod1 drives the OPPOSITE DNp26\n"
                 f"(Nod1 output {n1.get('nod1_output_crossing_pct')}% cross-midline; "
                 f"asymmetry {n1.get('asymmetry_ratio')}x)", fontsize=10)
    return _save(fig, out)


def heterolateral_bridges(run: dict, out: Path) -> Path:
    """Bridge cells onto the contralateral figure machinery, colored by input direction (layer)."""
    n2 = run["channels"].get("N2", {}).get("derived", {})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bridges = (n2 or {}).get("bridges", [])
    if not bridges:
        ax.text(0.5, 0.5, "no N2 bridge data", ha="center"); return _save(fig, out)
    bridges = sorted(bridges, key=lambda b: b["cross_syn_onto_machinery"], reverse=True)[:12]
    names = [b["bridge_type"] for b in bridges]
    syn = [b["cross_syn_onto_machinery"] for b in bridges]
    dirn = [(b.get("input_layer") or {}).get("dominant_direction") for b in bridges]
    color = {"regressive": "#C0392B", "progressive": "#2E86C1",
             "upward": "#27AE60", "downward": "#8E44AD", None: "#888"}
    cols = [color.get(d, "#888") for d in dirn]
    y = np.arange(len(names))
    ax.barh(y, syn, color=cols)
    for i, (s, d, b) in enumerate(zip(syn, dirn, bridges)):
        ax.text(s, i, f"  {d or '?'} ({b['nt']})", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("crossing synapses onto contralateral figure machinery")
    ax.set_title("Heterolateral input bridges, by input direction\n"
                 "(regressive/red = Egelhaaf's contralateral FD1 inhibition)", fontsize=10)
    for d, c in [("regressive", "#C0392B"), ("progressive", "#2E86C1")]:
        ax.bar(0, 0, color=c, label=d)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out)


def shared_convergence(run: dict, out: Path) -> Path:
    """Roles of the cells receiving from both readouts."""
    n4 = run["channels"].get("N4", {}).get("derived", {})
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    roles = (n4 or {}).get("shared_by_role", {})
    if not roles:
        ax.text(0.5, 0.5, "no N4 data", ha="center"); return _save(fig, out)
    items = sorted(roles.items(), key=lambda kv: kv[1], reverse=True)
    xs = np.arange(len(items))
    ax.bar(xs, [v for _, v in items], color="#34495E")
    for i, (_, v) in enumerate(items):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=8)
    ax.set_ylabel("shared cells")
    ax.set_xticks(xs)
    ax.set_xticklabels([k for k, _ in items], rotation=30, ha="right", fontsize=8)
    ax.set_title(f"Shared downstream convergence of the two readouts "
                 f"({n4.get('n_shared')} cells)", fontsize=10)
    return _save(fig, out)


def soma_vs_position(run: dict, out: Path) -> Path:
    """The methodological figure: soma-side vs position-based crossing for the key cells."""
    n3 = run["channels"].get("N3", {}).get("derived", {})
    n1 = run["channels"].get("N1", {}).get("derived", {})
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rows = []
    for t, g in (n3 or {}).get("gater_laterality", {}).items():
        rows.append((t, g.get("soma_contra_pct"), g.get("position_cross_pct")))
    # add the genuine crossers for contrast (from the bridge manifest if present)
    bm = run.get("bridge_manifest", {}).get("bridge_types", {})
    for t in ("Nod1", "H1", "H2", "LPT42_Nod4"):
        if t in bm:
            rows.append((t, None, bm[t].get("position_output_crossing_pct")))
    if not rows:
        ax.text(0.5, 0.5, "no data", ha="center"); return _save(fig, out)
    names = [r[0] for r in rows]
    soma = [r[1] if r[1] is not None else 0 for r in rows]
    pos = [r[2] if r[2] is not None else 0 for r in rows]
    x = np.arange(len(names))
    ax.bar(x - 0.2, soma, 0.4, label="soma-side contra %", color="#BDC3C7")
    ax.bar(x + 0.2, pos, 0.4, label="position-based crossing %", color="#C0392B")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("%"); ax.axhline(50, color="#888", ls="--", lw=0.7)
    ax.set_title("Soma-side vs position-based crossing: VCH/DCH are NOT axonal bridges\n"
                 "(high soma %, ~0 position %); Nod1/H1/H2/LPT42 genuinely cross", fontsize=9)
    ax.legend(fontsize=8)
    return _save(fig, out)


def render_all(run: dict, figdir: Path) -> list[Path]:
    figdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn, name in [(readout_crossing, "readout_crossing.png"),
                     (heterolateral_bridges, "heterolateral_bridges.png"),
                     (shared_convergence, "shared_convergence.png"),
                     (soma_vs_position, "soma_vs_position.png")]:
        try:
            paths.append(fn(run, figdir / name))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[figures_inter] {name} failed: {e}")
    return paths
