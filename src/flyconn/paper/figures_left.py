"""Stage-8 figures: the existence-test headline + mirror comparison + wing-flip.

Matplotlib Agg backend (headless). Each function is defensive: missing data -> a placeholder
panel, never a crash, so render_all always produces the figure set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_STAGE_COLOR = {"PRESENT": "#2E8B57", "WEAK": "#E1A100", "ABSENT": "#C0392B"}


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def stage_existence(run: dict, out: Path) -> Path:
    """The headline: 7-stage PRESENT/WEAK/ABSENT grid for left vs right."""
    es = run.get("existence_screen")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    if not es:
        ax.text(0.5, 0.5, "no existence-screen data", ha="center"); return _save(fig, out)
    rs, ls = es["right"]["stages"], es["left"]["stages"]
    stages = list(rs.keys())
    sides = ["right", "left"]
    grid = [[rs[s] for s in stages], [ls[s] for s in stages]]
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=_STAGE_COLOR.get(val, "#888"),
                                       ec="white", lw=2))
            ax.text(j + 0.5, 1 - i + 0.5, val[:4], ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(0, len(stages)); ax.set_ylim(0, 2)
    ax.set_xticks([j + 0.5 for j in range(len(stages))])
    ax.set_xticklabels([s.split("_", 1)[-1] for s in stages], rotation=35, ha="right", fontsize=8)
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["RIGHT\n(control)", "LEFT\n(mirror)"])
    dec = es.get("decision")
    ax.set_title(f"Left-mirror existence screen — decision: {dec}", fontsize=12, fontweight="bold")
    for v, c in _STAGE_COLOR.items():
        ax.bar(0, 0, color=c, label=v)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.18), ncol=3, fontsize=8, frameon=False)
    return _save(fig, out)


def mirror_comparison(run: dict, out: Path) -> Path:
    """Left vs right headline quantities (scale-robust + counts)."""
    m = (run.get("mirror", {}).get("derived", {}) or {}).get("mirror", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    if not m:
        ax1.text(0.5, 0.5, "no mirror data", ha="center"); return _save(fig, out)

    # scale-robust fractions / percentages
    frac_keys = [("recip_frac", "T4/T5->VCH\nrecip %"), ("vch_layer_a_pct", "VCH layer-a %"),
                 ("nod1_contra_pct", "Nod1 contra %"), ("fd3_contra_pct", "FD3 contra %")]
    labels, rvals, lvals = [], [], []
    for k, lab in frac_keys:
        v = m.get(k, {})
        if v.get("right") is not None and v.get("left") is not None:
            labels.append(lab); rvals.append(float(v["right"])); lvals.append(float(v["left"]))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, rvals, 0.4, label="right", color="#34495E")
    ax1.bar(x + 0.2, lvals, 0.4, label="left", color="#2E86C1")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("%"); ax1.set_title("Scale-robust statistics (left mirrors right)")
    ax1.legend(fontsize=8)

    # absolute counts (show the proofreading-completeness deficit honestly)
    cnt_keys = [("vch_in_syn", "VCH in"), ("vch_out_syn", "VCH out"),
                ("sheet_size", "LLPC1 sheet"), ("nod1_to_dnp26", "Nod1->DNp26")]
    labels2, r2, l2 = [], [], []
    for k, lab in cnt_keys:
        v = m.get(k, {})
        if v.get("right") and v.get("left"):
            labels2.append(lab); r2.append(float(v["right"])); l2.append(float(v["left"]))
    x2 = np.arange(len(labels2))
    # normalise each pair to right=1.0 to show the proportional deficit
    rn = [1.0] * len(r2); ln = [l / r for l, r in zip(l2, r2)]
    ax2.bar(x2 - 0.2, rn, 0.4, label="right (=1.0)", color="#34495E")
    ax2.bar(x2 + 0.2, ln, 0.4, label="left / right", color="#2E86C1")
    for i, (l, r) in enumerate(zip(l2, r2)):
        ax2.text(i + 0.2, l / r + 0.02, f"{int(l)}/{int(r)}", ha="center", fontsize=7)
    ax2.axhline(1.0, color="#888", lw=0.8, ls="--")
    ax2.set_xticks(x2); ax2.set_xticklabels(labels2, fontsize=8)
    ax2.set_ylabel("left / right"); ax2.set_title("Absolute counts (proofreading deficit, proportional)")
    ax2.legend(fontsize=8)
    return _save(fig, out)


def wing_flip(run: dict, out: Path) -> Path:
    """DNp26 per-body steering: each body's target physical wing (the complement)."""
    wf = (run.get("mirror", {}).get("derived", {}) or {}).get("wing_flip", {})
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if not wf.get("available"):
        ax.text(0.5, 0.5, f"wing-flip unavailable:\n{wf.get('reason')}", ha="center")
        return _save(fig, out)
    bodies = wf.get("dnp26_bodies", [])
    labels = [f"DNp26_{b['dn_soma_side']}\n(soma {b['dn_soma_side']})" for b in bodies]
    ipsi = [b["ipsi_frac"] or 0 for b in bodies]
    contra = [1 - (b["ipsi_frac"] or 0) for b in bodies]
    x = np.arange(len(bodies))
    ax.bar(x, ipsi, 0.5, label="ipsi wing", color="#2E86C1")
    ax.bar(x, contra, 0.5, bottom=ipsi, label="contra wing", color="#C0392B")
    for i, b in enumerate(bodies):
        ax.text(i, 1.02, f"-> {b['target_wing']} wing", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("fraction of steering output"); ax.set_ylim(0, 1.15)
    flip = wf.get("dnp26_bodies_opposite_wings")
    ax.set_title(f"DNp26 wing-flip — opposite wings: {flip} "
                 f"(shared muscles: {wf.get('dnp26_shared_muscles')})", fontsize=10)
    ax.legend(fontsize=8, loc="lower center")
    return _save(fig, out)


def cell_correspondence(run: dict, out: Path) -> Path:
    """Left+right counterpart with matching NT+super_class, per named circuit type."""
    cc = (run.get("mirror", {}).get("derived", {}) or {}).get("cell_correspondence", {})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    per = cc.get("per_type", {})
    if not per:
        ax.text(0.5, 0.5, "no correspondence data", ha="center"); return _save(fig, out)
    types = list(per.keys())
    y = np.arange(len(types))
    for i, t in enumerate(types):
        r = per[t]
        ok = r.get("both_present") and r.get("nt_match") and r.get("class_match")
        c = "#2E8B57" if ok else "#C0392B"
        ax.barh(i, 1, color=c)
        txt = (f"L{r['n_left']}/R{r['n_right']}  {r.get('nt', '')}/{r.get('super_class', '')}"
               if r.get("both_present") else f"L{r['n_left']}/R{r['n_right']} MISSING")
        ax.text(0.02, i, txt, va="center", fontsize=7, color="white")
    ax.set_yticks(y); ax.set_yticklabels(types, fontsize=8)
    ax.set_xticks([])
    ax.set_title(f"Cell-type correspondence: {cc.get('n_corresponding')}/{cc.get('n_types')} "
                 f"left+right with matching NT+class", fontsize=11)
    return _save(fig, out)


def render_all(run: dict, figdir: Path) -> list[Path]:
    figdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn, name in [(stage_existence, "stage_existence.png"),
                     (mirror_comparison, "mirror_comparison.png"),
                     (wing_flip, "wing_flip.png"),
                     (cell_correspondence, "per_celltype_bars.png")]:
        try:
            paths.append(fn(run, figdir / name))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[figures_left] {name} failed: {e}")
    return paths
