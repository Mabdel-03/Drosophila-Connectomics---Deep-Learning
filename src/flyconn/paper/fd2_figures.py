"""Data-driven figures for the FD2 = LPT21 full-circuit report.

Each function takes the assembled FD2 ``run`` dict (``run["derived"]["FD2"]``) and writes one PNG.
Anatomical neuron renders (skeletons / synapse clouds / hex maps) live in ``fd2_figures_anat.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .oracle import consts as C

_FD2C = "#1B7837"     # FD2 / LPT21 (green)
_FD1C = "#1f77b4"     # FD1 / Nod1 anchor (blue)
_FD3C = "#C0392B"     # FD3 / LPT42_Nod4 (red)


def _fd2(run: dict) -> dict:
    return run["derived"].get("FD2", {})


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Identity + confidence.
# ---------------------------------------------------------------------------
def fd2_confidence(run: dict, out: Path) -> Path:
    """The decomposed confidence: per-property matches (left) and the point estimate + interval with
    its uniqueness factor and discounts (right)."""
    conf = _fd2(run).get("confidence", {})
    props = conf.get("properties", [])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [3, 2]})

    labels = [p["property"] for p in props]
    matched = [1 if p["match"] else 0 for p in props]
    cols = [_FD2C if m else "#BBBBBB" for m in matched]
    y = range(len(labels))
    ax1.barh(list(y)[::-1], [1] * len(labels), color=cols)
    ax1.set_yticks(list(y)[::-1]); ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xticks([]); ax1.set_xlim(0, 1.05)
    for i, p in enumerate(props):
        ax1.text(0.02, len(labels) - 1 - i, p.get("support", ""), va="center", fontsize=6.5,
                 color="white" if p["match"] else "#333")
    ax1.set_title(f"FD2-defining properties matched by LPT21 "
                  f"({conf.get('n_matched')}/{conf.get('n_properties')})")

    ax2.axis("off")
    pt = conf.get("point_estimate"); iv = conf.get("interval", [None, None])
    ax2.barh([0], [pt or 0], color=_FD2C, height=0.4)
    ax2.errorbar([pt or 0], [0], xerr=[[(pt or 0) - (iv[0] or 0)], [(iv[1] or 0) - (pt or 0)]],
                 fmt="o", color="black", capsize=5)
    ax2.set_xlim(0, 1); ax2.set_ylim(-1.5, 1.5)
    ax2.axis("on"); ax2.set_yticks([]); ax2.set_xlabel("confidence that LPT21 is FD2")
    ax2.text(0.02, 0.9, f"point estimate: {pt}", fontsize=10, fontweight="bold")
    ax2.text(0.02, 0.6, f"interval: {iv}", fontsize=9)
    ax2.text(0.02, 0.3, f"uniqueness factor: {conf.get('uniqueness_factor')}", fontsize=8)
    disc = conf.get("discounts", {})
    ax2.text(0.02, 0.0, "discounts: " + ", ".join(f"{k} -{v}" for k, v in disc.items()),
             fontsize=6.5, color="#333", wrap=True)
    ax2.set_title("Confidence (property fraction x uniqueness - discounts)")
    return _save(fig, out)


def fd2_uniqueness_scatter(run: dict, out: Path) -> Path:
    """Every scanned LP-tangential cell placed by regressiveness (x, layer-b %) and output laterality
    (y, contra %). The FD2 corner (regressive + homolateral, lower right) should contain LPT21 alone;
    competitors that share the corner are flagged with why they are not FD2."""
    uniq = _fd2(run).get("uniqueness", {})
    rows = uniq.get("core_hits", []) + [r for r in uniq.get("full_hits", []) if r not in uniq.get("core_hits", [])]
    # Use the full scanned set if present in core+full; else just the hits we have.
    scanned = {r["type"]: r for r in uniq.get("core_hits", [])}
    for r in uniq.get("full_hits", []):
        scanned.setdefault(r["type"], r)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.axhspan(0, 15, color=_FD2C, alpha=0.06)
    ax.axvspan(80, 100, color="#d62728", alpha=0.04)
    for t, r in scanned.items():
        lb = r.get("layer_b_pct") or 0; contra = r.get("contra_pct") or 0
        is_lpt21 = (t == "LPT21")
        col = _FD2C if is_lpt21 else ("#C97A0A" if r.get("core_match") else "#999999")
        ax.scatter([lb], [contra], s=150 if is_lpt21 else 70, color=col,
                   edgecolor="black", zorder=3 if is_lpt21 else 2)
        if is_lpt21 or r.get("core_match"):
            note = ""
            if not is_lpt21:
                if not r.get("single_pair"):
                    note = f" (n={r.get('n_cells')})"
            ax.annotate(t + note, (lb, contra), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=col)
    ax.axhline(15, ls=":", c=_FD2C, lw=1)
    ax.set_xlabel("regressive drive: % of T4/T5 input that is layer-b")
    ax.set_ylabel("output laterality: % contralateral (low = homolateral, FD2)")
    ax.set_title(f"FD2 uniqueness scan ({uniq.get('n_fd2_hits')} full FD2 match(es); "
                 f"unique={uniq.get('unique')})")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Afferent.
# ---------------------------------------------------------------------------
def fd2_input_census(run: dict, out: Path) -> Path:
    """(a) LPT21 direct inputs by type, coloured motion/sheet/inhibitor; (b) its T4/T5 layer split."""
    census = _fd2(run).get("afferent", {}).get("census", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    types = census.get("top_input_types", [])[:12]
    names = [t["cell_type"] for t in types]; vals = [t["syn"] for t in types]

    def _klass(t):
        ct = t["cell_type"]
        if ct in ("T4b", "T5b", "T4a", "T5a", "T4c", "T5c", "T4d", "T5d"):
            return "#d62728"
        if ct.startswith(("LPC", "LLPC", "Tlp", "MeLp", "LPTe")):
            return "#1f77b4"
        if ct.startswith(("LPi", "CT1")) or t.get("nt") == "gaba":
            return "#9467bd"
        return "#999999"
    ax1.barh(range(len(names))[::-1], vals, color=[_klass(t) for t in types])
    ax1.set_yticks(range(len(names))[::-1]); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("input synapses onto LPT21")
    ax1.set_title(f"a  LPT21 direct inputs by type\n(T4/T5 motion = {census.get('t4t5_frac_of_total')}% of input)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#d62728", "#1f77b4", "#9467bd", "#999999")]
    ax1.legend(handles, ["T4/T5 motion", "LPC/LLPC sheets", "LPi inhibitors", "other"],
               fontsize=7, loc="lower right")

    layer = census.get("t4t5_layer_frac", {}) or {}
    lk = ["a", "b", "c", "d"]; lv = [layer.get(k) or 0 for k in lk]
    lcolor = {"a": "#1f77b4", "b": "#d62728", "c": "#2ca02c", "d": "#9467bd"}
    ax2.bar([f"{k}\n({C.LAYER_DIRECTION[k].replace('_', ' ')})" for k in lk], lv,
            color=[lcolor[k] for k in lk])
    ax2.set_ylabel("% of LPT21's T4/T5 input")
    ax2.set_title(f"b  LPT21's motion drive is layer-b / regressive\n"
                  f"(T4b ON + T5b OFF, {census.get('on_off_split', {}).get('t4_frac')}% T4)")
    for i, v in enumerate(lv):
        if v:
            ax2.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Efferent.
# ---------------------------------------------------------------------------
def fd2_dn_ranking(run: dict, out: Path) -> Path:
    """The descending neurons LPT21 contacts directly, ranked by synapses; wing-steering DNs marked."""
    direct = _fd2(run).get("efferent", {}).get("direct", {})
    rank = direct.get("ranking", [])[:12]
    names = [r["cell_type"] for r in rank]; vals = [r["syn"] for r in rank]
    steer = [r.get("is_steering") for r in rank]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cols = ["#C0392B" if s else "#4C72B0" for s in steer]
    ax.barh(range(len(names))[::-1], vals, color=cols)
    ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("synapses from LPT21")
    ax.set_title(f"Descending neurons contacted directly by LPT21\n"
                 f"(red = known wing-steering DN; total {direct.get('dn_syn')} syn)")
    return _save(fig, out)


def fd2_motor_systems(run: dict, out: Path) -> Path:
    """LPT21-weighted motor-system shares (male-CNS proxy) + DNp26's top wing-steering muscles."""
    mot = _fd2(run).get("efferent", {}).get("motor", {})
    pct = mot.get("motor_system_pct", {}) or {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    items = sorted(pct.items(), key=lambda kv: kv[1], reverse=True)
    names = [k.replace("_", " ") for k, _ in items]; vals = [v for _, v in items]
    cols = ["#C0392B" if k == "wing_steering" else "#888888" for k, _ in items]
    ax1.bar(names, vals, color=cols)
    ax1.set_ylabel("% of LPT21-weighted descending output")
    ax1.set_title(f"Motor-system proxy (dominant: {mot.get('dominant_motor_system')})")
    ax1.tick_params(axis="x", rotation=30, labelsize=8)
    dnp26 = (mot.get("per_dn", {}) or {}).get("DNp26", {})
    mus = dnp26.get("top_muscles", {}) or {}
    if mus:
        ax2.bar(list(mus.keys()), list(mus.values()), color="#C0392B")
        ax2.set_ylabel("synapses (male CNS)")
        ax2.set_title(f"DNp26 top muscles ({dnp26.get('wing')} wing)")
    else:
        ax2.axis("off"); ax2.text(0.5, 0.5, "DNp26 muscle data not available", ha="center")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Dual output (the FD2-unique second axonal branch).
# ---------------------------------------------------------------------------
def fd2_dual_output(run: dict, out: Path) -> Path:
    """Per LPT21 cell, the output-synapse distribution along the medio-lateral axis, showing the two
    separated terminal fields (the main ipsilateral POF terminal + the frontal branch)."""
    dual = _fd2(run).get("dual_output", {})
    cells = [c for c in dual.get("cells", []) if c.get("available")]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    labels, separation, minor = [], [], []
    for c in cells:
        labels.append(f"{c.get('side')}\ncell")
        # separation = how deep the valley is between the two fields (100% = the valley is empty).
        dip = c.get("ml_valley_dip_ratio")
        separation.append(100.0 * (1.0 - dip) if isinstance(dip, (int, float)) else 0)
        minor.append((c.get("minor_lobe_frac") or 0) * 100)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, separation, 0.4, color=_FD2C, label="separation between fields (100 - valley/peak, %)")
    ax.bar(x + 0.2, minor, 0.4, color="#C97A0A", label="smaller terminal field (% of output)")
    ax.axhline(75, ls=":", c=_FD2C, lw=1)
    ax.set_ylim(0, 105)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_title("Dual output: two separated terminal fields per LPT21 cell\n"
                 "(the connectome correlate of FD2's main + frontal axonal branches)")
    ax.legend(fontsize=8)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Comprehensive connectivity + compartment split.
# ---------------------------------------------------------------------------
def fd2_connectivity_wheel(run: dict, out: Path) -> Path:
    """LPT21's major input (left) and output (right) partner types by super-class."""
    census = _fd2(run).get("census", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, title in ((ax1, "input", "Inputs to LPT21"), (ax2, "output", "Outputs from LPT21")):
        block = census.get(key, {}) or {}
        ranked = block.get("ranked_types", [])[:10]
        names = [r.get("cell_type") for r in ranked]; vals = [r.get("syn") for r in ranked]
        ax.barh(range(len(names))[::-1], vals, color="#4C72B0" if key == "input" else "#2ca02c")
        ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("synapses"); ax.set_title(title)
    return _save(fig, out)


def fd2_compartment_split(run: dict, out: Path) -> Path:
    """Input synapses to each LPT21 cell split dendrite vs axon, by partner class."""
    comp = _fd2(run).get("compartment", {})
    fig, axes = plt.subplots(1, max(len(comp), 1), figsize=(6 * max(len(comp), 1), 4.2), squeeze=False)
    for ax, (root, cs) in zip(axes[0], comp.items()):
        cbc = (cs or {}).get("class_by_compartment", {}) or {}
        classes = sorted({k for v in cbc.values() for k in (v or {})})
        dend = [(cbc.get("dendrite", {}) or {}).get(k, 0) for k in classes]
        axon = [(cbc.get("axon", {}) or {}).get(k, 0) for k in classes]
        x = np.arange(len(classes))
        ax.bar(x - 0.2, dend, 0.4, label="dendrite", color="#4C72B0")
        ax.bar(x + 0.2, axon, 0.4, label="axon", color="#C0392B")
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, fontsize=7)
        ax.set_ylabel("input synapses")
        ax.set_title(f"cell {root}: input by compartment\n(dendrite frac {(cs or {}).get('dendrite_frac')})")
        ax.legend(fontsize=8)
    return _save(fig, out)
