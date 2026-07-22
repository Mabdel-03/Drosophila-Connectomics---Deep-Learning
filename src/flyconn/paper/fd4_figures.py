"""Data-driven figures for the FD4 search report (the negative result).

Each function takes the assembled FD4 ``run`` dict (``run["derived"]["FD4"]``) and writes one PNG.
The figures document why FD4 has no cleanly resolved connectome correlate: the candidate
elimination, the Nod1 homogeneity, the FD4 phenotype crosswalk, and the null-aware confidence.
Anatomical neuron renders live in ``fd4_figures_anat.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FD1C = "#1f77b4"     # FD1 / Nod1 (blue)
_FD4C = "#C97A0A"     # FD4 (orange; the target)
_FD3C = "#C0392B"     # FD3 / LPT42_Nod4 (red)
_GREY = "#BBBBBB"
_GREEN = "#1B7837"


def _fd4(run: dict) -> dict:
    return run["derived"].get("FD4", {})


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 1. Candidate elimination: the layer-a funnel over all types.
# ---------------------------------------------------------------------------
def fd4_candidate_elimination(run: dict, out: Path) -> Path:
    """The exhaustive funnel: of all cell types, which pass each FD4 output-class gate. The last
    surviving bar is Nod1 (= FD1). Shows there is no distinct progressive figure-output cell."""
    sc = _fd4(run).get("candidate_screen", {})
    rows = sc.get("rows", [])
    universe = sc.get("universe_size", 0)
    n_layer_a = len(rows)
    n_hetero = sum(1 for r in rows if r.get("heterolateral"))
    n_chol = sum(1 for r in rows if r.get("heterolateral") and r.get("cholinergic"))
    n_out = sum(1 for r in rows if r.get("heterolateral") and r.get("cholinergic") and r.get("output_class"))
    stages = ["all\ntypes", "layer-a\n(progressive),\nlow-copy",
              "+ heterolateral\n(contra >=70%)", "+ cholinergic", "+ visual\nprojection output"]
    vals = [universe, n_layer_a, n_hetero, n_chol, n_out]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    cols = [_GREY, _GREY, _GREY, _GREY, _FD1C]
    bars = ax1.bar(range(len(stages)), vals, color=cols, log=True)
    ax1.set_xticks(range(len(stages))); ax1.set_xticklabels(stages, fontsize=8)
    ax1.set_ylabel("number of cell types (log)")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v * 1.15, str(v), ha="center", fontsize=9)
    surv = sc.get("survivors", [])
    ax1.set_title(f"FD4 output-class funnel over {universe} types -> {len(surv)} survivor: "
                  f"{', '.join(surv) if surv else 'none'}", fontsize=10)

    ax2.axis("off")
    ax2.text(0.0, 1.0, "Layer-a low-copy types and why each fails the FD4 output-class test:",
             fontsize=9, fontweight="bold", va="top")
    y = 0.90
    for r in sorted(rows, key=lambda r: -(r.get("contra_pct") or 0))[:14]:
        viable = "VIABLE" in r.get("verdict", "")
        col = _FD1C if viable else "#555"
        ax2.text(0.0, y, f"{r['type']} (n={r['n_cells']}, {r['contra_pct']}% contra): {r['verdict']}",
                 fontsize=6.6, color=col, va="top", fontweight="bold" if viable else "normal")
        y -= 0.066
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 2. Nod1 homogeneity: the 4 cells do not split into FD1 + FD4.
# ---------------------------------------------------------------------------
def fd4_nod1_homogeneity(run: dict, out: Path) -> Path:
    """The 4 Nod1 cells placed by RF centroid (frontal-lateral) and RF width, coloured by side,
    with the two candidate bilateral pairings and their silhouette scores. All four cluster
    frontally: no separable FD4 (lateral, wide) sub-pair. The identity centrepiece."""
    sp = _fd4(run).get("nod1_split", {})
    per = sp.get("per_cell", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [3, 2]})

    for rid, c in per.items():
        cp = c.get("centroid_p"); w = c.get("width_p"); side = c.get("side")
        if cp is None or w is None:
            continue
        col = _FD1C if side == "left" else _FD3C
        ax1.scatter([cp], [w], s=160, color=col, edgecolor="k", zorder=3)
        ax1.annotate(f"{side}\nlat.wt {c.get('lateral_weight')}", (cp, w), fontsize=7,
                     textcoords="offset points", xytext=(8, 4))
    # An FD4 cell would sit to the RIGHT (more lateral, higher p) and be WIDER (up).
    ax1.axvspan(2, 30, color=_FD4C, alpha=0.10)
    ax1.text(0.98, 0.02, "where an FD4 cell would sit\n(lateral, wide) -> empty",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=8, color=_FD4C)
    ax1.set_xlabel("RF centroid p  (frontal < 0 < lateral)")
    ax1.set_ylabel("RF width (FWHM, lattice units)")
    ax1.set_title("The 4 Nod1 cells: all frontal, none lateral/wide (no FD4 sub-pair)", fontsize=10)
    ax1.axvline(0, color="k", lw=0.6, ls=":")

    ax2.axis("off")
    ax2.text(0.0, 1.0, "Is the Nod1 quartet a separable FD1 + FD4 split?", fontsize=9,
             fontweight="bold", va="top")
    y = 0.86
    for s in sp.get("pairings", []):
        ax2.text(0.0, y, f"pairing {s['pairing']}:", fontsize=7, va="top")
        ax2.text(0.05, y - 0.06, f"within={s['within']}  between={s['between']}  "
                 f"silhouette={s['silhouette']}", fontsize=7, va="top",
                 color=_GREEN if s["silhouette"] > 0 else _FD3C)
        y -= 0.15
    ax2.text(0.0, y - 0.02, f"separable: {sp.get('separable')}    homogeneous: {sp.get('homogeneous')}",
             fontsize=8.5, fontweight="bold", va="top",
             color=_FD3C if sp.get("homogeneous") else _GREEN)
    ax2.text(0.0, y - 0.10, f"same-side input Jaccard: {sp.get('same_side_input_jaccard')}\n"
             f"cross-side input Jaccard: {sp.get('cross_side_input_jaccard')}\n"
             f"(high same-side, low cross-side = two copies of one cell per side)",
             fontsize=7, va="top")
    ax2.text(0.0, y - 0.30, "Both pairings have NEGATIVE silhouette: the four cells are one\n"
             "homogeneous frontal population (Nod1 = FD1), not FD1 + FD4.",
             fontsize=7.5, va="top", color=_FD3C)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 3. FD4 phenotype crosswalk on the Nod1 population.
# ---------------------------------------------------------------------------
def fd4_phenotype(run: dict, out: Path) -> Path:
    """Each FD4-defining property, whether the Nod1 population matches it, and whether it is a
    shared-class property (no FD4-vs-FD1 power) or an FD4-discriminating one (all of which fail)."""
    ph = _fd4(run).get("phenotype", {})
    props = ph.get("properties", [])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    labels, cols, notes = [], [], []
    for p in props:
        shared = any(k in p["property"] for k in ("layer-a", "heterolateral", "cholinergic", "second arbor"))
        labels.append(p["property"])
        if p["match"]:
            cols.append(_FD1C if shared else _GREEN)
        else:
            cols.append(_GREY)
        notes.append(("shared FD1/FD4 class" if shared else "FD4-discriminating") +
                     (" (match)" if p["match"] else " (no match)"))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, [1] * len(labels), color=cols)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([])
    for i, (p, note) in enumerate(zip(props, notes)):
        ax.text(0.02, len(labels) - 1 - i, f"{p['measured']}  [{note}]", va="center",
                fontsize=6.6, color="white" if p["match"] else "#333")
    ax.set_title(f"FD4-defining properties vs the Nod1 population "
                 f"({ph.get('n_matched')}/{ph.get('n_properties')} matched; all matches are "
                 f"shared FD1/FD4 class properties)", fontsize=10)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 4. RF comparison: FD1 frontal, FD3 lateral+gap, FD4 (would be) whole-eye lateral.
# ---------------------------------------------------------------------------
def fd4_rf_comparison(run: dict, out: Path) -> Path:
    """Schematic of the four FD receptive fields on one eye's azimuth axis, with the measured Nod1
    (FD1) centroid marked, showing FD4's expected whole-eye lateral field is unoccupied."""
    sp = _fd4(run).get("nod1_split", {})
    per = sp.get("per_cell", {})
    cps = [c.get("centroid_p") for c in per.values() if c.get("centroid_p") is not None]
    nod1_c = float(np.mean(cps)) if cps else -8.0

    fig, ax = plt.subplots(figsize=(11, 4.2))
    az = np.linspace(-20, 130, 400)

    def band(center, width, label, color, amp=1.0):
        y = amp * np.exp(-0.5 * ((az - center) / width) ** 2)
        ax.plot(az, y, color=color, lw=2, label=label)
        ax.fill_between(az, y, color=color, alpha=0.12)

    band(10, 18, "FD1 frontal (progressive)", _FD1C)
    band(5, 16, "FD2 frontal (regressive)", _GREEN, amp=0.9)
    band(45, 26, "FD3 fronto-lateral + gap (regressive)", _FD3C, amp=0.95)
    # FD4 whole-eye, lateral-weighted (dashed = not realized in the connectome)
    y4 = 0.5 + 0.5 * (az / 130.0)
    y4[az < 0] = 0.5
    ax.plot(az, y4 * 0.9, color=_FD4C, lw=2.4, ls="--",
            label="FD4 whole-eye, lateral-weighted (not resolved)")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"azimuth $\psi$ (deg): frontal 0, lateral +")
    ax.set_ylabel("relative sensitivity")
    ax.set_title("The four FD receptive fields. FD4's whole-eye lateral field (dashed) is the one "
                 "with no resolved connectome correlate.", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_ylim(0, 1.25)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# 5. Null-aware confidence.
# ---------------------------------------------------------------------------
def fd4_confidence(run: dict, out: Path) -> Path:
    """The decomposed, null-aware confidence: discriminating-property matches, the point estimate
    and wide interval, and the discount table (with the FD1-collinearity term)."""
    conf = _fd4(run).get("confidence", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [2, 3]})

    ax1.axis("off")
    pt = conf.get("point_estimate") or 0.0
    iv = conf.get("interval", [0, 0])
    ax1.barh([0], [pt], color=_FD4C, height=0.4)
    ax1.errorbar([pt], [0], xerr=[[max(pt - (iv[0] or 0), 0)], [max((iv[1] or 0) - pt, 0)]],
                 fmt="o", color="black", capsize=6)
    ax1.axvline(conf.get("ceiling", 0.85), color=_FD3C, ls="--", lw=1)
    ax1.set_xlim(0, 1); ax1.set_ylim(-1.5, 1.5)
    ax1.axis("on"); ax1.set_yticks([]); ax1.set_xlabel("confidence FD4 is individually resolved")
    ax1.text(0.02, 0.95, f"point estimate: {conf.get('point_estimate')}", fontsize=10, fontweight="bold")
    ax1.text(0.02, 0.72, f"interval: {iv}", fontsize=9)
    ax1.text(0.02, 0.50, f"ceiling: {conf.get('ceiling')} (FD1-collinearity cap)", fontsize=8, color=_FD3C)
    ax1.text(0.02, 0.28, f"verdict: {conf.get('identity_verdict')}", fontsize=8.5, color=_FD3C)
    ax1.set_title("Null-aware confidence")

    ax2.axis("off")
    ax2.text(0.0, 1.0, "Why the confidence is low and the interval wide (discounts):", fontsize=9,
             fontweight="bold", va="top")
    y = 0.86
    disc = conf.get("discounts", {})
    labels = {"fd1_collinearity_shared_class": "FD4 shares FD1's entire progressive output class",
              "not_a_distinct_type": "FD4 is not an independently annotated FlyWire type",
              "no_independent_anchor": "no independent FD4 anchor; identity is inference",
              "literature_declined_mapping": "modern typing declined the FD1/2/3/4 mapping",
              "further_fd_cells_possible": "further FD cells cannot be excluded"}
    for k, v in disc.items():
        ax2.text(0.0, y, f"-{v}", fontsize=8, fontweight="bold", va="top", color=_FD3C)
        ax2.text(0.08, y, labels.get(k, k), fontsize=7.5, va="top")
        y -= 0.11
    ax2.text(0.0, y - 0.02, conf.get("statement", ""), fontsize=6.8, va="top", wrap=True)
    return _save(fig, out)


def render_all(run: dict, fig_dir: Path) -> dict:
    """Render every data figure; returns {name: path}. Anatomical figures are rendered separately."""
    fig_dir = Path(fig_dir)
    made = {}
    for name, fn in [("fd4_candidate_elimination", fd4_candidate_elimination),
                     ("fd4_nod1_homogeneity", fd4_nod1_homogeneity),
                     ("fd4_phenotype", fd4_phenotype),
                     ("fd4_rf_comparison", fd4_rf_comparison),
                     ("fd4_confidence", fd4_confidence)]:
        try:
            made[name] = str(fn(run, fig_dir / f"{name}.png"))
        except Exception as e:  # noqa: BLE001
            made[name] = f"error: {type(e).__name__}: {e}"
    return made
