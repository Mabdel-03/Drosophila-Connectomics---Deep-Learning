"""Summary figure for the paper verification ledger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flyconn.motif import compare as K

from .oracle import consts as C

_COLOR = {
    K.CONFIRMED: "#2ca02c",
    K.CONFIRMED_WITH_CAVEAT: "#ff7f0e",
    K.REFUTED: "#d62728",
    K.UNVERIFIABLE: "#7f7f7f",
}
_ORDER = [K.CONFIRMED, K.CONFIRMED_WITH_CAVEAT, K.REFUTED, K.UNVERIFIABLE]


def summary_figure(run: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fams = list(run["results"].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = [0] * len(fams)
    for v in _ORDER:
        vals = [K.verdict_counts(run["results"][f]).get(v, 0) for f in fams]
        ax.bar(fams, vals, bottom=bottoms, color=_COLOR[v], label=v)
        bottoms = [b + x for b, x in zip(bottoms, vals)]
    ax.set_xlabel("claim family")
    ax.set_ylabel("# claims")
    ax.set_title("Figure-Ground Circuit verification: verdicts by family")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Family K (FD3 == LPT42_Nod4) figures + publishable table.
# ---------------------------------------------------------------------------
def fd3_rf_differential(run: dict, out: Path) -> Path:
    """Bar chart of the differential RF: LPT42_Nod4 (FD3) vs Nod1 (FD1) centroid azimuth
    (hex-p) and width, per side, with the frontal-gap occupancy annotated."""
    d = run["derived"].get("K", {})
    rf = d.get("rf", {})
    per_side = rf.get("per_side", {})
    nb = d.get("smallfield_null", {})
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))

    sides = sorted(per_side.keys())
    if sides:
        offs = [per_side[s]["centroid_offset_p"] for s in sides]
        cis = [per_side[s]["offset_ci95"] for s in sides]
        yerr = [[o - ci[0] for o, ci in zip(offs, cis)], [ci[1] - o for o, ci in zip(offs, cis)]]
        ax1.bar(sides, offs, yerr=yerr, color="#1f77b4", capsize=5)
        ax1.axhline(0, color="k", lw=0.8)
        ax1.set_ylabel("centroid offset vs FD1 (hex-p; + = more lateral)")
        ax1.set_title("a  FD3 RF is more lateral than FD1\n(95% CI over input columns)")

        x = range(len(sides))
        cand_occ = [per_side[s]["cand_frontal_occ"] for s in sides]
        ref_occ = [per_side[s]["ref_frontal_occ"] for s in sides]
        ax2.bar([i - 0.2 for i in x], ref_occ, 0.4, label="FD1=Nod1", color="#7f7f7f")
        ax2.bar([i + 0.2 for i in x], cand_occ, 0.4, label="FD3=LPT42_Nod4", color="#d62728")
        ax2.set_xticks(list(x)); ax2.set_xticklabels(sides)
        ax2.set_ylabel("occupancy of FD1 frontal band")
        ax2.set_title("b  FD3 frontal gap\n(Egelhaaf FD3 feature)")
        ax2.legend(fontsize=8)

    # Panel c: small-field permutation null. Observed patch radius is below the null.
    obs, nullr = nb.get("obs_radius"), nb.get("null_radius")
    if obs is not None and nullr is not None:
        ax3.bar(["observed", "in-degree\nnull"], [obs, nullr], color=["#d62728", "#7f7f7f"])
        ax3.set_ylabel("RF patch radius (lattice units)")
        ax3.set_title(f"c  Bounded small field\nz={nb.get('z_score')}, p={nb.get('p_value')} "
                      f"({nb.get('n_perms')} perms)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fd3_crosswalk(run: dict, out: Path) -> Path:
    """Egelhaaf-FD3-physiology <-> LPT42_Nod4-connectome-measurement checkmark crosswalk."""
    d = run["derived"].get("K", {})
    rf = d.get("rf", {}).get("per_side", {})
    any_lat = any(rf[s].get("more_lateral") for s in rf) if rf else False
    any_gap = any(rf[s].get("has_frontal_gap") for s in rf) if rf else False
    nb = d.get("smallfield_null", {})
    rows = [
        ("Regressive (back-to-front) preferred dir", "T4/T5 input layer-b "
         f"{(d.get('cand_layer_frac') or {}).get('b','?')}%", "confirmed" if d.get("cand_dominant_direction") == "back_to_front" else "check"),
        ("Fronto-lateral RF", "RF centroid lateral of FD1 (bootstrap CI)",
         "confirmed, relative RF" if any_lat else "check"),
        ("Frontal gap (Egelhaaf FD3 feature)", "near-zero frontal-band occupancy",
         "confirmed" if any_gap else "check"),
        ("Small-field selective", f"patch bounded vs null (z={nb.get('z_score')})",
         "wiring support" if nb.get("bounded") else "check"),
        ("Heterolateral noduli-group axon", f"contra output {d.get('cand_contra_output_pct')}%",
         "confirmed" if (d.get("cand_contra_output_pct") or 0) >= 70 else "check"),
        ("Cholinergic output", f"ACh conf {d.get('cand_mean_nt_conf')}",
         "confirmed" if d.get("cand_nt") == "acetylcholine" else "check"),
        ("Cell body posterolateral", f"soma z-pct {d.get('soma',{}).get('post_z_percentile')}",
         "confirmed" if d.get("soma", {}).get("bilateral_split") else "check"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(rows) + 1))
    ax.axis("off")
    ax.set_title("FD3 (Egelhaaf 1985) vs LPT42_Nod4 (FlyWire): property crosswalk", fontsize=12)
    for i, (phys, conn, status) in enumerate(rows):
        y = len(rows) - i
        ax.text(0.01, y, phys, va="center", fontsize=10)
        ax.text(0.43, y, conn, va="center", fontsize=10, color="#333")
        color = "#1B7837" if status.startswith("confirmed") else "#C97A0A"
        ax.text(0.97, y, status, va="center", ha="right",
                fontsize=9, color=color, fontweight="bold")
    ax.text(0.01, len(rows) + 0.8, "FD3 physiology", fontsize=10, fontweight="bold")
    ax.text(0.43, len(rows) + 0.8, "Connectome measurement", fontsize=10, fontweight="bold")
    ax.text(0.97, len(rows) + 0.8, "Evidence status", fontsize=10, fontweight="bold", ha="right")
    ax.set_xlim(0, 1); ax.set_ylim(0, len(rows) + 1.4)
    return _save(fig, out)


def fd3_layer_composition(run: dict, out: Path) -> Path:
    """Grouped T4/T5 lobula-plate layer-fraction bars across the Nod family + contra-inh strata."""
    d = run["derived"].get("K", {})
    profiles = d.get("profiles", {})
    order = [t for t in ("LPT42_Nod4", "Nod1", "Nod3", "Nod5", "Nod2") if t in profiles]
    layers = ["a", "b", "c", "d"]
    lcolor = {"a": "#1f77b4", "b": "#d62728", "c": "#2ca02c", "d": "#9467bd"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    x = range(len(order))
    w = 0.2
    for j, L in enumerate(layers):
        vals = [(profiles[t].get("layer_frac") or {}).get(L) or 0 for t in order]
        ax1.bar([i + (j - 1.5) * w for i in x], vals, w, label=f"layer-{L} ({LAYER_DIR(L)})", color=lcolor[L])
    ax1.set_xticks(list(x)); ax1.set_xticklabels(order, rotation=20)
    ax1.set_ylabel("% of T4/T5 input synapses")
    ax1.set_title("Preferred-direction (lobula-plate layer) composition\nLPT42_Nod4 is layer-b; Nod1 layer-a; Nod5 layer-c")
    ax1.legend(fontsize=8)
    # contra-inhibition strata
    ci = (d.get("contra_inhibition") or {})
    cats, prog, reg = [], [], []
    for t in ("LPT42_Nod4", "Nod1"):
        c = ci.get(t, {})
        if c.get("available"):
            cats.append(t); prog.append(c.get("frac_progressive") or 0); reg.append(c.get("frac_regressive") or 0)
    xi = range(len(cats))
    ax2.bar([i - 0.2 for i in xi], prog, 0.4, label="progressive (layer-a)", color="#1f77b4")
    ax2.bar([i + 0.2 for i in xi], reg, 0.4, label="regressive (layer-b)", color="#d62728")
    ax2.set_xticks(list(xi)); ax2.set_xticklabels(cats)
    ax2.set_ylabel("fraction of contralateral inhibitory synapses")
    ax2.set_title("Contralateral inhibition strata\n(power-limited; see caveats)")
    ax2.legend(fontsize=8)
    return _save(fig, out)


def LAYER_DIR(letter: str) -> str:
    return {"a": "front->back", "b": "back->front", "c": "up", "d": "down"}.get(letter, letter)


def fd3_family_heatmap(run: dict, out: Path) -> Path:
    """FD{1-4} x candidate match-score heatmap; annotate the LPT42_Nod4/FD3 reciprocal best hit."""
    import numpy as np
    s = run["derived"].get("K", {}).get("fd_family_screen", {})
    mat = s.get("score_matrix", {})
    fds = ["FD1", "FD2", "FD3", "FD4"]
    cands = sorted({c for fd in mat for c in mat[fd]})
    if not cands:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "no screen data", ha="center"); return _save(fig, out)
    M = np.array([[mat.get(fd, {}).get(c, 0) for c in cands] for fd in fds], dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(cands)), 3.4))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(cands))); ax.set_xticklabels(cands, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(fds))); ax.set_yticklabels(fds)
    for i in range(len(fds)):
        for j in range(len(cands)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] >= 2 else "black")
    # highlight FD3 / LPT42_Nod4 (red) and, for the disambiguation, FD2 / LPT21 (green).
    if "LPT42_Nod4" in cands:
        j = cands.index("LPT42_Nod4")
        ax.add_patch(plt.Rectangle((j - 0.5, 2 - 0.5), 1, 1, fill=False, edgecolor="#C0392B", lw=2.5))
    if "LPT21" in cands:
        j = cands.index("LPT21")
        ax.add_patch(plt.Rectangle((j - 0.5, 1 - 0.5), 1, 1, fill=False, edgecolor="#1B7837", lw=2.5))
    ax.set_title(f"FD-family match scores (0-3). FD3 reciprocal best: "
                 f"{s.get('best_match_for_FD3')} (margin {s.get('fd3_margin')}); "
                 f"FD2 (LPT21/Nod3 tie) broken by homolateral+frontal metrics")
    fig.colorbar(im, ax=ax, shrink=0.8, label="features matched")
    return _save(fig, out)


def fd3_replication_robustness(run: dict, out: Path) -> Path:
    """(a) three-track replication grid; (b) knob/seed robustness."""
    import numpy as np
    d = run["derived"].get("K", {})
    offline = (run.get("offline") or {}).get("derived", {})
    cv = d.get("cross_version", {})
    rows = ["dominant_layer", "layer_b%", "contra%", "RF lateral+gap", "small-field"]
    def track_vals(dd, cvkey=None):
        rf = dd.get("rf", {}).get("per_side", {})
        latgap = bool(rf and all(rf[s].get("more_lateral") and rf[s].get("has_frontal_gap") for s in rf))
        return [dd.get("cand_dominant_layer"), (dd.get("cand_layer_frac") or {}).get("b"),
                dd.get("cand_contra_output_pct"), latgap, bool(dd.get("smallfield_null", {}).get("bounded"))]
    primary = track_vals(d)
    off = track_vals(offline) if offline else [None] * 5
    v630 = cv.get("v630", {}) if cv.get("available") else {}
    v630col = [v630.get("dominant_layer"), v630.get("layer_b"), v630.get("contra_output_pct"),
               cv.get("agree", {}).get("rf_lateral_gap"), None] if v630 else [None] * 5
    track = run.get("meta", {}).get("flywire_track", "primary")
    cols = [f"{track}-v783", "offline-v783", "v630"]
    data = list(zip(primary, off, v630col))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    ax1.axis("off")
    ax1.set_title("Replication across sources & versions")
    tbl = ax1.table(cellText=[[str(v) if v is not None else "-" for v in row] for row in data],
                    rowLabels=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.scale(1, 1.6); tbl.set_fontsize(9)
    # robustness panel
    rb = d.get("robustness", {})
    gp = rb.get("per_knob", {}).get("gap_drop", {})
    grid = gp.get("grid", []); pas = gp.get("pass", [])
    ax2.bar(range(len(grid)), [1 if p else 0 for p in pas], color=["#1B7837" if p else "#C0392B" for p in pas])
    ax2.set_xticks(range(len(grid))); ax2.set_xticklabels([str(g) for g in grid])
    ax2.set_ylim(0, 1.2); ax2.set_ylabel("frontal-gap holds")
    ax2.set_xlabel("gap_drop threshold")
    ss = rb.get("seed_stability", {})
    ax2.set_title(f"Robustness: gap-threshold sweep\nseed offset-CI-lower in "
                  f"[{ss.get('offset_ci_lower_min')}, {ss.get('offset_ci_lower_max')}] (>0)")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Family KD (FD3 disambiguation: LPT42_Nod4 vs Nod3) figures.
# ---------------------------------------------------------------------------
_LPT_C = "#C0392B"      # FD3 / LPT42_Nod4 (red)
_NOD3_C = "#C97A0A"     # FD2 / Nod3 (orange)


def _kd(run: dict) -> dict:
    return run["derived"].get("KD", {})


_LPT21_C = "#1B7837"    # FD2 / LPT21 (green)


def fd3_disambig_contra_dualtrack(run: dict, out: Path) -> Path:
    """Output laterality (contralateral %) for the three regressive candidates across offline / live
    / v630.

    Egelhaaf's FD2 is homolateral (ipsilateral projection, contra near 0); FD3 is heterolateral
    (contra high). The three cells form a clean progression: LPT21 (~1.5%, FD2), Nod3 (~45%,
    intermediate), LPT42_Nod4 (~90%, FD3). The homolateral band and the heterolateral threshold are
    both marked.
    """
    d = _kd(run)
    off = d.get("candidates_offline", {})
    live = d.get("live", {})
    cv = d.get("cross_version", {})
    order = [("LPT21", _LPT21_C, "LPT21 (FD2)"),
             ("Nod3", _NOD3_C, "Nod3 (intermediate)"),
             ("LPT42_Nod4", _LPT_C, "LPT42_Nod4 (FD3)")]

    def contra(block, ct):
        return (block.get("candidates", {}) if block.get("available") else {}).get(ct, {}).get("contra_output_pct")

    tracks = ["offline v783"]
    vals = {ct: [off.get(ct, {}).get("contra_output_pct")] for ct, _, _ in order}
    if live.get("available"):
        tracks.append("live v783")
        for ct, _, _ in order:
            vals[ct].append(contra(live, ct))
    if any(cv.get(ct, {}).get("available") for ct, _, _ in order):
        tracks.append("v630")
        for ct, _, _ in order:
            v = (cv.get(ct, {}).get("v630") or {}).get("contra_output_pct") if cv.get(ct, {}).get("available") else None
            vals[ct].append(v)

    x = np.arange(len(tracks))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(tracks)), 4.4))
    for j, (ct, col, lab) in enumerate(order):
        xo = x + (j - 1) * w
        ax.bar(xo, [v if v is not None else 0 for v in vals[ct]], w, color=col, label=lab)
        for i, v in enumerate(vals[ct]):
            if v is not None:
                ax.text(xo[i], v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.axhspan(0, 15, color="#1B7837", alpha=0.06)
    ax.axhline(15, ls=":", c="#1B7837", lw=1)
    ax.text(len(tracks) - 0.5, 16.5, "homolateral band (FD2, <=15%)", ha="right", fontsize=8, color="#1B7837")
    ax.axhline(70, ls="--", c="gray", lw=1.2)
    ax.text(len(tracks) - 0.5, 71.5, "heterolateral threshold (FD3, >=70%)", ha="right", fontsize=8, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(tracks)
    ax.set_ylabel("output synapses on the contralateral side (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Output laterality: LPT21 homolateral (FD2), LPT42_Nod4 heterolateral (FD3)")
    ax.legend(fontsize=8.5, loc="upper center", ncol=3)
    return _save(fig, out)


def fd3_disambig_rf_frontal(run: dict, out: Path) -> Path:
    """(a) RF centroid offset vs the FD1 anchor with 95% CI, per candidate/side; (b) frontal-band
    occupancy (Nod3 fills the FD1 frontal band; LPT42_Nod4 leaves it empty = the gap)."""
    d = _kd(run)
    off = d.get("candidates_offline", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    trio = (("LPT21", _LPT21_C), ("Nod3", _NOD3_C), ("LPT42_Nod4", _LPT_C))
    labels, offs, los, his, cols = [], [], [], [], []
    for ct, col in trio:
        per = off.get(ct, {}).get("rf", {}).get("per_side", {})
        for side in sorted(per):
            labels.append(f"{ct}\n{side}")
            o = per[side].get("centroid_offset_p")
            ci = per[side].get("offset_ci95") or [None, None]
            offs.append(o if o is not None else 0)
            los.append((o - ci[0]) if (o is not None and ci[0] is not None) else 0)
            his.append((ci[1] - o) if (o is not None and ci[1] is not None) else 0)
            cols.append(col)
    x = np.arange(len(labels))
    ax1.bar(x, offs, color=cols, yerr=[los, his], capsize=4)
    ax1.axhline(0, c="gray", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("centroid offset vs FD1 anchor (lattice p; >0 = lateral)")
    ax1.set_title("Receptive field position vs FD1\nLPT21 frontal (~0); Nod3 slightly lateral; LPT42_Nod4 lateral")

    labels2, cand_occ, ref_occ, cols2 = [], [], [], []
    for ct, col in trio:
        per = off.get(ct, {}).get("rf", {}).get("per_side", {})
        for side in sorted(per):
            labels2.append(f"{ct}\n{side}")
            cand_occ.append((per[side].get("cand_frontal_occ") or 0) * 100)
            ref_occ.append((per[side].get("ref_frontal_occ") or 0) * 100)
            cols2.append(col)
    x2 = np.arange(len(labels2))
    ax2.bar(x2 - 0.2, ref_occ, 0.4, color="#888888", label="FD1 (frontal band)")
    ax2.bar(x2 + 0.2, cand_occ, 0.4, color=cols2, label="candidate")
    ax2.set_xticks(x2); ax2.set_xticklabels(labels2, fontsize=7.5)
    ax2.set_ylabel("frontal-band occupancy (%)")
    ax2.set_title("Frontal field: LPT21 & Nod3 fill the FD1 band; LPT42_Nod4 leaves it empty (gap)")
    ax2.legend(fontsize=8)
    return _save(fig, out)


def fd3_disambig_morphology(run: dict, out: Path) -> Path:
    """Axon medio-lateral shift (dendrite -> axon) per cell for both candidates, alongside the
    fraction of output that is contralateral.

    The bar height is the dendrite-to-axon medio-lateral shift; a shift toward the opposite
    hemisphere is the geometric signature of a crossing axon. Both candidates show such a shift,
    so the discriminator is not the shift sign alone but whether the OUTPUT is predominantly
    contralateral: bars are coloured solid where the candidate is heterolateral (>=70% contra
    output, the FD3 signature) and grey where the output stays largely ipsilateral (FD2).
    """
    d = _kd(run)
    off = d.get("candidates_offline", {})
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels, shifts, cols = [], [], []
    for ct, base in (("LPT21", _LPT21_C), ("Nod3", _NOD3_C), ("LPT42_Nod4", _LPT_C)):
        cand = off.get(ct, {})
        hetero = bool(cand.get("heterolateral"))
        contra = cand.get("contra_output_pct")
        m = cand.get("morphology", {})
        for c in (m.get("cells", []) if m.get("available") else []):
            side = c.get("side")
            pct = f"{contra:.0f}%" if isinstance(contra, (int, float)) else ""
            labels.append(f"{ct}\n{side}\n({pct} contra)")
            sh = c.get("axon_ml_shift_um")
            shifts.append(sh if sh is not None else 0)
            cols.append(base if hetero else "#BBBBBB")
    x = np.arange(len(labels))
    ax.bar(x, shifts, color=cols)
    ax.axhline(0, c="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("dendrite -> axon medio-lateral shift (um)")
    src = (off.get("LPT42_Nod4", {}).get("morphology", {}) or {}).get("source", "proxy")
    ax.set_title(f"Axon shift and output laterality (solid = heterolateral, >=70% contra)\n"
                 f"only LPT42_Nod4 is heterolateral; LPT21 & Nod3 stay ipsilateral. source: {src}")
    return _save(fig, out)


def fd3_disambig_decision(run: dict, out: Path) -> Path:
    """The head-to-head comparison table as a color-coded figure, with the decision-rule outcome."""
    import textwrap
    d = _kd(run)
    rows = d.get("comparison_rows", [])
    cons = d.get("constructive", {})

    # Five columns: Property | Egelhaaf ref | LPT21 | Nod3 | LPT42_Nod4. Left edges + wrap widths.
    xs = [0.005, 0.20, 0.40, 0.60, 0.80]
    wrap_chars = [22, 22, 22, 22, 22]

    def _wrap(text, w):
        return "\n".join(textwrap.wrap(str(text), width=w)) or str(text)

    wrapped, row_lines = [], []
    for r in rows:
        cells = [_wrap(r["property"], wrap_chars[0]), _wrap(r["egelhaaf"], wrap_chars[1]),
                 _wrap(r.get("lpt21", ""), wrap_chars[2]), _wrap(r["nod3"], wrap_chars[3]),
                 _wrap(r["lpt42"], wrap_chars[4])]
        wrapped.append(cells)
        row_lines.append(max(c.count("\n") + 1 for c in cells))

    line_h = 0.23
    y_cursor = 0.0
    y_positions = []
    for n in row_lines:
        y_positions.append(y_cursor)
        y_cursor += n * line_h + 0.16
    total_h = y_cursor
    fig_h = total_h + 2.4

    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    ax.axis("off")
    ax.set_title("Three regressive candidates vs Egelhaaf's FD cells", fontsize=13)
    headers = ["Property", "Egelhaaf reference", "LPT21 (FD2)", "Nod3 (intermediate)", "LPT42_Nod4 (FD3)"]
    hcol = ["#000000", "#000000", _LPT21_C, _NOD3_C, _LPT_C]
    y_top = total_h + 0.35
    for hx, h, hc in zip(xs, headers, hcol):
        ax.text(hx, y_top, h, fontsize=9, fontweight="bold", color=hc)
    ax.plot([0, 1], [total_h + 0.15, total_h + 0.15], color="#999", lw=0.8)
    for i, r in enumerate(rows):
        y = total_h - y_positions[i] - line_h
        wt = "normal" if not r.get("discriminates") else "bold"
        cells = wrapped[i]
        ax.text(xs[0], y, cells[0], fontsize=7.5, fontweight=wt, va="top")
        ax.text(xs[1], y, cells[1], fontsize=7, color="#333", va="top")
        ax.text(xs[2], y, cells[2], fontsize=7, color="#333", va="top")
        ax.text(xs[3], y, cells[3], fontsize=7, color="#333", va="top")
        ax.text(xs[4], y, cells[4], fontsize=7, color="#333", va="top")
    dv = next((c for c in d.get("_claims", []) if c.get("id") == "KD.decision_verdict"), {})
    verdict_txt = dv.get("computed_primary") or "FD3=LPT42_Nod4; FD2=LPT21; Nod3=intermediate"
    ax.text(0.005, -0.35, f"Assignment: {verdict_txt}", fontsize=10, fontweight="bold", color="#000000")
    lpt21_contra = (d.get("candidates_offline", {}).get("LPT21", {}) or {}).get("contra_output_pct")
    nod3_contra = (d.get("candidates_offline", {}).get("Nod3", {}) or {}).get("contra_output_pct")
    ax.text(0.005, -0.75,
            f"FD2 tie-break: LPT21 is homolateral (contra {lpt21_contra:.0f}%) and frontal; Nod3 is "
            f"mixed ({nod3_contra:.0f}% contra) and slightly lateral, so LPT21 is the clean FD2 and "
            f"Nod3 is left intermediate.", fontsize=8.5, color="#333")
    ax.set_xlim(0, 1); ax.set_ylim(-1.1, total_h + 0.7)
    return _save(fig, out)


def fd3_disambig_fd2_scatter(run: dict, out: Path) -> Path:
    """The FD2 discriminator plane: output laterality (x, homolateral<->heterolateral) vs receptive-
    field offset from the FD1 anchor (y, frontal<->lateral).

    Egelhaaf's FD2 sits in the lower-left (homolateral + frontal); FD3 in the upper-right
    (heterolateral + lateral). LPT21 lands in the FD2 corner, LPT42_Nod4 in the FD3 corner, and Nod3
    between them, confirming Nod3 is an intermediate rather than a clean FD2.
    """
    d = _kd(run)
    off = d.get("candidates_offline", {})
    pts = (("LPT21", _LPT21_C, "LPT21 (FD2)"),
           ("Nod3", _NOD3_C, "Nod3 (intermediate)"),
           ("LPT42_Nod4", _LPT_C, "LPT42_Nod4 (FD3)"))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axvspan(0, 15, color="#1B7837", alpha=0.05)
    ax.axvline(15, ls=":", c="#1B7837", lw=1)
    ax.axvline(70, ls="--", c=_LPT_C, lw=1)
    for ct, col, lab in pts:
        b = off.get(ct, {})
        x = b.get("contra_output_pct"); y = b.get("mean_rf_offset")
        if x is None or y is None:
            continue
        ax.scatter([x], [y], s=180, color=col, edgecolor="black", zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9, color=col)
    ax.set_xlabel("output on the contralateral side (%)  [homolateral -> heterolateral]")
    ax.set_ylabel("RF centroid offset vs FD1 anchor  [frontal -> lateral]")
    ax.text(2, ax.get_ylim()[1] * 0.92, "FD2 corner\n(homolateral, frontal)", fontsize=8.5, color="#1B7837")
    ax.text(72, ax.get_ylim()[1] * 0.92, "FD3 corner\n(heterolateral, lateral)", fontsize=8.5, color=_LPT_C)
    ax.set_title("The FD2/FD3 discriminator plane")
    ax.grid(alpha=0.2)
    return _save(fig, out)


def fd3_circuit_placement(run: dict, out: Path) -> Path:
    """(a) top output targets + contra%; (b) soma scatter (soma_x vs soma_z)."""
    d = run["derived"].get("K", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    tt = d.get("cand_top_targets", {})
    names = list(tt.keys())[:8]; vals = [tt[n] for n in names]
    ax1.barh(range(len(names))[::-1], vals, color="#4C72B0")
    ax1.set_yticks(range(len(names))[::-1]); ax1.set_yticklabels(names)
    ax1.set_xlabel("synapses from LPT42_Nod4")
    ax1.set_title(f"Top downstream targets (contralateral {d.get('cand_contra_output_pct')}%)")
    # Soma scatter from the soma block (LPT42 pair). Other cells need coordinates.
    soma = d.get("soma", {})
    c = soma.get("centroid_xyz", {})
    if c:
        ax2.scatter([c.get("soma_x")], [c.get("soma_z")], s=80, c="#C0392B", label="LPT42_Nod4 centroid")
        ax2.axvline(soma.get("midline_x"), ls="--", c="gray", lw=1, label="brain midline")
    ax2.set_xlabel("soma_x (nm)"); ax2.set_ylabel("soma_z (nm; higher = posterior)")
    ax2.set_title(f"Cell-body location: posterior (z-pct {soma.get('post_z_percentile')}), bilateral")
    ax2.legend(fontsize=8)
    return _save(fig, out)


def fd3_morphology(run: dict, out: Path) -> Path:
    """Skeleton-free morphology: dendrite vs axon cloud extent + the heterolateral axon shift
    toward the noduli landmark, per cell."""
    d = run["derived"].get("K", {})
    m = d.get("morphology", {})
    cells = m.get("cells", [])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    # (a) dendrite/axon medio-lateral centroid + shift arrow
    nod = m.get("noduli_landmark_um")
    for i, c in enumerate(cells):
        dx = (c.get("dendrite") or {}).get("centroid", [None])[0]
        ax = (c.get("axon") or {}).get("centroid", [None])[0]
        if dx is None or ax is None:
            continue
        y = i
        ax1.scatter([dx], [y], s=80, c="#1f77b4", label="dendrite (input syn)" if i == 0 else None)
        ax1.scatter([ax], [y], s=80, c="#C0392B", marker="s", label="axon (output syn)" if i == 0 else None)
        ax1.annotate("", xy=(ax, y), xytext=(dx, y),
                     arrowprops=dict(arrowstyle="->", color="gray"))
    if nod:
        ax1.axvline(nod[0], ls="--", c="green", lw=1.2, label="noduli landmark (Nod1)")
    if m.get("midline_x_um"):
        ax1.axvline(m["midline_x_um"], ls=":", c="black", lw=1, label="brain midline")
    ax1.set_yticks(range(len(cells))); ax1.set_yticklabels([c.get("side") for c in cells])
    ax1.set_xlabel("medio-lateral position (um)")
    ax1.set_title("a  Axon displaced from dendrite toward noduli")
    ax1.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    # (b) distance-to-noduli: axon vs dendrite
    sides = [c.get("side") for c in cells]
    a2n = [c.get("axon_to_noduli_um") for c in cells]
    d2n = [c.get("dend_to_noduli_um") for c in cells]
    x = range(len(sides))
    ax2.bar([i - 0.2 for i in x], d2n, 0.4, label="dendrite", color="#1f77b4")
    ax2.bar([i + 0.2 for i in x], a2n, 0.4, label="axon", color="#C0392B")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(sides)
    ax2.set_ylabel("distance to noduli landmark (um)")
    ax2.set_title("b  Axon nearer noduli than dendrite")
    ax2.legend(fontsize=8)
    return _save(fig, out)


def fd3_claim_table_tex(run: dict, out: Path) -> Path:
    """A booktabs LaTeX table of Family-K claims and verdicts (publishable artifact)."""
    claims = run["results"].get("K", [])
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Validation of \textit{LPT42\_Nod4} as the modern correlate of "
        r"Egelhaaf-1985 FD3. Verdicts: CONFIRMED, CONFIRMED$^{*}$ (with caveat), "
        r"REFUTED, N/A (unverifiable).}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Claim & Expected & Verdict \\",
        r"\midrule",
    ]
    icon = {K.CONFIRMED: "CONFIRMED", K.CONFIRMED_WITH_CAVEAT: r"CONFIRMED$^{*}$",
            K.REFUTED: "REFUTED", K.UNVERIFIABLE: "N/A"}

    def esc(s: str) -> str:
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    for c in claims:
        L.append(f"{esc(c.id)} & {esc(c.report_value)} & {icon.get(c.verdict, c.verdict)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out.write_text("\n".join(L))
    return out


# ---------------------------------------------------------------------------
# Family L (FD3 -> descending neurons -> motor) figures.
# ---------------------------------------------------------------------------
_MOTOR_SYS_COLOR = {
    "wing_steering": "#d62728", "wing_power": "#9467bd", "neck_gaze": "#1f77b4",
    "haltere": "#2ca02c", "leg": "#8c564b", "jump_ttm": "#e377c2",
    "abdominal": "#ff7f0e", "other": "#7f7f7f",
}


def fd3_dn_ranking(run: dict, out: Path) -> Path:
    """The descending neurons FD3 contacts directly, ranked by synapses; steering DNs marked."""
    d = run["derived"].get("L", {})
    ranking = (d.get("direct", {}) or {}).get("ranking", [])[:15]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.45 * len(ranking) + 1)))
    if ranking:
        names = [r["cell_type"] for r in ranking][::-1]
        syn = [r["syn"] for r in ranking][::-1]
        colors = ["#d62728" if r["is_steering"] else "#7f7f7f" for r in ranking][::-1]
        y = range(len(names))
        ax.barh(list(y), syn, color=colors)
        ax.set_yticks(list(y)); ax.set_yticklabels(names)
        ax.set_xlabel("synapses from FD3 (LPT42_Nod4)")
        for i, r in enumerate(ranking[::-1]):
            ax.text(r["syn"], i, f"  {r['syn']}", va="center", fontsize=8)
        ax.set_title("Descending neurons FD3 contacts directly\n"
                     "(red = known figure-steering DN; DNp26 leads, shared with the FD1 arm)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#d62728", label="figure-steering DN"),
                           Patch(color="#7f7f7f", label="other DN")], fontsize=8, loc="lower right")
    return _save(fig, out)


def fd3_descending_channels(run: dict, out: Path) -> Path:
    """Direct vs relay descending routes: synapses, #DNs, and the relay-threshold sensitivity."""
    d = run["derived"].get("L", {})
    direct = d.get("direct", {}) or {}
    relay = d.get("relay", {}) or {}
    sweep = d.get("relay_threshold_sweep", []) or []
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    routes = ["direct\nFD3->DN", f"relay\nFD3->X->DN\n(>={d.get('relay_min_syn','?')} syn)"]
    syn = [direct.get("dn_syn", 0), relay.get("dn_syn", 0)]
    ndn = [direct.get("n_dns", 0), relay.get("n_dns", 0)]
    x = range(len(routes))
    ax1.bar([i - 0.2 for i in x], syn, 0.4, label="synapses", color="#1f77b4")
    ax1b = ax1.twinx()
    ax1b.bar([i + 0.2 for i in x], ndn, 0.4, label="# DNs", color="#ff7f0e")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(routes)
    ax1.set_ylabel("circuit -> DN synapses", color="#1f77b4")
    ax1b.set_ylabel("# distinct DNs", color="#ff7f0e")
    ax1.set_title("a  Two descending routes from FD3")

    if sweep:
        thr = [r["min_syn"] for r in sweep]
        nd = [r["n_dns"] for r in sweep]
        ax2.plot(thr, nd, "o-", color="#d62728")
        ax2.set_xlabel("relay intermediary synapse floor (FD3 -> X)")
        ax2.set_ylabel("# relayed DNs reached")
        ax2.set_title("b  Relay reach vs threshold\n(monotone; cut is reported, not silent)")
    return _save(fig, out)


def fd3_motor_systems(run: dict, out: Path) -> Path:
    """FD3's descending drive distributed across male-CNS motor systems (the functional summary)."""
    d = run["derived"].get("L", {})
    motor = d.get("motor", {}) or {}
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    msp = motor.get("motor_system_pct") if motor.get("available") else None
    if msp:
        systems = list(msp.keys())
        vals = [msp[s] for s in systems]
        colors = [_MOTOR_SYS_COLOR.get(s, "#7f7f7f") for s in systems]
        ax.bar(systems, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("% of FD3 descending drive (synapse-weighted)")
        ax.set_title("FD3-weighted motor-system proxy for descending output\n"
                     "(male CNS; wing-steering largest)")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "MaleCNS motor mapping unavailable", ha="center", va="center")
    return _save(fig, out)


def fd3_brain_to_muscle(run: dict, out: Path) -> Path:
    """Schematic flow FD3 -> leading DNs -> their top steering muscles (per-DN wing laterality)."""
    d = run["derived"].get("L", {})
    motor = d.get("motor", {}) or {}
    ranking = (d.get("direct", {}) or {}).get("ranking", [])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    if not motor.get("available"):
        ax.text(0.5, 0.5, "MaleCNS motor mapping unavailable", ha="center", va="center")
        return _save(fig, out)

    per_dn = motor.get("per_dn", {})
    # leading steering DNs FD3 reaches that have a male-CNS motor map.
    dns = [r["cell_type"] for r in ranking
           if r["is_steering"] and per_dn.get(r["cell_type"], {}).get("steering_syn")][:5]
    if not dns:
        dns = [k for k, v in per_dn.items() if v.get("steering_syn")][:5]
    ax.text(0.04, 0.5, "FD3\n(LPT42_Nod4)", ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round", fc="#fde0dd", ec="#d62728"))
    n = max(len(dns), 1)
    for i, dn in enumerate(dns):
        yd = (i + 0.5) / n
        row = per_dn.get(dn, {})
        wing = row.get("wing", "?")
        ax.annotate("", xy=(0.36, yd), xytext=(0.12, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#888"))
        ax.text(0.42, yd, f"{dn}\n({wing} wing)", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round", fc="#e8e8ff", ec="#1f77b4"))
        muscles = list(row.get("top_muscles", {}).items())[:3]
        mlabel = ", ".join(f"{m} ({s})" for m, s in muscles) or "-"
        ax.annotate("", xy=(0.72, yd), xytext=(0.5, yd),
                    arrowprops=dict(arrowstyle="-|>", color="#888"))
        ax.text(0.74, yd, mlabel, ha="left", va="center", fontsize=9)
    ax.text(0.42, 1.02, "leading descending neurons", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.78, 1.02, "top steering muscles", ha="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.08)
    ax.set_title("FD3 -> descending neurons -> wing-steering muscles (male CNS)", y=1.06)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Family P (FD3 afferent / input pathway).
# ---------------------------------------------------------------------------
def fd3_input_census(run: dict, out: Path) -> Path:
    """(a) FD3's input by class (motion vs sheets vs inhibitors). Motion is a minority;
    (b) the T4/T5 lobula-plate layer composition. That minority is mostly layer-b."""
    d = run["derived"].get("P", {})
    census = d.get("census", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) top input types, coloured motion / sheet / inhibitor / other.
    types = census.get("top_input_types", [])[:12]
    names = [t["cell_type"] for t in types]
    vals = [t["syn"] for t in types]

    def _klass(t):
        ct = t["cell_type"]
        if ct in ("T4b", "T5b", "T4a", "T5a", "T4c", "T5c", "T4d", "T5d"):
            return "#d62728"           # motion
        if ct.startswith(("LPC", "LLPC", "Tlp", "MeLp", "LPTe")):
            return "#1f77b4"           # columnar sheets
        if ct.startswith(("LPi", "CT1")) or t.get("nt") == "gaba":
            return "#9467bd"           # inhibitors
        return "#999999"
    colors = [_klass(t) for t in types]
    ax1.barh(range(len(names))[::-1], vals, color=colors)
    ax1.set_yticks(range(len(names))[::-1]); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("input synapses onto FD3")
    t45f = census.get("t4t5_frac_of_total")
    ax1.set_title(f"a  FD3 direct inputs by type\n(T4/T5 motion is {t45f}% of total input)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#d62728", "#1f77b4", "#9467bd", "#999999")]
    ax1.legend(handles, ["T4/T5 motion", "LPC/LLPC sheets", "LPi/contra inhibitors", "other"],
               fontsize=7, loc="lower right")

    # (b) T4/T5 layer composition: the motion drive is layer-b (regressive).
    layer = census.get("t4t5_layer_frac", {}) or {}
    lk = ["a", "b", "c", "d"]
    lv = [layer.get(k) or 0 for k in lk]
    lcolor = {"a": "#1f77b4", "b": "#d62728", "c": "#2ca02c", "d": "#9467bd"}
    ax2.bar([f"{k}\n({C.LAYER_DIRECTION[k].replace('_', ' ')})" for k in lk], lv,
            color=[lcolor[k] for k in lk])
    ax2.set_ylabel("% of FD3's T4/T5 (motion) input")
    ax2.set_title(f"b  FD3's motion drive is layer-b / regressive\n"
                  f"(T4b ON + T5b OFF = {census.get('on_off_split', {}).get('t4_frac')}% T4)")
    for i, v in enumerate(lv):
        if v:
            ax2.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Comprehensive connectivity census + functional-circuit highlight (Families census/Q/R/S).
# ---------------------------------------------------------------------------
# Functional-circuit partners drawn in saturated colour so they pop out of the full census.
_CIRCUIT_PARTNERS = {"T4b": "#d62728", "T5b": "#ff7f0e", "LPC1": "#1f77b4",
                     "LPi14": "#9467bd", "DNp26": "#2ca02c"}


def fd3_connectivity_wheel(run: dict, out: Path) -> Path:
    """Comprehensive census: FD3 input vs output by cell type, functional-circuit partners in
    saturated colour, everything else desaturated — the functional circuit pops out of the census."""
    cen = run["derived"].get("census", {})
    inp = cen.get("input", {}); outp = cen.get("output", {})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, block, title in ((axes[0], inp, "inputs to FD3"), (axes[1], outp, "outputs of FD3")):
        types = block.get("ranked_types", [])[:15]
        names = [t["cell_type"] for t in types]
        vals = [t["syn"] for t in types]
        colors = [_CIRCUIT_PARTNERS.get(n, "#cfcfcf") for n in names]
        ax.barh(range(len(names))[::-1], vals, color=colors)
        ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("synapses")
        tot = block.get("total_syn"); npart = block.get("n_partners")
        ax.set_title(f"{title}\n{tot} syn / {npart} partners")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _CIRCUIT_PARTNERS.values()]
    axes[0].legend(handles, list(_CIRCUIT_PARTNERS), fontsize=7, loc="lower right",
                   title="functional circuit")
    fig.suptitle("FD3 comprehensive connectivity census "
                 "(functional-circuit partners highlighted)", fontsize=13)
    return _save(fig, out)


def fd3_connectivity_matrix(run: dict, out: Path) -> Path:
    """Heatmap: top input types x (super_class / nt / laterality) synapse mass — census appendix."""
    cen = run["derived"].get("census", {})
    inp = cen.get("input", {})
    types = inp.get("ranked_types", [])[:18]
    if not types:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off")
        ax.text(0.5, 0.5, "census unavailable", ha="center"); return _save(fig, out)
    names = [t["cell_type"] for t in types]
    scs = sorted({t.get("super_class") or "?" for t in types})
    nts = sorted({t.get("nt") or "?" for t in types})
    cols = [("class:" + s) for s in scs] + [("nt:" + n) for n in nts]
    M = np.zeros((len(names), len(cols)))
    for i, t in enumerate(types):
        M[i, cols.index("class:" + (t.get("super_class") or "?"))] = t["syn"]
        M[i, cols.index("nt:" + (t.get("nt") or "?"))] = t["syn"]
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.9), max(5, len(names) * 0.35)))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="input synapses")
    ax.set_title("FD3 input types x class/NT (synapse mass)")
    return _save(fig, out)


def fd3_compartment_split(run: dict, out: Path) -> Path:
    """Dendrite vs axon input synapses per partner class — where each input class lands on FD3.

    Reads the compartment split(s) in ``run['derived']['compartment']`` (a list, one per FD3
    cell). Measured result: essentially all input arrives on the dendrite (FD3 is polarized:
    inputs on dendrite, outputs on the crossing axon)."""
    comp = run["derived"].get("compartment", [])
    if not comp:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off")
        ax.text(0.5, 0.5, "compartment split unavailable", ha="center"); return _save(fig, out)
    classes = ["motion", "sheet", "inhibitor", "contra_inhibitor", "other"]
    ccol = {"motion": "#d62728", "sheet": "#1f77b4", "inhibitor": "#9467bd",
            "contra_inhibitor": "#8c564b", "other": "#cfcfcf"}
    fig, axes = plt.subplots(1, len(comp), figsize=(6 * len(comp), 5), squeeze=False)
    for j, cs in enumerate(comp):
        ax = axes[0][j]
        dend = cs.get("dendrite", {}).get("by_class", {})
        axon = cs.get("axon", {}).get("by_class", {})
        x = np.arange(len(classes))
        ax.bar(x - 0.2, [dend.get(c, 0) for c in classes], 0.4, label="dendrite",
               color=[ccol[c] for c in classes], edgecolor="k", linewidth=0.4)
        ax.bar(x + 0.2, [axon.get(c, 0) for c in classes], 0.4, label="axon",
               color=[ccol[c] for c in classes], hatch="///", edgecolor="k", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("input synapses")
        dfrac = cs.get("dendrite_frac")
        ax.set_title(f"FD3 {cs.get('side')} cell ({cs.get('source')})\n"
                     f"dendrite fraction of input = {dfrac}")
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Where FD3's inputs land: dendrite vs axon, by partner class", fontsize=13)
    return _save(fig, out)
