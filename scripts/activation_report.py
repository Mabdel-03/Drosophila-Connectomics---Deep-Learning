"""Generate all figures + a metrics JSON for the activation-pathways report.

One GPU forward pass over the test set (via per_step_neuron_activity) yields the
per-(recurrence-step, neuron) mean |activation| for the frozen-connectome ReLU model;
everything else is derived from that array + the labelled neuron table. Produces:

  figs/activation/
    A_recurrence_vs_accuracy.png   recurrence step -> activation spread vs test accuracy
    B_wavefront_heatmap.png        cell-type x step staircase (one synaptic hop per step)
    C_superclass_over_depth.png    optic -> central -> descending handoff over depth
    D_anatomy_2d.png               2D anatomical projections, activity-colored, I/O marked
    E_retinotopy_hex.png           retinotopic "eye view" of the input photoreceptors
    F_activation_3d_interactive.html   rotatable 3D, hover neuron identity
    _metrics.json                  every number the report prose quotes
    table3_superclass_by_depth.csv super_class x step activation matrix

Run on a GPU node (consortium env): sbatch slurm/activation_report.sbatch
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from flyconn.models.activity import per_step_neuron_activity
from flyconn.paths import data_root

RUN = "eyeTact_whole_learned_relu_untrained_T12"
REPO = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics")
FIG = REPO / "3 - Modeling" / "figs" / "activation"
ACT = data_root() / "v783" / "activity"
RETINO = data_root() / "v783" / "processed" / "retinotopy_columns.parquet"
FIG.mkdir(parents=True, exist_ok=True)

# pathway stage for annotating cell types (rough, by leading token)
STAGE = [
    ("R1-6", "retina"), ("R7", "retina"), ("R8", "retina"),
    ("L1", "lamina"), ("L2", "lamina"), ("L3", "lamina"), ("Lai", "lamina"), ("Lawf", "lamina"),
    ("Mi", "medulla"), ("Tm", "medulla"), ("Dm", "medulla"), ("T1", "medulla"),
    ("T2", "medulla"), ("Pm", "medulla"), ("C2", "medulla"), ("C3", "medulla"), ("CT1", "medulla"),
    ("LC", "lobula"), ("LT", "lobula"), ("LPLC", "lobula"), ("Li", "lobula"),
    ("PVLP", "central"), ("AVLP", "central"), ("AOTU", "central"), ("TuBu", "central"),
    ("ER", "central"), ("ExR", "central"), ("CB", "central"),
    ("DN", "descending"),
]


def stage_of(ct: str) -> str:
    for pre, st in STAGE:
        if ct.startswith(pre):
            return st
    return "other"


def main():
    d = per_step_neuron_activity(RUN, max_batches=12)
    mas = d["mean_abs_step"]                 # [T, N] mean |activation| per step, neuron
    nrn = d["neurons"].reset_index(drop=True)
    T, N = d["T"], mas.shape[1]
    inp = np.asarray(d["input_ids"]); rdt = np.asarray(d["readout_ids"])
    is_in = np.zeros(N, bool); is_in[inp] = True
    is_out = np.zeros(N, bool); is_out[rdt] = True
    overall = mas.mean(axis=0)               # [N]
    peak_step = mas.argmax(axis=0) + 1       # [N]
    ct = nrn["cell_type"].astype(str).to_numpy()
    sc = nrn["super_class"].astype(str).to_numpy()
    root = nrn["root_id"].to_numpy()
    metrics = {"run": RUN, "T": int(T), "N": int(N),
               "n_input": int(is_in.sum()), "n_readout": int(is_out.sum())}

    # depth-pathway CSV (accuracy + n_active_types per t)
    dp = pd.read_csv(ACT / "depth_pathway_relu_untrained.csv")
    metrics["n_active_types_per_t"] = dp["n_active_types"].tolist()
    metrics["test_acc_per_t"] = dp["test_acc@T=t"].tolist()
    first_readout_t = int(dp.loc[dp["readout_reached"] == "yes", "t"].min())
    metrics["readout_first_reached_t"] = first_readout_t

    # ---------------- per-cell-type x step matrix (group-mean) ----------------
    df = pd.DataFrame({"cell_type": ct})
    type_step = np.zeros((T, 0))
    ctypes = []
    by = pd.DataFrame({"cell_type": ct})
    for t in range(T):
        by[f"s{t}"] = mas[t]
    g = by.groupby("cell_type").mean()       # [n_types, T] mean over neurons of each type
    peak_by_type = g.max(axis=1)
    # Force-include representative early-pathway types so the staircase's top-left (retina ->
    # lamina -> medulla -> lobula) is populated, not just the high-magnitude late central cells.
    early = ["R1-6", "R7", "R8", "L1", "L2", "L3", "Lai", "Mi1", "Tm2", "Tm1", "Dm6", "Dm17",
             "T1", "Pm12", "LT1a", "LT1b", "LC12", "TuBu06a"]
    early = [c for c in early if c in g.index]
    late = [c for c in peak_by_type.sort_values(ascending=False).index
            if c not in early][:max(0, 28 - len(early))]
    top_types = early + late
    gT = g.loc[top_types].to_numpy()
    # order rows by first step above threshold, then by peak step (the staircase)
    THR = 0.05
    arrival = []
    for row in gT:
        above = np.where(row > THR)[0]
        arrival.append(above[0] if len(above) else 99)
    order = np.argsort([(a, -gT[i].max()) for i, a in enumerate(arrival)],
                       axis=0)  # noqa: not used directly
    idx_order = sorted(range(len(top_types)), key=lambda i: (arrival[i], -gT[i].argmax()))
    top_types = [top_types[i] for i in idx_order]
    gT = gT[idx_order]

    # ============================ FIGURE A ============================
    fig, axL = plt.subplots(figsize=(9, 5.2))
    t = dp["t"].to_numpy()
    nact = dp["n_active_types"].to_numpy()
    acc = dp["test_acc@T=t"].to_numpy()
    cL, cR = "crimson", "navy"
    axL.plot(t, nact, "o-", color=cL, lw=2, label="active cell types")
    axL.set_yscale("log")
    axL.set_xlabel("recurrence step  t  (= synaptic hops propagated)")
    axL.set_ylabel("# active cell types  (mean|h|>0.05, log scale)", color=cL)
    axL.tick_params(axis="y", labelcolor=cL)
    axL.set_xticks(t)
    axR = axL.twinx()
    axR.plot(t, acc, "s-", color=cR, lw=2, label="test accuracy")
    axR.set_ylabel("test accuracy  (model trained at T=t)", color=cR)
    axR.tick_params(axis="y", labelcolor=cR)
    axR.set_ylim(0, 1.02)
    axR.axhline(0.1, ls=":", color="grey", lw=1)
    axL.axvline(first_readout_t, ls="--", color="green", lw=1.3)
    axL.annotate(f"readout reached (t={first_readout_t})\naccuracy jumps to ~0.98",
                 xy=(first_readout_t, nact[first_readout_t - 1]),
                 xytext=(first_readout_t + 1.2, nact[1] * 4),
                 fontsize=9, color="green",
                 arrowprops=dict(arrowstyle="->", color="green"))
    axL.set_title("Recurrence depth: activation spread explodes, accuracy saturates\n"
                  "frozen-connectome ReLU network (whole brain)")
    fig.tight_layout(); fig.savefig(FIG / "A_recurrence_vs_accuracy.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

    # ============================ FIGURE B ============================
    fig, ax = plt.subplots(figsize=(10, 8))
    floor = max(gT[gT > 0].min(), 1e-2)
    im = ax.imshow(np.clip(gT, floor, None), aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=floor, vmax=gT.max()))
    ax.set_xticks(range(T)); ax.set_xticklabels(range(1, T + 1))
    ax.set_yticks(range(len(top_types)))
    ax.set_yticklabels([f"{c}  ·{stage_of(c)}" for c in top_types], fontsize=8)
    ax.set_xlabel("recurrence step  t"); ax.set_ylabel("cell type (ordered by arrival hop)")
    ax.set_title("Activation wavefront: one synaptic hop per recurrence step\n"
                 "(diagonal staircase = signal marching retina -> lamina -> medulla -> "
                 "lobula -> central -> descending)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="mean |activation| (log)")
    fig.tight_layout(); fig.savefig(FIG / "B_wavefront_heatmap.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

    # ============================ FIGURE C ============================
    classes = ["sensory", "optic", "visual_projection", "central", "descending"]
    palette = {"sensory": "#4C72B0", "optic": "#55A868", "visual_projection": "#8C8C3A",
               "central": "#DD8452", "descending": "#C44E52", "other": "#BBBBBB"}
    sc_step = np.zeros((len(classes) + 1, T))
    for t in range(T):
        sums = pd.Series(mas[t]).groupby(pd.Series(sc)).sum()
        for ci, c in enumerate(classes):
            sc_step[ci, t] = sums.get(c, 0.0)
        sc_step[-1, t] = sums.sum() - sc_step[:-1, t].sum()   # other
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.stackplot(range(1, T + 1), sc_step,
                 labels=classes + ["other"],
                 colors=[palette[c] for c in classes] + [palette["other"]], alpha=0.9)
    ax.set_xlabel("recurrence step  t"); ax.set_ylabel("total |activation| (summed over neurons)")
    ax.set_xticks(range(1, T + 1))
    ax.set_title("Where the activity lives over depth: optic -> central -> descending handoff")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "C_superclass_over_depth.png", dpi=150,
                                    bbox_inches="tight"); plt.close(fig)
    # table 3 csv
    pd.DataFrame(sc_step, index=classes + ["other"],
                 columns=[f"t{t+1}" for t in range(T)]).to_csv(
        FIG / "table3_superclass_by_depth.csv")

    # ============================ FIGURE D ============================
    # Both panels use the XY (dorsal) plane — the only informative projection, since pos_z is
    # a thin ~7k-nm A-P slab. Left = activity magnitude; right = peak recurrence step (the
    # spatial wavefront). I/O populations marked on both.
    px = nrn["pos_x"].to_numpy(float) / 1e3   # -> microns for readability
    py = nrn["pos_y"].to_numpy(float) / 1e3
    pz = nrn["pos_z"].to_numpy(float) / 1e3
    hot_order = np.argsort(overall)[::-1][:4000]
    show_wave = overall > np.quantile(overall, 0.88)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # panel 1: activity magnitude
    ax = axes[0]
    ax.scatter(px[::6], py[::6], s=0.5, c="lightgrey", alpha=0.12, linewidths=0)
    scd = ax.scatter(px[hot_order], py[hot_order], c=overall[hot_order], cmap="hot",
                     s=7, alpha=0.8, linewidths=0)
    ax.scatter(px[is_in], py[is_in], s=9, c="dodgerblue", alpha=0.5, linewidths=0,
               label=f"input photoreceptors (n={int(is_in.sum())})")
    ax.scatter(px[is_out], py[is_out], s=9, c="limegreen", alpha=0.45, linewidths=0,
               label=f"output VPN/descending (n={int(is_out.sum())})")
    ax.set_title("Activation magnitude (mean |h| over the unroll)")
    ax.set_aspect("equal"); ax.legend(fontsize=8, markerscale=2, loc="upper right")
    ax.set_xlabel("medio-lateral  (x, µm)"); ax.set_ylabel("dorso-ventral  (y, µm)")
    cb1 = fig.colorbar(scd, ax=ax, shrink=0.6); cb1.set_label("mean |activation|")
    # panel 2: peak recurrence step (the spatial wavefront)
    ax2 = axes[1]
    ax2.scatter(px[::6], py[::6], s=0.5, c="lightgrey", alpha=0.1, linewidths=0)
    scw = ax2.scatter(px[show_wave], py[show_wave], c=peak_step[show_wave], cmap="viridis",
                      s=7, alpha=0.8, linewidths=0, vmin=1, vmax=T)
    ax2.scatter(px[is_in], py[is_in], s=9, c="red", alpha=0.45, linewidths=0, label="input")
    ax2.scatter(px[is_out], py[is_out], s=9, marker="^", c="black", alpha=0.4, linewidths=0,
                label="output")
    ax2.set_title("Wavefront: recurrence step at which each neuron peaks")
    ax2.set_aspect("equal"); ax2.legend(fontsize=8, markerscale=2, loc="upper right")
    ax2.set_xlabel("medio-lateral  (x, µm)"); ax2.set_ylabel("dorso-ventral  (y, µm)")
    cb2 = fig.colorbar(scw, ax=ax2, shrink=0.6); cb2.set_label("peak recurrence step t")
    fig.suptitle("Anatomy of activation (frozen-connectome ReLU, dorsal view): hot integrator "
                 "core in central brain; wavefront sweeps optic lobes -> centre", y=1.01)
    fig.savefig(FIG / "D_anatomy_2d.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ============================ FIGURE E ============================
    try:
        rt = pd.read_parquet(RETINO)
        act_df = pd.DataFrame({"root_id": root, "act": overall, "is_in": is_in})
        m = rt.merge(act_df[act_df["is_in"]], on="root_id", how="inner")
        # axial hex (p,q) -> cartesian
        m["hx"] = m["p"] + m["q"] / 2.0
        m["hy"] = m["q"] * np.sqrt(3) / 2.0
        # per-column mean activation
        col = m.groupby(["eye", "column_id"]).agg(
            hx=("hx", "mean"), hy=("hy", "mean"), act=("act", "mean")).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, eye in zip(axes, [0, 1]):
            e = col[col["eye"] == eye]
            scm = ax.scatter(e["hx"], e["hy"], c=e["act"], cmap="viridis", marker="h",
                             s=40, linewidths=0)
            ax.set_title(f"{'left' if eye == 0 else 'right'} eye  "
                         f"({len(e)} columns)")
            ax.set_aspect("equal"); ax.set_xlabel("retinotopic p"); ax.set_ylabel("q")
        fig.colorbar(scm, ax=axes, shrink=0.6, label="mean photoreceptor |activation|")
        fig.suptitle("Retinotopic 'eye view': input activation on the FlyWire hex lattice",
                     y=1.02)
        fig.savefig(FIG / "E_retinotopy_hex.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        metrics["retinotopy_columns_plotted"] = int(len(col))
    except Exception as ex:
        print(f"[warn] retinotopy figure E skipped: {ex}")

    # ============================ FIGURE F (plotly) ============================
    try:
        import plotly.graph_objects as go
        show = np.argsort(overall)[::-1][:9000]
        bg = np.arange(0, N, 20)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter3d(
            x=px[bg], y=py[bg], z=pz[bg], mode="markers",
            marker=dict(size=1, color="lightgrey", opacity=0.08), hoverinfo="skip",
            name="background"))
        fig3.add_trace(go.Scatter3d(
            x=px[show], y=py[show], z=pz[show], mode="markers",
            marker=dict(size=2.5, color=peak_step[show], colorscale="Viridis",
                        colorbar=dict(title="peak step"), opacity=0.8),
            text=[f"{ct[i]} | {sc[i]} | root {int(root[i])} | peak t{peak_step[i]} "
                  f"| |h|={overall[i]:.2f}" for i in show],
            hoverinfo="text", name="active neurons"))
        fig3.update_layout(title="Activation wavefront (rotatable) — color = peak recurrence step",
                           scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
        fig3.write_html(str(FIG / "F_activation_3d_interactive.html"), include_plotlyjs="cdn")
    except Exception as ex:
        print(f"[warn] interactive figure F skipped: {ex}")

    # ---------------- metrics for the prose ----------------
    tbl = pd.DataFrame({"root_id": root, "cell_type": ct, "super_class": sc,
                        "side": nrn["side"].astype(str).to_numpy(),
                        "mean_abs": overall, "is_in": is_in, "is_out": is_out})
    metrics["top20_neurons"] = (tbl.sort_values("mean_abs", ascending=False)
                                .head(20)[["root_id", "cell_type", "super_class", "side",
                                           "mean_abs", "is_in", "is_out"]]
                                .to_dict("records"))
    # super_class peak step
    metrics["superclass_peak_step"] = {
        c: int(sc_step[ci].argmax() + 1) for ci, c in enumerate(classes)}
    metrics["wavefront_top_types_order"] = top_types
    with open(FIG / "_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    print(f"[activation_report] wrote figures + metrics to {FIG}")
    print(f"  active types per t: {metrics['n_active_types_per_t']}")
    print(f"  acc per t:          {[round(a,3) for a in metrics['test_acc_per_t']]}")
    print(f"  readout first reached t={first_readout_t}")
    print(f"  superclass peak step: {metrics['superclass_peak_step']}")


if __name__ == "__main__":
    main()
