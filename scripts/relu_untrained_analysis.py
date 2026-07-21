"""Deep-dive on the UNTRAINED ReLU frozen-connectome model (whole brain).

Produces, for eyeTact_whole_learned_relu_untrained_T12:
  (1) most-activated neurons overall (exact root_id + cell_type), saved + printed
  (2) most-activated neurons at EACH recurrence depth t=1..T (exact neurons)
  (3) a 3D spatial visualization of the most-active neurons over the unroll, with the
      input photoreceptors and output (VPN/descending) readout neurons marked.

All from one forward pass over the test set (per_step_neuron_activity).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from flyconn.models.activity import per_step_neuron_activity
from flyconn.paths import data_root

RUN = "eyeTact_whole_learned_relu_untrained_T12"
OUT = data_root() / "v783" / "activity"
OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics/3 - Modeling")


def main():
    d = per_step_neuron_activity(RUN, max_batches=12)
    mas = d["mean_abs_step"]               # [T, N]
    nrn = d["neurons"]                     # local-order labelled table
    T, N = d["T"], len(nrn)
    inp = set(d["input_ids"].tolist())     # photoreceptors
    rdt = set(d["readout_ids"].tolist())   # VPN + descending
    overall = mas.mean(axis=0)             # [N] mean |h| over the whole unroll

    # ---- (1) overall top neurons ----
    tbl = nrn.copy()
    tbl["mean_abs"] = overall
    tbl["is_input"] = tbl["local_idx"].isin(inp)
    tbl["is_readout"] = tbl["local_idx"].isin(rdt)
    top = tbl.sort_values("mean_abs", ascending=False)
    cols = ["root_id", "cell_type", "super_class", "side", "mean_abs", "is_input", "is_readout"]
    top[cols].head(200).to_csv(OUT / f"{RUN}__top_neurons.csv", index=False)
    print("="*88)
    print("(1) MOST-ACTIVATED NEURONS OVERALL (untrained ReLU, whole brain)")
    print("="*88)
    print(top[cols].head(25).to_string(index=False))

    # ---- (2) top neurons at each depth ----
    print("\n" + "="*88)
    print("(2) TOP-5 NEURONS AT EACH RECURRENCE DEPTH t")
    print("="*88)
    rows = []
    for t in range(T):
        a = mas[t]
        idx = np.argsort(a)[::-1][:5]
        for rank, i in enumerate(idx):
            r = nrn.iloc[i]
            rows.append({"t": t + 1, "rank": rank + 1, "root_id": int(r["root_id"]),
                         "cell_type": str(r["cell_type"]), "super_class": str(r["super_class"]),
                         "side": str(r["side"]), "mean_abs": float(a[i]),
                         "is_input": i in inp, "is_readout": i in rdt})
        names = ", ".join(f'{str(nrn.iloc[i]["cell_type"])}({int(nrn.iloc[i]["root_id"])})'
                          for i in idx)
        tag = "IN" if any(i in inp for i in idx) else ("OUT" if any(i in rdt for i in idx) else "")
        print(f"  t={t+1:2d} {tag:3s} | {names}")
    pd.DataFrame(rows).to_csv(OUT / f"{RUN}__per_depth_neurons.csv", index=False)

    # ---- (3) 3D spatial visualization ----
    px = nrn["pos_x"].to_numpy(float); py = nrn["pos_y"].to_numpy(float)
    pz = nrn["pos_z"].to_numpy(float)
    is_in = tbl["is_input"].to_numpy(); is_out = tbl["is_readout"].to_numpy()
    # rank-based size/alpha for active neurons (top 2000 by overall activity)
    active_order = np.argsort(overall)[::-1]
    top_active = active_order[:2000]

    fig = plt.figure(figsize=(16, 7))
    # ---- Panel A: overall activity (input/output marked) ----
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    # faint grey background of all neurons
    ax.scatter(px[::6], py[::6], pz[::6], s=0.5, c="lightgrey", alpha=0.12, linewidths=0)
    # active neurons coloured by activity
    sc = ax.scatter(px[top_active], py[top_active], pz[top_active],
                    c=overall[top_active], cmap="hot", s=6, alpha=0.7, linewidths=0)
    # input photoreceptors (blue) + output readout (lime), marked distinctly
    ax.scatter(px[is_in], py[is_in], pz[is_in], s=10, c="dodgerblue", alpha=0.6,
               linewidths=0, label=f"input: photoreceptors (n={int(is_in.sum())})")
    ax.scatter(px[is_out], py[is_out], pz[is_out], s=10, c="limegreen", alpha=0.5,
               linewidths=0, label=f"output: VPN/descending (n={int(is_out.sum())})")
    ax.set_title("Untrained ReLU connectome — mean |activation| over the unroll\n"
                 "(hot = most active; blue = input, green = output)")
    ax.legend(loc="upper left", fontsize=8, markerscale=2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="mean |activation|")

    # ---- Panel B: depth coloring (which hop a neuron peaks at = the wavefront) ----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    peak_step = mas.argmax(axis=0) + 1                 # [N] the step each neuron peaks
    # only show neurons that ever meaningfully activate
    show = overall > np.quantile(overall, 0.90)
    ax2.scatter(px[::6], py[::6], pz[::6], s=0.5, c="lightgrey", alpha=0.1, linewidths=0)
    sc2 = ax2.scatter(px[show], py[show], pz[show], c=peak_step[show], cmap="viridis",
                      s=6, alpha=0.7, linewidths=0, vmin=1, vmax=T)
    ax2.scatter(px[is_in], py[is_in], pz[is_in], s=10, c="red", alpha=0.5, linewidths=0,
                label="input")
    ax2.scatter(px[is_out], py[is_out], pz[is_out], s=10, marker="^", c="black",
                alpha=0.4, linewidths=0, label="output")
    ax2.set_title("Activation wavefront: recurrence step at which each neuron peaks\n"
                  "(dark = early/input side, bright = late/output side)")
    ax2.legend(loc="upper left", fontsize=8, markerscale=2)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    fig.colorbar(sc2, ax=ax2, shrink=0.5, label="peak recurrence step t")

    fig.tight_layout()
    out_png = FIG / "relu_untrained_activity_3d.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\n(3) saved 3D visualization -> {out_png}")

    # small JSON summary
    json.dump({
        "run": RUN, "test_acc": 0.9818,
        "top10_overall": top[cols].head(10).to_dict("records"),
    }, open(OUT / f"{RUN}__deepdive.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
