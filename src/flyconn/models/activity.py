"""Rank neurons by firing activity in a trained connectome model.

The model's hidden state ``h`` is a per-neuron activation vector ([B, N], one continuous
rate per neuron). "Most frequently firing" = highest mean activation over a batch of test
images. This module reloads a trained run, pushes the test set through it while capturing
``h`` at every recurrence step, and ranks neurons (and aggregated cell types) by activity.

Because the connectome is signed (inhibitory neurons go strongly negative), we report BOTH
the signed mean and the mean magnitude (|h|). We aggregate over the whole T-step unroll
(``mode='accum'``-style), since with ff_unroll a neuron's signal peaks at its BFS-hop step.

Usage:
    python -m flyconn.models.activity --run eyeTact_whole_learned_relu_trained_T08
    python -m flyconn.models.activity --run <name> --out-dir <dir> --max-batches 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..paths import data_root
from .connectome_net import ConnectomeClassifier, RigidEyeClassifier


def _inject(model, pixels: torch.Tensor) -> torch.Tensor:
    """Build the external-current vector x:[B,N] for either classifier type."""
    if isinstance(model, RigidEyeClassifier):
        return model.eye(pixels)                                   # [B, N]
    # ConnectomeClassifier: encode -> scatter onto photoreceptor input nodes.
    cur = model.encoder(pixels)                                    # [B, n_input]
    x = pixels.new_zeros(pixels.shape[0], model.core.N)
    x[:, model.input_local] = cur
    return x


@torch.no_grad()
def neuron_activity(run_name: str, *, device: str | None = None,
                    max_batches: int | None = None, batch_size: int = 256) -> pd.DataFrame:
    """Return a per-neuron activity table for a trained run, ranked by mean |activation|.

    Columns: local_idx, root_id, cell_type, super_class, side, mean_act (signed),
    mean_abs (|h|), peak_abs (max |h| over the unroll), frac_active (fraction of images
    where |h| exceeds the per-neuron median |h| — a crude "firing frequency").
    """
    from .train import build_classifier
    from .data import get_loaders

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = data_root() / "v783" / "models" / run_name
    ckpt = torch.load(run / "ckpt_best.pt", map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    model, sub, info = build_classifier(cfg, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, _, test = get_loaders(batch_size=batch_size,
                             dataset=cfg.get("dataset", "mnist"),
                             color=cfg.get("color", "luma"),
                             num_workers=cfg.get("num_workers", 4),
                             download=cfg.get("download", True))

    N, T = sub.N, model.core.cfg.T
    w = model.core.edge_weight()
    persistent = model.core.cfg.inject == "persistent"

    # Streaming accumulators over the whole test set.
    sum_signed = torch.zeros(N, device=device)         # sum of trajectory-mean h
    sum_abs = torch.zeros(N, device=device)            # sum of trajectory-mean |h|
    peak_abs = torch.zeros(N, device=device)           # running max |h| anywhere
    n_imgs = 0
    for bi, (pixels, _) in enumerate(test):
        if max_batches is not None and bi >= max_batches:
            break
        pixels = pixels.to(device)
        x = _inject(model, pixels)                     # [B, N] external current
        h = x.new_zeros(pixels.shape[0], N)
        zero = torch.zeros_like(x)
        traj_abs = torch.zeros_like(h)                 # accumulate |h| across the unroll
        traj_signed = torch.zeros_like(h)
        for t in range(T):
            x_t = x if (persistent or t == 0) else zero
            h = model.core._step(h, x_t, w)
            traj_signed += h
            ah = h.abs()
            traj_abs += ah
            peak_abs = torch.maximum(peak_abs, ah.max(dim=0).values)
        # per-image trajectory means, then sum over images
        sum_signed += (traj_signed / T).sum(dim=0)
        sum_abs += (traj_abs / T).sum(dim=0)
        n_imgs += pixels.shape[0]

    mean_signed = (sum_signed / n_imgs).cpu().numpy()
    mean_abs = (sum_abs / n_imgs).cpu().numpy()
    peak = peak_abs.cpu().numpy()

    nrn = sub.neurons.reset_index(drop=True)
    out = pd.DataFrame({
        "local_idx": np.arange(N),
        "root_id": nrn["root_id"].to_numpy(),
        "cell_type": nrn["cell_type"].astype(str).to_numpy(),
        "super_class": nrn["super_class"].astype(str).to_numpy(),
        "side": nrn["side"].astype(str).to_numpy(),
        "mean_act": mean_signed,
        "mean_abs": mean_abs,
        "peak_abs": peak,
    })
    return out.sort_values("mean_abs", ascending=False).reset_index(drop=True)


@torch.no_grad()
def per_step_activity(run_name: str, *, device: str | None = None,
                      max_batches: int | None = 8, batch_size: int = 256,
                      top_k: int = 6) -> pd.DataFrame:
    """Per-recurrence-step activation wavefront for a trained run.

    For each step t = 1..T, report the mean |activation| of every cell TYPE and which types
    are newly recruited at that depth (their first step above a small threshold). This shows
    how activity propagates one synaptic hop per step from the photoreceptors toward the
    readout. Returns a tidy DataFrame: rows (t, cell_type) with mean_abs at that step, plus a
    helper column ``readout`` flagging the readout super-classes.

    Columns: t, cell_type, super_class, n, mean_abs (avg |h| over neurons of that type at step
    t), is_readout. The caller can pivot to "top-k types per depth".
    """
    from .train import build_classifier
    from .data import get_loaders

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = data_root() / "v783" / "models" / run_name
    ckpt = torch.load(run / "ckpt_best.pt", map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model, sub, info = build_classifier(cfg, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _, _, test = get_loaders(batch_size=batch_size,
                             dataset=cfg.get("dataset", "mnist"),
                             color=cfg.get("color", "luma"),
                             num_workers=cfg.get("num_workers", 4),
                             download=cfg.get("download", True))

    N, T = sub.N, model.core.cfg.T
    w = model.core.edge_weight()
    persistent = model.core.cfg.inject == "persistent"

    # sum_abs_step[t] = sum over images of |h_t| per neuron -> mean after.
    sum_abs_step = torch.zeros(T, N, device=device)
    n_imgs = 0
    for bi, (pixels, _) in enumerate(test):
        if max_batches is not None and bi >= max_batches:
            break
        pixels = pixels.to(device)
        x = _inject(model, pixels)
        h = x.new_zeros(pixels.shape[0], N)
        zero = torch.zeros_like(x)
        for t in range(T):
            x_t = x if (persistent or t == 0) else zero
            h = model.core._step(h, x_t, w)
            sum_abs_step[t] += h.abs().sum(dim=0)
        n_imgs += pixels.shape[0]
    mean_abs_step = (sum_abs_step / n_imgs).cpu().numpy()        # [T, N]

    nrn = sub.neurons.reset_index(drop=True)
    ct = nrn["cell_type"].astype(str).to_numpy()
    sc = nrn["super_class"].astype(str).to_numpy()
    # which super-classes are the biological readout for this subgraph
    from . import io_inject
    rdt = set(io_inject.readout_local_ids(sub).tolist())
    is_rdt_type = np.zeros(N, bool); is_rdt_type[list(rdt)] = True

    rows = []
    for t in range(T):
        df = pd.DataFrame({"cell_type": ct, "super_class": sc,
                           "mean_abs": mean_abs_step[t],
                           "is_readout_node": is_rdt_type})
        g = df.groupby("cell_type").agg(
            super_class=("super_class", "first"),
            n=("mean_abs", "size"),
            mean_abs=("mean_abs", "mean"),
            readout_frac=("is_readout_node", "mean"),
        ).reset_index()
        g["t"] = t + 1                                           # 1-indexed recurrence step
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    return out[["t", "cell_type", "super_class", "n", "mean_abs", "readout_frac"]]


@torch.no_grad()
def per_step_neuron_activity(run_name: str, *, device: str | None = None,
                             max_batches: int | None = 8, batch_size: int = 256):
    """Per-(step, neuron) mean |activation| for a run, plus the labelled neuron table.

    Returns (mean_abs_step [T, N] numpy, neurons DataFrame in local order with root_id /
    cell_type / super_class / side / pos_x,y,z, input_ids, readout_ids). This is the raw
    material for ranking exact neurons at each depth AND for the 3D spatial plot.
    """
    from .train import build_classifier
    from .data import get_loaders
    from . import io_inject

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = data_root() / "v783" / "models" / run_name
    ckpt = torch.load(run / "ckpt_best.pt", map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model, sub, info = build_classifier(cfg, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _, _, test = get_loaders(batch_size=batch_size,
                             dataset=cfg.get("dataset", "mnist"),
                             color=cfg.get("color", "luma"),
                             num_workers=cfg.get("num_workers", 4),
                             download=cfg.get("download", True))

    N, T = sub.N, model.core.cfg.T
    w = model.core.edge_weight()
    persistent = model.core.cfg.inject == "persistent"
    sum_abs_step = torch.zeros(T, N, device=device)
    n_imgs = 0
    for bi, (pixels, _) in enumerate(test):
        if max_batches is not None and bi >= max_batches:
            break
        pixels = pixels.to(device)
        x = _inject(model, pixels)
        h = x.new_zeros(pixels.shape[0], N)
        zero = torch.zeros_like(x)
        for t in range(T):
            x_t = x if (persistent or t == 0) else zero
            h = model.core._step(h, x_t, w)
            sum_abs_step[t] += h.abs().sum(dim=0)
        n_imgs += pixels.shape[0]
    mean_abs_step = (sum_abs_step / n_imgs).cpu().numpy()        # [T, N]

    nrn = sub.neurons.reset_index(drop=True).copy()
    nrn["local_idx"] = np.arange(N)
    return {
        "mean_abs_step": mean_abs_step,                          # [T, N]
        "neurons": nrn,
        "input_ids": io_inject.input_local_ids(sub),             # photoreceptors
        "readout_ids": io_inject.readout_local_ids(sub),         # VPN + descending
        "T": T,
    }


def celltype_summary(table: pd.DataFrame, by: str = "cell_type") -> pd.DataFrame:
    """Aggregate neuron activity to cell types: total + mean |activation| and neuron count."""
    g = table.groupby(by).agg(
        n=("local_idx", "size"),
        total_abs=("mean_abs", "sum"),
        mean_abs=("mean_abs", "mean"),
        mean_signed=("mean_act", "mean"),
    ).sort_values("total_abs", ascending=False)
    return g.reset_index()


def run(run_name: str, *, out_dir: str | None = None, top: int = 40,
        max_batches: int | None = None) -> dict:
    """Compute + save the activity ranking for a run. Returns a small summary dict."""
    table = neuron_activity(run_name, max_batches=max_batches)
    by_type = celltype_summary(table, "cell_type")
    by_super = celltype_summary(table, "super_class")

    out = Path(out_dir) if out_dir else (data_root() / "v783" / "activity")
    out.mkdir(parents=True, exist_ok=True)
    table.head(500).to_csv(out / f"{run_name}__neurons.csv", index=False)
    by_type.to_csv(out / f"{run_name}__celltypes.csv", index=False)
    by_super.to_csv(out / f"{run_name}__superclass.csv", index=False)

    summary = {
        "run": run_name,
        "top_neurons": table.head(top)[
            ["root_id", "cell_type", "super_class", "side", "mean_abs", "mean_act"]
        ].to_dict("records"),
        "top_cell_types": by_type.head(20).to_dict("records"),
        "top_super_classes": by_super.head(12).to_dict("records"),
    }
    with open(out / f"{run_name}__activity.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[activity] {run_name}: wrote rankings to {out}")
    print(f"  top cell types: "
          + ", ".join(f"{r['cell_type']}({r['total_abs']:.1f})"
                      for r in summary['top_cell_types'][:8]))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flyconn.models.activity")
    ap.add_argument("--run", required=True, help="run_name under v783/models/")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="limit test batches (None = full test set)")
    args = ap.parse_args(argv)
    run(args.run, out_dir=args.out_dir, top=args.top, max_batches=args.max_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
