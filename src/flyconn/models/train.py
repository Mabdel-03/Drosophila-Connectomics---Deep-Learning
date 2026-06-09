"""Build + train a connectome-constrained MNIST classifier from a config dict.

Glues subgraph -> ConnectomeNet -> init mode -> I/O sets -> MNIST loop. Checkpoints,
metrics, and the resolved config are written to scratch under
``$FLYCONN_DATA_ROOT/v783/models/<run_name>/``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..paths import data_root
from . import io_inject
from .connectome_net import ConnectomeClassifier, ConnectomeNet, NetConfig
from .init_modes import apply_init
from .subgraphs import build_subgraph


def run_dir(run_name: str) -> Path:
    d = data_root() / "v783" / "models" / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_classifier(cfg: dict, device: str = "cpu"):
    """Construct the full classifier + return (model, sub, info) from a config dict."""
    subgraph_id = cfg["subgraph_id"]
    arch = cfg["arch"]
    init_mode = cfg["init_mode"]
    policy = cfg.get("policy", "flyvis_standard")

    sub = build_subgraph(subgraph_id, policy=policy)
    # Start from per-arch defaults, then apply explicit overrides from the config.
    net_cfg = NetConfig.for_arch(arch)
    if cfg.get("alpha") is not None:
        net_cfg.alpha = float(cfg["alpha"])
    if cfg.get("nonlinearity"):
        net_cfg.nonlinearity = cfg["nonlinearity"]
    if cfg.get("target_radius") is not None:
        net_cfg.target_radius = float(cfg["target_radius"])
    if cfg.get("T") is not None:
        net_cfg.T = int(cfg["T"])

    input_ids = io_inject.input_local_ids(sub)
    readout_ids = io_inject.readout_local_ids(sub)
    if input_ids.size == 0:
        raise ValueError(f"{subgraph_id}: no photoreceptor input nodes found")

    # Recommend / validate T against BFS depth to readout.
    rec_T, bfs = io_inject.recommend_T(sub, input_ids, readout_ids,
                                       floor=cfg.get("T_floor", 10))
    if cfg.get("T") is None:
        net_cfg.T = rec_T
    elif net_cfg.T < (bfs["max_hops"] or 0):
        print(f"[warn] T={net_cfg.T} < BFS max_hops={bfs['max_hops']}; readout may be starved")

    core = ConnectomeNet(sub, net_cfg).to(device)
    init_report = apply_init(core, sub, init_mode,
                             alpha0=cfg.get("alpha0", 0.01), seed=cfg.get("seed", 0))
    model = ConnectomeClassifier(core, input_ids, readout_ids).to(device)

    info = {
        "subgraph_id": subgraph_id, "N": sub.N, "E": sub.E, "arch": arch,
        "init_mode": init_mode, "net_cfg": asdict(net_cfg),
        "n_input": int(input_ids.size), "n_readout": int(readout_ids.size),
        "bfs": bfs, "init": init_report,
    }
    return model, sub, info


def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(crit(logits, y))
            correct += int((logits.argmax(1) == y).sum())
            total += y.numel()
    return correct / max(total, 1), loss_sum / max(total, 1)


def train(cfg: dict) -> dict:
    """Full training run. Returns a summary dict; writes artifacts to scratch."""
    from .data import get_loaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = cfg.get("run_name") or f"{cfg['subgraph_id']}_{cfg['arch']}_{cfg['init_mode']}"
    out = run_dir(run_name)
    torch.manual_seed(cfg.get("seed", 0))

    model, sub, info = build_classifier(cfg, device=device)
    print(f"[train] {run_name}  N={info['N']} E={info['E']} "
          f"T={info['net_cfg']['T']} device={device}")
    print(f"[train] BFS to readout: {info['bfs']}  init rho_after={info['init'].get('rho_after')}")

    epochs = int(cfg.get("epochs", 20))
    batch = int(cfg.get("batch_size", 128))
    tr, va, te = get_loaders(batch_size=batch, num_workers=cfg.get("num_workers", 4),
                             download=cfg.get("download", True))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    clip = cfg.get("grad_clip", 1.0)

    metrics_path = out / "metrics.jsonl"
    best_val = -1.0
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        sched.step()
        val_acc, val_loss = _evaluate(model, va, device)
        rec = {"epoch": ep, "val_acc": val_acc, "val_loss": val_loss,
               "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[train] ep{ep:02d} val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg, "info": info},
                       out / "ckpt_best.pt")

    test_acc, test_loss = _evaluate(model, te, device)
    summary = {"run_name": run_name, **{k: cfg.get(k) for k in
               ("subgraph_id", "arch", "init_mode")},
               "N": info["N"], "E": info["E"], "T": info["net_cfg"]["T"],
               "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
               "best_val_acc": best_val, "test_acc": test_acc, "test_loss": test_loss}
    with open(out / "summary.json", "w") as fh:
        json.dump({**summary, "info": info, "history": history}, fh, indent=2, default=str)
    # Resolved config for reproducibility.
    with open(out / "config_resolved.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    print(f"[train] DONE {run_name}: best_val={best_val:.4f} test={test_acc:.4f}")
    return summary
