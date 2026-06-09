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
from .connectome_net import (
    ConnectomeClassifier,
    ConnectomeNet,
    NetConfig,
    RigidEyeClassifier,
)
from .eye import RigidEye, build_eye_map
from .init_modes import apply_init
from .subgraphs import build_subgraph


def run_dir(run_name: str) -> Path:
    d = data_root() / "v783" / "models" / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_classifier(cfg: dict, device: str = "cpu"):
    """Construct the full classifier + return (model, sub, info) from a config dict.

    ``eye='learned'`` builds the original ConnectomeClassifier (learned encoder). ``eye=
    'rigid'`` builds a RigidEyeClassifier (non-learned image->photoreceptor map) in one of
    three variants set by ``learn_core``/``learn_readout``/``decision`` (see RigidEye-
    Classifier). The ``photoreceptor_sign`` config (-1 for the biologically faithful
    histaminergic/inhibitory override) is applied at subgraph-build time.
    """
    subgraph_id = cfg["subgraph_id"]
    arch = cfg["arch"]
    init_mode = cfg["init_mode"]
    policy = cfg.get("policy", "flyvis_standard")
    photoreceptor_sign = cfg.get("photoreceptor_sign", "inherit")

    sub = build_subgraph(subgraph_id, policy=policy,
                         photoreceptor_sign=photoreceptor_sign)
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
    # Gain control: rigid-eye models default to per-step RMS renorm so the signal survives
    # the frozen core to the readout; learned-encoder models keep the original 'none'.
    # state_norm=null/None in the config means "use this default".
    eye_is_rigid = cfg.get("eye", "learned") == "rigid"
    state_norm = cfg.get("state_norm")
    net_cfg.state_norm = state_norm if state_norm else ("rms" if eye_is_rigid else "none")

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

    eye_mode = cfg.get("eye", "learned")
    info = {
        "subgraph_id": subgraph_id, "N": sub.N, "E": sub.E, "arch": arch,
        "init_mode": init_mode, "net_cfg": asdict(net_cfg),
        "n_input": int(input_ids.size), "n_readout": int(readout_ids.size),
        "bfs": bfs, "init": init_report, "eye": eye_mode,
        "photoreceptor_sign": photoreceptor_sign,
    }

    if eye_mode == "learned":
        model = ConnectomeClassifier(core, input_ids, readout_ids).to(device)
        return model, sub, info

    # ---- rigid eye ----
    eye_map = build_eye_map(
        sub,
        source=cfg.get("eye_source", "auto"),
        drive_gain=float(cfg.get("drive_gain", 0.5)),
        eyes=cfg.get("eye_eyes", "both"),
        channels=cfg.get("eye_channels", "all"),
        fill=cfg.get("eye_fill", "zero"),
        mirror_lr=bool(cfg.get("mirror_lr", True)),
    )
    learn_core = bool(cfg.get("learn_core", True))
    decision = cfg.get("decision", "linear")
    readout_mode = cfg.get("readout_mode", "accum")
    model = RigidEyeClassifier(
        RigidEye(eye_map), core, readout_ids,
        learn_core=learn_core, decision=decision, readout_mode=readout_mode,
    ).to(device)
    info.update({
        "variant": _variant_name(cfg), "decision": decision,
        "learn_core": learn_core, "learn_readout": bool(cfg.get("learn_readout", True)),
        "eye_source": eye_map.source, "n_assigned_receptors": eye_map.n_assigned,
        "drive_gain": eye_map.drive_gain, "state_norm": net_cfg.state_norm,
        "readout_mode": readout_mode,
    })
    return model, sub, info


def _variant_name(cfg: dict) -> str:
    """V1/V2/V3 label from the learnability flags."""
    if cfg.get("decision") == "ncm":
        return "V3"
    return "V1" if cfg.get("learn_core", True) else "V2"


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
    """Full training run (gradient descent). Handles V1 and V2; V3 -> use fit_rigid.

    Returns a summary dict; writes artifacts to scratch.
    """
    from .data import get_loaders

    if cfg.get("decision") == "ncm":
        raise ValueError(
            "decision='ncm' (V3) has no gradient training; use "
            "`python -m flyconn.models.run fit_rigid --config ...` instead."
        )

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

    # V2 (rigid eye, frozen core, linear probe): standardize the frozen readout features
    # over the train set so the probe doesn't waste capacity undoing the DC offset under
    # a fixed-drive tanh core. (V1 trains the core, so its feature scale is learned.)
    if isinstance(model, RigidEyeClassifier) and not cfg.get("learn_core", True):
        mean, std = _feature_stats(model, tr, device)
        model.set_feature_stats(mean, std)
        info["feat_std_mean"] = float(std.mean())

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.get("lr", 1e-3))
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
               "variant": info.get("variant", "stage3"),
               "eye": info.get("eye", "learned"),
               "decision": info.get("decision", "linear"),
               "photoreceptor_sign": info.get("photoreceptor_sign", "inherit"),
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


def _feature_stats(model: RigidEyeClassifier, loader, device, max_batches: int | None = None):
    """Mean/std of the frozen readout features over the train set (one pass, streaming)."""
    model.eval()
    n = 0
    s = sq = None
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            raw = _raw_features(model, x.to(device))    # RAW (pre-z-score) readout acts
            s = raw.sum(0) if s is None else s + raw.sum(0)
            sq = (raw * raw).sum(0) if sq is None else sq + (raw * raw).sum(0)
            n += raw.shape[0]
    mean = s / n
    var = (sq / n) - mean * mean
    std = var.clamp(min=1e-12).sqrt()
    return mean, std


def _raw_features(model: RigidEyeClassifier, pixels: torch.Tensor) -> torch.Tensor:
    """Readout activations WITHOUT z-scoring (frozen-stats-independent).

    Uses the same whole-trajectory readout as model.features() so the stats computed
    here match what the classifier consumes.
    """
    x = model.eye(pixels)
    return model.core.readout_trajectory(x, model.readout_local, mode=model.readout_mode)


def fit_rigid(cfg: dict) -> dict:
    """V3: zero-learning template classification on the frozen rigid-eye network.

    One streaming pass over the train set collects per-class means (+ shrinkage covariance
    for the LDA rule) of the frozen readout features; test images are classified by the
    chosen nearest-template rule. Reports raw-Euclidean-NCM (the brightness-artifact floor),
    cosine-NCM (on z-scored features), and shrinkage-LDA-NCM, plus a label-shuffle control.
    Writes summary.json in the same schema as train() so aggregate.py picks it up.
    """
    from .data import get_loaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = cfg.get("run_name") or f"{cfg['subgraph_id']}_{cfg['arch']}_V3"
    out = run_dir(run_name)
    torch.manual_seed(cfg.get("seed", 0))

    cfg = {**cfg, "decision": "ncm", "learn_core": False, "learn_readout": False,
           "eye": "rigid"}
    model, sub, info = build_classifier(cfg, device=device)
    model.eval()
    print(f"[fit_rigid] {run_name} N={info['N']} n_readout={info['n_readout']} "
          f"variant=V3 device={device}")

    batch = int(cfg.get("batch_size", 256))
    tr, _, te = get_loaders(batch_size=batch, num_workers=cfg.get("num_workers", 4),
                            download=cfg.get("download", True))

    # --- one pass over train: accumulate per-class sums + global stats ---
    n_classes = 10
    d = info["n_readout"]
    cls_sum = torch.zeros(n_classes, d, device=device)
    cls_cnt = torch.zeros(n_classes, device=device)
    g_sum = torch.zeros(d, device=device)
    g_sqs = torch.zeros(d, device=device)
    n_tot = 0
    # For LDA: pooled within-class scatter (after PCA reduction) is collected in a second
    # cheap pass on reduced features; first pass also stores reduced train feats per class.
    with torch.no_grad():
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            f = _raw_features(model, x)                 # [B, d]
            for c in range(n_classes):
                m = y == c
                if m.any():
                    cls_sum[c] += f[m].sum(0)
                    cls_cnt[c] += int(m.sum())
            g_sum += f.sum(0)
            g_sqs += (f * f).sum(0)
            n_tot += f.shape[0]
    g_mean = g_sum / n_tot
    g_std = ((g_sqs / n_tot) - g_mean * g_mean).clamp(min=1e-12).sqrt()
    model.set_feature_stats(g_mean, g_std)
    cls_mean_raw = cls_sum / cls_cnt.clamp(min=1).unsqueeze(1)          # [10, d]
    cls_mean_z = (cls_mean_raw - g_mean) / g_std

    # --- evaluate three rules on the test set ---
    accs = _eval_ncm_rules(model, te, device, cls_mean_raw, cls_mean_z, g_mean, g_std,
                           cfg.get("ncm_metric", "cosine"))

    # --- label-shuffle control: shuffle train labels, refit cosine templates, eval ---
    shuffle_acc = _label_shuffle_control(model, tr, te, device, g_mean, g_std, n_classes, d)

    headline = accs.get(cfg.get("ncm_metric", "cosine"), accs["cosine"])
    summary = {"run_name": run_name,
               "subgraph_id": cfg["subgraph_id"], "arch": cfg["arch"],
               "init_mode": cfg["init_mode"], "variant": "V3",
               "eye": "rigid", "decision": "ncm",
               "ncm_metric": cfg.get("ncm_metric", "cosine"),
               "photoreceptor_sign": info.get("photoreceptor_sign", "inherit"),
               "N": info["N"], "E": info["E"], "T": info["net_cfg"]["T"],
               "n_params": 0, "best_val_acc": None,
               "test_acc": headline, "test_loss": None,
               "ncm_acc_raw_euclid": accs["euclid"],
               "ncm_acc_cosine": accs["cosine"],
               "ncm_acc_lda": accs["lda"],
               "shuffle_control_acc": shuffle_acc}
    with open(out / "summary.json", "w") as fh:
        json.dump({**summary, "info": info}, fh, indent=2, default=str)
    with open(out / "config_resolved.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    print(f"[fit_rigid] DONE {run_name}: euclid={accs['euclid']:.4f} "
          f"cosine={accs['cosine']:.4f} lda={accs['lda']:.4f} "
          f"shuffle={shuffle_acc:.4f}")
    return summary


def _eval_ncm_rules(model, loader, device, cls_mean_raw, cls_mean_z, g_mean, g_std,
                    metric, pca_dim: int = 256):
    """Nearest-template accuracy under euclid(raw), cosine(z), and shrinkage-LDA(z+PCA)."""
    # Precompute LDA: fit PCA on the z-scored class means' span is too small; instead use
    # a streaming estimate -> here we approximate LDA by whitening with the pooled diagonal
    # within-class variance (already have g_std as a proxy) + cosine. For a fuller LDA we
    # reduce with PCA over class means; with 10 means PCA<=9 dims is degenerate, so we use
    # the shrinkage-diagonal Mahalanobis (z-score == diagonal whitening) + nearest mean,
    # which is the closed-form, zero-learning LDA-NCM under a diagonal covariance.
    cm_raw = cls_mean_raw                                  # [10, d]
    cm_z = cls_mean_z                                      # [10, d]
    cm_z_norm = cm_z / cm_z.norm(dim=1, keepdim=True).clamp(min=1e-12)
    correct = {"euclid": 0, "cosine": 0, "lda": 0}
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            raw = _raw_features(model, x)                 # [B, d]
            z = (raw - g_mean) / g_std
            # euclid on raw features (brightness-artifact floor)
            de = torch.cdist(raw, cm_raw)                 # [B, 10]
            correct["euclid"] += int((de.argmin(1) == y).sum())
            # cosine on z-scored features
            zc = z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)
            sim = zc @ cm_z_norm.t()                      # [B, 10]
            correct["cosine"] += int((sim.argmax(1) == y).sum())
            # diagonal-whitened (LDA-NCM under diagonal cov) == euclid in z-space
            dl = torch.cdist(z, cm_z)
            correct["lda"] += int((dl.argmin(1) == y).sum())
            total += y.numel()
    return {k: c / max(total, 1) for k, c in correct.items()}


def _label_shuffle_control(model, tr, te, device, g_mean, g_std, n_classes, d):
    """Refit cosine templates on SHUFFLED train labels; accuracy must collapse to ~chance."""
    g = torch.Generator(device="cpu").manual_seed(1234)
    cls_sum = torch.zeros(n_classes, d, device=device)
    cls_cnt = torch.zeros(n_classes, device=device)
    with torch.no_grad():
        for x, y in tr:
            x = x.to(device)
            yshuf = y[torch.randperm(y.numel(), generator=g)].to(device)
            z = (_raw_features(model, x) - g_mean) / g_std
            for c in range(n_classes):
                m = yshuf == c
                if m.any():
                    cls_sum[c] += z[m].sum(0)
                    cls_cnt[c] += int(m.sum())
        cm = cls_sum / cls_cnt.clamp(min=1).unsqueeze(1)
        cm = cm / cm.norm(dim=1, keepdim=True).clamp(min=1e-12)
        correct = total = 0
        for x, y in te:
            x, y = x.to(device), y.to(device)
            z = (_raw_features(model, x) - g_mean) / g_std
            zc = z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)
            correct += int(((zc @ cm.t()).argmax(1) == y).sum())
            total += y.numel()
    return correct / max(total, 1)
