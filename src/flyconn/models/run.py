"""Stage-3 CLI: build/smoke/train connectome-constrained MNIST models.

Usage:
  python -m flyconn.models.run train   --config configs/experiments/optic_left_ff_initA.yaml
  python -m flyconn.models.run build   --config ...     # construct model, print info, no train
  python -m flyconn.models.run smoke   --config ...     # 1 epoch, tiny, sanity
  python -m flyconn.models.run grid    --out configs/experiments   # (re)generate the 24 leaf configs

A leaf config is merged on top of configs/model_base.yaml.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

from ..paths import REPO_ROOT

BASE = REPO_ROOT / "configs" / "model_base.yaml"

SUBGRAPHS = ["whole", "whole_right", "whole_left", "optic", "optic_right", "optic_left"]
ARCHES = ["ff_unroll", "rnn"]
INITS = {"initA": "from_data", "initB": "random"}


def load_merged(config_path: str | Path) -> dict:
    with open(BASE) as fh:
        cfg = yaml.safe_load(fh) or {}
    with open(config_path) as fh:
        leaf = yaml.safe_load(fh) or {}
    cfg.update(leaf)
    return cfg


def generate_grid(out_dir: str | Path) -> list[str]:
    """Write the 24 leaf experiment configs (subgraph x arch x init)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for sg, arch, (itag, imode) in itertools.product(SUBGRAPHS, ARCHES, INITS.items()):
        name = f"{sg}_{arch}_{itag}"
        leaf = {"run_name": name, "subgraph_id": sg, "arch": arch, "init_mode": imode}
        p = out / f"{name}.yaml"
        with open(p, "w") as fh:
            yaml.safe_dump(leaf, fh, sort_keys=False)
        written.append(str(p))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flyconn.models.run")
    ap.add_argument("stage", choices=["build", "smoke", "train", "grid", "aggregate"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "configs" / "experiments"))
    args = ap.parse_args(argv)

    if args.stage == "grid":
        paths = generate_grid(args.out)
        print(f"[grid] wrote {len(paths)} configs to {args.out}")
        return 0

    if args.stage == "aggregate":
        from .aggregate import write_summary
        write_summary()
        return 0

    if not args.config:
        ap.error("--config is required for build/smoke/train")
    cfg = load_merged(args.config)

    if args.stage == "build":
        import torch
        from .train import build_classifier
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, info = build_classifier(cfg, device=device)
        import json
        print(json.dumps(info, indent=2, default=str))
        return 0

    if args.stage == "smoke":
        cfg = {**cfg, "epochs": 1, "batch_size": cfg.get("batch_size", 128)}

    from .train import train
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
