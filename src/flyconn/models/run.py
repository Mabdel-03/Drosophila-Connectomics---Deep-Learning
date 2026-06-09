"""Stage-3 CLI: build/smoke/train connectome-constrained MNIST models.

Usage:
  python -m flyconn.models.run train     --config configs/experiments/optic_left_ff_initA.yaml
  python -m flyconn.models.run fit_rigid --config configs/eye_experiments/optic_left_V3.yaml
  python -m flyconn.models.run build     --config ...     # construct model, print info, no train
  python -m flyconn.models.run smoke     --config ...     # 1 epoch, tiny, sanity
  python -m flyconn.models.run grid      --out configs/experiments     # the 24 stage-3 configs
  python -m flyconn.models.run eye_grid  --out configs/eye_experiments # the faithful-eye configs

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


# Faithful-eye variants. V1 trains the core; V2/V3 freeze it. All use the rigid eye and
# the from_data init (the "as-measured" wiring). The 'stage' field routes to train vs
# fit_rigid in train_array; it is informational in the leaf config.
EYE_VARIANTS = {
    "V1": {"eye": "rigid", "learn_core": True,  "learn_readout": True,  "decision": "linear"},
    "V2": {"eye": "rigid", "learn_core": False, "learn_readout": True,  "decision": "linear"},
    "V3": {"eye": "rigid", "learn_core": False, "learn_readout": False, "decision": "ncm"},
}


def generate_eye_grid(out_dir: str | Path, *, full: bool = False) -> list[str]:
    """Write faithful-eye leaf configs.

    Default (phased) = Phase 1: the cheapest, most-informative subset that answers
    "is there signal?" — V2+V3 x {sign inherit, -1} x {optic_left, optic_right} plus
    V1 x {sign -1} x {optic_left, optic_right}. ``full=True`` expands to all variants x
    all subgraphs x {inherit, -1} (the Phase-2 superset).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arch = "ff_unroll"          # FF is the natural feedforward visual sweep; cheapest core
    init = "from_data"          # the "as-measured" wiring (V2/V3 require it)
    written = []

    def emit(variant, sg, sign):
        flags = EYE_VARIANTS[variant]
        sign_tag = "fix" if sign == -1 else "raw"
        name = f"{sg}_{variant}_{sign_tag}"
        leaf = {"run_name": name, "subgraph_id": sg, "arch": arch,
                "init_mode": init, "photoreceptor_sign": sign,
                "stage": "fit_rigid" if variant == "V3" else "train", **flags}
        p = out / f"{name}.yaml"
        with open(p, "w") as fh:
            yaml.safe_dump(leaf, fh, sort_keys=False)
        written.append(str(p))

    if full:
        for sg in SUBGRAPHS:
            for variant in EYE_VARIANTS:
                for sign in ("inherit", -1):
                    emit(variant, sg, sign)
    else:
        small = ["optic_left", "optic_right"]
        for sg in small:
            for sign in ("inherit", -1):
                emit("V2", sg, sign)
                emit("V3", sg, sign)
            emit("V1", sg, -1)            # V1 only with the faithful sign (it's the slow one)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flyconn.models.run")
    ap.add_argument("stage", choices=["build", "smoke", "train", "fit_rigid",
                                       "grid", "eye_grid", "aggregate"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "configs" / "experiments"))
    ap.add_argument("--full", action="store_true",
                    help="eye_grid: emit the full Phase-2 superset instead of Phase 1")
    args = ap.parse_args(argv)

    if args.stage == "grid":
        paths = generate_grid(args.out)
        print(f"[grid] wrote {len(paths)} configs to {args.out}")
        return 0

    if args.stage == "eye_grid":
        out = args.out
        if out == str(REPO_ROOT / "configs" / "experiments"):
            out = str(REPO_ROOT / "configs" / "eye_experiments")
        paths = generate_eye_grid(out, full=args.full)
        print(f"[eye_grid] wrote {len(paths)} configs to {out} "
              f"({'full' if args.full else 'phase-1'})")
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

    if args.stage == "fit_rigid":
        from .train import fit_rigid
        fit_rigid(cfg)
        return 0

    from .train import train
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
