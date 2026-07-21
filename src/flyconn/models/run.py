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

# Family 3 "recmul": an RNN OF the connectome — repeatedly multiply the node state by the
# connectivity matrix W (T=10). Three step rules, frozen-W vs trainable-W, from_data/random.
RECMUL_SUBGRAPHS = ["optic_left", "optic_right"]
# display rule -> (dynamics, state_norm, target_radius). 'linear' is the raw pure-matmul (set
# rho=1 so the repeated multiply is scale-neutral); 'linear_rms' bounds it with per-step RMS
# renorm; 'tanh' is the bounded leaky-integrator step reused from Families 1/2.
RECMUL_RULES = {
    "linear":     {"dynamics": "linear", "state_norm": None,  "target_radius": 1.0},
    "linear_rms": {"dynamics": "linear", "state_norm": "rms", "target_radius": 0.9},
    "tanh":       {"dynamics": "tanh",   "state_norm": None,  "target_radius": 0.9},
}
RECMUL_CORES = {"frozenW": False, "trainW": True}   # run-name tag -> learn_core

# Family 4 "eyeTact": a sweep over the FULL connectome (whole, N=139,255, signed+weighted)
# crossing input front-end x recurrence depth T(1..12) x activation x connectome-trainability.
# Each cell is run BOTH "untrained" (connectome frozen exactly as measured from the fly; only
# the input encoder + output readout are trained) and "trained" (connectome edge magnitudes
# also learn). So every config reports an untrained AND a trained accuracy. 2x2x12x3 = 144.
EYETACT_SUBGRAPH = "whole"
EYETACT_T_VALUES = list(range(1, 13))                  # 1..12 recurrences
EYETACT_EYES = {                                        # eye tag -> eye-specific leaf fields
    "learned": {"eye": "learned"},
    "rigid":   {"eye": "rigid", "eye_source": "columns",   # faithful FlyWire v783 retinotopy
                "learn_readout": True, "decision": "linear", "readout_mode": "accum"},
}
# core tag -> learn_core. 'untrained' = connectome frozen as-measured (I/O heads still train);
# 'trained' = connectome magnitudes also learn (Dale sign/mask stay fixed either way).
EYETACT_CORES = {"untrained": False, "trained": True}
# activation tag -> core fields. state_norm/target_radius are chosen PER ACTIVATION (eye-
# independent) so a cell's bound behavior tracks only the activation axis. 'linear' is kept
# FAITHFUL (no renorm) — the literal "multiply by W" the user asked for; the T=12 smoke checks
# it stays finite, and 'rms' is the documented fallback. 'relu' is unbounded-above on a deep
# trainable core, so it is rms-guarded; 'tanh' is self-bounding.
EYETACT_ACTS = {
    "linear": {"dynamics": "linear", "state_norm": "none", "target_radius": 0.9},
    "tanh":   {"dynamics": "tanh", "nonlinearity": "tanh", "state_norm": "none", "target_radius": 0.9},
    "relu":   {"dynamics": "tanh", "nonlinearity": "relu", "state_norm": "rms",  "target_radius": 0.9},
}


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


def generate_recmul_grid(out_dir: str | Path, *, full: bool = False) -> list[str]:
    """Write the Family-3 "recmul" leaf configs (recurrent-multiply, depth T=10).

    Each cell is a learned-encoder ConnectomeClassifier whose core forward is the
    ``ff_unroll`` path (alpha=1, inject t=0, T=10) with the chosen step ``dynamics``. The
    grid crosses {optic_left, optic_right} x {linear, linear_rms, tanh} x {frozenW, trainW}
    x init. Phase 1 (default) uses init=from_data only (the "as-measured" wiring, 12 cells);
    ``full=True`` adds the random-init control (24 cells).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inits = INITS.items() if full else [("initA", "from_data")]
    written = []
    for sg in RECMUL_SUBGRAPHS:
        for rule, params in RECMUL_RULES.items():
            for core_tag, learn_core in RECMUL_CORES.items():
                for itag, imode in inits:
                    name = f"{sg}_{rule}_{core_tag}_{itag}"
                    leaf = {
                        "run_name": name, "subgraph_id": sg, "arch": "ff_unroll",
                        "dynamics": params["dynamics"], "state_norm": params["state_norm"],
                        "init_mode": imode, "learn_core": learn_core, "T": 10,
                        "target_radius": params["target_radius"], "eye": "learned",
                        "family": "recmul",
                    }
                    p = out / f"{name}.yaml"
                    with open(p, "w") as fh:
                        yaml.safe_dump(leaf, fh, sort_keys=False)
                    written.append(str(p))
    return written


def generate_eyeTact_grid(out_dir: str | Path, *, full: bool = False) -> list[str]:
    """Write the Family-4 "eyeTact" leaf configs: eye x activation x depth x trainability.

    The full connectome (``whole``, signed + magnitude-weighted) is run across every cell of
    {learned eye, rigid eye} x {linear, tanh, relu} x T(1..12) x {untrained, trained} = 144
    runs. 'untrained' freezes the connectome as-measured (only the I/O heads train); 'trained'
    also learns the edge magnitudes. ``full`` is accepted for dispatch parity (no-op).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for eye_tag, eye_fields in EYETACT_EYES.items():            # 2 eyes
        for act_tag, act_fields in EYETACT_ACTS.items():        # 3 activations
            for core_tag, learn_core in EYETACT_CORES.items():  # untrained / trained
                for T in EYETACT_T_VALUES:                      # 12 depths -> 144
                    name = (f"eyeTact_{EYETACT_SUBGRAPH}_{eye_tag}_{act_tag}"
                            f"_{core_tag}_T{T:02d}")
                    leaf = {
                        "run_name": name, "subgraph_id": EYETACT_SUBGRAPH, "arch": "ff_unroll",
                        "init_mode": "from_data", "T": T, "learn_core": learn_core,
                        "epochs": 25, "family": "eyeTact",
                        **eye_fields, **act_fields,
                    }
                    p = out / f"{name}.yaml"
                    with open(p, "w") as fh:
                        yaml.safe_dump(leaf, fh, sort_keys=False)
                    written.append(str(p))
    return written


# CIFAR-10 mirror of the eyeTact grid (same four axes, harder natural-image task). The full
# sweep runs on LUMINANCE (single channel, 32x32=1024-d) so the rigid eye stays biologically
# faithful (one brightness per ommatidial column); a small set of LEARNED-eye-only RGB
# controls (3072-d) measures what the colorblind constraint costs. Plus a tanh state_norm=rms
# ablation arm so any frozen-tanh collapse can be attributed to "the connectome can't separate
# the classes" vs "the signal decayed before reaching the readout" (the MNIST failure mode).
EYETACT_CIFAR_RGB_TS = [4, 12]                 # T values sampled for the rgb controls
EYETACT_CIFAR_PHASE1_TS = [1, 4, 8, 12]        # representative depths for the 48-cell Phase 1


# The faithful SPECTRAL ("color") rigid eye: each photoreceptor type reads its own image
# channel (R1-6<-luminance, R7<-blue, R8<-green; see eye.SPECTRAL_CHANNELS). It consumes the
# raw 3-channel image, so its cells carry color='rgb'. Same eye fields as the rigid eye but
# eye='rigid_color'. Added as a NEW eye axis (the luma rigid sweep is untouched).
EYETACT_CIFAR_RIGID_COLOR = {"eye": "rigid_color", "eye_source": "columns",
                             "learn_readout": True, "decision": "linear",
                             "readout_mode": "accum"}


def _eyeTact_cifar_leaf(eye_tag, eye_fields, act_tag, act_fields, core_tag, learn_core, T,
                        *, color="luma", state_norm_override=None, name_suffix="",
                        eye_label=None):
    """Build one CIFAR eyeTact leaf-config dict + its run-name."""
    if eye_label is None:
        eye_label = "learnedRGB" if color == "rgb" else eye_tag
    name = (f"eyeTact_cifar_{EYETACT_SUBGRAPH}_{eye_label}_{act_tag}"
            f"_{core_tag}_T{T:02d}{name_suffix}")
    leaf = {
        "run_name": name, "subgraph_id": EYETACT_SUBGRAPH, "arch": "ff_unroll",
        "init_mode": "from_data", "T": T, "learn_core": learn_core,
        "epochs": 25, "family": "eyeTact_cifar",
        "dataset": "cifar10", "color": color,
        **eye_fields, **act_fields,
    }
    if state_norm_override is not None:
        leaf["state_norm"] = state_norm_override
    return name, leaf


def generate_eyeTact_cifar_grid(out_dir: str | Path, *, full: bool = False) -> list[str]:
    """Write the CIFAR-10 eyeTact leaf configs (+ a curated Phase-1 subset).

    Mirrors ``generate_eyeTact_grid`` exactly on the four axes (eye x act x T(1..12) x
    {untrained,trained} = 144 luminance cells), then adds:
      * RGB color-ablation controls: learned eye only, {linear,tanh,relu} x {untrained,
        trained} x T in {4,12} = 12 cells.
      * A tanh state_norm=rms ablation arm: the untrained-tanh cells (both eyes, all T = 24)
        re-run with rms gain control, tagged ``_rmsabl``.
      * The faithful SPECTRAL ("color") rigid eye as a NEW eye axis: {linear,tanh,relu} x
        {untrained,trained} x T(1..12) = 72 cells (eye='rigid_color', color='rgb'; each
        receptor type reads its own channel R1-6<-luma/R7<-blue/R8<-green).
    = 252 configs total. Also writes the 48-cell luma Phase-1 subset (T in {1,4,8,12}) to a
    sibling ``eyeTact_cifar_phase1/`` and the 24-cell color Phase-1 subset to
    ``eyeTact_cifar_color_phase1/`` so the phased rollout selects cells by directory, not by
    array-index range (the sorted-glob order interleaves the trainability axis across T).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    phase1 = out.parent / "eyeTact_cifar_phase1"
    phase1.mkdir(parents=True, exist_ok=True)
    written = []

    def emit(name, leaf, *, in_phase1):
        p = out / f"{name}.yaml"
        with open(p, "w") as fh:
            yaml.safe_dump(leaf, fh, sort_keys=False)
        written.append(str(p))
        if in_phase1:
            with open(phase1 / f"{name}.yaml", "w") as fh:
                yaml.safe_dump(leaf, fh, sort_keys=False)

    # --- 144 luminance cells (the faithful four-axis mirror) ---
    for eye_tag, eye_fields in EYETACT_EYES.items():            # 2 eyes
        for act_tag, act_fields in EYETACT_ACTS.items():        # 3 activations
            for core_tag, learn_core in EYETACT_CORES.items():  # untrained / trained
                for T in EYETACT_T_VALUES:                      # 12 depths
                    name, leaf = _eyeTact_cifar_leaf(
                        eye_tag, eye_fields, act_tag, act_fields, core_tag, learn_core, T)
                    emit(name, leaf, in_phase1=(T in EYETACT_CIFAR_PHASE1_TS))

    # --- 12 RGB color-ablation controls (learned eye only) ---
    learned_fields = EYETACT_EYES["learned"]
    for act_tag, act_fields in EYETACT_ACTS.items():
        for core_tag, learn_core in EYETACT_CORES.items():
            for T in EYETACT_CIFAR_RGB_TS:
                name, leaf = _eyeTact_cifar_leaf(
                    "learned", learned_fields, act_tag, act_fields, core_tag, learn_core, T,
                    color="rgb")
                emit(name, leaf, in_phase1=False)              # controls are Phase-2 only

    # --- tanh state_norm=rms ablation arm (untrained tanh, both eyes, all T) ---
    tanh_fields = EYETACT_ACTS["tanh"]
    for eye_tag, eye_fields in EYETACT_EYES.items():
        for T in EYETACT_T_VALUES:
            name, leaf = _eyeTact_cifar_leaf(
                eye_tag, eye_fields, "tanh", tanh_fields, "untrained", False, T,
                state_norm_override="rms", name_suffix="_rmsabl")
            emit(name, leaf, in_phase1=False)                  # ablation is Phase-2 only

    # --- faithful SPECTRAL ("color") rigid eye: a NEW eye axis, full 4-axis depth sweep ---
    # {linear,tanh,relu} x {untrained,trained} x T(1..12) = 72 cells. eye='rigid_color',
    # color='rgb' (per-receptor-type channel: R1-6<-luma, R7<-blue, R8<-green). The
    # representative T in {1,4,8,12} also go to a dedicated configs/eyeTact_cifar_color_phase1/
    # so this faithful-color arm can be launched as its own first wave.
    color_p1 = out.parent / "eyeTact_cifar_color_phase1"
    color_p1.mkdir(parents=True, exist_ok=True)
    for act_tag, act_fields in EYETACT_ACTS.items():
        for core_tag, learn_core in EYETACT_CORES.items():
            for T in EYETACT_T_VALUES:
                name, leaf = _eyeTact_cifar_leaf(
                    "rigid_color", EYETACT_CIFAR_RIGID_COLOR, act_tag, act_fields,
                    core_tag, learn_core, T, color="rgb", eye_label="rigidcolor")
                emit(name, leaf, in_phase1=False)
                if T in EYETACT_CIFAR_PHASE1_TS:
                    with open(color_p1 / f"{name}.yaml", "w") as fh:
                        yaml.safe_dump(leaf, fh, sort_keys=False)

    return written


# Family 5 "initablation": decompose the connectome's computation into MAGNITUDE (synapse
# counts) vs DIRECTION (excitatory/inhibitory sign) by re-initializing the SAME fixed mask
# three ways, on the full connectome (whole) with the faithful rigid eye on MNIST, each run
# both untrained (core frozen as-measured; only the linear readout trains) and trained (edge
# magnitudes also learn). Two null controls sharpen the contrast (shuffled-sign, random-mag).
#   faithful : sign(data) + magnitude(counts)   -> init_mode from_data, force_sign none
#   nomag    : sign(data) + CONSTANT magnitude   -> init_mode constant,  force_sign none
#   nodirmag : +1 sign     + CONSTANT magnitude  -> init_mode constant,  force_sign +1
#   shufsign : sign permuted + magnitude(counts) -> init_mode from_data, sign_shuffle=seed
#   randmag  : sign(data) + lognormal magnitude  -> init_mode random,    force_sign none
# All five share ONE edge set (built with photoreceptor_sign=-1 so the ~956 dropped photo-
# receptor edges are restored identically). 5 variants x {relu, linearrms} x {untrained,
# trained} x {seed 0,1,2} = 60 cells. T=8 (BFS input->readout max_hops=7, on the plateau).
INITABLATION_SUBGRAPH = "whole"
INITABLATION_T = 8
INITABLATION_SEEDS = [0, 1, 2]
# variant tag -> (init_mode, force_sign, uses_sign_shuffle). When uses_sign_shuffle is True the
# per-cell sign_shuffle is set to the cell's seed (so the 3 seeds give 3 distinct permutations).
INITABLATION_VARIANTS = {
    "faithful": ("from_data", "none", False),
    "nomag":    ("constant",  "none", False),
    "nodirmag": ("constant",  1,      False),
    "shufsign": ("from_data", "none", True),
    "randmag":  ("random",    "none", False),
}
INITABLATION_EYE = {"eye": "rigid", "eye_source": "columns",
                    "learn_readout": True, "decision": "linear", "readout_mode": "accum"}
# activation tag -> core fields. relu (headline; highest ceiling/dynamic range to separate the
# variants) and linearrms (interpretable: literally R . normalize(W^T) . S). Both use rms gain
# control so the frozen core's signal survives to the VPN/descending readout.
INITABLATION_ACTS = {
    "relu":      {"dynamics": "tanh", "nonlinearity": "relu", "state_norm": "rms",
                  "target_radius": 0.9},
    "linearrms": {"dynamics": "linear", "state_norm": "rms", "target_radius": 0.9},
}
INITABLATION_CORES = {"untrained": False, "trained": True}   # tag -> learn_core


def generate_initablation_grid(out_dir: str | Path, *, full: bool = False) -> list[str]:
    """Write the Family-5 'initablation' leaf configs: magnitude vs direction on ``whole``.

    5 init variants x {relu, linearrms} x {untrained, trained} x {seed 0,1,2} = 60 cells, all
    on the full connectome with the faithful rigid eye (photoreceptor_sign=-1) on MNIST, T=8.
    ``full`` is accepted for dispatch parity (no-op).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for var_tag, (imode, fsign, uses_shuffle) in INITABLATION_VARIANTS.items():
        for act_tag, act_fields in INITABLATION_ACTS.items():
            for core_tag, learn_core in INITABLATION_CORES.items():
                for seed in INITABLATION_SEEDS:
                    name = (f"initablation_{INITABLATION_SUBGRAPH}_{var_tag}_{act_tag}"
                            f"_{core_tag}_s{seed}")
                    leaf = {
                        "run_name": name, "subgraph_id": INITABLATION_SUBGRAPH,
                        "arch": "ff_unroll", "init_mode": imode, "force_sign": fsign,
                        "sign_shuffle": (seed if uses_shuffle else "none"),
                        "photoreceptor_sign": -1, "T": INITABLATION_T,
                        "learn_core": learn_core, "seed": seed, "epochs": 25,
                        "family": "initablation", "dataset": "mnist", "color": "luma",
                        **INITABLATION_EYE, **act_fields,
                    }
                    p = out / f"{name}.yaml"
                    with open(p, "w") as fh:
                        yaml.safe_dump(leaf, fh, sort_keys=False)
                    written.append(str(p))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flyconn.models.run")
    ap.add_argument("stage", choices=["build", "smoke", "train", "fit_rigid",
                                       "grid", "eye_grid", "recmul_grid", "eyeTact_grid",
                                       "eyeTact_cifar_grid", "initablation_grid",
                                       "aggregate"])
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

    if args.stage == "recmul_grid":
        out = args.out
        if out == str(REPO_ROOT / "configs" / "experiments"):
            out = str(REPO_ROOT / "configs" / "recmul_experiments")
        paths = generate_recmul_grid(out, full=args.full)
        print(f"[recmul_grid] wrote {len(paths)} configs to {out} "
              f"({'full' if args.full else 'phase-1'})")
        return 0

    if args.stage == "eyeTact_grid":
        out = args.out
        if out == str(REPO_ROOT / "configs" / "experiments"):
            out = str(REPO_ROOT / "configs" / "eyeTact_experiments")
        paths = generate_eyeTact_grid(out, full=args.full)
        print(f"[eyeTact_grid] wrote {len(paths)} configs to {out}")
        return 0

    if args.stage == "eyeTact_cifar_grid":
        out = args.out
        if out == str(REPO_ROOT / "configs" / "experiments"):
            out = str(REPO_ROOT / "configs" / "eyeTact_cifar_experiments")
        paths = generate_eyeTact_cifar_grid(out, full=args.full)
        print(f"[eyeTact_cifar_grid] wrote {len(paths)} configs to {out} "
              f"(+ Phase-1 subset to {Path(out).parent / 'eyeTact_cifar_phase1'})")
        return 0

    if args.stage == "initablation_grid":
        out = args.out
        if out == str(REPO_ROOT / "configs" / "experiments"):
            out = str(REPO_ROOT / "configs" / "initablation_experiments")
        paths = generate_initablation_grid(out, full=args.full)
        print(f"[initablation_grid] wrote {len(paths)} configs to {out}")
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
