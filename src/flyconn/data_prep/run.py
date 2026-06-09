"""Stage-1 CLI dispatcher.

Usage:
  python -m flyconn.data_prep.run download [--only KEY ...] [--force]
  python -m flyconn.data_prep.run build
  python -m flyconn.data_prep.run validate
  python -m flyconn.data_prep.run all
  python -m flyconn.data_prep.run neurons   # build node table only
  python -m flyconn.data_prep.run edges     # build edges/adjacency only

All stages take --config (defaults to configs/data_v783.yaml).
"""

from __future__ import annotations

import argparse

from ..config import load_config
from . import build_edges, build_neurons, download, validate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flyconn.data_prep.run")
    ap.add_argument(
        "stage",
        choices=["download", "neurons", "edges", "build", "validate", "all"],
        help="Which stage to run. 'build' = neurons+edges; 'all' = download+build+validate.",
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="(download) restrict to keys")
    ap.add_argument("--force", action="store_true", help="(download) redownload valid files")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.stage in ("download", "all"):
        download.download_all(cfg, only=args.only, force=args.force)
    if args.stage in ("neurons", "build", "all"):
        build_neurons.run(cfg)
    if args.stage in ("edges", "build", "all"):
        build_edges.run(cfg)
    if args.stage in ("validate", "all"):
        validate.run(cfg)

    print(f"[run] stage '{args.stage}' complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
