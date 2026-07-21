"""CLI: build the Stage-12 FD4 search report PDF from the FD4 verification JSON.

  python -m flyconn.fd4_report combined \
      [--results "12 - FD4 Identification/fd4_circuit.json"] \
      [--stage "12 - FD4 Identification"] [--with-figures]

``--with-figures`` renders the anatomical neuron figures (needs a FlyWire source; run in the
flyconn_cave env with a token for live skeletons, else cached skeletons / synapse clouds are used).
"""

from __future__ import annotations

import argparse
import sys

from . import combined


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flyconn.fd4_report")
    ap.add_argument("command", choices=["combined"], nargs="?", default="combined")
    ap.add_argument("--results", default="12 - FD4 Identification/fd4_circuit.json")
    ap.add_argument("--stage", default="12 - FD4 Identification")
    ap.add_argument("--with-figures", action="store_true",
                    help="render the anatomical neuron figures (needs a FlyWire source)")
    args = ap.parse_args(argv)

    src, meta = (None, None)
    if args.with_figures:
        sys.path.insert(0, "src")
        from flyconn.paper import fw_access as FW
        src, meta = FW.make_source("auto"), FW.NeuronMeta.load("783")
    out = combined.build_report(args.stage, args.results, src=src, meta=meta)
    print(f"tex      = {out.get('tex')}")
    print(f"pdf      = {out.get('pdf')}")
    print(f"compiled = {out.get('compiled')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
