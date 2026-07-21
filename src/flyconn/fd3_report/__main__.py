"""CLI: build the Stage-7 FD3 report PDFs from the verification JSON.

  # the FD3 identity report (Family K)
  python -m flyconn.fd3_report report \
      [--results "5 - Paper Verification/verification_results.json"] \
      [--offline-results "5 - Paper Verification/figures/K_offline.json"] \
      [--stage "7 - FD3 Identification"]

  # the FD3 input-pathway report (Family P)  -- anatomical figures if --with-figures
  python -m flyconn.fd3_report input [--results ...] [--stage ...] [--with-figures]

  # the FD3 descending-neuron report (Family L)
  python -m flyconn.fd3_report descending [--results ...] [--stage ...]

  # the combined 3-part circuit report (Identity + Inputs + Outputs)
  python -m flyconn.fd3_report combined \
      [--k-results FILE] [--p-results FILE] [--l-results FILE] [--with-figures]

Each ``--only`` verification run overwrites the shared verification_results.json with just its
family, so for ``combined`` you can point --k/--p/--l-results at per-family copies. ``--with-
figures`` builds the real anatomical figures (needs a FlyWire source; run in the flyconn_cave env
with a token for live skeletons, else cached skeletons + synapse clouds are used).
"""

from __future__ import annotations

import argparse
import sys

from . import builder, combined, descending, disambig, functional, input as input_report


def _source(with_figures: bool):
    """A (src, meta) pair for anatomical figures, or (None, None) to skip them."""
    if not with_figures:
        return None, None
    import sys as _sys
    _sys.path.insert(0, "src")
    from flyconn.paper import fw_access as FW
    return FW.make_source("auto"), FW.NeuronMeta.load("783")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flyconn.fd3_report")
    ap.add_argument("command",
                    choices=["report", "input", "descending", "functional", "combined", "disambig"],
                    nargs="?", default="report")
    ap.add_argument("--results", default="5 - Paper Verification/verification_results.json")
    ap.add_argument("--offline-results", default="5 - Paper Verification/figures/K_offline.json")
    ap.add_argument("--stage", default="7 - FD3 Identification")
    ap.add_argument("--disambig-results", default=None,
                    help="disambig: the fd3_disambig.json (default <stage>/fd3_disambig.json)")
    ap.add_argument("--with-figures", action="store_true",
                    help="render the real anatomical figures (needs a FlyWire source)")
    ap.add_argument("--k-results", default=None, help="combined: Family-K JSON (default --results)")
    ap.add_argument("--p-results", default=None, help="combined: Family-P JSON (default --results)")
    ap.add_argument("--l-results", default=None, help="combined: Family-L JSON (default --results)")
    args = ap.parse_args(argv)

    if args.command == "disambig":
        stage = args.stage if args.stage != "7 - FD3 Identification" else "10 - FD3 vs Nod3 Disambiguation"
        results = args.disambig_results or f"{stage}/fd3_disambig.json"
        out = disambig.build_report(stage, results)
    elif args.command == "descending":
        out = descending.build_report(args.stage, args.results)
    elif args.command == "input":
        src, meta = _source(args.with_figures)
        out = input_report.build_report(args.stage, args.results, src=src, meta=meta)
    elif args.command == "functional":
        src, meta = _source(args.with_figures)
        out = functional.build_report(args.stage, args.results, src=src, meta=meta)
    elif args.command == "combined":
        src, meta = _source(args.with_figures)
        out = combined.build_report(args.stage, args.results, k_path=args.k_results,
                                    p_path=args.p_results, l_path=args.l_results, src=src, meta=meta)
    else:
        out = builder.build_report(args.stage, args.results, args.offline_results)
    print(f"tex      = {out.get('tex')}")
    print(f"pdf      = {out.get('pdf')}")
    print(f"compiled = {out.get('compiled')}")
    if out.get("missing"):
        print(f"missing  = {out['missing']}")
    return 0  # never fail the caller; inspect compiled / .compile_error.txt


if __name__ == "__main__":
    sys.exit(main())
