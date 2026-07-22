"""Stage 5 — Paper Verification runner.

Re-derives every quantitative claim in `Figure_Ground_Circuit.pdf` across the FlyWire
FAFB v783 (live CAVE) and male CNS (MaleCNS v1.0, offline bulk) connectomes, compares to
the paper's claims (the oracle), and emits the per-claim ledger:

  5 - Paper Verification/verification_results.json
  5 - Paper Verification/VERIFICATION.md

Usage:
  python scripts/paper_verify.py                 # all families, live CAVE primary
  python scripts/paper_verify.py --only A,E,J    # subset
  python scripts/paper_verify.py --offline       # force offline FlyWire (no live counts)
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from flyconn.paper import ledger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated family ids, e.g. A,E,J")
    ap.add_argument("--offline", action="store_true", help="force offline FlyWire source")
    args = ap.parse_args()

    only = args.only.split(",") if args.only else None
    prefer = "offline" if args.offline else "auto"
    run = ledger.run_all(prefer=prefer, only=only)
    ledger.emit(run)


if __name__ == "__main__":
    main()
