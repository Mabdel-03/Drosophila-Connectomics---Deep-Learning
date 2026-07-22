"""Stage 9 — Inter-Hemispheric Coupling runner.

Identifies and quantifies how the left and right figure-ground circuits connect and communicate,
across five channels (readout crossing, heterolateral bridges, centrifugal gating, shared
convergence, systematic discovery), and emits the per-channel ledger + bridge manifest.

Usage:
  python scripts/interhemispheric_verify.py                 # all channels, live CAVE primary
  python scripts/interhemispheric_verify.py --only N1,N3     # a subset of channels
  python scripts/interhemispheric_verify.py --offline        # offline FlyWire
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from flyconn.paper import interhemispheric as IHM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated channel ids: N1..N5, MANIFEST")
    ap.add_argument("--offline", action="store_true", help="force offline FlyWire source")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    prefer = "offline" if args.offline else "auto"
    run = IHM.run_all(prefer=prefer, only=only)
    IHM.emit(run)


if __name__ == "__main__":
    main()
