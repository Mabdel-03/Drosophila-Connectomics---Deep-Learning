"""Stage 8 — Left-Hemisphere Figure-Ground Circuit runner.

Tests whether a faithful LEFT mirror of the right figure-ground circuit exists, and if so
characterizes it. Runs (live CAVE v783 primary, cache-backed):
  Phase 0    right Nod1=FD1 anchor audit
  Phase 0.5  left-mirror existence screen (7-stage PRESENT/WEAK/ABSENT go/no-go)
  Families   A-I + K for BOTH sides (cfg=RIGHT positive control, cfg=LEFT mirror candidate)
  Family M   mirror symmetry + negative controls + wing-flip (MaleCNS)
  Manifest   every left+right circuit root id

Emits to "8 - Left Hemisphere Circuit/": left_hemisphere_results.json,
cell_identity_manifest.json, VERIFICATION.md, REPORT.md, figures/.

Usage:
  python scripts/left_hemisphere_verify.py                 # all, live CAVE primary
  python scripts/left_hemisphere_verify.py --only A,M       # subset (e.g. just family A + mirror)
  python scripts/left_hemisphere_verify.py --offline        # force offline FlyWire
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from flyconn.paper import left_hemisphere as LH  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated section ids: K1 (Phase0), SCREEN, A..K, M, MANIFEST")
    ap.add_argument("--offline", action="store_true", help="force offline FlyWire source")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    prefer = "offline" if args.offline else "auto"
    run = LH.run_all(prefer=prefer, only=only)
    LH.emit(run)


if __name__ == "__main__":
    main()
