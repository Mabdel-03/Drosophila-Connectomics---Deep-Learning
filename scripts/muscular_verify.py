"""Stage-5 verification (offline): re-derive S14/S15, run nulls + coverage, emit deliverables.

Thin runner over ``python -m flyconn.muscular verify``. Reads only the muscular/
intermediate parquets the extract step wrote (cache-only), so it runs in seconds with no
network and no token. Writes the '5 - Muscular Projection/' dataset + gate files.

    python -u scripts/muscular_verify.py
"""

from __future__ import annotations

from flyconn.muscular.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["verify"]))
