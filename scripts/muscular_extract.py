"""Stage-5 extraction (network): pull DN->MN edges + annotations from MCNS via live CAVE.

Thin runner over ``python -m flyconn.muscular extract``. Needs the [cave] extra + a CAVE
token + outbound network. Populates the scratch cache and the muscular/ intermediate
parquets, so the cache-only verify step can run offline afterwards.

    python -u scripts/muscular_extract.py
"""

from __future__ import annotations

from flyconn.muscular.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["extract"]))
