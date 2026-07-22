"""Download the public male CNS (MaleCNS v1.0) bulk flat files used by family J.

No authentication required — the files are public on Google Cloud Storage. Idempotent:
skips files already present with the expected size.

Usage: python scripts/malecns_prep.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")
from flyconn.paper.malecns import RAW  # noqa: E402

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "body-annotations.feather": ("body-annotations-male-cns-v1.0-minconf-0.5.feather", 14_483_314),
    "body-neurotransmitters.feather": ("body-neurotransmitters-male-cns-v1.0.feather", 43_282_834),
    "connectome-weights.feather": ("connectome-weights-male-cns-v1.0-minconf-0.5.feather", 1_051_241_946),
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for local, (remote, size) in FILES.items():
        dest = RAW / local
        if dest.exists() and dest.stat().st_size == size:
            print(f"[malecns_prep] {local}: present ({size:,} bytes), skip")
            continue
        url = f"{BASE}/{remote}"
        print(f"[malecns_prep] downloading {local} <- {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"[malecns_prep] wrote {dest} ({dest.stat().st_size:,} bytes)")
    print("[malecns_prep] done")


if __name__ == "__main__":
    main()
