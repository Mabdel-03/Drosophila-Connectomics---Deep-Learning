"""Stage 7 — build the FD3 report PDFs from the verification JSON.

  python scripts/fd3_report.py report        # FD3 identity report (Family K)
  python scripts/fd3_report.py descending     # FD3 descending-neuron report (Family L)
  python scripts/fd3_report.py report --results "..." --offline-results "..." --stage "..."

Reads only the JSON (token-free): no live CAVE, no feathers.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.fd3_report.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
