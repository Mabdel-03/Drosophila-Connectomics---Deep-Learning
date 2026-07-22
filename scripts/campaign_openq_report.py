"""Stage-6 `report_rN` wrapper: compile the LaTeX -> PDF deliverable (runs only post-gate).

Reaches this stage only when verify_rN passed the gate (no refuted claims, full coverage).
Runs ``python -m flyconn.openq report`` to render ``open_questions_report.tex`` and compile it
to ``open_questions_report.pdf`` in '6 - Open Questions/', then assembles a dataset manifest
(``open_questions_dataset.json``) and recomputes the verdict tally from claims[] as an
integrity check (so a tampered refuted list cannot slip the gate).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _campaign_common import (  # noqa: E402
    CONNECTOMICS_REPO, cave_python, env, parse_workspace_args, read_json, write_json,
)

OPENQ_DATA_ROOT = "/orcd/scratch/orcd/012/mabdel03/connectome_data"
STAGE6_DIR = CONNECTOMICS_REPO / "6 - Open Questions"


def _run_openq_report() -> int:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(CONNECTOMICS_REPO / "src") + os.pathsep + e.get("PYTHONPATH", "")
    e["FLYCONN_DATA_ROOT"] = env("FLYCONN_DATA_ROOT", OPENQ_DATA_ROOT) or OPENQ_DATA_ROOT
    if "scratch" not in e["FLYCONN_DATA_ROOT"]:
        e["FLYCONN_DATA_ROOT"] = OPENQ_DATA_ROOT
    # Ensure the TeX engines on the miniforge bin are reachable from the subprocess PATH.
    e["PATH"] = "/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin" + os.pathsep + e.get("PATH", "")
    return subprocess.call([cave_python(), "-u", "-m", "flyconn.openq", "report"],
                           cwd=str(CONNECTOMICS_REPO), env=e)


def main(argv=None) -> int:
    args = parse_workspace_args(argv)
    ws = Path(args.workspace)
    rnd = env("CONNECTOME_ROUND", "1")

    rc = _run_openq_report()  # best-effort PDF; non-zero only on a true crash
    tex = STAGE6_DIR / "open_questions_report.tex"
    pdf = STAGE6_DIR / "open_questions_report.pdf"

    # Integrity check: recompute the verdict tally from claims[]; the certified dataset must
    # match the flat refuted list (defends against a repair that zeroed it).
    vr = read_json(STAGE6_DIR / "openq_results.json") if (STAGE6_DIR / "openq_results.json").exists() else {}
    cov = read_json(STAGE6_DIR / "coverage.json") if (STAGE6_DIR / "coverage.json").exists() else {}
    claims = vr.get("claims", [])
    recomputed = dict(Counter(c.get("verdict") for c in claims))
    declared_refuted = set(vr.get("refuted_claims", []))
    actual_refuted = {c["id"] for c in claims if c.get("verdict") == "REFUTED"}
    integrity_ok = declared_refuted == actual_refuted

    dataset = {
        "stage": "6 - Open Questions",
        "round": rnd,
        "verdict_summary": recomputed,
        "coverage": cov,
        "provenance": {
            "datasets": vr.get("meta", {}).get("datasets", {}),
            "seeds": vr.get("meta", {}).get("seeds", []),
            "tracks": vr.get("meta", {}).get("tracks", {}),
        },
        "deliverables": {
            "tex": str(tex) if tex.exists() else None,
            "pdf": str(pdf) if pdf.exists() else None,
            "pdf_compiled": pdf.exists(),
        },
        "integrity_check": {"declared_eq_actual_refuted": integrity_ok,
                            "actual_refuted": sorted(actual_refuted)},
    }
    write_json(STAGE6_DIR / "open_questions_dataset.json", dataset)

    # Surface success artifacts into the workspace for the runner's validators.
    for name in ("open_questions_dataset.json", "open_questions_report.tex",
                 "open_questions_report.pdf", "OPENQ.md"):
        s = STAGE6_DIR / name
        if s.exists():
            shutil.copy2(s, ws / name)

    if not integrity_ok:
        print(f"[campaign_openq_report] INTEGRITY FAIL: declared={declared_refuted} "
              f"actual={actual_refuted}")
        return 3
    if not pdf.exists():
        # The .tex is always emitted; a failed PDF compile is degraded, not fatal -- but flag it.
        print(f"[campaign_openq_report] WARNING: PDF not produced (tex at {tex}); rc={rc}")
    print(f"[campaign_openq_report] round {rnd}: certified; verdicts={recomputed} "
          f"pdf={'yes' if pdf.exists() else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
