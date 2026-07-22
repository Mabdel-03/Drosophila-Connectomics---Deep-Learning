"""Stage-6 `run_rN` wrapper: resolve the three open questions, gate on no-refuted.

Runs ``python -m flyconn.openq run`` (Q1 single-animal closure, Q2 bilateral generalization,
Q3 escape-route separability) DETERMINISTICALLY -- no LLM authors verdicts; only the
re-derivation + oracle decide CONFIRMED/REFUTED/UNVERIFIABLE. Copies the verifier's
``openq_results.json`` + ``coverage.json`` into the workspace, where the campaign's
artifact_validators act as the GATE (refuted_claims == 0, bilateral_coverage == 1.0,
muscle_chain_complete == 1.0).

Why a custom data root: unlike Stage 5 (which only WRITES the CAVE cache, on the group
volume), Stage 6 also READS the offline FlyWire annotation table ``neurons.parquet``, which
lives on SCRATCH. So this wrapper points FLYCONN_DATA_ROOT at the scratch connectome_data
root (writable, 250T free, holds both neurons.parquet and the live-CAVE cache) rather than
the group volume. The MCNS feathers are read via their own absolute paths in
flyconn.paper.malecns, independent of FLYCONN_DATA_ROOT.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _campaign_common import (  # noqa: E402
    CONNECTOMICS_REPO, cave_python, env, parse_workspace_args, write_json,
)

# Stage-6 reads the offline annotation table from SCRATCH (the 7MB neurons.parquet + the
# 48G FAFB feather live there); it also writes the live-CAVE cache + intermediates there.
OPENQ_DATA_ROOT = "/orcd/scratch/orcd/012/mabdel03/connectome_data"
STAGE6_DIR = CONNECTOMICS_REPO / "6 - Open Questions"


def _run_openq(subcmd: list[str]) -> int:
    """Run ``python -m flyconn.openq <subcmd>`` with the Stage-6 data root + src on path."""
    e = dict(os.environ)
    e["PYTHONPATH"] = str(CONNECTOMICS_REPO / "src") + os.pathsep + e.get("PYTHONPATH", "")
    # Stage 6 needs the SCRATCH root (neurons.parquet); override whatever common.sh set.
    e["FLYCONN_DATA_ROOT"] = env("FLYCONN_DATA_ROOT", OPENQ_DATA_ROOT) or OPENQ_DATA_ROOT
    if "scratch" not in e["FLYCONN_DATA_ROOT"]:
        e["FLYCONN_DATA_ROOT"] = OPENQ_DATA_ROOT
    return subprocess.call(
        [cave_python(), "-u", "-m", "flyconn.openq", *subcmd],
        cwd=str(CONNECTOMICS_REPO), env=e,
    )


def main(argv=None) -> int:
    args = parse_workspace_args(argv)
    ws = Path(args.workspace)
    rnd = env("CONNECTOME_ROUND", "1")
    n_perm = env("CONNECTOME_N_PERM", "200")

    rc = _run_openq(["run", "--n-perm", str(n_perm)])
    res_src = STAGE6_DIR / "openq_results.json"
    cov_src = STAGE6_DIR / "coverage.json"

    if rc != 0 or not res_src.exists() or not cov_src.exists():
        # Infra/config failure (not a scientific "REFUTED"): emit a failure record so no
        # stale 'ok' result survives, and exit non-zero so the overseer classifies it.
        write_json(ws / "openq_results.json",
                   {"status": "run_failed", "rc": rc, "round": rnd,
                    "verdict_counts": {}, "refuted_claims": ["__run_crashed__"],
                    "unverifiable_claims": [], "claims": []})
        write_json(ws / "coverage.json",
                   {"bilateral_coverage": 0.0, "muscle_chain_complete": 0.0,
                    "dn_to_mn_edges": 0, "mn_to_muscle_edges": 0, "missing": []})
        print(f"[campaign_openq_run] run failed rc={rc} (infra/config)")
        return rc or 1

    # Mirror the canonical Stage-6 artifacts into the workspace so the runner's
    # artifact_validators (the gate) act on this round's fresh files.
    copied = []
    for name in ("openq_results.json", "coverage.json", "openq_raw.json", "OPENQ.md"):
        s = STAGE6_DIR / name
        if s.exists():
            shutil.copy2(s, ws / name)
            copied.append(name)
    print(f"[campaign_openq_run] round {rnd}: copied {copied} from {STAGE6_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
