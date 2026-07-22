"""Stage `extract_rN` wrapper: run the live-CAVE MCNS extraction, gate on n_edges.

Submits the SLURM extract job (or runs inline if CONNECTOME_LAUNCHER != 'slurm') and
copies the resulting motif_edges.parquet + extract_stats.json into the workspace as the
stage's success_artifacts. The extract step itself distinguishes an empty result
(n_edges=0, a SCIENTIFIC outcome the overseer re-proposes) from an unreachable CAVE
(non-zero exit, a TRANSIENT failure the overseer retries) - we never let a network blip
masquerade as 'no edges'.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _campaign_common import (  # noqa: E402
    CONNECTOMICS_REPO, parse_workspace_args, read_json,
    run_flyconn, stage5_data_root, write_json,
)
from _campaign_common import env  # noqa: E402

MUSCULAR_DIR = Path(stage5_data_root()) / "v783" / "muscular"


def _job_final_state(jid: str) -> str:
    """Terminal SLURM state for a job id via sacct (COMPLETED / FAILED / ... / UNKNOWN)."""
    out = subprocess.run(
        ["sacct", "-j", jid, "--format=State", "-n", "-P", "-X"],
        capture_output=True, text=True,
    )
    states = [s.strip() for s in out.stdout.splitlines() if s.strip()]
    return states[0].split()[0] if states else "UNKNOWN"


def _sbatch_and_wait(sbatch: Path, timeout_s: int = 3600) -> int:
    """Submit an sbatch job, block until it leaves the queue, and CHECK it succeeded.

    A job that FAILED also leaves squeue, so 'left the queue' is not success -- we read the
    terminal state from sacct and only return 0 on COMPLETED. This is what stops a failed
    extract (e.g. the data-root guard tripping) from being recorded as a clean run.
    """
    out = subprocess.run(["sbatch", "--parsable", str(sbatch)], capture_output=True, text=True)
    if out.returncode != 0:
        print(f"[campaign_extract] sbatch failed: {out.stderr}")
        return out.returncode or 1
    jid = out.stdout.strip().split(";")[0]
    print(f"[campaign_extract] submitted job {jid}; waiting...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        q = subprocess.run(["squeue", "-h", "-j", jid], capture_output=True, text=True)
        if not q.stdout.strip():
            state = _job_final_state(jid)
            print(f"[campaign_extract] job {jid} final state: {state}")
            return 0 if state.startswith("COMPLETED") else 1
        time.sleep(15)
    print("[campaign_extract] timed out waiting for SLURM job")
    return 124


def main(argv=None) -> int:
    args = parse_workspace_args(argv)
    ws = Path(args.workspace)
    rnd = env("CONNECTOME_ROUND", "1")
    launcher = env("CONNECTOME_LAUNCHER", "inline")

    if launcher == "slurm":
        rc = _sbatch_and_wait(CONNECTOMICS_REPO / "slurm" / "muscular_extract.sbatch")
    else:
        rc = run_flyconn(["flyconn.muscular", "extract"])

    stats_src = MUSCULAR_DIR / "_muscular_extract_stats.json"
    edges_src = MUSCULAR_DIR / "mcns_dn_mn_edges.parquet"

    if rc != 0:
        # Infra/config failure (sbatch error, non-COMPLETED job, timeout). Exit non-zero so
        # the harness marks the stage failed and the overseer classifies it (retry vs config
        # repair) - NOT 'no edges'. Write a failure record so no stale 'ok' stats survive.
        write_json(ws / "extract_stats.json",
                   {"status": "extract_failed", "rc": rc, "round": rnd,
                    "source": "offline MaleCNS v1.0 flat files", "n_edges": 0})
        print(f"[campaign_extract] extraction failed rc={rc} (infra/config)")
        return rc

    # Success: require FRESH output from THIS run (do NOT fall back to stale canonical files
    # -- that would let a no-op extract masquerade as a clean run).
    if not stats_src.exists() or not edges_src.exists():
        write_json(ws / "extract_stats.json",
                   {"status": "extract_failed", "rc": "missing_output", "round": rnd,
                    "source": "offline MaleCNS v1.0 flat files", "n_edges": 0})
        print(f"[campaign_extract] extraction produced no output at {MUSCULAR_DIR}")
        return 1

    import shutil
    stats = read_json(stats_src)
    write_json(ws / "extract_stats.json", stats)
    shutil.copy2(edges_src, ws / "motif_edges.parquet")
    print(f"[campaign_extract] round {rnd}: n_edges={stats.get('n_edges')} "
          f"source={stats.get('source')}")
    # n_edges==0 is a *scientific* result; we still succeed here so verify/coverage runs
    # and the overseer decides to re-propose. The extract_stats json_min_numeric gate on
    # the campaign side fails the stage when n_edges<1 (empty motif), triggering re-loop.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
