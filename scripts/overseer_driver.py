"""Drive the GLM campaign-overseer on a fixed cadence, bypassing OpenClaw's cron.

OpenClaw 2026.3.8's cron SCHEDULER arms its timer but never fires jobs (enqueue returns ok
but the turn never executes; the gateway logs no dispatch). DIRECT agent turns
(`openclaw agent --agent campaign-overseer ...`) work perfectly, so this driver invokes the
overseer that way on each tick — giving the full autonomous loop (GLM reasoning + Telegram
announcement + campaign progression) without the broken cron path.

Each tick the overseer runs its heartbeat workflow (campaign_heartbeat.py advances the DAG;
the overseer reasons over the verdicts per the playbook). The driver stops when the campaign
converges (CAMPAIGN_COMPLETE.md appears) or escalates after a guard limit, and sleeps the
interval between ticks.

Run via slurm/overseer_driver.sbatch (survives logout) or directly:
    python scripts/overseer_driver.py --interval 1800 --max-ticks 48
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OC = "/orcd/home/002/mabdel03/conda_envs/consortium/bin/openclaw"
REPO = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics")
MSC = "/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal"
CONSORTIUM_PY = "/orcd/home/002/mabdel03/conda_envs/consortium/bin/python"
CAMPAIGN = "campaigns/campaign_muscular_projection.yaml"
WORKSPACE_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/connectome_campaigns/muscular_projection_v1")
AGENT_ID = "campaign-overseer"

TICK_MESSAGE = (
    "CONNECTOME SUPERVISION TICK — Muscular Projection Campaign. "
    "Follow knowledge/muscular_projection_playbook.md exactly. Run from "
    f"{REPO}: (1) {CONSORTIUM_PY} {MSC}/scripts/campaign_heartbeat.py --campaign {CAMPAIGN} "
    "(capture EXIT_CODE); (2) "
    f"{CONSORTIUM_PY} {MSC}/scripts/campaign_cli.py --campaign {CAMPAIGN} status. "
    "Apply the verdict-driven decision tree: 0=converged (announce dataset path); "
    "1/3=monitor; 2/4=verdict triage (REFUTED-by-biology or coverage gap => re-loop in "
    "place: rewrite-task propose_r1 + set-stage-status propose_r1 pending; config bug => "
    "escalate; NEVER edit the verifier or fake a CONFIRMED). Append your reasoning to "
    "knowledge/campaign_journal.md and reply with a one-line stage-status summary."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stage_statuses() -> dict:
    """Read campaign stage statuses via the campaign CLI."""
    try:
        out = subprocess.run(
            [CONSORTIUM_PY, f"{MSC}/scripts/campaign_cli.py", "--campaign", CAMPAIGN, "status"],
            cwd=str(REPO), capture_output=True, text=True, timeout=120,
            env={"PYTHONPATH": MSC, "PATH": "/usr/bin:/bin"},
        )
        d = json.loads(out.stdout)
        return {k: v.get("status") for k, v in d.get("stages", {}).items()}
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)[:120]}


def gateway_up() -> bool:
    out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True)
    return ":18789" in out.stdout


def run_overseer_tick() -> tuple[int, str]:
    """Invoke the GLM overseer via the working direct-agent path. Returns (rc, reply)."""
    p = subprocess.run(
        [OC, "agent", "--agent", AGENT_ID, "-m", TICK_MESSAGE, "--json"],
        cwd=str(REPO), capture_output=True, text=True, timeout=600,
    )
    reply = ""
    try:
        d = json.loads(p.stdout)
        r = d.get("result", d)
        pl = r.get("payloads") if isinstance(r, dict) else None
        reply = pl[0]["text"] if pl else str(r.get("text", ""))[:300]
    except Exception:
        reply = (p.stdout or p.stderr)[:300]
    return p.returncode, reply


def run_heartbeat_direct() -> int:
    """Run the deterministic heartbeat directly (advances the DAG; no gateway/GLM needed).

    This GUARANTEES campaign progression even if the gateway or GLM is unavailable. The GLM
    overseer turn (run separately) layers reasoning + Telegram on top. Returns the heartbeat
    exit code (0 complete, 1 in-progress, 2 failed, 3 advanced, 4 idle).
    """
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = MSC
    env.setdefault("FLYCONN_DATA_ROOT", "/orcd/data/tpoggio/001/mabdel03/connectome_data")
    # CAVE_TOKEN for ${CAVE_TOKEN} expansion in the YAML, if a repo secret exists
    sec = REPO / ".cave_secret"
    if sec.is_file() and "CAVE_TOKEN" not in env:
        env["CAVE_TOKEN"] = sec.read_text().strip()
    p = subprocess.run(
        [CONSORTIUM_PY, f"{MSC}/scripts/campaign_heartbeat.py", "--campaign", CAMPAIGN],
        cwd=str(REPO), capture_output=True, text=True, timeout=1200, env=env,
    )
    tail = "\n".join((p.stdout or "").splitlines()[-3:])
    log(f"heartbeat rc={p.returncode}: {tail}")
    return p.returncode


def converged() -> bool:
    return (WORKSPACE_ROOT / "CAMPAIGN_COMPLETE.md").is_file()


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=int, default=1800, help="seconds between ticks")
    ap.add_argument("--max-ticks", type=int, default=48, help="hard stop after N ticks")
    args = ap.parse_args(argv)

    log(f"overseer driver start: interval={args.interval}s max_ticks={args.max_ticks}")
    if converged():
        log("campaign already converged (CAMPAIGN_COMPLETE.md present). nothing to do.")
        return 0

    for tick in range(1, args.max_ticks + 1):
        # Layer 1 (always): deterministic heartbeat advances the DAG — guarantees the
        # science completes regardless of gateway/GLM health.
        log(f"tick {tick}: deterministic heartbeat…")
        run_heartbeat_direct()
        # Layer 2 (best-effort): GLM overseer for reasoning + Telegram + verdict triage.
        if gateway_up():
            log(f"tick {tick}: invoking GLM overseer (supervision + announce)…")
            try:
                rc, reply = run_overseer_tick()
                log(f"tick {tick}: overseer rc={rc} reply={reply!r}")
            except Exception as e:  # noqa: BLE001 - supervision is best-effort
                log(f"tick {tick}: overseer turn errored (non-fatal): {str(e)[:150]}")
        else:
            log(f"tick {tick}: gateway down — skipping GLM supervision (heartbeat still ran).")
        st = stage_statuses()
        log(f"tick {tick}: stages={st}")
        if converged():
            log("CONVERGED — CAMPAIGN_COMPLETE.md present. driver exiting 0.")
            return 0
        if tick < args.max_ticks:
            time.sleep(args.interval)

    log(f"max_ticks={args.max_ticks} reached without convergence. exiting 1 for review.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
