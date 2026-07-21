"""Stage `propose_rN` wrapper: produce proposal.json for this round.

Reads seed_spec.json and, if present, the prior round's verification_results.json +
coverage.json (copied into the workspace by the runner), then asks a Claude Code agent to
write a concrete, runnable motif proposal. If the Claude CLI is unavailable, falls back to
a deterministic proposal derived from the seed spec so the pipeline still progresses
(round 1 = the minimal DNbe001/DNp26 -> wing-MN -> muscle trace for both wings).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _campaign_common import (  # noqa: E402
    claude_code, env, parse_workspace_args, read_json, task_file_text, write_json,
)


def _deterministic_proposal(ws: Path, rnd: str) -> dict:
    seed = read_json(ws / "seed_spec.json") if (ws / "seed_spec.json").exists() else {}
    prior_cov = ws / "coverage.json"
    seeds = seed.get("seed_neurons", ["DNbe001", "DNp26"])
    rationale = "Round 1: minimal DN->MN->muscle trace for both wings."
    if prior_cov.exists():
        cov = read_json(prior_cov)
        missing = cov.get("missing", [])
        extra = sorted({m["dn"] for m in missing if m.get("dn")})
        if extra:
            seeds = sorted(set(seeds) | set(extra))
        rationale = f"Re-loop: widen to cover gaps {missing[:5]}"
    return {
        "motif": "figure-ground DN -> wing motor-neuron -> wing muscle, both wings (MCNS)",
        "seed_neurons": seeds,
        "dataset": seed.get("dataset", "MCNS"),
        "edges_to_trace": [
            {"from": "DN", "to": "motor_neuron", "dataset": "MCNS"},
            {"from": "motor_neuron", "to": "muscle", "dataset": "MCNS"},
        ],
        "thresholds": seed.get("thresholds", {"min_synapses": 1, "cleft_score_min": 0}),
        "coverage_targets": {"wings": ["ipsi", "contra"], "chain": ["DN", "MN", "muscle"]},
        "rationale": rationale,
        "round": rnd,
    }


def main(argv=None) -> int:
    args = parse_workspace_args(argv)
    ws = Path(args.workspace)
    rnd = env("CONNECTOME_ROUND", "1")
    out = ws / "proposal.json"

    # The OpenClaw overseer runs on GLM (OpenRouter); the propose stage uses the
    # DETERMINISTIC proposal by default (the proven path that passes the gate). The Claude
    # CLI is only invoked if CONNECTOME_USE_LLM=1 is explicitly set (it cannot route to GLM
    # -- the local `claude` binary is Claude-only -- so it is off by default here).
    rc = 127
    if env("CONNECTOME_USE_LLM", "0") == "1":
        task = task_file_text("connectome_propose.txt")
        if task:
            rc = claude_code(task, workspace=ws, allow_edits=True,
                             model=env("CONNECTOME_MODEL", "claude-opus-4-6"), max_turns=20,
                             budget_cents=int(env("CONNECTOME_PROPOSE_CENTS", "500")))
    if rc != 0 or not out.exists():
        # deterministic proposal (default): keeps the loop fully model-agnostic.
        write_json(out, _deterministic_proposal(ws, rnd))
        print(f"[campaign_propose] round {rnd}: wrote deterministic proposal.json")
    else:
        print(f"[campaign_propose] round {rnd}: agent wrote proposal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
