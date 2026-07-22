"""Stage `verify_rN` wrapper: run the flyconn verifier DETERMINISTICALLY (no LLM verdicts).

This stage runs ``python -m flyconn.muscular verify`` directly - it does NOT spawn an LLM
agent to author verdicts. That is a deliberate guardrail: a REFUTED claim or a coverage
gap is a real scientific outcome, and only code (re-derivation + compare.py) decides it.
The wrapper copies the verifier's verification_results.json + coverage.json into the
workspace, where the campaign's artifact_validators act as the GATE.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _campaign_common import (  # noqa: E402
    copy_into_workspace, env, parse_workspace_args, run_flyconn, stage_artifacts, write_json,
)


def main(argv=None) -> int:
    args = parse_workspace_args(argv)
    ws = Path(args.workspace)
    rnd = env("CONNECTOME_ROUND", "1")

    rc = run_flyconn(["flyconn.muscular", "verify"])
    if rc != 0:
        write_json(ws / "verification_results.json",
                   {"status": "verify_failed", "rc": rc, "verdict_counts": {},
                    "refuted_claims": ["__verify_crashed__"], "unverifiable_claims": [], "claims": []})
        write_json(ws / "coverage.json",
                   {"bilateral_coverage": 0.0, "muscle_chain_complete": 0.0,
                    "dn_to_mn_edges": 0, "mn_to_muscle_edges": 0, "missing": []})
        print(f"[campaign_verify] verify crashed rc={rc}")
        return rc

    copied = copy_into_workspace(ws, ["verification_results.json", "coverage.json"])
    src = stage_artifacts()
    print(f"[campaign_verify] round {rnd}: copied {copied} from {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
