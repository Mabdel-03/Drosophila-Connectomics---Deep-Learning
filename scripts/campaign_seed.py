"""Stage `seed_spec` wrapper: write seed_spec.json from the campaign env.

Captures the scientific target (seed DNs, dataset, target muscles, thresholds) so every
downstream round reads a single source of truth. Deterministic; no LLM needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _campaign_common import env, parse_workspace_args, write_json  # noqa: E402


def main(argv=None) -> int:
    args = parse_workspace_args(argv)
    ws = Path(args.workspace)
    seed_neurons = [s.strip() for s in (env("CONNECTOME_SEED_NEURONS", "DNbe001,DNp26") or "").split(",") if s.strip()]
    spec = {
        "seed_neurons": seed_neurons,
        "dataset": env("CONNECTOME_DATASET", "MCNS"),
        "target_muscles": (env("CONNECTOME_TARGET", "wing_muscles_bilateral")),
        "thresholds": {
            "min_synapses": int(env("CONNECTOME_MIN_SYN", "1")),
            "cleft_score_min": int(env("CONNECTOME_CLEFT_MIN", "0")),
            "robustness_tolerance_pct": float(env("CONNECTOME_TOL_PCT", "20")),
        },
        "hops_budget": int(env("CONNECTOME_HOPS", "2")),
        "goal": ("Complete Figure_Ground_Circuit.pdf: connect VCH->T4/T5->LLPC1->Nod1->DNs "
                 "to the full bilateral wing musculature by tracing DN->motor-neuron->wing-muscle "
                 "in the male-CNS (MCNS) connectome, for BOTH wings."),
    }
    write_json(ws / "seed_spec.json", spec)
    print(f"[campaign_seed] wrote seed_spec.json: seeds={seed_neurons} dataset={spec['dataset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
