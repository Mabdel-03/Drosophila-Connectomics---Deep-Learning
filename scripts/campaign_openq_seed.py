"""Stage-6 `seed_spec` wrapper: capture the three-open-questions spec as seed_spec.json.

Deterministic single source of truth every round reads. The three questions and their
seeds/datasets are fixed; the spec records them so the proposal/verify stages stay consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _campaign_common import env, parse_workspace_args, write_json  # noqa: E402


def main(argv=None) -> int:
    args = parse_workspace_args(argv)
    ws = Path(args.workspace)
    spec = {
        "seed_neurons": ["LLPC1", "Nod1", "DNp26", "LPLC2", "LC4",
                         "DNp01", "DNp03", "DNp04", "DNp06"],
        "dataset": env("CONNECTOME_DATASET", "FAFB+MCNS"),
        "target_muscles": "escape (tergotrochanter/TTM) vs wing-steering (hg/i/b)",
        "questions": {
            "q1": "Single-animal closure: does LLPC1->Nod1->DNp26 reproduce within MCNS alone?",
            "q2": "Bilateral/4-direction generalization: does the right-sheet circuit generalize "
                  "to the left sheet (and other directions) in FlyWire?",
            "q3": "Escape-route census to muscle: is the LPLC2/LC4 escape arm anatomically "
                  "separable (distinct DNs + muscles) from the LLPC1 steering arm?",
        },
        "thresholds": {
            "min_synapses": int(env("CONNECTOME_MIN_SYN", "1")),
            "escape_syn_floor": int(env("CONNECTOME_ESCAPE_FLOOR", "5")),
            "n_perm": int(env("CONNECTOME_N_PERM", "200")),
        },
        "goal": ("Resolve the three open anatomical questions left by Figure_Ground_Circuit.pdf "
                 "from connectome data (FlyWire FAFB v783 live CAVE + MCNS v1.0 offline), each "
                 "yielding a neutral falsifiable verdict, and emit a LaTeX->PDF report."),
    }
    write_json(ws / "seed_spec.json", spec)
    print(f"[campaign_openq_seed] wrote seed_spec.json: {len(spec['seed_neurons'])} seeds, 3 questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
