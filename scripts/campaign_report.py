"""Stage `report_rN` wrapper: compile the extended verified dataset (runs only post-gate).

Reaches this stage only when verify_rN passed the gate (all CONFIRMED/with-caveat, full
bilateral coverage). Assembles muscular_projection_dataset.json from the verifier outputs
and the Stage-5 parquets, and recomputes verdict_counts from claims[] as an integrity
check (so a tampered refuted_claims list cannot slip through).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _campaign_common import (  # noqa: E402
    copy_into_workspace, env, parse_workspace_args, read_json, stage_artifacts, write_json,
)


def main(argv=None) -> int:
    args = parse_workspace_args(argv)
    ws = Path(args.workspace)
    rnd = env("CONNECTOME_ROUND", "1")
    src = stage_artifacts()

    # Read THIS round's verify outputs from the workspace (the runner copies verify_rN's
    # success_artifacts here via context_from). Fall back to the canonical stage dir only
    # for manual runs. Reading the workspace avoids a stale read when rounds interleave.
    def _round_file(name: str):
        w = ws / name
        if w.exists():
            return read_json(w)
        c = src / name
        return read_json(c) if c.exists() else {}

    vr = _round_file("verification_results.json")
    cov = _round_file("coverage.json")

    # Integrity check: recompute verdict_counts from claims[]; refuse to certify if the
    # flat refuted_claims list disagrees (defends against a repair that zeroed the list).
    claims = vr.get("claims", [])
    recomputed = dict(Counter(c.get("verdict") for c in claims))
    declared_refuted = set(vr.get("refuted_claims", []))
    actual_refuted = {c["id"] for c in claims if c.get("verdict") == "REFUTED"}
    integrity_ok = declared_refuted == actual_refuted

    dataset = {
        "stage": "5 - Muscular Projection",
        "round": rnd,
        "verdict_summary": recomputed,
        "coverage": cov,
        "provenance": {
            "datasets": vr.get("meta", {}).get("datasets", {}),
            "seeds": vr.get("meta", {}).get("seeds", []),
            "tracks": vr.get("meta", {}).get("tracks", {}),
        },
        "integrity_check": {"declared_eq_actual_refuted": integrity_ok,
                            "actual_refuted": sorted(actual_refuted)},
    }
    # roll in the per-table parquet record counts if present
    for name in ("edges.parquet", "mn_targets.parquet", "bilateral_split.parquet",
                 "muscle_assignments.parquet"):
        p = src / name
        if p.exists():
            try:
                import pyarrow.parquet as pq
                dataset.setdefault("tables", {})[name] = pq.ParquetFile(p).metadata.num_rows
            except Exception:
                pass

    write_json(ws / "muscular_projection_dataset.json", dataset)
    # VERIFICATION.md already authored by the verifier; surface it as a success artifact.
    copy_into_workspace(ws, ["VERIFICATION.md", "muscular_projection_dataset.json"])
    # write the dataset manifest into the canonical stage dir too
    write_json(src / "muscular_projection_dataset.json", dataset)

    if not integrity_ok:
        print(f"[campaign_report] INTEGRITY FAIL: refuted list mismatch "
              f"declared={declared_refuted} actual={actual_refuted}")
        return 3
    print(f"[campaign_report] round {rnd}: dataset certified; verdicts={recomputed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
