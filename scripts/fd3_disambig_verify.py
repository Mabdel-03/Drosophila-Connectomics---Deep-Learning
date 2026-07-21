"""Stage 10 — FD3 disambiguation: is FD3 the FlyWire cell LPT42_Nod4 or Nod3?

Runs the KD family (``src/flyconn/paper/derive/kd_fd3_disambig.py`` + its oracle) which measures every
FD3-defining property for BOTH candidates on the offline (primary), live (confirm) and v630
(cross-version) tracks, then applies a pre-registered, two-sided decision rule. Writes a self-contained
JSON to "10 - FD3 vs Nod3 Disambiguation/fd3_disambig.json" that the report reads token-free.

The offline track is the primary source of the numbers; the live/v630 tracks confirm the laterality
separation (the contra% metric is data-source sensitive). If no CAVE token is present the live column
is omitted and the report says so. The cached Family-K derived block (from the paper-verification JSON)
is copied into the output so the report's shared layer-composition and family-screen figures render.

Usage (flyconn_cave env + token for the live/v630 tracks):
  /orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python scripts/fd3_disambig_verify.py
  python scripts/fd3_disambig_verify.py --offline           # offline only (no live/v630)
  python scripts/fd3_disambig_verify.py --report            # also build the PDF after verifying
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from flyconn.io import read_json, write_json  # noqa: E402
from flyconn.paper import fw_access as FW  # noqa: E402
from flyconn.paper.derive import kd_fd3_disambig as KD  # noqa: E402
from flyconn.paper.oracle import kd_fd3_disambig as KDO  # noqa: E402

STAGE = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics/10 - FD3 vs Nod3 Disambiguation")
K_RESULTS = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics/5 - Paper Verification/verification_results.json")


def _k_derived() -> dict | None:
    """The cached Family-K derived block (for the report's shared layer/screen figures)."""
    for p in (K_RESULTS, Path("5 - Paper Verification/verification_results.json")):
        try:
            r = read_json(p)
            fam = r.get("families", {}).get("K")
            if fam and "derived" in fam:
                return {"derived": fam["derived"], "claims": fam.get("claims", [])}
        except FileNotFoundError:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="offline primary only; skip the live/v630 confirmation tracks")
    ap.add_argument("--report", action="store_true", help="also compile the PDF after verifying")
    ap.add_argument("--stage", default=str(STAGE))
    args = ap.parse_args()

    meta = FW.NeuronMeta.load("783")

    # Primary track: offline (reproducible, no token). Live/v630 confirm when a token is present.
    off_src = FW.make_source("offline")
    live_src = None
    if not args.offline:
        try:
            cand = FW.make_source("live")
            if getattr(cand, "track", None) == "live":
                live_src = cand
            else:
                print("[fd3-disambig] no live CAVE source (token absent); offline only")
        except Exception as e:  # noqa: BLE001
            print(f"[fd3-disambig] live source unavailable ({type(e).__name__}: {e}); offline only")

    print(f"[fd3-disambig] running KD on offline primary"
          + (" + live/v630 confirm" if live_src is not None else " only") + " ...")
    derived = KD.run(off_src, meta, live_src=live_src)
    claims = KDO.build_claims(derived)

    verdict = next((c for c in claims if c.id == "KD.decision_verdict"), None)
    if verdict is not None:
        print(f"[fd3-disambig] DECISION: {verdict.computed_primary}")
    cons = derived.get("constructive", {})
    print(f"[fd3-disambig] FD2 candidate = {cons.get('fd2_candidate')} "
          f"(LPT21 fine {cons.get('lpt21_fd2_fine')}/2, Nod3 fine {cons.get('nod3_fd2_fine')}/2); "
          f"Nod3 identity = {cons.get('nod3_identity')}; FD3 = {cons.get('fd3_best_candidate')}")
    off = derived.get("candidates_offline", {})
    print("[fd3-disambig] offline contra%: "
          f"LPT21={off.get('LPT21', {}).get('contra_output_pct')} "
          f"Nod3={off.get('Nod3', {}).get('contra_output_pct')} "
          f"LPT42_Nod4={off.get('LPT42_Nod4', {}).get('contra_output_pct')}")

    # Assemble the token-free report JSON: KD block + a copy of the cached K block.
    families = {"KD": {"title": "Disambiguation: FD3 is LPT42_Nod4, not Nod3 (Nod3 = FD2)",
                       "derived": derived,
                       "claims": [c.to_dict() for c in claims]}}
    k_block = _k_derived()
    if k_block is not None:
        families["K"] = {"title": "LPT42_Nod4 is the modern correlate of Egelhaaf-1985 FD3",
                         **k_block}
    else:
        print("[fd3-disambig] WARNING: cached Family-K block not found; the shared "
              "layer-composition / family-heatmap figures will be skipped in the report")

    out = {"meta": {"flywire_version": "783", "flywire_track": derived.get("track"),
                    "artifact": "fd3_disambig"},
           "families": families}

    stage = Path(args.stage)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    out_path = stage / "fd3_disambig.json"
    write_json(out_path, out)
    print(f"[fd3-disambig] wrote {out_path}")

    if args.report:
        from flyconn.fd3_report import disambig as REP
        res = REP.build_report(str(stage), str(out_path))
        print(f"[fd3-disambig] tex={res.get('tex')} pdf={res.get('pdf')} compiled={res.get('compiled')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
