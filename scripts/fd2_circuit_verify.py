"""Stage 11 - FD2 identification and full circuit: LPT21 is Egelhaaf-1985 FD2.

Runs the FD2 family (``src/flyconn/paper/derive/fd2_lpt21.py`` + its oracle): the LPT21 identity with a
decomposed confidence interval, a global uniqueness scan, the dual-output (frontal-branch) test, and
the full photoreceptor-to-motor circuit (afferent P, sheet Q, gate R, efferent L + male-CNS motor).
Writes a self-contained JSON to "11 - FD2 Identification/fd2_circuit.json" that the report reads
token-free.

The offline track is the primary source of the numbers; the live/v630 tracks confirm the key
laterality (contra%) measurement, which is data-source sensitive. If no CAVE token is present the live
confirmation is omitted and the report says so.

Usage (flyconn_cave env; token for the live/v630 confirmation):
  /orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python scripts/fd2_circuit_verify.py
  python scripts/fd2_circuit_verify.py --offline            # offline only (no live/v630)
  python scripts/fd2_circuit_verify.py --report --with-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from flyconn.io import write_json  # noqa: E402
from flyconn.paper import fw_access as FW  # noqa: E402
from flyconn.paper.derive import fd2_lpt21 as F2  # noqa: E402
from flyconn.paper.derive import k_fd3_lpt42 as K  # noqa: E402
from flyconn.paper.oracle import fd2_lpt21 as F2O  # noqa: E402

STAGE = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics/11 - FD2 Identification")


def _contra_on(src, meta, candidate: str):
    """LPT21 contra-output % on a given source (live/v630 identity confirmation)."""
    try:
        prof = K._type_profile(K._CachedSource(src), meta, candidate)
        return prof.get("contra_output_pct")
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="offline only; skip live/v630 confirmation")
    ap.add_argument("--report", action="store_true", help="also compile the PDF after verifying")
    ap.add_argument("--with-figures", action="store_true", help="render anatomical figures (needs source)")
    ap.add_argument("--stage", default=str(STAGE))
    args = ap.parse_args()

    meta = FW.NeuronMeta.load("783")
    off_src = FW.make_source("offline")

    print("[fd2] running the FD2=LPT21 identity + full circuit (offline primary) ...")
    derived = F2.run(off_src, meta)
    claims = F2O.build_claims(derived)

    conf = derived.get("confidence", {})
    uniq = derived.get("uniqueness", {})
    eff = derived.get("efferent", {})
    print(f"[fd2] confidence point={conf.get('point_estimate')} interval={conf.get('interval')} "
          f"({conf.get('n_matched')}/{conf.get('n_properties')} properties)")
    print(f"[fd2] uniqueness: fd2_hits={uniq.get('n_fd2_hits')} unique={uniq.get('unique')} "
          f"competitors={[c['type'] for c in uniq.get('competitors', [])]}")
    d = eff.get("direct", {})
    top = d.get("ranking", [{}])[0].get("cell_type") if d.get("ranking") else None
    print(f"[fd2] afferent T4/T5={derived['afferent']['census'].get('t4t5_frac_of_total')}%; "
          f"sheet={derived['sheet'].get('named_sheet')}; gate={derived['gate'].get('named_inhibitor')}; "
          f"top DN={top}; motor={eff.get('motor', {}).get('dominant_motor_system')}; "
          f"dual_output={derived['dual_output'].get('both_bimodal')}")

    # live / v630 contra confirmation of the homolateral (FD2) projection.
    confirm = {"offline": derived["identity"].get("contra_output_pct")}
    if not args.offline:
        try:
            live = FW.make_source("live")
            if getattr(live, "track", None) == "live":
                confirm["live"] = _contra_on(live, meta, F2.CANDIDATE)
                confirm["v630"] = _contra_on(FW.LiveCaveFlyWire(mat_version=630), meta, F2.CANDIDATE)
            else:
                print("[fd2] no live CAVE source (token absent); offline only")
        except Exception as e:  # noqa: BLE001
            print(f"[fd2] live confirmation unavailable ({type(e).__name__}: {e})")
    derived["identity_contra_tracks"] = confirm
    print(f"[fd2] contra% tracks (homolateral confirmation): {confirm}")

    out = {"meta": {"flywire_version": "783", "flywire_track": derived.get("track"),
                    "artifact": "fd2_circuit"},
           "families": {"FD2": {"title": "LPT21 is the modern correlate of Egelhaaf-1985 FD2",
                                "derived": derived,
                                "claims": [c.to_dict() for c in claims]}}}

    stage = Path(args.stage)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    out_path = stage / "fd2_circuit.json"
    write_json(out_path, out)
    print(f"[fd2] wrote {out_path}")

    if args.report:
        from flyconn.fd2_report import combined as REP
        src, m = (FW.make_source("auto"), meta) if args.with_figures else (None, None)
        res = REP.build_report(str(stage), str(out_path), src=src, meta=m)
        print(f"[fd2] tex={res.get('tex')} pdf={res.get('pdf')} compiled={res.get('compiled')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
