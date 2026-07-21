"""Stage 12 - the connectomic search for Egelhaaf-1985 FD4, and its (negative) result.

Runs the FD4 family (``src/flyconn/paper/derive/fd4_nod1.py`` + its oracle). FD4 shares FD1's
entire progressive, heterolateral, cholinergic, noduli-group output class, so it cannot be
separated from FD1 by direction, transmitter, or output side. This script measures, on offline
v783 (primary) and optionally live CAVE + v630 (confirmation):
  * the exhaustive candidate elimination (the layer-a low-copy funnel over all 8,806 types),
  * the Nod1 homogeneity test (are the 4 Nod1 cells a separable FD1 + FD4 split, or one
    homogeneous FD1 population?),
  * the FD4 phenotype crosswalk on the best available candidate,
  * a decomposed, null-aware confidence interval, and
  * the progressive figure arm (T4a/T5a -> LLPC1 -> Nod1/FD1 -> DNp26 -> wing, VCH gate) that any
    FD4 correlate would use.

Writes a self-contained JSON to "12 - FD4 Identification/fd4_circuit.json" that the report reads
token-free.

Usage (flyconn_cave env; token for the live/v630 confirmation):
  /orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python scripts/fd4_circuit_verify.py
  python scripts/fd4_circuit_verify.py --offline            # offline only (no live/v630)
  python scripts/fd4_circuit_verify.py --report --with-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from flyconn.io import write_json  # noqa: E402
from flyconn.paper import fw_access as FW  # noqa: E402
from flyconn.paper.derive import fd4_nod1 as F4  # noqa: E402
from flyconn.paper.derive import k_fd3_lpt42 as K  # noqa: E402
from flyconn.paper.oracle import fd4_nod1 as F4O  # noqa: E402

STAGE = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics/12 - FD4 Identification")


def _split_on(src, meta):
    """Re-run the Nod1 homogeneity test on a given source (live/v630 confirmation)."""
    try:
        sp = F4._nod1_split(K._CachedSource(src), meta)
        return {"separable": sp.get("separable"), "homogeneous": sp.get("homogeneous"),
                "best_silhouette": sp.get("best_pairing", {}).get("silhouette"),
                "same_side_jac": sp.get("same_side_input_jaccard"),
                "cross_side_jac": sp.get("cross_side_input_jaccard")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="offline only; skip live/v630 confirmation")
    ap.add_argument("--no-circuit", action="store_true", help="skip the progressive-arm circuit trace")
    ap.add_argument("--report", action="store_true", help="also compile the PDF after verifying")
    ap.add_argument("--with-figures", action="store_true", help="render anatomical figures (needs source)")
    ap.add_argument("--stage", default=str(STAGE))
    args = ap.parse_args()

    meta = FW.NeuronMeta.load("783")
    off_src = FW.make_source("offline")

    print("[fd4] running the FD4 search + progressive-arm circuit (offline primary) ...")
    derived = F4.run(off_src, meta, with_circuit=not args.no_circuit)
    claims = F4O.build_claims(derived)

    sp = derived["nod1_split"]; sc = derived["candidate_screen"]; cf = derived["confidence"]
    print(f"[fd4] candidate elimination: {sc.get('n_survivors')} survivor(s) {sc.get('survivors')} "
          f"of {sc.get('n_layer_a_lowcopy_screened')} layer-a low-copy types "
          f"(only Nod1 = {sc.get('only_survivor_is_prog_type')})")
    print(f"[fd4] Nod1 split: separable={sp.get('separable')} homogeneous={sp.get('homogeneous')} "
          f"best_silhouette={sp.get('best_pairing', {}).get('silhouette')} "
          f"same/cross-side Jaccard={sp.get('same_side_input_jaccard')}/{sp.get('cross_side_input_jaccard')}")
    print(f"[fd4] confidence: point={cf.get('point_estimate')} interval={cf.get('interval')} "
          f"verdict={cf.get('identity_verdict')}")
    if derived.get("afferent"):
        af = derived["afferent"]["census"]
        print(f"[fd4] progressive arm: layer-a T4/T5={af.get('t4t5_layer_frac', {}).get('a')}%; "
              f"sheet={derived['sheet'].get('named_sheet')}; gate={derived['gate'].get('named_inhibitor')}; "
              f"top DN={(derived['efferent'].get('direct', {}).get('ranking', [{}]) or [{}])[0].get('cell_type')}")

    # live / v630 confirmation of the homogeneity finding (the load-bearing negative result).
    confirm = {"offline": {"separable": sp.get("separable"), "homogeneous": sp.get("homogeneous"),
                           "best_silhouette": sp.get("best_pairing", {}).get("silhouette")}}
    if not args.offline:
        try:
            live = FW.make_source("live")
            if getattr(live, "track", None) == "live":
                confirm["live"] = _split_on(live, meta)
                confirm["v630"] = _split_on(FW.LiveCaveFlyWire(mat_version=630), meta)
            else:
                print("[fd4] no live CAVE source (token absent); offline only")
        except Exception as e:  # noqa: BLE001
            print(f"[fd4] live confirmation unavailable ({type(e).__name__}: {e})")
    derived["split_tracks"] = confirm
    print(f"[fd4] homogeneity tracks: {confirm}")

    out = {"meta": {"flywire_version": "783", "flywire_track": derived.get("track"),
                    "artifact": "fd4_circuit"},
           "families": {"FD4": {"title": ("The connectomic search for Egelhaaf-1985 FD4: a "
                                          "homogeneous Nod1/FD1 population and an honest null"),
                                "derived": derived,
                                "claims": [c.to_dict() for c in claims]}}}

    stage = Path(args.stage)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    out_path = stage / "fd4_circuit.json"
    write_json(out_path, out)
    print(f"[fd4] wrote {out_path}")

    # Evidence audit (claim -> verdict -> derived quantity), beside the report.
    try:
        from flyconn.fd4_report import audit as AUD
        aud = AUD.build_audit(out)
        write_json(stage / "fd4_full_circuit_audit.json", aud)
        (stage / "fd4_full_circuit_audit.md").write_text(AUD.audit_markdown(aud))
        print(f"[fd4] audit: {aud['verdict_counts']}")
    except Exception as e:  # noqa: BLE001
        print(f"[fd4] audit skipped ({type(e).__name__}: {e})")

    if args.report:
        from flyconn.fd4_report import combined as REP
        src, m = (FW.make_source("auto"), meta) if args.with_figures else (None, None)
        res = REP.build_report(str(stage), str(out_path), src=src, meta=m)
        print(f"[fd4] tex={res.get('tex')} pdf={res.get('pdf')} compiled={res.get('compiled')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
