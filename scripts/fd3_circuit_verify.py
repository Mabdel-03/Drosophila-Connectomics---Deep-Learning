"""Stage 7 — FD3 full-circuit assembly: existence screen + cell-identity manifest.

Composes the already-verified FD3 families into a single input->output artifact:
  * a 7-stage circuit EXISTENCE SCREEN (photoreceptor -> lamina/medulla -> layer-b T4b/T5b ->
    FD3 -> descending -> wing motor), each PRESENT/WEAK/ABSENT with a GO/PARTIAL/NO_GO decision;
  * a CELL-IDENTITY MANIFEST naming every root id of FD3's afferent circuit with provenance.

The input tiers (stages 1-4) are derived here from Family P; the identity/output tiers (5-7)
are read from the K/L blocks of the verification JSON (no recompute). Emits to
"7 - FD3 Identification/": fd3_circuit_screen.json, cell_identity_manifest.json.

Usage:
  # live CAVE primary (flyconn_cave env + token)
  /orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python scripts/fd3_circuit_verify.py
  python scripts/fd3_circuit_verify.py --offline
  python scripts/fd3_circuit_verify.py --results "5 - Paper Verification/verification_results.json"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from flyconn.io import read_json, write_json  # noqa: E402
from flyconn.paper import fw_access as FW  # noqa: E402
from flyconn.paper.derive import fd3_circuit_screen as SC  # noqa: E402
from flyconn.paper.derive import identities as ID  # noqa: E402

STAGE = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics/7 - FD3 Identification")


def _family_block(results: dict, fam: str) -> dict:
    if not results:
        return {}
    if "families" in results and fam in results["families"]:
        return results["families"][fam]
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="force offline FlyWire source")
    ap.add_argument("--results", default="5 - Paper Verification/verification_results.json",
                    help="verification JSON providing the K/L blocks for screen stages 5-7")
    ap.add_argument("--stage", default=str(STAGE))
    args = ap.parse_args()

    prefer = "offline" if args.offline else "auto"
    src = FW.make_source(prefer)
    meta = FW.NeuronMeta.load("783")

    # K/L/Q/R blocks for the named stages (optional; screen reports ABSENT if missing).
    try:
        results = read_json(Path(args.results))
    except FileNotFoundError:
        results = {}
    k_block = _family_block(results, "K")
    l_block = _family_block(results, "L")
    q_block = _family_block(results, "Q")
    r_block = _family_block(results, "R")

    # The identity stage reads K's aggregate verdict from its CLAIMS; the descending/motor, sheet
    # (Q) and inhibitor (R) stages read their DERIVED blocks.
    k_for_screen = {"claims": k_block.get("claims", [])} if k_block else {}
    l_for_screen = l_block.get("derived", {}) if l_block else {}
    q_for_screen = q_block.get("derived", {}) if q_block else {}
    r_for_screen = r_block.get("derived", {}) if r_block else {}

    print(f"[fd3-circuit] source track = {src.track}")
    print("[fd3-circuit] running the named-circuit existence screen ...")
    screen = SC.run(src, meta, k_derived=k_for_screen, l_derived=l_for_screen,
                    q_derived=q_for_screen, r_derived=r_for_screen)
    print(f"[fd3-circuit] decision = {screen['input_decision']} "
          f"({screen['n_present']} stages PRESENT; first_break={screen['first_break']})")
    for s in screen["stages"]:
        print(f"    {s['stage']:20s} {s['label']}  {s.get('detail','')}")

    print("[fd3-circuit] building the cell-identity manifest ...")
    manifest = ID.build_fd3_input_manifest(src, meta)

    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    write_json(stage / "fd3_circuit_screen.json", screen)
    write_json(stage / "cell_identity_manifest.json", manifest)
    print(f"[fd3-circuit] wrote {stage/'fd3_circuit_screen.json'} and "
          f"{stage/'cell_identity_manifest.json'}")


if __name__ == "__main__":
    main()
