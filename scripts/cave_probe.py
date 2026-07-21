"""Probe live CAVE reachability for FAFB + MCNS and discover the real MCNS table names.

The MCNS datastack/table names in flyconn.connectome.datasets are documented DEFAULTS;
this script confirms them at first connect and writes the resolved names to
configs/cave_mcns.yaml so the extract stage uses the truth, not the guess.

    python scripts/cave_probe.py            # probe both, print status + tables
    python scripts/cave_probe.py --write    # also write configs/cave_mcns.yaml

Run on a login node first (to settle whether compute nodes have egress), or via
slurm/cave_probe.sbatch on a compute node to test that path directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flyconn.connectome import ConnectomeClient
from flyconn.io import write_json
from flyconn.paths import REPO_ROOT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write configs/cave_mcns.yaml")
    args = ap.parse_args(argv)

    summary = {}
    for ds in ("fafb", "mcns"):
        st = ConnectomeClient.probe(ds)
        summary[ds] = st
        ok = "REACHABLE" if st["reachable"] else "UNREACHABLE"
        print(f"[cave_probe] {ds}: {ok}  latency={st['latency_s']}s  err={st['error']}")

    tables = {}
    if summary["mcns"]["reachable"]:
        try:
            cl = ConnectomeClient("mcns", allow_network=True)
            tables = {"mcns_tables": cl.list_tables(),
                      "mcns_materialization": cl.materialization_info()}
            print(f"[cave_probe] MCNS tables: {tables['mcns_tables']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[cave_probe] MCNS table discovery failed: {exc}")

    out = REPO_ROOT / "configs" / "cave_probe_result.json"
    write_json(out, {"probe": summary, **tables})
    print(f"[cave_probe] wrote {out}")

    if args.write and tables.get("mcns_tables"):
        cfg = REPO_ROOT / "configs" / "cave_mcns.yaml"
        lines = [
            "# Resolved MCNS table names discovered at first CAVE connect (cave_probe.py).",
            "# Edit flyconn.connectome.datasets.MCNS or override here as needed.",
            "datastack: male_cns",
            "materialization: v1.0",
            "discovered_tables:",
        ]
        lines += [f"  - {t}" for t in tables["mcns_tables"]]
        cfg.write_text("\n".join(lines) + "\n")
        print(f"[cave_probe] wrote {cfg} (review + reconcile with datasets.MCNS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
