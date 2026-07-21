"""Prefetch real FlyWire skeletons for the FD4 search report figures.

Skeletonizes all four Nod1 (= FD1) cells and the progressive-arm partners (T4a, T5a, the layer-a
sheet LLPC1, and the shared steering neuron DNp26) via ``fafbseg.flywire.skeletonize_neuron`` and
caches them to the project-tree live cache (``.flyconn_cache``). The report then renders real neuron
morphology (the four-cell quartet, the arbor+inputs, the circuit) from the cache.

All four Nod1 cells are fetched (``--per-type 0`` for Nod1) so the quartet figure showing the
population is homogeneous is skeleton-backed rather than a synapse-cloud proxy.

REQUIREMENTS (heavy deps live only in the flyconn_cave env):
  - run with: /orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python
  - FlyWire token at ~/.cloudvolume/secrets/chunkedgraph-secret.json

Usage:
  /orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python scripts/fetch_fd4_skeletons.py
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

from flyconn.paper import fw_access as FW  # noqa: E402
from flyconn.paper import skeleton_fetch as SK  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-nod1", type=int, default=4,
                    help="max Nod1 cells to skeletonize (default all 4, for the quartet figure)")
    ap.add_argument("--partners", default="T4a,T5a,LLPC1,DNp26,LPT42_Nod4",
                    help="progressive-arm partners (+ FD3 for the RF contrast) to skeletonize")
    ap.add_argument("--per-partner", type=int, default=2)
    ap.add_argument("--version", default="783")
    args = ap.parse_args()

    try:
        from fafbseg import flywire
        flywire.set_default_dataset("public")
    except Exception as e:  # noqa: BLE001
        print(f"[fetch] fafbseg unavailable: {e}")
        return 1

    meta = FW.NeuronMeta.load(args.version)
    src = FW.OfflineFlyWire()

    roots = sorted(int(r) for r in meta.root_ids_of_type(["Nod1"]))[: args.all_nod1 or None]
    for t in [p.strip() for p in args.partners.split(",") if p.strip()]:
        tr = sorted(int(r) for r in meta.root_ids_of_type([t]))
        roots.extend(tr[: args.per_partner] if args.per_partner else tr)
    roots = sorted(set(roots))
    print(f"[fetch] skeletonizing {len(roots)} cells (4 Nod1 + progressive-arm partners)")

    ok = 0
    for r in roots:
        t0 = time.time()
        skel = SK.fetch_skeleton(src, r)
        if skel is not None and skel.ok:
            ok += 1
            print(f"  [{ok}/{len(roots)}] {r}: {len(skel.vertices_um)} nodes ({time.time()-t0:.0f}s)")
        else:
            print(f"  {r}: FAILED ({getattr(SK.fetch_skeleton, 'last_skip_reason', '?')}) "
                  f"({time.time()-t0:.0f}s)")
    print(f"[fetch] cached {ok}/{len(roots)} skeletons")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
