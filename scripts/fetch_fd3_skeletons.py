"""Prefetch real FlyWire skeletons for the FD3 morphology claims (Family K).

Skeletonizes the FD3 candidate (LPT42_Nod4) + the FD1 anchor (Nod1) cells via
``fafbseg.flywire.skeletonize_neuron`` (meshes through CloudVolume, skeletonizes locally with
skeletor) and caches them to the project-tree live cache (``.flyconn_cache``). The verification
pipeline then reads the cache in any env and computes skeleton-backed morphology claims.

REQUIREMENTS (the heavy deps live ONLY in the flyconn_cave env):
  - run with: /orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python
  - FlyWire token at ~/.cloudvolume/secrets/chunkedgraph-secret.json (see README / the
    set-token step); the same 32-char token as cave-secret.json works.

Usage:
  /orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python scripts/fetch_fd3_skeletons.py
  # optional: --types LPT42_Nod4,Nod1   (default) ; --version 783
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
    ap.add_argument("--types", default="LPT42_Nod4,Nod1,T4b,T5b,LPC1,LPi14,DNp26",
                    help="comma-separated cell types to skeletonize (FD3 + FD1 anchor + the "
                         "afferent/efferent chain partners for the anatomical circuit figures)")
    ap.add_argument("--per-type", type=int, default=2,
                    help="max cells to skeletonize per type (exemplars for the render; 0 = all)")
    ap.add_argument("--version", default="783")
    args = ap.parse_args()

    try:
        from fafbseg import flywire
        flywire.set_default_dataset("public")
    except Exception as e:  # noqa: BLE001
        print(f"[fetch] fafbseg unavailable: {e}\n"
              f"        install it in flyconn_cave: "
              f"/orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/pip install fafbseg")
        return 1

    meta = FW.NeuronMeta.load(args.version)
    # A dummy source object: fetch_skeleton's fafbseg path needs no live client.
    src = FW.OfflineFlyWire()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    roots = []
    for t in types:
        t_roots = sorted(int(r) for r in meta.root_ids_of_type([t]))
        if args.per_type and len(t_roots) > args.per_type:
            t_roots = t_roots[:args.per_type]  # exemplars are enough for the circuit render
        roots.extend(t_roots)
    roots = sorted(set(roots))
    print(f"[fetch] skeletonizing {len(roots)} cells across types {types} "
          f"(<= {args.per_type or 'all'} per type)")

    ok = 0
    for r in roots:
        t0 = time.time()
        skel = SK.fetch_skeleton(src, r)
        if skel is not None and skel.ok:
            ok += 1
            print(f"  [{ok}/{len(roots)}] {r}: {len(skel.vertices_um)} nodes, "
                  f"{len(skel.edges)} edges, source={skel.source} ({time.time()-t0:.0f}s)")
        else:
            print(f"  {r}: FAILED ({SK.fetch_skeleton.last_skip_reason}) ({time.time()-t0:.0f}s)")
    print(f"[fetch] cached {ok}/{len(roots)} skeletons to the project-tree live cache")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
