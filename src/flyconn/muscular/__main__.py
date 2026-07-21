"""Stage-5 CLI: trace the figure-ground circuit to the bilateral wing musculature.

  python -m flyconn.muscular extract   # live CAVE (MCNS): DN->MN edges + annotations -> parquets
  python -m flyconn.muscular verify    # offline: re-derive S14/S15, run nulls + coverage, emit deliverables

``extract`` needs the [cave] extra + a token and network; it populates the scratch cache
and the muscular/ intermediate parquets. ``verify`` reads only those parquets (cache-only),
so it runs in seconds with no network. The campaign wrappers call these two entrypoints.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..circuit import report as R
from ..io import read_parquet, write_json, write_parquet
from ..motif import compare as K
from . import muscular_config as C
from . import spec as S
from . import trace as T
from . import verify as V

REPO = Path(__file__).resolve().parents[3]


def _stage_dir() -> Path:
    d = REPO / "5 - Muscular Projection"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _df_to_md(df: pd.DataFrame) -> str:
    """Markdown table without depending on the optional ``tabulate`` package."""
    if df.empty:
        return "_no rows_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _channel_map() -> dict:
    """dn cell_type -> descending channel, from the oracle channel tables."""
    m = {}
    for ch, rows in (("direct", C.DIRECT_CHANNEL), ("nodtype", C.NODTYPE_CHANNEL),
                     ("broadcast", C.BROADCAST_CHANNEL)):
        for dn, *_ in rows:
            m.setdefault(dn, ch)
    return m


# ---------------------------------------------------------------------------
# extract (offline MaleCNS flat files)
# ---------------------------------------------------------------------------
# The male-CNS connectome is distributed as public bulk flat files (NOT a CAVE datastack),
# already downloaded to scratch and read via flyconn.paper.malecns. This is the same
# offline source the paper-verification track used. The brain-side figure->DN edges (FAFB)
# are the only live-CAVE part of the pipeline; the DN->MN->muscle motor map is offline.
def cmd_extract(args) -> int:
    import pandas as pd

    from ..paper.malecns import client as M  # offline MaleCNS flat-file access

    t0 = time.time()
    out = C.muscular_dir(C.VERSION)
    dns = list(C.FIGURE_DNS)

    # 1) seed DN type -> MaleCNS bodies (with side). The DN-name bridge is the `type` string.
    dn_rows = []
    for dn in dns:
        b = M.bodies_of_type(dn)
        for _, r in b.iterrows():
            dn_rows.append({"root_id": int(r["bodyId"]), "cell_type": dn,
                            "somaSide": r.get("somaSide")})
    dn_roots = pd.DataFrame(dn_rows, columns=["root_id", "cell_type", "somaSide"])
    write_parquet(dn_roots, out / "mcns_dn_root_ids.parquet")

    # 2) DN -> motor-neuron edges (onto annotated motor neurons), normalised to the
    #    verifier's column vocabulary (pre/post_pt_root_id, syn_count).
    motor = M.motor_neurons()
    e = M.dn_to_motor([int(x) for x in dn_roots["root_id"]], motor_df=motor)
    edges = pd.DataFrame({
        "pre_pt_root_id": e["body_pre"].astype("int64"),
        "post_pt_root_id": e["body_post"].astype("int64"),
        "syn_count": e["syn"].astype("int64"),
    }) if not e.empty else pd.DataFrame(columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    write_parquet(edges, out / "mcns_dn_mn_edges.parquet")

    # 3) MN annotations for the contacted motor neurons: root_id, cell_type(=muscle),
    #    somaSide, cell_sub_class(=subclass) -- the columns trace.dn_to_motor_neurons reads.
    contacted = sorted(set(int(x) for x in edges["post_pt_root_id"])) if not edges.empty else []
    mset = motor.set_index("bodyId")
    ann_rows = []
    for bid in contacted:
        if bid in mset.index:
            r = mset.loc[bid]
            ann_rows.append({"root_id": bid, "cell_type": r.get("muscle"),
                             "somaSide": r.get("somaSide"), "cell_sub_class": r.get("subclass")})
    mn_ann = pd.DataFrame(ann_rows, columns=["root_id", "cell_type", "somaSide", "cell_sub_class"])
    write_parquet(mn_ann, out / "mcns_mn_annotations.parquet")

    stats = {
        "dataset": "mcns",
        "cave_materialization": "v1.0",
        "synapse_table": "connectome-weights (flat file)",
        "n_seed_dns": int(len(dns)),
        "n_dn_cells": int(len(dn_roots)),
        "n_edges": int(len(edges)),
        "n_motor_neurons_contacted": int(len(contacted)),
        "source": "offline MaleCNS v1.0 flat files",
        "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
        "status": "ok",
    }
    write_json(out / "_muscular_extract_stats.json", stats)
    print(f"[muscular extract] n_edges={stats['n_edges']} dns={stats['n_dn_cells']} "
          f"mns={stats['n_motor_neurons_contacted']} in {stats['elapsed_s']}s")
    return 0


# ---------------------------------------------------------------------------
# verify (offline)
# ---------------------------------------------------------------------------
def cmd_verify(args) -> int:
    t0 = time.time()
    src = C.muscular_dir(C.VERSION)
    stage = _stage_dir()
    figs = stage / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    dn_roots = read_parquet(src / "mcns_dn_root_ids.parquet")
    edges = read_parquet(src / "mcns_dn_mn_edges.parquet")
    mn_ann = read_parquet(src / "mcns_mn_annotations.parquet")

    dn_mn = T.dn_to_motor_neurons(edges, dn_roots, mn_ann)
    dn_mn_wing = T.split_by_wing(dn_mn)

    res = V.run_all(dn_mn_wing, n_perm=args.n_perm)
    claims = res["claims"]
    cov = res["coverage"]

    # Final dataset tables
    lat = T.dn_wing_laterality(dn_mn_wing)
    bilateral = T.bilateral_muscle_map(dn_mn_wing, channel_map=_channel_map())
    write_parquet(bilateral, stage / "edges.parquet")
    write_parquet(dn_mn_wing, stage / "mn_targets.parquet")
    write_parquet(lat, stage / "bilateral_split.parquet")
    if "muscle" in dn_mn_wing.columns:
        muscle_assign = (dn_mn_wing.groupby(["dn", "muscle", "motor_system", "wing_rel"],
                                            dropna=False)["syn_count"].sum().reset_index())
        write_parquet(muscle_assign, stage / "muscle_assignments.parquet")

    meta = {
        "stage": "5 - Muscular Projection",
        "datasets": {"mcns": {"materialization": "v1.0"}, "fafb": {"materialization": 783}},
        "seeds": list(C.SEED_DNS),
        "tracks": {"primary": "all-synapse CAVE", "secondary": "min_syn>=5"},
        "verdict_counts": K.verdict_counts(claims),
        "elapsed_s": round(time.time() - t0, 1),
    }
    payload = R.write_results_json(stage / "verification_results.json", meta, claims,
                                   details=res["details"])
    R.write_coverage_json(stage / "coverage.json",
                          bilateral_coverage=cov["bilateral_coverage"],
                          muscle_chain_complete=cov["muscle_chain_complete"],
                          dn_to_mn_edges=cov["dn_to_mn_edges"],
                          mn_to_muscle_edges=cov["mn_to_muscle_edges"],
                          missing=cov["missing"])
    R.write_markdown(stage / "VERIFICATION.md", "Muscular Projection", meta, claims,
                     sections=[("Per-DN wing laterality (S14)", _df_to_md(lat))])

    print(f"[muscular verify] verdicts={meta['verdict_counts']} "
          f"refuted={payload['refuted_claims']} "
          f"bilateral_coverage={cov['bilateral_coverage']:.2f} "
          f"chain={cov['muscle_chain_complete']:.2f}")
    return 0


def _check_data_root_writable() -> None:
    """Fail fast with an actionable message if the resolved data root is not writable.

    Stage 5 writes the CAVE cache + MCNS intermediates under FLYCONN_DATA_ROOT. The repo
    default (flyconn.paths) is personal scratch, which is over quota; Stage-5 callers
    export FLYCONN_DATA_ROOT to the group volume. If that didn't happen we get a cryptic
    'Disk quota exceeded' deep in a parquet write -- catch it here instead.
    """
    from ..paths import data_root  # lazy
    root = data_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"[flyconn.muscular] data root not writable: {root}\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "  Set FLYCONN_DATA_ROOT to a writable location (Stage 5 uses the group volume\n"
            "  /orcd/data/tpoggio/001/mabdel03/connectome_data; personal scratch is over quota)."
        ) from exc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flyconn.muscular", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="live CAVE: DN->MN edges + annotations -> parquets")
    pe.add_argument("--cache-only", action="store_true",
                    help="read from the populated cache instead of the network")
    pe.set_defaults(func=cmd_extract)

    pv = sub.add_parser("verify", help="offline: re-derive S14/S15, nulls, coverage, deliverables")
    pv.add_argument("--n-perm", type=int, default=500)
    pv.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    _check_data_root_writable()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
