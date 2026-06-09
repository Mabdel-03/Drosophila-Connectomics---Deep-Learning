"""Validate the built artifacts and emit a data card + machine-readable schema.

Hard asserts (raise on failure):
  * N neurons == cfg.expected_neurons (139,255 for v783)
  * idx is exactly 0..N-1, unique root_ids
  * no dangling edge endpoints (0 <= idx < N)
  * thresholded (>=5 syn) connection count ~= cfg.expected_connections
    (2,700,513 for v783; checked with a small tolerance)

Soft reports (logged + written to the data card, never fatal): density, degree
stats, self-loops, NT coverage, signed weight mass per policy, reciprocity,
annotation-join health.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import Config
from ..io import load_csr, read_json, read_parquet, write_json
from . import nt_signs, schemas


class ValidationError(AssertionError):
    pass


def _git_commit(repo_root) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _hard(cond: bool, msg: str) -> None:
    if not cond:
        raise ValidationError(msg)


def validate(cfg: Config, *, now_iso: str | None = None) -> dict:
    paths = cfg.paths()
    neurons = read_parquet(paths.neurons)
    edges = read_parquet(paths.edges)
    N = len(neurons)

    # ---------------- HARD asserts ----------------
    _hard(
        N == cfg.expected_neurons,
        f"neuron count {N} != expected {cfg.expected_neurons}",
    )
    idx = neurons["idx"].to_numpy()
    _hard(np.array_equal(idx, np.arange(N)), "neurons.idx is not exactly 0..N-1 in order")
    _hard(neurons["root_id"].is_unique, "duplicate root_id in neuron table")

    for col in ("pre_idx", "post_idx"):
        lo, hi = int(edges[col].min()), int(edges[col].max())
        _hard(lo >= 0 and hi < N, f"edges.{col} out of range [0,{N}): saw [{lo},{hi}]")

    # Connection count: exact-ish with a small tolerance. The threshold logic is
    # deterministic, but pinning to a hand-copied figure is brittle across FlyWire
    # re-materializations, so allow +/-0.5% (still catches wrong-column / failed-
    # threshold bugs, which are off by 2-5x).
    n_edges = len(edges)
    tol = max(1000, int(0.005 * cfg.expected_connections))
    _hard(
        abs(n_edges - cfg.expected_connections) <= tol,
        f"thresholded (>= {cfg.synapse_threshold} syn) connection count {n_edges:,} "
        f"differs from expected {cfg.expected_connections:,} by more than {tol:,}",
    )

    # no-threshold edge count (informational; the full graph)
    n_edges_full = None
    if paths.edges_full.exists():
        n_edges_full = int(len(read_parquet(paths.edges_full)))

    # ---------------- SOFT reports ----------------
    self_loops = int((edges["pre_idx"] == edges["post_idx"]).sum())
    out_deg = np.asarray(np.bincount(edges["pre_idx"].to_numpy(), minlength=N))
    in_deg = np.asarray(np.bincount(edges["post_idx"].to_numpy(), minlength=N))

    nt_counts = (
        neurons["nt_canonical"].value_counts(dropna=False).to_dict()
        if "nt_canonical" in neurons else {}
    )
    nt_counts = {str(k): int(v) for k, v in nt_counts.items()}
    annotated = int(neurons["super_class"].notna().sum()) if "super_class" in neurons else 0

    # reciprocity: fraction of directed edges whose reverse also exists.
    pair = set(zip(edges["pre_idx"].tolist(), edges["post_idx"].tolist()))
    sample = edges if n_edges <= 2_000_000 else edges.sample(2_000_000, random_state=0)
    recip = np.mean([(b, a) in pair for a, b in zip(sample["pre_idx"], sample["post_idx"])])

    # signed weight mass per policy (read back the saved matrices).
    policy_stats = {}
    for policy in nt_signs.available_policies(cfg):
        A: sp.csr_matrix = load_csr(paths.adjacency_signed(policy))
        d = A.data
        policy_stats[policy] = {
            "nnz": int(A.nnz),
            "pos_mass": float(d[d > 0].sum()),
            "neg_mass": float(d[d < 0].sum()),
        }
    counts_A = load_csr(paths.adjacency_counts)

    report = {
        "version": cfg.version,
        "generated_utc": now_iso or datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(_repo_root()),
        "neurons": N,
        "annotated_neurons": annotated,
        "synapse_threshold": cfg.synapse_threshold,
        "edges": n_edges,
        "edges_no_threshold": n_edges_full,
        "total_synapses": int(edges["syn_count"].sum()),
        "density": n_edges / (N * N),
        "self_loops": self_loops,
        "out_degree": {"mean": float(out_deg.mean()), "max": int(out_deg.max()),
                       "zero": int((out_deg == 0).sum())},
        "in_degree": {"mean": float(in_deg.mean()), "max": int(in_deg.max()),
                      "zero": int((in_deg == 0).sum())},
        "reciprocity": float(recip),
        "nt_distribution": nt_counts,
        "counts_matrix_nnz": int(counts_A.nnz),
        "signed_policies": policy_stats,
    }
    return report


def _repo_root():
    from ..paths import REPO_ROOT
    return REPO_ROOT


def write_schema(cfg: Config) -> None:
    paths = cfg.paths()
    schema = {
        "version": cfg.version,
        "neurons.parquet": schemas.NEURONS_SCHEMA,
        "node_index_map.parquet": {"idx": "int64", "root_id": "int64"},
        "edges.parquet": schemas.EDGES_SCHEMA,
        "edges_thresholding": schemas.EDGES_NOTE,
        "synapse_threshold": cfg.synapse_threshold,
        "adjacency_*.npz / adjacency.pt": schemas.ADJACENCY_NOTE,
        "nt_policies": cfg.nt_policies,
        "default_nt_policy": cfg.default_nt_policy,
        "citation": cfg.citation,
    }
    write_json(paths.schema, schema)


def write_data_card(cfg: Config, report: dict) -> None:
    paths = cfg.paths()
    c = cfg.citation
    lines = []
    lines.append(f"# FlyWire FAFB Connectome — Data Card (v{cfg.version})\n")
    lines.append(f"_Generated {report['generated_utc']} · code commit `{report['git_commit']}`_\n")
    lines.append("## Provenance\n")
    lines.append(f"- **License:** {c.get('license', '?')} — attribution required.")
    lines.append(f"- **Connectivity:** {c.get('connectivity', '')}")
    lines.append(f"  (Zenodo {cfg.raw['zenodo']['doi']})")
    lines.append(f"- **Annotations:** {c.get('annotations', '')}")
    lines.append(f"  (GitHub `{cfg.raw['github']['repo']}` @ `{cfg.raw['github']['ref']}`)\n")
    lines.append("### Source files (md5-verified)\n")
    lines.append("| file | size (bytes) | md5 |")
    lines.append("|---|---|---|")
    for f in cfg.zenodo_files:
        lines.append(f"| `{f.key}` | {f.size} | `{f.md5}` |")
    lines.append("")
    lines.append("## Summary statistics\n")
    lines.append(f"- **Neurons (nodes):** {report['neurons']:,} "
                 f"({report['annotated_neurons']:,} with a super_class label)")
    lines.append(f"- **Directed connections (canonical, >= {report['synapse_threshold']} "
                 f"synapses):** {report['edges']:,}")
    if report.get("edges_no_threshold") is not None:
        lines.append(f"- **Directed edges (no threshold, full graph):** "
                     f"{report['edges_no_threshold']:,} "
                     f"(in `edges_full.parquet` / `adjacency_counts_full_csr.npz`)")
    lines.append(f"- **Total synapses (in edges):** {report['total_synapses']:,}")
    lines.append(f"- **Density:** {report['density']:.2e}")
    lines.append(f"- **Self-loops (autapses):** {report['self_loops']:,}")
    lines.append(f"- **Out-degree:** mean {report['out_degree']['mean']:.1f}, "
                 f"max {report['out_degree']['max']:,}, "
                 f"{report['out_degree']['zero']:,} sources with 0 out")
    lines.append(f"- **In-degree:** mean {report['in_degree']['mean']:.1f}, "
                 f"max {report['in_degree']['max']:,}, "
                 f"{report['in_degree']['zero']:,} sinks with 0 in")
    lines.append(f"- **Reciprocity:** {report['reciprocity']:.3f}\n")
    lines.append("### Neurotransmitter distribution (canonical)\n")
    for k, v in sorted(report["nt_distribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v:,}")
    lines.append("")
    lines.append("### Signed adjacency by policy\n")
    lines.append("| policy | nnz | +mass | -mass |")
    lines.append("|---|---|---|---|")
    for p, s in report["signed_policies"].items():
        lines.append(f"| {p} | {s['nnz']:,} | {s['pos_mass']:,.0f} | {s['neg_mass']:,.0f} |")
    lines.append("")
    lines.append("## Artifact orientation\n")
    lines.append(schemas.ADJACENCY_NOTE)
    lines.append("")
    paths.data_card.parent.mkdir(parents=True, exist_ok=True)
    paths.data_card.write_text("\n".join(lines))


def run(cfg: Config) -> dict:
    cfg.paths().ensure()
    write_schema(cfg)
    report = validate(cfg)
    write_data_card(cfg, report)
    print(f"[validate] PASS — N={report['neurons']:,}, E={report['edges']:,}")
    print(f"[validate] data card: {cfg.paths().data_card}")
    # Persist the raw report alongside the card for programmatic use.
    write_json(cfg.paths().reports / f"validation_report_v{cfg.version}.json", report)
    return report
