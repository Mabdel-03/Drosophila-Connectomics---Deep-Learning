"""Generic extract -> derive -> compare driver for a CircuitSpec.

Consumes a ``CircuitSpec`` plus one ``ConnectomeClient`` per dataset and produces
``ClaimResult``s via the existing ``motif.compare`` verdict engine. There is deliberately
no new verdict logic here: ``evaluate_claims`` is a thin dispatcher onto
``compare.compare_count/pct/ratio/categorical`` and ``check_table_sum``.

"Two tracks" mirrors the Stage-4 design: a PRIMARY track (all synapses / dataset default
threshold = the published-total analog) and a SECONDARY track (cleft/min-syn thresholded
= the robustness analog). The same derivation runs on both so a claim can be confirmed,
confirmed-with-caveat, or refuted with the source difference made explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..motif import compare as K
from .spec import CircuitSpec, EdgeClaim, NullModel, Seed

# Per-edge synapse floor for the secondary (robustness) track.
SECONDARY_MIN_SYN = 5


# ---------------------------------------------------------------------------
# Seed resolution
# ---------------------------------------------------------------------------
def resolve_selector(client, selector: dict) -> np.ndarray:
    """Resolve a selector dict to an array of root_ids against one dataset."""
    if "root_ids" in selector:
        return np.asarray(sorted(int(x) for x in selector["root_ids"]), dtype=np.int64)
    if "cell_type" in selector:
        ann = client.cell_annotations()
        ct = str(selector["cell_type"])
        ids = ann.loc[ann.get("cell_type", pd.Series(dtype=object)) == ct, "root_id"]
        return np.asarray(sorted(int(x) for x in ids), dtype=np.int64)
    raise ValueError(f"Unsupported selector {selector!r}; expected 'root_ids' or 'cell_type'")


def resolve_seeds(spec: CircuitSpec, clients: dict) -> dict[str, np.ndarray]:
    """Map each Seed.name -> resolved root_ids (cached through the client)."""
    out: dict[str, np.ndarray] = {}
    for s in spec.seeds:
        out[s.name] = resolve_selector(clients[s.dataset], s.selector)
    return out


# ---------------------------------------------------------------------------
# Edge derivation (two tracks)
# ---------------------------------------------------------------------------
def _edge_syn_count(client, src_ids, dst_ids, *, min_syn: int) -> int:
    """Total synapses from src set -> dst set, at a per-edge synapse floor."""
    if len(src_ids) == 0 or len(dst_ids) == 0:
        return 0
    edges = client.connectivity(list(src_ids), direction="downstream", min_syn=min_syn)
    if edges.empty:
        return 0
    dst = set(int(x) for x in dst_ids)
    hit = edges[edges["post_pt_root_id"].isin(dst)]
    return int(hit["syn_count"].sum())


def _edge_n_partners(client, src_ids, dst_ids, *, min_syn: int) -> int:
    if len(src_ids) == 0 or len(dst_ids) == 0:
        return 0
    edges = client.connectivity(list(src_ids), direction="downstream", min_syn=min_syn)
    if edges.empty:
        return 0
    dst = set(int(x) for x in dst_ids)
    return int(edges[edges["post_pt_root_id"].isin(dst)]["post_pt_root_id"].nunique())


def derive_edge(spec, clients, claim: EdgeClaim, seeds: dict) -> dict:
    """Derive one EdgeClaim's metric on both tracks. Returns {primary, secondary}."""
    client = clients[claim.dataset]
    src = _ids_for(claim.src, seeds, clients[claim.dataset])
    dst = _ids_for(claim.dst, seeds, clients[claim.dataset])
    metric = claim.metric

    def _val(min_syn: int):
        if metric in ("syn_count", "frac", "ipsi_frac"):
            return _edge_syn_count(client, src, dst, min_syn=min_syn)
        if metric == "n_partners":
            return _edge_n_partners(client, src, dst, min_syn=min_syn)
        raise ValueError(f"Unknown metric {metric!r} for claim {claim.id}")

    return {"primary": _val(1), "secondary": _val(SECONDARY_MIN_SYN)}


def _ids_for(selector: dict, seeds: dict, client) -> np.ndarray:
    """A selector may reference a named seed ({'seed': 'DNp26'}) or be resolvable directly."""
    if "seed" in selector:
        return seeds[selector["seed"]]
    return resolve_selector(client, selector)


# ---------------------------------------------------------------------------
# Claim evaluation (thin dispatch onto motif.compare)
# ---------------------------------------------------------------------------
def evaluate_claim(claim: EdgeClaim, derived: dict) -> K.ClaimResult:
    primary = derived["primary"]
    secondary = derived.get("secondary")
    kw = dict(claim.compare_kwargs)
    if claim.compare == "count":
        return K.compare_count(claim.id, claim.description, claim.report_value,
                               primary, secondary, **kw)
    if claim.compare == "pct":
        return K.compare_pct(claim.id, claim.description, claim.report_value,
                             primary, secondary, **kw)
    if claim.compare == "ratio":
        lo, hi = claim.report_value
        return K.compare_ratio(claim.id, claim.description, lo, hi, primary)
    if claim.compare == "categorical":
        return K.compare_categorical(claim.id, claim.description, claim.report_value,
                                     primary, **kw)
    if claim.compare == "table_sum":
        return K.check_table_sum(claim.id, claim.description, primary,
                                 claim.report_value, **kw)
    raise ValueError(f"Unknown compare kind {claim.compare!r} for claim {claim.id}")


def evaluate_claims(spec: CircuitSpec, clients: dict, seeds: dict | None = None
                    ) -> list[K.ClaimResult]:
    """Derive + evaluate every EdgeClaim in the spec."""
    seeds = seeds if seeds is not None else resolve_seeds(spec, clients)
    out = []
    for claim in spec.edge_claims:
        derived = derive_edge(spec, clients, claim, seeds)
        out.append(evaluate_claim(claim, derived))
    return out


# ---------------------------------------------------------------------------
# Robustness sweep
# ---------------------------------------------------------------------------
def run_threshold_sweep(spec: CircuitSpec, clients: dict, seeds: dict | None = None
                        ) -> dict:
    """Per syn-edge floor, total synapses of each claimed edge - does it survive thinning?"""
    seeds = seeds if seeds is not None else resolve_seeds(spec, clients)
    sweep: dict[str, dict[int, int]] = {}
    for claim in spec.edge_claims:
        if claim.metric not in ("syn_count", "frac", "ipsi_frac"):
            continue
        client = clients[claim.dataset]
        src = _ids_for(claim.src, seeds, client)
        dst = _ids_for(claim.dst, seeds, client)
        sweep[claim.id] = {
            int(f): _edge_syn_count(client, src, dst, min_syn=int(f))
            for f in spec.thresholds.syn_edge_floors
        }
    return sweep


# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------
def label_permutation_null(values: np.ndarray, labels: np.ndarray, *,
                           statistic, n_perm: int, rng_seed: int = 0) -> dict:
    """Generic label-permutation null: shuffle ``labels``, recompute ``statistic``.

    ``statistic(values, labels) -> float``. Returns observed value, null mean/std,
    z-score and a two-sided permutation p-value. Deterministic (seeded) so cached runs
    reproduce. Used by the muscular verifier for the somaSide -> ipsi_frac test.
    """
    rng = np.random.default_rng(rng_seed)
    values = np.asarray(values)
    labels = np.asarray(labels)
    observed = float(statistic(values, labels))
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = float(statistic(values, rng.permutation(labels)))
    mean, std = float(null.mean()), float(null.std())
    z = (observed - mean) / std if std > 0 else float("inf") if observed != mean else 0.0
    p = float((np.abs(null - mean) >= abs(observed - mean)).mean())
    return {"observed": observed, "null_mean": mean, "null_std": std,
            "z": z, "p": p, "n_perm": int(n_perm)}


# ---------------------------------------------------------------------------
# Oracle cross-check (paper tables as ground truth)
# ---------------------------------------------------------------------------
def crosscheck_oracle(spec: CircuitSpec, computed_rows: dict) -> list[K.ClaimResult]:
    """Compare computed table rows against the spec's oracle_tables.

    ``computed_rows`` maps oracle table id (e.g. "S14") -> {row_key: {field: value}}.
    The oracle is the same shape. Each numeric field becomes a compare_count claim and
    each string field a compare_categorical claim. This is the agent's self-check before
    extending results beyond what the paper sampled.
    """
    out: list[K.ClaimResult] = []
    for tid, oracle in spec.oracle_tables.items():
        comp = computed_rows.get(tid, {})
        for row_key, fields in oracle.items():
            crow = comp.get(row_key, {})
            for fname, oval in fields.items():
                cval = crow.get(fname)
                cid = f"{tid}.{row_key}.{fname}"
                desc = f"{tid} {row_key} {fname}"
                if cval is None:
                    out.append(K.unverifiable(cid, desc, oval, "not computed"))
                elif isinstance(oval, str):
                    out.append(K.compare_categorical(cid, desc, oval, cval))
                elif isinstance(oval, float) and 0.0 <= oval <= 1.0:
                    out.append(K.compare_pct(cid, desc, oval * 100.0,
                                             float(cval) * 100.0, pp=5.0))
                else:
                    out.append(K.compare_count(cid, desc, oval, cval,
                                               rel=0.15, abs_floor=3, drift_dir="down"))
    return out
