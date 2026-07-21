"""Declarative description of a circuit to extract + verify.

The Stage-4 VCH pipeline hardcodes its circuit. To let an agent search for *any* motif
and verify it with the same machinery, a circuit is expressed as data: which neurons to
seed, which edges/claims to check, which robustness sweeps and null models to run, and an
oracle of expected values (e.g. the paper's tables) to cross-check against. The generic
``circuit.engine`` consumes a ``CircuitSpec`` + a ``ConnectomeClient`` per dataset and
emits ``ClaimResult``s through the existing ``motif.compare`` verdict engine - no new
verdict logic.

A ``Selector`` is a small dict the engine knows how to resolve to root_ids against a
dataset, e.g. ``{"cell_type": "DNp26"}`` or ``{"root_ids": [720...]}``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Seed:
    """A named neuron set to anchor the search."""

    name: str               # human label, e.g. "DNbe001"
    selector: dict          # {"cell_type": "DNbe001"} | {"root_ids": [...]}
    dataset: str            # "fafb" | "mcns"


@dataclass(frozen=True)
class EdgeClaim:
    """One quantitative claim about an edge / set of edges, with its expected value.

    ``compare`` selects the verdict function in ``motif.compare``:
      "count"       -> compare_count(report, primary, secondary, **kwargs)
      "pct"         -> compare_pct(report, primary, secondary, **kwargs)
      "ratio"       -> compare_ratio(report_lo, report_hi, primary)  [report_value=(lo,hi)]
      "categorical" -> compare_categorical(report, computed, **kwargs)
      "table_sum"   -> check_table_sum(rows_total, stated_total, **kwargs)
    """

    id: str
    description: str
    src: dict               # presynaptic selector
    dst: dict               # postsynaptic selector
    dataset: str
    metric: str             # "syn_count" | "n_partners" | "frac" | "ipsi_frac" | ...
    report_value: Any
    compare: str            # one of the keys above
    compare_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Threshold:
    """Robustness sweep grids (cleft-score floors + per-edge synapse floors)."""

    cleft_score_floors: tuple[int, ...] = (0, 50, 100, 140)
    syn_edge_floors: tuple[int, ...] = (1, 5, 10)


@dataclass(frozen=True)
class NullModel:
    """A permutation / null model to test that a statistic is non-random."""

    kind: str               # "degree_preserving" | "label_permutation"
    statistic: str          # name of the per-permutation statistic (engine-specific)
    n_perm: int = 500
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CircuitSpec:
    """A full, self-contained description of a circuit to extract and verify."""

    name: str
    datasets: tuple[str, ...]
    seeds: tuple[Seed, ...]
    edge_claims: tuple[EdgeClaim, ...] = ()
    thresholds: Threshold = field(default_factory=Threshold)
    nulls: tuple[NullModel, ...] = ()
    oracle_tables: dict = field(default_factory=dict)   # {"S14": [...], ...} paper tables
    stage_dir: str = ""                                 # e.g. "5 - Muscular Projection"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seed(d: dict) -> Seed:
    return Seed(name=d["name"], selector=d["selector"], dataset=d["dataset"])


def _edge_claim(d: dict) -> EdgeClaim:
    return EdgeClaim(
        id=d["id"], description=d.get("description", d["id"]),
        src=d["src"], dst=d["dst"], dataset=d["dataset"],
        metric=d.get("metric", "syn_count"), report_value=d["report_value"],
        compare=d.get("compare", "count"), compare_kwargs=d.get("compare_kwargs", {}),
    )


def spec_from_dict(d: dict) -> CircuitSpec:
    """Build a CircuitSpec from a plain dict (parsed YAML/JSON or a config module)."""
    thr = d.get("thresholds", {})
    nulls = tuple(
        NullModel(kind=n["kind"], statistic=n.get("statistic", ""),
                  n_perm=n.get("n_perm", 500), params=n.get("params", {}))
        for n in d.get("nulls", [])
    )
    return CircuitSpec(
        name=d["name"],
        datasets=tuple(d.get("datasets", ())),
        seeds=tuple(_seed(s) for s in d.get("seeds", [])),
        edge_claims=tuple(_edge_claim(e) for e in d.get("edge_claims", [])),
        thresholds=Threshold(
            cleft_score_floors=tuple(thr.get("cleft_score_floors", (0, 50, 100, 140))),
            syn_edge_floors=tuple(thr.get("syn_edge_floors", (1, 5, 10))),
        ),
        nulls=nulls,
        oracle_tables=d.get("oracle_tables", {}),
        stage_dir=d.get("stage_dir", ""),
    )
