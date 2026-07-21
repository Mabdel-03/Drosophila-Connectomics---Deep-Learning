"""Build the CircuitSpec instance for the muscular-projection completion.

Seeds are the two named endpoints (DNbe001 + DNp26) expanded to the seven figure DNs;
edge claims come from S15-S18; the oracle is S14-S19; the null is a somaSide label
permutation on the wing-laterality statistic.
"""

from __future__ import annotations

from ..circuit.spec import CircuitSpec, EdgeClaim, NullModel, Seed, Threshold
from . import muscular_config as C


def _seeds() -> tuple[Seed, ...]:
    return tuple(
        Seed(name=dn, selector={"cell_type": dn}, dataset="mcns") for dn in C.FIGURE_DNS
    )


def _edge_claims() -> tuple[EdgeClaim, ...]:
    """S15 DNp26->muscle synapse claims (the per-muscle exemplars we re-derive)."""
    claims = []
    for muscle, info in C.DNP26_MUSCLES.items():
        claims.append(
            EdgeClaim(
                id=f"S15.DNp26.{muscle}",
                description=f"DNp26 -> {muscle} steering MN synapses",
                src={"seed": "DNp26"},
                dst={"cell_type": muscle},
                dataset="mcns",
                metric="syn_count",
                report_value=info["syn"],
                compare="count",
                compare_kwargs={"rel": 0.2, "abs_floor": 5, "drift_dir": "down"},
            )
        )
    return tuple(claims)


def _oracle_tables() -> dict:
    """S14 + S15 as the cross-check oracle (the rows the trace must reproduce)."""
    s14 = {
        dn: {"ipsi_frac": info["ipsi_frac"], "wing": info["wing"],
             "steering_syn": info["steering_syn"]}
        for dn, info in C.DN_LATERALITY.items()
    }
    s15 = {m: {"syn": info["syn"], "wing": info["wing"]} for m, info in C.DNP26_MUSCLES.items()}
    return {"S14": s14, "S15": s15}


def build_spec() -> CircuitSpec:
    return CircuitSpec(
        name="muscular_projection",
        datasets=("fafb", "mcns"),
        seeds=_seeds(),
        edge_claims=_edge_claims(),
        thresholds=Threshold(cleft_score_floors=(0, 50, 100, 140), syn_edge_floors=(1, 5, 10)),
        nulls=(NullModel(kind="label_permutation", statistic="ipsi_frac", n_perm=500),),
        oracle_tables=_oracle_tables(),
        stage_dir="5 - Muscular Projection",
    )
