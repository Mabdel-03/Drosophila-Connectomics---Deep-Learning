"""Q1 — SINGLE-ANIMAL CLOSURE: does LLPC1->Nod1->DNp26 reproduce WITHIN MCNS alone?

The published circuit is split across two animals of different sex (female FlyWire FAFB
for the optic-lobe/DN readout, male MCNS for the motor side). But LLPC1 (285 bodies),
Nod1 (4), DNp26 (2), DNa04 (2), DNbe001 (2) all EXIST in the male CNS — so the readout
half can in principle be closed in ONE animal. This module tests the WIRING, not the
existence (existence is a recon fact): from ``M.weights()`` (body-to-body synapse counts)
it sums LLPC1->Nod1 and Nod1->DNp26 synapses, counts how many LLPC1 actually contact
Nod1, and asks whether Nod1 is the dominant excitatory readout of LLPC1 in MCNS too (the
paper's headline: Nod1 the dominant excitatory LLPC1 readout, Nod1->DNp26 the strongest
convergent steering target).

Verdict logic mirrors the rest of the pipeline (``motif.compare``): each of the three
motif edges that holds in-animal contributes 1/3 to a ``chain_reproduced`` fraction, and
the overall verdict is CONFIRMED if all three hold, CONFIRMED_WITH_CAVEAT if the chain is
present but weaker / partially split, REFUTED if a chain edge is categorically absent.
Pure pandas; importable and runnable standalone.
"""

from __future__ import annotations

import pandas as pd

from ..motif import compare as K

# The readout cells that exist in MCNS (recon-confirmed body counts, used as a positive
# control on ``M.bodies_of_type``).
READOUT_TYPES = ("LLPC1", "Nod1", "DNp26", "DNa04", "DNbe001")

# Paper FlyWire headline numbers the in-animal chain is compared against (for narrative;
# absolute MCNS counts differ by animal/sex, so the load-bearing test is the MOTIF
# topology — chain present + Nod1 dominant — not digit-equality).
PAPER_LLPC1_TO_NOD1_SYN = 4228       # Nod1 dominant excitatory readout, from 91/100 LLPC1
PAPER_NOD1_TO_DNP26_SYN = 448        # Nod1 -> DNp26 relay


def _edge_syn(weights: pd.DataFrame, pre_set: set[int], post_set: set[int]
              ) -> tuple[int, int]:
    """(total synapses, n distinct pre contacting any post) for pre_set -> post_set."""
    if not pre_set or not post_set:
        return 0, 0
    e = weights[weights["body_pre"].isin(pre_set) & weights["body_post"].isin(post_set)]
    return int(e["weight"].sum()), int(e["body_pre"].nunique())


def _nod1_rank_among_llpc1_targets(weights: pd.DataFrame, M, llpc1: set[int],
                                   nod1: set[int]) -> dict:
    """Rank Nod1 among LLPC1's TARGET cell types by total synapses (excitatory readout).

    Aggregates LLPC1 output by the target body's ``type`` and reports where the Nod1 type
    sits in that ranking — the in-animal analog of the paper's "Nod1 is the dominant
    excitatory readout". LLPC1's own lateral connections are excluded so the ranking is of
    downstream readouts, not the sheet's self-coupling.
    """
    out = weights[weights["body_pre"].isin(llpc1)].copy()
    ann = M.annotations()[["bodyId", "type"]].rename(columns={"bodyId": "body_post"})
    out = out.merge(ann, on="body_post", how="left")
    out = out[out["type"].notna() & (out["type"] != "LLPC1")]
    by_type = (out.groupby("type")["weight"].sum().sort_values(ascending=False)
               .reset_index())
    by_type["rank"] = range(1, len(by_type) + 1)
    nod1_row = by_type[by_type["type"] == "Nod1"]
    nod1_rank = int(nod1_row["rank"].iloc[0]) if len(nod1_row) else None
    nod1_syn = int(nod1_row["weight"].iloc[0]) if len(nod1_row) else 0
    return {
        "nod1_rank": nod1_rank,
        "nod1_syn": nod1_syn,
        "n_target_types": int(len(by_type)),
        "top5_targets": by_type.head(5)[["type", "weight", "rank"]].to_dict("records"),
        # "dominant excitatory readout" in-animal = top-ranked downstream readout type.
        "nod1_is_dominant_readout": nod1_rank is not None and nod1_rank <= 3,
    }


def run_q1(source_mcns_via_M) -> dict:
    """Test the LLPC1->Nod1->DNp26 readout within MCNS alone.

    ``source_mcns_via_M`` is the ``flyconn.paper.malecns.client`` module (M): it exposes
    ``bodies_of_type``, ``annotations`` and ``weights``. Returns existence counts, the raw
    in-animal wiring numbers, Nod1's rank among LLPC1 targets, a ``chain_reproduced``
    fraction (how many of the 3 motif facts hold) and the ``ClaimResult``s.
    """
    M = source_mcns_via_M
    weights = M.weights()

    # 1. Existence (positive control on the recon fact).
    existence = {t: int(len(M.bodies_of_type(t))) for t in READOUT_TYPES}
    llpc1 = set(int(b) for b in M.bodies_of_type("LLPC1")["bodyId"])
    nod1 = set(int(b) for b in M.bodies_of_type("Nod1")["bodyId"])
    dnp26 = set(int(b) for b in M.bodies_of_type("DNp26")["bodyId"])

    # 2. In-animal wiring: LLPC1 -> Nod1 -> DNp26.
    llpc1_nod1_syn, llpc1_contacting = _edge_syn(weights, llpc1, nod1)
    nod1_dnp26_syn, nod1_contacting = _edge_syn(weights, nod1, dnp26)

    # 3. Is Nod1 the dominant excitatory LLPC1 readout in MCNS too?
    rank = _nod1_rank_among_llpc1_targets(weights, M, llpc1, nod1)

    # The three motif facts that must hold for in-animal closure.
    edge1 = llpc1_nod1_syn > 0 and llpc1_contacting > 0      # LLPC1 -> Nod1 present
    edge2 = nod1_dnp26_syn > 0 and nod1_contacting > 0       # Nod1 -> DNp26 present
    edge3 = bool(rank["nod1_is_dominant_readout"])           # Nod1 dominant readout
    chain_reproduced = round(sum([edge1, edge2, edge3]) / 3.0, 3)

    claims = [
        K.ClaimResult(
            id="Q1.exist", description="LLPC1/Nod1/DNp26 readout cells exist in MCNS",
            report_value="present", computed_primary=existence,
            tolerance="all >0", verdict=K.CONFIRMED if all(existence.values()) else K.REFUTED,
            numeric_outcome=K.MATCH if all(existence.values()) else K.MISMATCH,
            notes="recon positive control",
        ),
        K.ClaimResult(
            id="Q1.llpc1_nod1", description="LLPC1 -> Nod1 synaptic edge present in MCNS",
            report_value=f"~{PAPER_LLPC1_TO_NOD1_SYN} (FlyWire)",
            computed_primary=llpc1_nod1_syn,
            tolerance=">0 syn; >0 LLPC1 contacting Nod1",
            verdict=K.CONFIRMED if edge1 else K.REFUTED,
            numeric_outcome=K.MATCH if edge1 else K.MISMATCH,
            notes=f"{llpc1_contacting} LLPC1 bodies contact Nod1 ({llpc1_nod1_syn} syn)",
        ),
        K.ClaimResult(
            id="Q1.nod1_dnp26", description="Nod1 -> DNp26 relay present in MCNS",
            report_value=f"~{PAPER_NOD1_TO_DNP26_SYN} (FlyWire)",
            computed_primary=nod1_dnp26_syn,
            tolerance=">0 syn; >0 Nod1 contacting DNp26",
            verdict=K.CONFIRMED if edge2 else K.REFUTED,
            numeric_outcome=K.MATCH if edge2 else K.MISMATCH,
            notes=f"{nod1_contacting} Nod1 bodies contact DNp26 ({nod1_dnp26_syn} syn)",
        ),
        K.ClaimResult(
            id="Q1.nod1_dominant",
            description="Nod1 is a dominant excitatory LLPC1 readout in MCNS",
            report_value="rank 1 (FlyWire)",
            computed_primary=f"rank {rank['nod1_rank']} of {rank['n_target_types']}",
            tolerance="Nod1 in top-3 LLPC1 target types",
            verdict=K.CONFIRMED if edge3 else K.CONFIRMED_WITH_CAVEAT,
            numeric_outcome=K.MATCH if edge3 else K.MINOR_DIFF,
            notes=f"Nod1 syn from LLPC1 = {rank['nod1_syn']}",
        ),
    ]

    # Overall: all three edges -> CONFIRMED; chain present but Nod1 not top -> caveat.
    if edge1 and edge2 and edge3:
        verdict = K.CONFIRMED
    elif edge1 and edge2:
        verdict = K.CONFIRMED_WITH_CAVEAT
    else:
        verdict = K.REFUTED

    return {
        "question": "Q1_single_animal_closure",
        "existence": existence,
        "wiring": {
            "llpc1_to_nod1_syn": llpc1_nod1_syn,
            "llpc1_contacting_nod1": llpc1_contacting,
            "nod1_to_dnp26_syn": nod1_dnp26_syn,
            "nod1_contacting_dnp26": nod1_contacting,
        },
        "nod1_readout_rank": rank,
        "chain_reproduced": chain_reproduced,
        "verdict": verdict,
        "claims": [c.to_dict() for c in claims],
        "track": "mcns",
    }


if __name__ == "__main__":  # standalone smoke test
    import json

    from ..paper.malecns import client as M

    print(json.dumps(run_q1(M), indent=2, default=str))
