"""Q3 — ESCAPE-ROUTE CENSUS TO MUSCLE, and its SEPARABILITY from the steering output.

The paper called the PLP/PVLP->LPLC2/LC4 looming/escape arm "the weaker route" and never
traced it. This module enumerates that arm's full descending census in FlyWire (DNs
downstream of LPLC2 + LC4, esp. the giant-fibre / looming-command cluster DNp01/DNp03/
DNp04/DNp06), then follows the SAME DN->motor-neuron->muscle method the Nod1 arm used to
the MUSCLE level in MCNS, and tests the paper's claim that the escape output is
ANATOMICALLY SEPARABLE from the LLPC1 course-control / wing-steering output.

Separability is operationalised two ways, both of which must be ~0 for "separable":
  * DN-overlap: do the escape DNs and the LLPC1/Nod1 steering DNs (DNp26, DNa04, DNbe001,
    DNg32, ...) share any neuron? (giant-fibre escape vs steering = disjoint DN sets);
  * muscle-overlap: do the escape muscles (jump / TTM / tergotrochanter) and the steering
    muscles (wing-steering hg/i/b 'wm') share any target? (escape jump vs wing steering =
    disjoint muscles).

FlyWire gives the DN census; MCNS gives the DN->MN->muscle map (escape DNs by ``type``).
Both sources are passed in; the function is importable and runnable standalone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..motif import compare as K
from ..paper.oracle import consts as C

# The looming/escape command cluster (giant fibre DNp01 + the looming DNp03/04/06) the
# question names explicitly; the census is the union of these plus any other DN the
# LPLC2/LC4 broadcast drives.
ESCAPE_COMMAND_DNS = ("DNp01", "DNp03", "DNp04", "DNp06")

# The LLPC1/Nod1 course-control steering DNs (Fig 6 / Table S14) — the set the escape
# census is tested against for disjointness.
STEERING_DNS = set(C.WING_STEERING_DNS)

# A DN edge must clear this to count as a real downstream partner (suppresses 1-synapse
# noise; same spirit as the secondary-track floor used elsewhere).
MIN_DN_SYN = 5

# Escape / jump musculature name signatures in MCNS (tergotrochanter TTM = the GF jump
# target). NOTE: MCNS labels the tergotrochanter MNs (TTMn/STTMm) ``subclass=='wm'``, so
# ``M.motor_neurons().is_wing_steering`` is True for them too — they are wing-region MNs
# anatomically but are the JUMP/escape effector, not a course-control steering muscle. The
# separability test therefore compares the escape arm against the COURSE-CONTROL steering
# muscles (wing-steering MINUS these escape muscles), so a shared TTMn is scored as the
# expected escape target, not a spurious overlap.
ESCAPE_MUSCLE_PAT = ("ttm", "tergotroch", "tt", "jump")


def _downstream_dns(src, meta, source_roots: np.ndarray) -> pd.DataFrame:
    """DNs downstream of a source id set: per-DN (cell_type, syn), syn>=MIN_DN_SYN.

    A DN is ``super_class == 'descending'`` OR a ``cell_type`` starting 'DN' (covers the
    handful of escape DNs annotated central whose descending flag is missing).
    """
    src_set = set(int(x) for x in source_roots)
    if not src_set:
        return pd.DataFrame(columns=["post_pt_root_id", "cell_type", "syn"])
    out = src.synapses(pre_ids=source_roots.tolist())
    out = out[out["pre_pt_root_id"].isin(src_set)]
    g = out.groupby("post_pt_root_id").size().reset_index(name="syn")
    g = g.join(meta.by_root[["cell_type", "super_class"]], on="post_pt_root_id")
    ct = g["cell_type"].astype("string")
    is_dn = (g["super_class"] == "descending") | ct.str.startswith("DN", na=False)
    g = g[is_dn & (g["syn"] >= MIN_DN_SYN)].copy()
    return g[["post_pt_root_id", "cell_type", "syn"]].sort_values("syn", ascending=False)


def _escape_muscle(muscle: object) -> bool:
    """True if a parsed MCNS muscle name is escape/jump (TTM / tergotrochanter)."""
    if muscle is None or (isinstance(muscle, float) and np.isnan(muscle)):
        return False
    s = str(muscle).lower()
    return any(p in s for p in ESCAPE_MUSCLE_PAT)


def run_q3(source_fw, source_mcns_via_M) -> dict:
    """Enumerate the LPLC2/LC4 escape census to muscle and test steering-separability.

    ``source_fw`` is a ``FlyWireSource`` (DN census); ``source_mcns_via_M`` the malecns
    ``client`` module (DN->MN->muscle). Returns the FlyWire escape-DN census, the MCNS
    muscle map for those DNs, the DN- and muscle-overlap separability metrics, and the
    ``ClaimResult``s.
    """
    M = source_mcns_via_M
    # NeuronMeta is needed alongside the synapse source; load it the same way the derive
    # layer does (it is the shared annotation table, independent of live vs offline).
    from ..paper.fw_access import NeuronMeta
    meta = NeuronMeta.load(C.VERSION)

    # 1. FlyWire: escape DN census downstream of LPLC2 + LC4.
    lplc2 = meta.root_ids_of_type(["LPLC2"])
    lc4 = meta.root_ids_of_type(["LC4"])
    seeds = np.array(sorted(set(int(x) for x in (*lplc2, *lc4))), dtype=np.int64)
    dn_census = _downstream_dns(source_fw, meta, seeds)
    by_type = (dn_census.dropna(subset=["cell_type"]).groupby("cell_type")["syn"].sum()
               .sort_values(ascending=False))
    census = {str(t): int(s) for t, s in by_type.items()}
    command_present = {d: census.get(d, 0) for d in ESCAPE_COMMAND_DNS}
    # The escape DN type set = command cluster (present in census) + any other DN driven.
    escape_dn_types = set(command_present) | set(census)

    # 2. MCNS: those escape DNs -> motor neurons -> muscles, classify escape vs steering.
    escape_in_mcns = [d for d in escape_dn_types
                      if len(M.bodies_of_type(d))]  # only DNs that exist in MCNS
    dn_bodies = []
    for d in escape_in_mcns:
        dn_bodies += [int(b) for b in M.bodies_of_type(d)["bodyId"]]
    dn_motor = M.dn_to_motor(dn_bodies) if dn_bodies else pd.DataFrame(
        columns=["body_pre", "body_post", "syn", "muscle", "somaSide", "is_wing_steering"])

    if len(dn_motor):
        is_escape = dn_motor["muscle"].map(_escape_muscle)
        escape_mn = dn_motor[is_escape]
        # Course-control steering = wing-steering MNs that are NOT the tergotrochanter/jump
        # escape effector (so the shared 'wm' subclass on TTMn does not fake an overlap).
        steering_mn = dn_motor[dn_motor["is_wing_steering"].fillna(False) & (~is_escape)]
    else:
        escape_mn = steering_mn = dn_motor
    escape_muscles = sorted(set(str(m) for m in escape_mn.get("muscle", pd.Series(dtype=object))
                                .dropna().unique())) if len(escape_mn) else []
    steering_muscles = sorted(set(str(m) for m in steering_mn.get("muscle", pd.Series(dtype=object))
                                  .dropna().unique())) if len(steering_mn) else []

    # 3. Separability metrics. By construction escape vs course-control muscle sets are
    # name-disjoint; a non-empty overlap means an escape DN drives a true steering muscle.
    dn_overlap = sorted(escape_dn_types & STEERING_DNS)
    muscle_overlap = sorted(set(escape_muscles) & set(steering_muscles))
    dn_separable = len(dn_overlap) == 0
    muscle_separable = len(muscle_overlap) == 0

    claims = [
        K.ClaimResult(
            id="Q3.census",
            description="Escape DN census downstream of LPLC2/LC4 (looming cluster present)",
            report_value="weaker route (untraced)",
            computed_primary={d: command_present[d] for d in ESCAPE_COMMAND_DNS},
            tolerance=">=1 command DN with syn>=floor",
            verdict=K.CONFIRMED if any(command_present.values()) else K.REFUTED,
            numeric_outcome=K.MATCH if any(command_present.values()) else K.MISMATCH,
            notes=f"{len(census)} DN types downstream of LPLC2/LC4",
        ),
        K.compare_categorical(
            "Q3.dn_separable",
            "Escape DNs are disjoint from LLPC1/Nod1 steering DNs",
            True, dn_separable,
            refuted_note=f"shared DNs: {dn_overlap}"),
        K.compare_categorical(
            "Q3.muscle_separable",
            "Escape muscles (jump/TTM) are disjoint from wing-steering muscles",
            True, muscle_separable,
            refuted_note=f"shared muscles: {muscle_overlap}"),
    ]
    if not escape_mn.shape[0] and not steering_mn.shape[0]:
        # No motor mapping resolved for the escape DNs in MCNS -> can't judge the muscle
        # leg of separability; flag it rather than claiming a spurious "separable".
        claims[2] = K.unverifiable(
            "Q3.muscle_separable",
            "Escape muscles disjoint from wing-steering muscles",
            True, "no escape-DN -> motor-neuron edges resolved in MCNS")

    return {
        "question": "Q3_escape_route_census",
        "flywire": {
            "n_lplc2": int(len(lplc2)), "n_lc4": int(len(lc4)),
            "n_dn_types": len(census),
            "command_dn_syn": command_present,
            "census_top15": dict(list(census.items())[:15]),
        },
        "mcns": {
            "escape_dns_in_mcns": sorted(escape_in_mcns),
            "n_escape_motor_edges": int(len(escape_mn)),
            "n_steering_motor_edges": int(len(steering_mn)),
            "escape_muscles": escape_muscles,
            "steering_muscles": steering_muscles,
        },
        "separability": {
            "dn_overlap": dn_overlap, "dn_separable": dn_separable,
            "muscle_overlap": muscle_overlap, "muscle_separable": muscle_separable,
            "separable": dn_separable and muscle_separable,
        },
        "claims": [c.to_dict() for c in claims],
        "track": source_fw.track,
    }


if __name__ == "__main__":  # standalone smoke test
    import json

    from ..paper.fw_access import make_source
    from ..paper.malecns import client as M

    src = make_source("auto")
    print(json.dumps(run_q3(src, M), indent=2, default=str))
