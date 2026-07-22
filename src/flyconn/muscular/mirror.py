"""Stage-8 wing-flip helper: does the LEFT figure-ground command steer the OPPOSITE wing?

The crux of "complement". The right circuit's DNp26 drives the CONTRALATERAL wing (hg1/i1);
the mirror left circuit's command must drive the OPPOSITE physical wing. In MaleCNS a motor
neuron's ``somaSide`` is the wing it innervates, and DNp26 is a bilateral pair (DNp26_L,
DNp26_R). The right brain's Nod1 preferentially drives the somaSide-R DNp26 body; the left
brain's Nod1 drives the somaSide-L body. So the test is per-DNp26-body:

  * each DNp26 body is contralateral-steering (intrinsic DNp26 laterality, conserved), and
  * the two bodies steer OPPOSITE physical wings (their dominant target wing differs), via
    the SAME muscle identities (hg1/i1/hg2).

Plus a somaSide label-permutation null (NC3): the observed per-body contralateral bias is a
tail outlier vs permuted motor-neuron somaSide labels.

Uses the offline ``paper.malecns.client`` (the same access the motor map uses). Returns a
structured dict; the m_mirror oracle turns it into claims. No FlyWire pull needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DNP26 = "DNp26"
# The figure-driven wing-steering DNs whose intrinsic laterality the paper reports (S14).
FIGURE_DNS = ("DNa04", "DNbe001", "DNge107", "DNbe005", "DNp26", "DNg32", "DNge094")
SEED = 12345
N_PERM = 500


def _client():
    from ..paper.malecns import client as MC
    return MC


def _steering_edges_for_bodies(MC, bodies: list[int]) -> pd.DataFrame:
    """DN-body -> wing-steering motor-neuron edges (power muscles excluded), with sides."""
    motor = MC.motor_neurons()
    steering = motor[motor["is_wing_steering"]]
    edges = MC.dn_to_motor(bodies, motor_df=steering)
    # edges: body_pre, body_post, syn, type/muscle/somaSide/is_wing_steering
    return edges[edges["is_wing_steering"].fillna(False)] if "is_wing_steering" in edges.columns else edges


def _per_body_wing(MC, dn_type: str) -> list[dict]:
    """For each MCNS body of a DN type: dominant target physical wing + ipsi/contra fraction.

    ipsi = mn somaSide == dn somaSide; contra = differ. ``target_wing`` is the physical side
    (mn somaSide) carrying the most steering synapses.
    """
    bodies_df = MC.bodies_of_type(dn_type)
    out = []
    for _, b in bodies_df.iterrows():
        body = int(b["bodyId"])
        dn_side = b.get("somaSide")
        e = _steering_edges_for_bodies(MC, [body])
        if e.empty:
            out.append({"bodyId": body, "dn_soma_side": dn_side, "steering_syn": 0,
                        "ipsi_frac": float("nan"), "target_wing": None, "category": "none",
                        "top_muscles": {}})
            continue
        e = e.copy()
        e["mn_side"] = e["somaSide"]
        total = float(e["syn"].sum())
        ipsi = float(e.loc[e["mn_side"].astype("string") == str(dn_side), "syn"].sum())
        frac = ipsi / total if total else float("nan")
        by_wing = e.groupby(e["mn_side"].astype("string"))["syn"].sum().sort_values(ascending=False)
        target_wing = str(by_wing.index[0]) if len(by_wing) else None
        cat = ("ipsilateral" if frac >= 0.6 else "contralateral" if frac <= 0.4 else "bilateral")
        top_muscles = (e.groupby(e["muscle"].astype("string"))["syn"].sum()
                       .sort_values(ascending=False).head(5).astype(int).to_dict())
        out.append({"bodyId": body, "dn_soma_side": dn_side, "steering_syn": int(total),
                    "ipsi_frac": round(frac, 3) if np.isfinite(frac) else None,
                    "target_wing": target_wing, "category": cat, "top_muscles": top_muscles})
    return out


def _permutation_null(MC, dn_type: str) -> dict:
    """NC3: permute motor-neuron somaSide; is the observed DNp26 contralateral bias a tail outlier?

    Pools all bodies of the DN type, computes the observed ipsi_frac, then recomputes it under
    n_perm random shuffles of the motor neurons' somaSide labels. p = fraction of permutations
    with ipsi_frac <= observed (DNp26 is contralateral => observed should sit in the LOW tail).
    """
    bodies = [int(x) for x in MC.bodies_of_type(dn_type)["bodyId"]]
    if not bodies:
        return {"available": False}
    motor = MC.motor_neurons()
    steering = motor[motor["is_wing_steering"]].copy()
    edges = MC.dn_to_motor(bodies, motor_df=steering)
    if edges.empty:
        return {"available": False}
    bodies_df = MC.bodies_of_type(dn_type).set_index("bodyId")
    edges = edges.copy()
    edges["dn_side"] = edges["body_pre"].map(bodies_df["somaSide"])
    edges["mn_side"] = edges["somaSide"].astype("string")
    total = float(edges["syn"].sum())
    obs_ipsi = float(edges.loc[edges["mn_side"] == edges["dn_side"].astype("string"), "syn"].sum())
    obs_frac = obs_ipsi / total if total else float("nan")

    # permute the motor neurons' somaSide labels across the steering MN pool
    mn_sides = steering["somaSide"].astype("string").to_numpy()
    mn_ids = steering["bodyId"].to_numpy()
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for j in range(N_PERM):
        perm = rng.permutation(mn_sides)
        side_map = dict(zip(mn_ids, perm))
        pm = edges["body_post"].map(side_map).astype("string")
        ipsi = float(edges.loc[pm == edges["dn_side"].astype("string"), "syn"].sum())
        null[j] = ipsi / total if total else float("nan")
    p_low = max(float((null <= obs_frac).mean()), 1.0 / N_PERM)
    z = (obs_frac - float(np.mean(null))) / (float(np.std(null)) or float("nan"))
    return {"available": True, "obs_ipsi_frac": round(obs_frac, 3),
            "null_mean": round(float(np.mean(null)), 3), "null_std": round(float(np.std(null)), 3),
            "z": round(z, 2) if np.isfinite(z) else None,
            "p_low_tail": p_low, "n_perm": N_PERM,
            "significant_contra": bool(p_low < 0.01 and obs_frac < 0.5)}


def run() -> dict:
    """Wing-flip analysis for the figure-driven DNs (MCNS). Returns None-safe structure."""
    try:
        MC = _client()
        MC.annotations()  # probe availability
    except Exception as e:  # MCNS not present
        return {"available": False, "reason": f"MaleCNS not available: {type(e).__name__}: {e}"}

    dnp26_bodies = _per_body_wing(MC, DNP26)
    # The flip: the two DNp26 bodies (somaSide L vs R) steer OPPOSITE physical wings.
    targets = {str(b["dn_soma_side"]): b["target_wing"] for b in dnp26_bodies if b["target_wing"]}
    flip = bool(len(targets) >= 2 and len(set(targets.values())) >= 2)
    # each body contralateral?
    each_contra = all(b["category"] == "contralateral" for b in dnp26_bodies if b["steering_syn"] > 0)
    # same muscle identities across bodies?
    muscle_sets = [set(b["top_muscles"].keys()) for b in dnp26_bodies if b["top_muscles"]]
    shared_muscles = sorted(set.intersection(*muscle_sets)) if len(muscle_sets) >= 2 else []

    null = _permutation_null(MC, DNP26)

    # intrinsic laterality of all figure DNs (homology, S14 reproduction).
    fig = {}
    for dn in FIGURE_DNS:
        per = _per_body_wing(MC, dn)
        # pooled ipsi_frac across the type's bodies
        tot = sum(b["steering_syn"] for b in per)
        ipsi = sum((b["ipsi_frac"] or 0) * b["steering_syn"] for b in per if b["ipsi_frac"] is not None)
        pooled = round(ipsi / tot, 3) if tot else None
        cat = (None if pooled is None else
               "ipsilateral" if pooled >= 0.6 else "contralateral" if pooled <= 0.4 else "bilateral")
        fig[dn] = {"pooled_ipsi_frac": pooled, "category": cat, "steering_syn": int(tot),
                   "per_body": per}

    return {
        "available": True,
        "dnp26_bodies": dnp26_bodies,
        "dnp26_target_wings_by_soma": targets,
        "dnp26_bodies_opposite_wings": flip,
        "dnp26_each_body_contralateral": each_contra,
        "dnp26_shared_muscles": shared_muscles,
        "permutation_null": null,
        "figure_dn_laterality": fig,
    }
