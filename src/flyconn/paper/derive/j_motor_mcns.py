"""Derive family J: wing-steering DN -> motor-neuron mapping in the male CNS connectome.

For each of the 7 figure-driven wing-steering DN types we resolve its MaleCNS bodies by
``type``, pull its synapses onto wing-steering motor neurons (subclass == 'wm', excluding
DLM/DVM power), and partition that output into the ipsilateral and contralateral wing
(DN somaSide vs motor-neuron somaSide). We then read DNp26's strongest steering muscles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..malecns import ANNOTATIONS, client as MC
from ..oracle import consts as C


def _laterality_category(ipsi_frac: float) -> str:
    if ipsi_frac >= 0.60:
        return "ipsilateral"
    if ipsi_frac <= 0.40:
        return "contralateral"
    return "bilateral"


def run(src=None, meta=None) -> dict:
    if not ANNOTATIONS.exists():
        return {"available": False, "reason": f"MaleCNS annotations not found at {ANNOTATIONS}"}

    motor = MC.motor_neurons()
    steering = motor[motor["is_wing_steering"]]

    dns = {}
    for dn in C.WING_STEERING_DNS:
        bodies = MC.bodies_of_type(dn)
        if len(bodies) == 0:
            dns[dn] = {"n_bodies": 0, "category": None, "ipsi_frac": None,
                       "steering_syn": 0, "note": "no MaleCNS body of this type"}
            continue
        edges = MC.dn_to_motor(bodies["bodyId"].tolist(), steering)
        edges = edges[edges["is_wing_steering"]]
        if len(edges) == 0:
            dns[dn] = {"n_bodies": int(len(bodies)), "category": None, "ipsi_frac": None,
                       "steering_syn": 0}
            continue
        # ipsi/contra: compare each DN body's somaSide to the motor neuron's somaSide.
        dn_side = bodies.set_index("bodyId")["somaSide"]
        edges = edges.copy()
        edges["dn_side"] = edges["body_pre"].map(dn_side)
        # a DN's "ipsilateral" wing is the one on its own soma side.
        ipsi_mask = (edges["dn_side"] == edges["somaSide"])
        ipsi = int(edges.loc[ipsi_mask, "syn"].sum())
        total = int(edges["syn"].sum())
        ipsi_frac = round(ipsi / total, 2) if total else None
        dns[dn] = {
            "n_bodies": int(len(bodies)),
            "steering_syn": total,
            "ipsi_frac": ipsi_frac,
            "category": _laterality_category(ipsi_frac) if ipsi_frac is not None else None,
        }

    # DNp26's strongest steering muscles (across both DNp26 bodies).
    dnp26 = MC.bodies_of_type("DNp26")
    dnp26_top = []
    if len(dnp26):
        e = MC.dn_to_motor(dnp26["bodyId"].tolist(), steering)
        e = e[e["is_wing_steering"]]
        by_muscle = e.groupby("muscle")["syn"].sum().sort_values(ascending=False)
        dnp26_top = list(by_muscle.head(6).index)

    return {
        "available": True,
        "dns": dns,
        "dnp26_top_muscles": dnp26_top,
        "n_steering_motor_neurons": int(len(steering)),
        "track": "malecns_offline",
    }
