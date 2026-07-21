"""Offline MaleCNS data access: annotations + weighted edge list, with motor-neuron
classification.

Wing-steering motor neurons are the curated subclass ``wm`` (excluding the DLM/DVM power
muscles, which the paper excludes as "wing-power"). Steering muscle identity is read from
the ``type`` string (e.g. 'hg1 MN', 'i1 MN', 'hg2 MN', 'b3 MN').
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from . import ANNOTATIONS, WEIGHTS

# Wing-power muscles to EXCLUDE from "wing-steering" (paper Methods: DLM/DVM power).
POWER_MUSCLE_PREFIXES = ("DLM", "DVM")


@lru_cache(maxsize=1)
def annotations() -> pd.DataFrame:
    df = feather.read_feather(ANNOTATIONS)
    return df


@lru_cache(maxsize=1)
def weights() -> pd.DataFrame:
    """Body-to-body weighted edges (body_pre, body_post, weight)."""
    return feather.read_feather(WEIGHTS)


def bodies_of_type(dn_type: str) -> pd.DataFrame:
    """All MaleCNS bodies whose ``type`` matches a FlyWire DN type name."""
    ann = annotations()
    m = ann["type"].astype("string") == dn_type
    return ann.loc[m, ["bodyId", "type", "instance", "somaSide", "superclass", "subclass", "class"]]


def motor_neurons() -> pd.DataFrame:
    """All motor neurons (superclass in {vnc_motor, cb_motor}), with a parsed muscle name
    and a wing-steering flag (subclass == 'wm', excluding DLM/DVM power)."""
    ann = annotations()
    mot = ann[ann["superclass"].astype("string").str.contains("motor", na=False)].copy()
    mot["muscle"] = mot["type"].astype("string").str.replace(" MN", "", regex=False)
    is_power = mot["muscle"].fillna("").str.startswith(POWER_MUSCLE_PREFIXES)
    mot["is_wing_steering"] = (mot["subclass"].astype("string") == "wm") & (~is_power)
    return mot[["bodyId", "type", "instance", "muscle", "subclass", "somaSide",
                "is_wing_steering"]]


def dn_to_motor(dn_bodies: list[int], motor_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Edges from the given DN bodies onto motor neurons, annotated with muscle & side.

    Returns one row per (DN body -> motor body) with weight, the motor muscle, the motor
    somaSide, and the wing-steering flag.
    """
    if motor_df is None:
        motor_df = motor_neurons()
    w = weights()
    sub = w[w["body_pre"].isin(set(int(b) for b in dn_bodies))]
    motor_set = motor_df.set_index("bodyId")
    sub = sub[sub["body_post"].isin(set(motor_set.index))].copy()
    sub = sub.join(motor_set, on="body_post")
    return sub.rename(columns={"weight": "syn"})
