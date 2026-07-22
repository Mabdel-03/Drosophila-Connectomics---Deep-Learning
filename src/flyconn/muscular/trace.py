"""DN -> motor-neuron -> muscle tracer, split by wing. Pure pandas on CAVE/cache frames.

These functions are the agent's core tools. They take normalised DataFrames (from
``flyconn.connectome.ConnectomeClient`` or the cache) and reproduce the paper's S14/S15
schemas, then extend the map to BOTH wings and all seven figure DNs - the completion the
paper only sampled.

Wing convention (S10.2): wing is relative to the DN's own side. A motor neuron is on the
``ipsi`` wing if ``mn.somaSide == dn.somaSide``, else ``contra``. Power muscles (DLM/DVM)
are excluded from the steering-laterality denominator.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import muscular_config as C

# Map an MCNS motor-neuron subclass / muscle label to one of MOTOR_SYSTEMS.
_SYSTEM_PATTERNS = [
    ("jump_ttm", re.compile(r"\bTTM|tergotrochanter|jump", re.I)),
    ("haltere", re.compile(r"\bMNhm|haltere|hm\d", re.I)),
    ("wing_power", re.compile(r"\bDLM|DVM|power", re.I)),
    ("neck_gaze", re.compile(r"\bADNM|MNnm|neck|gaze", re.I)),
    ("leg", re.compile(r"\bleg|tibia|femur|tarsus|sternal|trochanter|coxa", re.I)),
    ("abdominal", re.compile(r"\bMNad|abdomin|ps\d", re.I)),
    ("wing_steering", re.compile(r"\bhg\d|\bb\d|\bi\d|\btp\d|steering|\bwm\b", re.I)),
]


def classify_motor_system(subclass_or_muscle: "pd.Series | str") -> "pd.Series | str":
    """Map a subclass/muscle label (or a Series of them) to a MOTOR_SYSTEMS value."""
    def _one(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "other"
        s = str(val)
        # explicit subclass wins
        if s.strip().lower() == C.WING_STEERING_SUBCLASS:
            return "wing_steering"
        for system, pat in _SYSTEM_PATTERNS:
            if pat.search(s):
                return system
        return "other"

    if isinstance(subclass_or_muscle, pd.Series):
        return subclass_or_muscle.map(_one)
    return _one(subclass_or_muscle)


def expand_seeds_to_root_ids(annotations: pd.DataFrame, dn_names) -> pd.DataFrame:
    """cell_type -> root_ids in MCNS. A DN type may have multiple cells (e.g. 2 DNp26)."""
    names = list(dn_names)
    hit = annotations[annotations["cell_type"].isin(names)]
    cols = [c for c in ("root_id", "cell_type", "somaSide") if c in hit.columns]
    return hit[cols].drop_duplicates().reset_index(drop=True)


def dn_to_motor_neurons(dn_mn_edges: pd.DataFrame, dn_roots: pd.DataFrame,
                        mn_annotations: pd.DataFrame) -> pd.DataFrame:
    """Join DN->MN synapse edges with DN side and MN subclass/muscle/somaSide.

    ``dn_mn_edges``: columns pre_pt_root_id, post_pt_root_id, syn_count (post restricted
    to motor neurons). ``dn_roots``: root_id, cell_type, somaSide (DN side). ``mn_annotations``:
    root_id, cell_type|muscle, cell_sub_class, somaSide (MN side).
    Returns one row per (dn cell_type, mn root_id) with motor_system + wing tags.
    """
    e = dn_mn_edges.rename(columns={"pre_pt_root_id": "dn_root_id",
                                    "post_pt_root_id": "mn_root_id"})
    dn = dn_roots.rename(columns={"root_id": "dn_root_id", "cell_type": "dn",
                                  "somaSide": "dn_soma_side"})
    e = e.merge(dn[["dn_root_id", "dn", "dn_soma_side"]], on="dn_root_id", how="left")

    mn_cols = {"root_id": "mn_root_id"}
    if "cell_type" in mn_annotations.columns:
        mn_cols["cell_type"] = "muscle"
    if "somaSide" in mn_annotations.columns:
        mn_cols["somaSide"] = "mn_soma_side"
    mn = mn_annotations.rename(columns=mn_cols)
    keep = [c for c in ("mn_root_id", "muscle", "mn_soma_side", "cell_sub_class") if c in mn.columns]
    e = e.merge(mn[keep], on="mn_root_id", how="left")

    subclass = e["cell_sub_class"] if "cell_sub_class" in e.columns else e.get("muscle")
    e["motor_system"] = classify_motor_system(subclass.fillna(e.get("muscle")))
    return e


def split_by_wing(dn_mn: pd.DataFrame) -> pd.DataFrame:
    """Add ``wing_rel`` (ipsi/contra/unknown) and ``is_ipsi`` relative to the DN side."""
    out = dn_mn.copy()

    def _wing(row):
        d, m = row.get("dn_soma_side"), row.get("mn_soma_side")
        if d is None or m is None or pd.isna(d) or pd.isna(m):
            return "unknown"
        return "ipsi" if str(d) == str(m) else "contra"

    out["wing_rel"] = out.apply(_wing, axis=1)
    out["is_ipsi"] = out["wing_rel"].eq("ipsi")
    return out


def dn_wing_laterality(dn_mn_wing: pd.DataFrame) -> pd.DataFrame:
    """Per DN: steering_syn (wing-steering MNs only, power excluded), ipsi_frac, wing.

    Reproduces S14's schema so the result can be cross-checked against the oracle before
    extension. ``wing`` is "ipsilateral" (ipsi_frac>=0.6), "contralateral" (<=0.4), else
    "bilateral".
    """
    df = dn_mn_wing.copy()
    steering = df[(df["motor_system"] == "wing_steering")
                  & (~df.get("muscle", pd.Series(dtype=object)).map(C.is_power_muscle).fillna(False))]
    rows = []
    for dn, g in steering.groupby("dn"):
        total = float(g["syn_count"].sum())
        ipsi = float(g.loc[g["wing_rel"] == "ipsi", "syn_count"].sum())
        contra = float(g.loc[g["wing_rel"] == "contra", "syn_count"].sum())
        frac = ipsi / total if total else 0.0
        wing = "ipsilateral" if frac >= 0.6 else "contralateral" if frac <= 0.4 else "bilateral"
        rows.append({"dn": dn, "steering_syn": int(total), "ipsi_syn": int(ipsi),
                     "contra_syn": int(contra), "ipsi_frac": round(frac, 3), "wing": wing})
    return pd.DataFrame(rows).sort_values("steering_syn", ascending=False).reset_index(drop=True)


def muscle_table_for_dn(dn_mn_wing: pd.DataFrame, dn: str) -> pd.DataFrame:
    """Per (muscle, wing_rel): summed synapses for one DN. Reproduces S15."""
    g = dn_mn_wing[dn_mn_wing["dn"] == dn]
    if g.empty or "muscle" not in g.columns:
        return pd.DataFrame(columns=["muscle", "wing_rel", "syn"])
    tab = (g.groupby(["muscle", "wing_rel"])["syn_count"].sum()
           .reset_index().rename(columns={"syn_count": "syn"}))
    return tab.sort_values("syn", ascending=False).reset_index(drop=True)


def bilateral_muscle_map(dn_mn_wing: pd.DataFrame, *, channel_map: dict | None = None
                         ) -> pd.DataFrame:
    """The deliverable: every (DN, channel, MN, muscle, motor_system, wing_rel, syn).

    ``channel_map``: optional {dn_cell_type -> channel} to tag the descending channel
    (direct/nodtype/broadcast). The map covers BOTH wings (both somaSides) for all DNs -
    the extension beyond the paper's sampled exemplars.
    """
    df = dn_mn_wing.copy()
    if channel_map:
        df["channel"] = df["dn"].map(channel_map).fillna("unknown")
    else:
        df["channel"] = "unknown"
    wanted = ("dn", "channel", "dn_root_id", "mn_root_id", "muscle", "motor_system",
              "dn_soma_side", "mn_soma_side", "wing_rel", "syn_count")
    cols = [c for c in wanted if c in df.columns]
    return df[cols].sort_values(["dn", "syn_count"], ascending=[True, False]).reset_index(drop=True)
