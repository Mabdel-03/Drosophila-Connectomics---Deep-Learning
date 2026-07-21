"""Derive family L: the descending neurons the FD3 cell (LPT42_Nod4) connects to.

Family K established that LPT42_Nod4 IS Egelhaaf's FD3. This family asks the next question:
which DESCENDING NEURONS (DNs) carry the FD3 figure signal out of the brain, and what do they
actuate? DNs are the sole brain->ventral-nerve-cord conduit, so naming FD3's DN targets and
following them (in the male CNS) to motor neurons and muscles turns the identity result into a
functional sensorimotor circuit.

Two synaptic routes are enumerated, mirroring the established FD1=Nod1 arm (families G/H/I/J):

  * DIRECT   FD3 -> DN          (monosynaptic; the cell's own descending output)
  * RELAYED  FD3 -> X -> DN     (one hop through FD3's strongest non-DN central partners)

For every identified DN type we then cross into the male CNS (offline MaleCNS tables, bridged by
the shared Janelia DN ``type`` name) and read its motor-neuron / muscle targets, the motor system
it drives (wing-steering / neck-gaze / haltere / leg / jump / abdominal), and its wing laterality.

Reuses, verbatim, the primitives the figure-ground families already use:
  - synapse access + metadata: ``fw_access`` (``src.synapses`` / ``meta``)
  - partner aggregation:       ``common.partner_counts`` / ``common.attach_meta``
  - DN-target enumeration:     the ``i_descending_channels`` ``super_class == 'descending'`` idiom
  - DN -> motor -> muscle:     ``malecns.client`` + the Family-J wing-laterality convention
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as CM
from ..oracle import consts as C

SEED = 12345

CANDIDATE = "LPT42_Nod4"  # = Egelhaaf FD3 (family K)
STEERING_DN_TYPES = set(C.WING_STEERING_DNS)

# A central partner of FD3 counts as a "relay" only if FD3 drives it with at least this many
# synapses (summed over both FD3 cells). The cut is reported and swept (see ``_relay_threshold_sweep``)
# so nothing is silently dropped; 30 is the default working floor.
RELAY_MIN_SYN = 30
# Smaller floors used only for the reported sensitivity sweep.
RELAY_SWEEP = (10, 20, 30, 50, 100)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fd3_roots(meta, candidate: str = CANDIDATE) -> list[int]:
    return sorted(int(x) for x in meta.root_ids_of_type([candidate]))


def _output_partner_table(src, meta, roots: list[int]) -> pd.DataFrame:
    """Per postsynaptic partner of ``roots``: syn count + cell_type/side/super_class/nt."""
    out = src.synapses(pre_ids=roots)
    out = out[out["pre_pt_root_id"].isin(set(int(x) for x in roots))]
    return CM.attach_meta(CM.partner_counts(out, "post_pt_root_id"), meta)


def _dn_block(partner_table: pd.DataFrame) -> dict:
    """Summarise the descending rows of a partner table (synapses, #DNs, per-type ranking)."""
    dn = partner_table[partner_table["super_class"].values == "descending"]
    total_syn = int(dn["syn"].sum())
    n_dns = int(dn["root_id"].nunique())
    by = (dn.dropna(subset=["cell_type"]).groupby("cell_type")
          .agg(syn=("syn", "sum"), n_cells=("root_id", "nunique"),
               nt=("nt_canonical", "first")).reset_index()
          .sort_values("syn", ascending=False))
    by["is_steering"] = by["cell_type"].isin(STEERING_DN_TYPES)
    steering_syn = int(by.loc[by["is_steering"], "syn"].sum())
    ranking = [{"cell_type": str(r.cell_type), "syn": int(r.syn), "n_cells": int(r.n_cells),
                "nt": (None if pd.isna(r.nt) else str(r.nt)), "is_steering": bool(r.is_steering)}
               for r in by.itertuples(index=False)]
    return {
        "dn_syn": total_syn,
        "n_dns": n_dns,
        "steering_syn": steering_syn,
        "steering_frac": round(100.0 * steering_syn / total_syn, 1) if total_syn else float("nan"),
        "ranking": ranking,
        "top_dn": ranking[0]["cell_type"] if ranking else None,
    }


def _direct(src, meta, roots: list[int]) -> dict:
    """DIRECT FD3 -> DN, plus the per-FD3-cell (left/right) split of the descending output."""
    pt = _output_partner_table(src, meta, roots)
    block = _dn_block(pt)
    block["total_output_syn"] = int(pt["syn"].sum())
    block["total_partners"] = int(pt["root_id"].nunique())

    # Per-side (which physical FD3 cell drives the descending output).
    per_side = {}
    out = src.synapses(pre_ids=roots)
    out = out[out["pre_pt_root_id"].isin(set(int(x) for x in roots))]
    post_sc = meta.by_root.reindex(out["post_pt_root_id"].values)["super_class"].to_numpy()
    out = out.assign(_post_is_dn=(post_sc == "descending"))
    for r in roots:
        side = str(meta.by_root.loc[r, "side"]) if r in meta.by_root.index else str(r)
        sub = out[out["pre_pt_root_id"] == r]
        per_side[side] = {"dn_syn": int(sub["_post_is_dn"].sum()),
                          "total_syn": int(len(sub))}
    block["per_side"] = per_side
    return block


def _relay_seeds(partner_table: pd.DataFrame, min_syn: int) -> pd.DataFrame:
    """FD3's strongest NON-descending central partners (the relay intermediaries)."""
    inter = partner_table[partner_table["super_class"].values != "descending"]
    inter = inter.dropna(subset=["cell_type"])
    return inter[inter["syn"] >= min_syn]


# The live CAVE synapse_query truncates at this many rows (server-side default), logging
# "Limited query to 500000 rows". The relay's widest seed set can exceed it in one call, which
# would SILENTLY undercount; we query in batches small enough to stay clear of the cap.
_CAVE_ROW_CAP = 500_000
_SEED_BATCH = 12


def _second_hop(src, meta, partner_table: pd.DataFrame):
    """Pull FD3's second-hop synapses ONCE for the widest seed set (lowest sweep floor).

    Every threshold is then computed by filtering this single pull in memory. The query is
    BATCHED over seed roots so the live CAVE 500k-row cap is never hit in a single call (which
    would silently truncate the relay tally). Returns (seed_table, second_hop_partner_rows,
    truncated_flag); ``truncated_flag`` is True if any batch came back at the cap.
    """
    widest = min(RELAY_SWEEP + (RELAY_MIN_SYN,))
    seeds_all = _relay_seeds(partner_table, widest)
    seed_roots = [int(x) for x in seeds_all["root_id"]]
    empty = pd.DataFrame(columns=["pre_pt_root_id", "post_pt_root_id", "syn",
                                  "post_super_class", "post_cell_type", "post_nt"])
    if not seed_roots:
        return seeds_all, empty, False

    seed_set = set(seed_roots)
    parts, truncated = [], False
    for i in range(0, len(seed_roots), _SEED_BATCH):
        batch = seed_roots[i:i + _SEED_BATCH]
        out = src.synapses(pre_ids=batch)
        out = out[out["pre_pt_root_id"].isin(seed_set)]
        if len(out) >= _CAVE_ROW_CAP:           # a batch came back at the cap -> truncation risk
            truncated = True
        parts.append(out.groupby(["pre_pt_root_id", "post_pt_root_id"])
                     .size().rename("syn").reset_index())
    rows = (pd.concat(parts, ignore_index=True) if parts
            else empty[["pre_pt_root_id", "post_pt_root_id", "syn"]])
    # attach the postsynaptic (DN) metadata once.
    pm = meta.by_root.reindex(rows["post_pt_root_id"].values)
    rows = rows.assign(post_super_class=pm["super_class"].to_numpy(),
                       post_cell_type=pm["cell_type"].to_numpy(),
                       post_nt=pm["nt_canonical"].to_numpy())
    return seeds_all, rows, truncated


def _relay_at(seeds_all: pd.DataFrame, hop_rows: pd.DataFrame, min_syn: int) -> dict:
    """RELAYED FD3 -> intermediary -> DN at one threshold, by filtering the cached hop rows."""
    seeds = seeds_all[seeds_all["syn"] >= min_syn]
    seed_roots = set(int(x) for x in seeds["root_id"])
    if not seed_roots:
        return {"min_syn": min_syn, "n_seed_cells": 0, "n_seed_types": 0, "seed_types": [],
                "dn_syn": 0, "n_dns": 0, "steering_syn": 0, "steering_frac": float("nan"),
                "ranking": [], "top_dn": None}
    hop = hop_rows[hop_rows["pre_pt_root_id"].isin(seed_roots)]
    dn = hop[hop["post_super_class"].values == "descending"]
    total_syn = int(dn["syn"].sum())
    n_dns = int(dn["post_pt_root_id"].nunique())
    by = (dn.dropna(subset=["post_cell_type"]).groupby("post_cell_type")
          .agg(syn=("syn", "sum"), n_cells=("post_pt_root_id", "nunique"),
               nt=("post_nt", "first")).reset_index()
          .rename(columns={"post_cell_type": "cell_type"})
          .sort_values("syn", ascending=False))
    by["is_steering"] = by["cell_type"].isin(STEERING_DN_TYPES)
    steering_syn = int(by.loc[by["is_steering"], "syn"].sum())
    ranking = [{"cell_type": str(r.cell_type), "syn": int(r.syn), "n_cells": int(r.n_cells),
                "nt": (None if pd.isna(r.nt) else str(r.nt)), "is_steering": bool(r.is_steering)}
               for r in by.itertuples(index=False)]
    seed_by = (seeds.groupby("cell_type")
               .agg(fd3_syn=("syn", "sum"), n=("root_id", "nunique"),
                    sc=("super_class", "first")).reset_index()
               .sort_values("fd3_syn", ascending=False))
    return {
        "min_syn": min_syn,
        "n_seed_cells": int(len(seed_roots)),
        "n_seed_types": int(seeds["cell_type"].nunique()),
        "seed_types": [{"cell_type": str(r.cell_type), "fd3_syn": int(r.fd3_syn),
                        "n_cells": int(r.n), "super_class": str(r.sc)}
                       for r in seed_by.itertuples(index=False)],
        "dn_syn": total_syn,
        "n_dns": n_dns,
        "steering_syn": steering_syn,
        "steering_frac": round(100.0 * steering_syn / total_syn, 1) if total_syn else float("nan"),
        "ranking": ranking,
        "top_dn": ranking[0]["cell_type"] if ranking else None,
    }


# ---------------------------------------------------------------------------
# MaleCNS motor extension (the functional payload)
# ---------------------------------------------------------------------------
def _wing_category(ipsi_frac: float) -> str:
    if ipsi_frac >= 0.60:
        return "ipsilateral"
    if ipsi_frac <= 0.40:
        return "contralateral"
    return "bilateral"


def _motor_profile(dn_types: list[str], drive_by_type: dict[str, int]) -> dict:
    """For each DN type, read its MaleCNS motor targets; aggregate FD3's motor-system reach.

    ``drive_by_type``: FD3's (direct, synapse-weighted) drive onto each DN type, used to weight
    the brain-wide motor-system distribution. Gated on the offline MaleCNS tables being present;
    returns ``{"available": False, ...}`` if not (family then degrades to UNVERIFIABLE).
    """
    from ..malecns import ANNOTATIONS, client as MC

    if not ANNOTATIONS.exists():
        return {"available": False, "reason": f"MaleCNS annotations not found at {ANNOTATIONS}"}

    motor = MC.motor_neurons()
    steering = motor[motor["is_wing_steering"]]

    per_dn = {}
    # synapse-weighted motor-system totals across DNs FD3 drives (weight = FD3->DN drive).
    sys_weighted: dict[str, float] = {}
    sys_raw: dict[str, float] = {}
    for dn in dn_types:
        bodies = MC.bodies_of_type(dn)
        if len(bodies) == 0:
            per_dn[dn] = {"n_bodies": 0, "note": "no MaleCNS body of this type"}
            continue
        edges = MC.dn_to_motor(bodies["bodyId"].tolist(), motor)
        if len(edges) == 0:
            per_dn[dn] = {"n_bodies": int(len(bodies)), "motor_syn": 0,
                          "note": "no motor-neuron output in MaleCNS"}
            continue
        # motor-system distribution for this DN (by MaleCNS subclass; wm=wing-steering etc.).
        sys_for_dn = _system_breakdown(edges)
        # wing laterality over wing-steering output (DN body side vs MN side; Family J idiom).
        steer = edges[edges["is_wing_steering"]]
        ipsi_frac, category = None, None
        if len(steer):
            dn_side = bodies.set_index("bodyId")["somaSide"]
            sd = steer.copy()
            sd["dn_side"] = sd["body_pre"].map(dn_side)
            total = int(sd["syn"].sum())
            ipsi = int(sd.loc[sd["dn_side"] == sd["somaSide"], "syn"].sum())
            ipsi_frac = round(ipsi / total, 2) if total else None
            category = _wing_category(ipsi_frac) if ipsi_frac is not None else None
        top_muscles = (steer.groupby("muscle")["syn"].sum().sort_values(ascending=False)
                       .head(4).to_dict() if len(steer) else {})
        per_dn[dn] = {
            "n_bodies": int(len(bodies)),
            "motor_syn": int(edges["syn"].sum()),
            "steering_syn": int(steer["syn"].sum()) if len(steer) else 0,
            "ipsi_frac": ipsi_frac,
            "wing": category,
            "top_muscles": {str(k): int(v) for k, v in top_muscles.items()},
            "motor_systems": sys_for_dn,
            "dominant_motor_system": (max(sys_for_dn, key=sys_for_dn.get) if sys_for_dn else None),
            "is_steering_dn": dn in STEERING_DN_TYPES,
        }
        # accumulate the FD3-weighted brain-wide motor-system reach.
        dn_total = float(sum(sys_for_dn.values())) or 1.0
        w = float(drive_by_type.get(dn, 0))
        for syst, syn in sys_for_dn.items():
            sys_raw[syst] = sys_raw.get(syst, 0.0) + syn
            sys_weighted[syst] = sys_weighted.get(syst, 0.0) + w * (syn / dn_total)

    wtot = sum(sys_weighted.values()) or 1.0
    rtot = sum(sys_raw.values()) or 1.0
    motor_system_pct = {k: round(100.0 * v / wtot, 1) for k, v in
                        sorted(sys_weighted.items(), key=lambda kv: -kv[1])}
    motor_system_pct_raw = {k: round(100.0 * v / rtot, 1) for k, v in
                            sorted(sys_raw.items(), key=lambda kv: -kv[1])}
    return {
        "available": True,
        "per_dn": per_dn,
        "n_dns_mapped": int(sum(1 for v in per_dn.values() if v.get("n_bodies"))),
        "motor_system_pct": motor_system_pct,            # FD3-drive-weighted
        "motor_system_pct_raw": motor_system_pct_raw,    # unweighted (per-MN-synapse)
        "dominant_motor_system": (next(iter(motor_system_pct)) if motor_system_pct else None),
        "track": "malecns_offline",
    }


# MaleCNS motor-neuron subclass code -> functional motor system. The curated subclass is the
# authoritative tag (more reliable than name regex); 'wm' (wing muscle) is split into
# wing-steering vs wing-power (DLM/DVM/TTMn) per the paper's Methods.
_SUBCLASS_SYSTEM = {
    "nm": "neck_gaze", "hm": "haltere", "am": "abdominal",
    "fl": "leg", "hl": "leg", "ml": "leg", "ad": "abdominal",
    "pm": "other", "rm": "other", "xm": "other",
}


def _system_breakdown(edges: pd.DataFrame) -> dict[str, float]:
    """Synapse totals per functional motor system for one DN's MaleCNS motor output.

    Classifies by the curated MaleCNS subclass code (authoritative), splitting the wing-muscle
    subclass into wing-steering vs wing-power (DLM/DVM power excluded from steering).
    """
    from ...muscular import muscular_config as MCfg

    e = edges.copy()
    is_power = e["muscle"].fillna("").map(MCfg.is_power_muscle)
    sc = e["subclass"].astype("string")

    def _sys(row_sc, steering, power):
        if str(row_sc) == MCfg.WING_STEERING_SUBCLASS:   # 'wm'
            return "wing_power" if power else "wing_steering"
        return _SUBCLASS_SYSTEM.get(str(row_sc), "other")

    syst = [_sys(s, st, p) for s, st, p in zip(sc, e["is_wing_steering"], is_power)]
    e = e.assign(_sys=syst)
    return {str(k): int(v) for k, v in e.groupby("_sys")["syn"].sum().items()}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(src, meta, cfg: C.SideConfig = C.RIGHT, *, candidate: str = CANDIDATE) -> dict:
    # Bilateral by construction (the figure cell is a left/right pair); cfg accepted for a uniform
    # signature. ``candidate`` defaults to FD3 (LPT42_Nod4); pass "LPT21" for the FD2 descending trace.
    del cfg
    # Memoize synapse pulls: the offline source rescans the whole feather per call, and the
    # direct/relay blocks query the cell's output twice. Same wrapper Family K uses.
    from .k_fd3_lpt42 import _CachedSource
    src = _CachedSource(src)
    roots = _fd3_roots(meta, candidate)

    partner_table = _output_partner_table(src, meta, roots)
    direct = _direct(src, meta, roots)

    # Second hop pulled ONCE (batched) for the widest seed set; thresholds filter it in memory.
    seeds_all, hop_rows, hop_truncated = _second_hop(src, meta, partner_table)
    relay = _relay_at(seeds_all, hop_rows, RELAY_MIN_SYN)
    relay["query_truncated"] = bool(hop_truncated)
    relay_sweep = [{"min_syn": r["min_syn"], "n_seed_cells": r["n_seed_cells"],
                    "n_dns": r["n_dns"], "dn_syn": r["dn_syn"]}
                   for r in (_relay_at(seeds_all, hop_rows, t) for t in RELAY_SWEEP)]

    # direct vs relay overlap (which DNs FD3 reaches directly, which only via a relay).
    direct_types = {r["cell_type"] for r in direct["ranking"]}
    relay_types = {r["cell_type"] for r in relay["ranking"]}
    overlap = sorted(direct_types & relay_types)
    direct_only = sorted(direct_types - relay_types)
    relay_only = sorted(relay_types - direct_types)

    # MaleCNS motor extension for the union of identified DN types (direct gets weighting).
    union_types = sorted(direct_types | relay_types)
    direct_drive = {r["cell_type"]: r["syn"] for r in direct["ranking"]}
    motor = _motor_profile(union_types, direct_drive)

    return {
        "candidate": candidate,
        "fd3_roots": [int(x) for x in roots],
        "direct": direct,
        "relay": relay,
        "relay_min_syn": RELAY_MIN_SYN,
        "relay_threshold_sweep": relay_sweep,
        "overlap": {
            "n_overlap": len(overlap), "overlap": overlap,
            "n_direct_only": len(direct_only), "direct_only": direct_only,
            "n_relay_only": len(relay_only), "relay_only": relay_only,
        },
        "motor": motor,
        "track": src.track,
    }
