"""Derive family P: the AFFERENT (input-side) pathway of the FD3 cell (LPT42_Nod4).

Family K established that LPT42_Nod4 IS Egelhaaf's FD3. Family L followed FD3's output to the
descending neurons and wing muscles. This family answers the remaining question: WHAT DRIVES
FD3 -- the full input pathway, from the photoreceptors through the motion detectors to FD3's
dendrite, plus FD3's non-motion central inputs and its contralateral inhibition.

FD3 is a REGRESSIVE (lobula-plate layer-b, back-to-front) cell, so its input side is NOT a copy
of the FD1=Nod1 arm (which is a VCH-gated, layer-a, progressive LLPC1 sheet). We therefore
trace and grade the input in four tiers, tagged by evidence strength so nothing is overclaimed:

  Tier A  IMMEDIATE INPUT CENSUS (measured ON FD3)
          Every direct presynaptic partner of FD3, split into: T4/T5 motion drive (with its
          lobula-plate layer composition), sibling columnar projection sheets (LPC/LLPC/Tlp),
          lobula-plate intrinsic + centrifugal inhibitors (LPi/CT1), and contra-vs-ipsi. We
          report BOTH the layer-b fraction WITHIN the T4/T5 subset (~98%) AND the T4/T5 fraction
          of TOTAL input (~25%) so "98% layer-b" is never misread as "98% motion-driven".

  Tier B  UPSTREAM COLUMNAR CASCADE (per-type canonical wiring, traced hop-by-hop)
          Starting from the layer-b T4b/T5b cells that actually synapse onto FD3, walk one hop up
          to their medulla drivers (T4b<-Mi/C3/Tm3 ON limb; T5b<-Tm1/Tm2/Tm4/Tm9 OFF limb), then
          note the lamina (L1/L2/L3) and photoreceptor (R1-6) layers that complete the cascade.
          These are per-TYPE facts (existence + dominance), NOT FD3-specific synapse counts.

  Tier C  NON-T4/T5 CENTRAL INPUTS (measured ON FD3)
          FD3's dominant non-motion inputs (LPC/LLPC sibling sheets). We test whether they carry
          the same regressive (layer-b) motion channel FD3 reads.

  Tier D  CONTRALATERAL INHIBITION (measured ON FD3, power-capped)
          FD3's contralateral GABAergic inputs stratified progressive vs regressive (Egelhaaf
          1985 p.203: the contra inhibition to FD3 is BIDIRECTIONAL, unlike FD1). Reuses the
          shared ``common.contra_inhibition_profile``.

  Negative controls: FD3 has NO VCH gate and NO layer-a drive -- the two facts that prove it is
  a PARALLEL regressive arm, not a copy of the FD1 sheet.

Source: Egelhaaf 1985 Biol. Cybern. 52:195-209 "The FD3-Cell" (Part II, p.202-204 + Discussion
p.206-208; the dedicated input-circuitry Part III 52:267-280 is cited but not re-derived here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as CM
from . import interhemi_common as IC
from .k_fd3_lpt42 import _CachedSource
from ..oracle import consts as C

CANDIDATE = "LPT42_Nod4"          # = Egelhaaf FD3 (family K)
ANCHOR = "Nod1"                   # = FD1, the progressive/layer-a comparison

# The FD1 gate FD3 must NOT share (negative control): the crossing centrifugal cells.
VCH_TYPES = ("VCH", "DCH")

# Canonical medulla drivers of the two layer-b elementary-motion detectors (Fischbach &
# Dittrich 1989; Takemura 2013/2017; Shinomiya 2019). Used to LABEL the Tier-B relay, not to
# gate it -- the gate is the measured presence/dominance of Mi/Tm input to FD3's T4b/T5b.
T4_ON_MEDULLA = ("Mi1", "Mi4", "Mi9", "Tm3", "C3", "C2")     # ON limb -> T4
T5_OFF_MEDULLA = ("Tm1", "Tm2", "Tm4", "Tm9")                # OFF limb -> T5
LAMINA_TYPES = ("L1", "L2", "L3", "L4", "L5")
PHOTORECEPTOR_TYPES = ("R1-6", "R7", "R8")

# A presynaptic T4/T5 cell is a Tier-B seed only if it drives FD3 with at least this many
# synapses (summed over both FD3 cells). Reported + swept so nothing is silently dropped.
RELAY_MIN_SYN = 3
# Live CAVE truncates a single synapse_query at this many rows; batch the upstream pull to
# stay clear (T4b/T5b presynaptic to FD3 can be hundreds of cells).
_CAVE_ROW_CAP = 500_000
_SEED_BATCH = 12

# How many top non-T4/T5 central input types to characterise for Tier C.
N_CENTRAL_TYPES = 6


def _fd3_roots(meta, candidate: str = CANDIDATE) -> list[int]:
    return sorted(int(x) for x in meta.root_ids_of_type([candidate]))


# ---------------------------------------------------------------------------
# Tier A -- immediate presynaptic census.
# ---------------------------------------------------------------------------
def _input_partner_table(src, meta, roots: list[int]) -> pd.DataFrame:
    """Per presynaptic partner of ``roots``: syn count + cell_type/side/super_class/nt + is_t4t5."""
    rset = set(int(x) for x in roots)
    syn = src.synapses(post_ids=roots)
    syn = syn[syn["post_pt_root_id"].isin(rset)]
    return CM.attach_meta(CM.partner_counts(syn, "pre_pt_root_id"), meta)


def _input_census(src, meta, roots: list[int], detectors: tuple = ("T4b", "T5b")) -> dict:
    """Split FD3's direct inputs into motion / columnar / inhibitory, with fractions + laterality.

    ``detectors`` names the ON/OFF elementary-motion detector pair whose synapse counts are
    reported in the ON/OFF split (``("T4b", "T5b")`` for the regressive/layer-b arm, or
    ``("T4a", "T5a")`` for the progressive/layer-a arm). The full four-layer composition is
    computed regardless, so only the ON/OFF split key depends on this argument.
    """
    pt = _input_partner_table(src, meta, roots)
    total_syn = int(pt["syn"].sum())

    # T4/T5 motion drive + its lobula-plate layer composition.
    t45 = pt[pt["is_t4t5"]]
    t45_syn = int(t45["syn"].sum())
    layer = {k: CM.layer_fraction(t45, k) for k in ("a", "b", "c", "d")}
    dom_layer = CM.dominant_layer(t45)
    # ON (T4) vs OFF (T5) split of the dominant-direction drive.
    on_type, off_type = detectors
    t4b = int(t45.loc[t45["cell_type"] == on_type, "syn"].sum())
    t5b = int(t45.loc[t45["cell_type"] == off_type, "syn"].sum())

    # Input laterality: fraction of input synapses from each partner soma side (side-robust;
    # FD3 is a bilateral pair, so per-cell contra/ipsi is resolved in the manifest, not here).
    by_side = pt[pt["side"].notna()].groupby("side")["syn"].sum().to_dict()

    # By super_class (optic = columnar visual, visual_projection = sibling sheets, etc.).
    by_sc = pt.groupby("super_class")["syn"].sum().sort_values(ascending=False).to_dict()

    # Top input types overall (for the report + the census figure).
    by_ct = (pt.dropna(subset=["cell_type"]).groupby("cell_type")
             .agg(syn=("syn", "sum"), n=("root_id", "nunique"),
                  sc=("super_class", "first"), nt=("nt_canonical", "first")).reset_index()
             .sort_values("syn", ascending=False))
    top_types = [{"cell_type": str(r.cell_type), "syn": int(r.syn), "n_cells": int(r.n),
                  "super_class": (None if pd.isna(r.sc) else str(r.sc)),
                  "nt": (None if pd.isna(r.nt) else str(r.nt))}
                 for r in by_ct.head(15).itertuples(index=False)]

    return {
        "total_input_syn": total_syn,
        "n_partners": int(pt["root_id"].nunique()),
        "t4t5_syn": t45_syn,
        "t4t5_frac_of_total": round(100.0 * t45_syn / total_syn, 1) if total_syn else float("nan"),
        "t4t5_layer_frac": {k: (round(v, 2) if v == v else None) for k, v in layer.items()},
        "t4t5_dominant_layer": dom_layer,
        "t4t5_dominant_direction": C.LAYER_DIRECTION.get(dom_layer) if dom_layer else None,
        "layer_b_frac_of_t4t5": round(layer.get("b", float("nan")), 2) if layer.get("b") == layer.get("b") else None,
        "t4b_syn": t4b, "t5b_syn": t5b,
        "detectors": {"ON": on_type, "OFF": off_type},
        "on_off_split": {f"{on_type}_ON": t4b, f"{off_type}_OFF": t5b,
                         "t4_frac": round(100.0 * t4b / (t4b + t5b), 1) if (t4b + t5b) else float("nan")},
        "input_syn_by_side": {str(k): int(v) for k, v in by_side.items()},
        "input_syn_by_super_class": {str(k): int(v) for k, v in by_sc.items()},
        "top_input_types": top_types,
    }


# ---------------------------------------------------------------------------
# Tier B -- upstream columnar cascade (hop-by-hop from FD3's T4b/T5b drivers).
# ---------------------------------------------------------------------------
def _presyn_t4t5_seeds(pt: pd.DataFrame, subtype: str) -> pd.DataFrame:
    """The presynaptic T4/T5 cells of one subtype (e.g. 'T4b') that drive FD3 >= RELAY_MIN_SYN."""
    sub = pt[(pt["cell_type"] == subtype) & (pt["syn"] >= RELAY_MIN_SYN)]
    return sub


def _second_hop_up(src, meta, seed_roots: list[int]):
    """Pull the INPUT synapses of ``seed_roots`` (one hop upstream), batched under the 500k cap.

    Mirrors ``l_fd3_descending._second_hop`` but on the input side: for a set of T4b/T5b cells
    it returns their presynaptic partner rows (pre metadata attached) + a truncation flag.
    """
    empty = pd.DataFrame(columns=["pre_pt_root_id", "post_pt_root_id", "syn",
                                  "pre_super_class", "pre_cell_type", "pre_nt"])
    seed_roots = [int(x) for x in seed_roots]
    if not seed_roots:
        return empty, False
    seed_set = set(seed_roots)
    parts, truncated = [], False
    for i in range(0, len(seed_roots), _SEED_BATCH):
        batch = seed_roots[i:i + _SEED_BATCH]
        inp = src.synapses(post_ids=batch)
        inp = inp[inp["post_pt_root_id"].isin(seed_set)]
        if len(inp) >= _CAVE_ROW_CAP:
            truncated = True
        parts.append(inp.groupby(["pre_pt_root_id", "post_pt_root_id"])
                     .size().rename("syn").reset_index())
    rows = (pd.concat(parts, ignore_index=True) if parts
            else empty[["pre_pt_root_id", "post_pt_root_id", "syn"]])
    pm = meta.by_root.reindex(rows["pre_pt_root_id"].values)
    rows = rows.assign(pre_super_class=pm["super_class"].to_numpy(),
                       pre_cell_type=pm["cell_type"].to_numpy(),
                       pre_nt=pm["nt_canonical"].to_numpy())
    return rows, truncated


def _drivers_of(hop_rows: pd.DataFrame, expected: tuple[str, ...]) -> dict:
    """Rank the presynaptic types driving a T4/T5 subtype; flag which expected medulla types hit."""
    by = (hop_rows.dropna(subset=["pre_cell_type"]).groupby("pre_cell_type")["syn"]
          .sum().sort_values(ascending=False))
    total = float(by.sum())
    top = [{"cell_type": str(k), "syn": int(v),
            "frac": round(100.0 * v / total, 1) if total else float("nan")}
           for k, v in by.head(10).items()]
    present = sorted(set(expected) & set(by.index))
    exp_syn = float(by.reindex(list(expected)).fillna(0).sum())
    return {
        "top_drivers": top,
        "expected_medulla_present": present,
        "n_expected_present": len(present),
        "expected_frac_of_input": round(100.0 * exp_syn / total, 1) if total else float("nan"),
        "total_input_syn": int(total),
    }


def _upstream_cascade(src, meta, pt: pd.DataFrame, detectors: tuple = ("T4b", "T5b")) -> dict:
    """Trace one hop up from the figure cell's ON (T4) and OFF (T5) drivers to their medulla
    inputs, and record the lamina + photoreceptor layers that complete the cascade (existence,
    per type). ``detectors`` selects the ON/OFF detector subtypes for the arm being traced."""
    on_type, off_type = detectors
    t4b_seeds = _presyn_t4t5_seeds(pt, on_type)
    t5b_seeds = _presyn_t4t5_seeds(pt, off_type)
    t4b_roots = [int(x) for x in t4b_seeds["root_id"]]
    t5b_roots = [int(x) for x in t5b_seeds["root_id"]]

    t4b_hop, tr1 = _second_hop_up(src, meta, t4b_roots)
    t5b_hop, tr2 = _second_hop_up(src, meta, t5b_roots)
    on = _drivers_of(t4b_hop, T4_ON_MEDULLA)
    off = _drivers_of(t5b_hop, T5_OFF_MEDULLA)

    # Presence of the lamina + photoreceptor layers in the connectome (the cascade completes).
    present_types = set(meta.df["cell_type"].dropna().unique())
    lamina_present = sorted(t for t in LAMINA_TYPES if t in present_types)
    photoreceptor_present = sorted(t for t in PHOTORECEPTOR_TYPES if t in present_types)

    return {
        "n_t4b_seeds": len(t4b_roots), "n_t5b_seeds": len(t5b_roots),
        "on_limb_t4b": on,      # T4b <- Mi/Tm3/C3 (ON)
        "off_limb_t5b": off,    # T5b <- Tm1/Tm2/Tm4/Tm9 (OFF)
        "lamina_present": lamina_present,
        "photoreceptor_present": photoreceptor_present,
        "cascade_complete": bool(on["n_expected_present"] and off["n_expected_present"]
                                 and lamina_present and photoreceptor_present),
        "query_truncated": bool(tr1 or tr2),
    }


# ---------------------------------------------------------------------------
# Tier C -- non-T4/T5 central inputs.
# ---------------------------------------------------------------------------
def _central_inputs(src, meta, pt: pd.DataFrame) -> dict:
    """FD3's dominant non-T4/T5 inputs, and whether they carry the same layer-b motion channel."""
    non = pt[(~pt["is_t4t5"]) & (pt["super_class"].isin(["optic", "visual_projection"]))]
    by_ct = (non.dropna(subset=["cell_type"]).groupby("cell_type")
             .agg(syn=("syn", "sum"), n=("root_id", "nunique"),
                  sc=("super_class", "first"), nt=("nt_canonical", "first")).reset_index()
             .sort_values("syn", ascending=False))
    types = []
    for r in by_ct.head(N_CENTRAL_TYPES).itertuples(index=False):
        roots = sorted(int(x) for x in meta.root_ids_of_type([str(r.cell_type)]))
        prof = IC.input_layer_profile(src, meta, roots) if roots else {"has_t4t5": False}
        types.append({
            "cell_type": str(r.cell_type), "syn": int(r.syn), "n_cells": int(r.n),
            "super_class": (None if pd.isna(r.sc) else str(r.sc)),
            "nt": (None if pd.isna(r.nt) else str(r.nt)),
            "reads_layer_b": bool(prof.get("dominant_layer") == "b") if prof.get("has_t4t5") else None,
            "dominant_layer": prof.get("dominant_layer"),
        })
    layerb_carriers = [t for t in types if t.get("reads_layer_b")]
    return {
        "n_central_types": int(len(by_ct)),
        "top_types": types,
        "n_layerb_carriers": len(layerb_carriers),
        "any_layerb_carrier": bool(layerb_carriers),
        "total_central_syn": int(by_ct["syn"].sum()),
    }


# ---------------------------------------------------------------------------
# Negative controls -- FD3 is NOT the FD1 sheet.
# ---------------------------------------------------------------------------
def _neg_controls(src, meta, pt: pd.DataFrame, census: dict) -> dict:
    """FD3 has (1) no VCH/DCH gate and (2) no layer-a (progressive) drive -- unlike FD1."""
    vch_syn = int(pt.loc[pt["cell_type"].isin(VCH_TYPES), "syn"].sum())
    total = int(pt["syn"].sum())
    layer_a = census.get("t4t5_layer_frac", {}).get("a")
    return {
        "vch_input_syn": vch_syn,
        "vch_input_frac": round(100.0 * vch_syn / total, 3) if total else float("nan"),
        "no_vch_gate": bool(vch_syn <= max(3, 0.01 * total)),
        "layer_a_frac_of_t4t5": layer_a,
        "no_layer_a_drive": bool((layer_a or 0) < 10.0),
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run(src, meta, cfg: C.SideConfig = C.RIGHT, *, candidate: str = CANDIDATE,
        anchor: str = ANCHOR, detectors: tuple = ("T4b", "T5b"),
        roots: list[int] | None = None) -> dict:
    # Family P is intrinsically BILATERAL (it scores the figure cell pair's afferents per type
    # already), so it does not branch on cfg; the parameter is accepted for a uniform call
    # signature. ``candidate`` defaults to the FD3 cell (LPT42_Nod4); pass "LPT21" for FD2, etc.
    # ``detectors`` selects the ON/OFF motion-detector pair for the arm (layer-b "T4b"/"T5b" for
    # FD2/FD3, layer-a "T4a"/"T5a" for the FD1/FD4 progressive arm). ``roots`` overrides the type
    # lookup when the figure cell is a sub-set of a type (e.g. an FD4 sub-pair of Nod1).
    del cfg
    src = _CachedSource(src)  # memoize synapse pulls across the tiers
    roots = [int(x) for x in roots] if roots is not None else _fd3_roots(meta, candidate)
    pt = _input_partner_table(src, meta, roots)

    census = _input_census(src, meta, roots, detectors=detectors)
    cascade = _upstream_cascade(src, meta, pt, detectors=detectors)
    central = _central_inputs(src, meta, pt)
    contra_inh = CM.contra_inhibition_profile(src, meta, candidate)
    neg = _neg_controls(src, meta, pt, census)

    return {
        "candidate": candidate,
        "anchor": anchor,
        "fd3_roots": roots,
        "census": census,
        "upstream_cascade": cascade,
        "central_inputs": central,
        "contra_inhibition": contra_inh,
        "negative_controls": neg,
        "query_truncated": bool(cascade.get("query_truncated")),
        "track": src.track,
    }
