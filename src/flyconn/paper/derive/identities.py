"""Stage-8 cell-identity manifest: every root id of the figure-ground circuit, per side.

The prior bilateral pass (openq/q2_bilateral.py) worked only by cell_type+side and never
reported a single left-side root id, so the left circuit's cells were never NAMED. This
module fixes that: for a given ``SideConfig`` it resolves and records the concrete root ids of
every circuit element, with provenance, so "which exact neurons constitute the left circuit"
is answerable and re-auditable.

Read-only. Reuses ``derive.sheet.get_sheet`` for the gated T4a + LLPC1 sheet, and
``NeuronMeta.root_ids_of_type`` for the named cells. For DNs it also records MaleCNS bodyIds +
somaSide (the wing innervated) when the offline MCNS tables are present (for the wing-flip).
"""

from __future__ import annotations

import numpy as np

from . import common as CM
from . import sheet as SH
from ..oracle import consts as C

# Named circuit cells (single-or-few per side in FlyWire). DNs are matched by type in MCNS.
SHEET_INHIBITORS = ("LPi15", "PVLP011", "PLP249", "PLP163", "Am1", "LPi14")
NOD_TYPES = ("Nod1", "Nod2", "Nod3", "Nod5", "LPT42_Nod4")
CIRCUIT_DNS = ("DNp26", "DNbe001", "DNa04", "DNg32", "DNge094", "DNbe005", "DNge107", "DNae002")


def _cell_record(meta, root: int) -> dict:
    row = meta.by_root.loc[int(root)]
    return {
        "root_id": int(root),
        "cell_type": str(row.get("cell_type")),
        "side": str(row.get("side")),
        "super_class": str(row.get("super_class")),
        "nt_canonical": str(row.get("nt_canonical")),
    }


def _type_roots(meta, cell_type: str, side: str | None) -> list[dict]:
    roots = meta.root_ids_of_type([cell_type], side=side)
    return [_cell_record(meta, r) for r in sorted(int(x) for x in roots)]


def build_manifest(src, meta, cfg: C.SideConfig) -> dict:
    """All circuit root ids for one side, with provenance."""
    side = cfg.sheet_side

    # Gating centrifugal cells (the crossing VCH/DCH), re-verified at runtime.
    vch_roots = sorted(int(x) for x in meta.root_ids_of_type(["VCH"], side=cfg.gating_soma_side))
    dch_roots = sorted(int(x) for x in meta.root_ids_of_type(["DCH"], side=cfg.gating_soma_side))
    gating_vch = {
        "root": cfg.vch_root,
        "matches_annotation": bool(vch_roots == [cfg.vch_root]),
        "soma_side": cfg.gating_soma_side,
        **(_cell_record(meta, cfg.vch_root)),
    }
    gating_dch = {
        "root": cfg.dch_root,
        "matches_annotation": bool(dch_roots == [cfg.dch_root]),
        "soma_side": cfg.gating_soma_side,
        **(_cell_record(meta, cfg.dch_root)),
    }

    # The gated sheet: VCH-gated same-side T4a -> same-side LLPC1.
    sheet = SH.get_sheet(src, meta, cfg)
    llpc1_annotated = sorted(int(x) for x in meta.root_ids_of_type(["LLPC1"], side=side))

    # Nod-types and sheet inhibitors on this side.
    nods = {t: _type_roots(meta, t, side) for t in NOD_TYPES}
    inhibitors = {t: _type_roots(meta, t, side) for t in SHEET_INHIBITORS}

    # Nod1 cells driven BY this sheet (the sheet-driven relay subset).
    nod1_roots = [r["root_id"] for r in nods.get("Nod1", [])]
    nod1_sheet_driven = _nod1_sheet_driven(src, sheet, nod1_roots)

    # DNs: FlyWire left+right roots (matched by type) + MCNS bodies/somaSide for the wing-flip.
    dns = {}
    for dn in CIRCUIT_DNS:
        dns[dn] = {
            "flywire_left": [r["root_id"] for r in _type_roots(meta, dn, "left")],
            "flywire_right": [r["root_id"] for r in _type_roots(meta, dn, "right")],
            "mcns_bodies": _mcns_bodies(dn),
        }

    return {
        "sheet_side": side,
        "gating_soma_side": cfg.gating_soma_side,
        "gating_vch": gating_vch,
        "gating_dch": gating_dch,
        "t4a_gated": {"n": sheet.n_t4a, "roots": [int(x) for x in sheet.t4a_roots]},
        "llpc1_sheet": {
            "n": sheet.n_llpc1,
            "roots": [int(x) for x in sheet.llpc1_roots],
            "n_annotated_side": len(llpc1_annotated),
            "annotated_roots": llpc1_annotated,
            "sheet_is_subset_of_annotated": set(int(x) for x in sheet.llpc1_roots).issubset(set(llpc1_annotated)),
        },
        "nod_types": nods,
        "nod1_sheet_driven_roots": nod1_sheet_driven,
        "sheet_inhibitors": inhibitors,
        "descending_neurons": dns,
        "track": getattr(src, "track", None),
    }


def build_fd3_input_manifest(src, meta) -> dict:
    """Every root id of the FD3 (LPT42_Nod4) afferent circuit, with provenance.

    The output side (FD3 -> DN -> muscle) is already named by ``build_manifest``'s DN block and
    Family L; this fills the INPUT side. It records:
      * fd3                the LPT42_Nod4 pair
      * presynaptic_t4t5   the ACTUAL T4b/T5b cells synapsing onto FD3 (not the whole type),
                           per subtype, capped-list + count
      * central_inputs     the top columnar-sheet (LPC/LLPC) partners presynaptic to FD3
      * contra_inhibitors  the contralateral GABA partners (LPi/CT1/...) presynaptic to FD3
      * cascade_types      per-type presence of the upstream lamina/medulla/photoreceptor layers
    Reuses ``derive.p_fd3_input`` primitives (the input partner table) + ``_cell_record``.
    """
    from . import p_fd3_input as P

    roots = P._fd3_roots(meta)
    fd3 = [_cell_record(meta, r) for r in roots]
    pt = P._input_partner_table(src, meta, roots)

    def _partner_records(mask, cap: int = 60) -> dict:
        sub = pt[mask].sort_values("syn", ascending=False)
        sub_roots = [int(x) for x in sub["root_id"]]
        return {
            "n_cells": len(sub_roots),
            "total_syn": int(sub["syn"].sum()),
            "roots": [_cell_record(meta, r) for r in sub_roots[:cap]],
            "roots_truncated": len(sub_roots) > cap,
        }

    presyn_t4t5 = {sub: _partner_records(pt["cell_type"] == sub)
                   for sub in ("T4b", "T5b")}

    central_mask = ((~pt["is_t4t5"]) & pt["super_class"].isin(["optic", "visual_projection"])
                    & pt["cell_type"].notna())
    # top central types by synapse, then their partner roots
    top_central_types = (pt[central_mask].groupby("cell_type")["syn"].sum()
                         .sort_values(ascending=False).head(P.N_CENTRAL_TYPES).index.tolist())
    central_inputs = {t: _partner_records(pt["cell_type"] == t) for t in top_central_types}

    # Contralateral GABA partners presynaptic to FD3. FD3 is a bilateral pair, so laterality is
    # per-FD3-cell (a partner is contra to the specific FD3 cell it targets); we union across
    # both cells. Aggregating across the pair first (as ``pt`` does) would erase laterality.
    contra_roots: dict[int, int] = {}  # partner root -> summed contra syn
    for r in roots:
        r_side = meta.by_root.loc[r, "side"] if r in meta.by_root.index else None
        din = src.synapses(post_ids=[r]); din = din[din["post_pt_root_id"] == r]
        counts = CM.attach_meta(CM.partner_counts(din, "pre_pt_root_id"), meta)
        contra = counts[(counts["nt_canonical"] == "gaba") & counts["side"].notna()
                        & (counts["side"] != r_side)]
        for pr, sy in zip(contra["root_id"], contra["syn"]):
            contra_roots[int(pr)] = contra_roots.get(int(pr), 0) + int(sy)
    ranked = sorted(contra_roots.items(), key=lambda kv: kv[1], reverse=True)
    contra_inhibitors = {
        "n_cells": len(ranked),
        "total_syn": int(sum(contra_roots.values())),
        "roots": [{**_cell_record(meta, pr), "contra_syn": sy} for pr, sy in ranked[:60]],
        "roots_truncated": len(ranked) > 60,
    }

    present_types = set(meta.df["cell_type"].dropna().unique())
    cascade_types = {
        "photoreceptor": [t for t in P.PHOTORECEPTOR_TYPES if t in present_types],
        "lamina": [t for t in P.LAMINA_TYPES if t in present_types],
        "medulla_on": [t for t in P.T4_ON_MEDULLA if t in present_types],
        "medulla_off": [t for t in P.T5_OFF_MEDULLA if t in present_types],
    }

    return {
        "circuit": "FD3 afferent pathway (photoreceptor -> T4/T5 -> LPT42_Nod4)",
        "fd3": fd3,
        "presynaptic_t4t5": presyn_t4t5,
        "central_inputs": central_inputs,
        "contra_inhibitors": contra_inhibitors,
        "cascade_types": cascade_types,
        "track": getattr(src, "track", None),
    }


def _nod1_sheet_driven(src, sheet, nod1_roots: list[int]) -> list[int]:
    """Which of this side's Nod1 cells actually receive synapses from the sheet."""
    if not nod1_roots:
        return []
    sheet_set = set(int(x) for x in sheet.llpc1_roots)
    out = src.synapses(pre_ids=list(sheet_set))
    out = out[out["pre_pt_root_id"].isin(sheet_set)]
    driven = set(int(x) for x in out["post_pt_root_id"].unique()) & set(nod1_roots)
    return sorted(driven)


def _mcns_bodies(dn_type: str) -> list[dict]:
    """MaleCNS bodyIds + somaSide for a DN type, if the offline MCNS tables are present.

    Uses ``paper.malecns.client.bodies_of_type`` (the same access the motor map uses). Returns
    [] (no error) when MCNS is not available, so the manifest still builds FlyWire-only; the
    wing-flip (which needs MCNS) reports any gap separately.
    """
    try:
        from ..malecns import client as MC
        df = MC.bodies_of_type(dn_type)
    except Exception:
        return []
    out = []
    for _, r in df.iterrows():
        out.append({"bodyId": int(r["bodyId"]), "somaSide": r.get("somaSide"),
                    "instance": r.get("instance")})
    return out
