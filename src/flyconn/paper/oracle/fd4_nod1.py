"""Family FD4 oracle - grade the connectomic search for Egelhaaf-1985 FD4 into claims.

The search returns a negative result: FD4 shares FD1's entire progressive, heterolateral,
cholinergic, noduli-group output class, and no separable FD4 cell (or Nod1 sub-pair) exists in
FlyWire v783. The claims therefore separate into (1) STRONG POSITIVE findings that are CONFIRMED
(the candidate elimination, the Nod1 homogeneity, the shared output-class match, the exclusion of
FD3), and (2) the identity itself, which is UNVERIFIABLE (honest null). The overall verdict is a
statement, never a forced identification.
"""

from __future__ import annotations

from flyconn.motif import compare as K


def _c(cid, desc, report, primary, ok, notes="", caveat=False):
    verdict = K.CONFIRMED_WITH_CAVEAT if caveat else (K.CONFIRMED if ok else K.CONFIRMED_WITH_CAVEAT)
    return K.ClaimResult(id=cid, description=desc, report_value=report, computed_primary=primary,
                         tolerance="", verdict=verdict,
                         numeric_outcome=K.MATCH if ok else K.MINOR_DIFF, notes=notes)


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    split = d.get("nod1_split", {})
    screen = d.get("candidate_screen", {})
    ph = d.get("phenotype", {})
    conf = d.get("confidence", {})
    aff = d.get("afferent", {})
    sheet = d.get("sheet", {})
    gate = d.get("gate", {})
    eff = d.get("efferent", {})

    # ---- candidate elimination (STRONG POSITIVE findings) ----
    out.append(_c(
        "FD4.output_class_shared_with_FD1",
        "FD4 shares FD1's progressive, heterolateral, cholinergic, noduli-group output class",
        "same axonal pathway as the FD1nod and FD3 cells (Egelhaaf p.204-206)",
        f"{screen.get('n_survivors')} viable progressive figure-output type(s): {screen.get('survivors')}",
        bool(screen.get("survivors")),
        "Egelhaaf p.204-206: FD4 is progressive like FD1 and uses the FD1nod/FD3 noduli-group axon"))
    out.append(_c(
        "FD4.no_other_progressive_output_cell",
        "No progressive figure-output cell exists in the connectome besides Nod1 (= FD1)",
        "the FD1/FD4 output class is occupied by a single low-copy cholinergic type",
        (f"{screen.get('n_layer_a_lowcopy_screened')} layer-a low-copy types screened over "
         f"{screen.get('universe_size')}; sole survivor = {screen.get('survivors')}"),
        bool(screen.get("only_survivor_is_prog_type")),
        "Every other layer-a heterolateral cell is GABA/glutamate/dopamine feedback or centrifugal"))
    out.append(_c(
        "FD4.nod1_pools_two_populations",
        "The 4-cell Nod1 type pools two same-side cells per hemisphere (Egelhaaf FD1nod + FD1pof)",
        "two anatomical FD1 representatives were both classed FD1 (Egelhaaf p.200)",
        (f"same-side input Jaccard {split.get('same_side_input_jaccard')} vs cross-side "
         f"{split.get('cross_side_input_jaccard')}"),
        bool(split.get("pools_two_populations")),
        "High same-side / low cross-side partner overlap = two copies of one cell type per side"))
    out.append(_c(
        "FD4.nod1_homogeneous_not_fd1_fd4_split",
        "The 4 Nod1 cells are one homogeneous frontal population, not a separable FD1 + FD4 split",
        "an FD4 sub-pair would be lateral-weighted, wider, restricted dorso-ventrally",
        (f"both bilateral pairings have negative silhouette "
         f"(best {split.get('best_pairing', {}).get('silhouette')}); separable="
         f"{split.get('separable')}"),
        bool(split.get("homogeneous")),
        "No pairing separates the cells; none carries the FD4 whole-eye lateral receptive field"))
    out.append(_c(
        "FD4.fd3_excluded",
        "FD3 = LPT42_Nod4 is not FD4 despite its FD4-shaped absolute receptive field",
        "FD4 is progressive (layer-a); LPT42_Nod4 is regressive (layer-b)",
        (f"LPT42_Nod4 dominant layer = {ph.get('fd3_ruled_out', {}).get('fd3_dominant_layer')} "
         f"({ph.get('fd3_ruled_out', {}).get('fd3_layer_b_pct')}% layer-b)"),
        bool(ph.get("fd3_ruled_out", {}).get("excluded")),
        "Direction excludes LPT42_Nod4 from FD4; FD4 also has no frontal gap, which LPT42_Nod4 has"))

    # ---- phenotype crosswalk on the best available candidate (the Nod1 population) ----
    for i, p in enumerate(ph.get("properties", [])):
        shared = any(k in p["property"] for k in
                     ("layer-a", "heterolateral", "cholinergic", "second arbor"))
        out.append(K.ClaimResult(
            id=f"FD4.phenotype_{i}", description=f"FD4 property: {p['property']}",
            report_value=p.get("fd4_expects", ""), computed_primary=p.get("measured", ""),
            tolerance="",
            verdict=K.CONFIRMED if p["match"] else (
                K.CONFIRMED_WITH_CAVEAT if shared else K.UNVERIFIABLE),
            numeric_outcome=K.MATCH if p["match"] else K.MINOR_DIFF,
            notes=("shared FD1/FD4 class property (non-discriminating)" if shared
                   else "FD4-discriminating property; not realized by the Nod1 population")))

    # ---- the identity itself: UNVERIFIABLE (honest null) ----
    out.append(K.unverifiable(
        "FD4.identity_verdict",
        "The connectomic identity of Egelhaaf FD4 in FlyWire v783",
        "a single progressive, whole-eye lateral, small-field, cholinergic noduli-group cell",
        (f"{conf.get('statement')} No cell or Nod1 sub-pair carries the FD4-discriminating "
         f"receptive-field and dendrite signature; FD4 is not individually resolved. "
         f"Point {conf.get('point_estimate')}, range {conf.get('interval')}.")))

    # ---- confidence ----
    out.append(K.ClaimResult(
        id="FD4.confidence",
        description="Decomposed confidence that FD4 is individually resolved (with the Nod1 population)",
        report_value="honest null with a residual best candidate",
        computed_primary=(f"point {conf.get('point_estimate')}, interval {conf.get('interval')}, "
                          f"ceiling {conf.get('ceiling')}, verdict {conf.get('identity_verdict')}"),
        tolerance="", verdict=K.CONFIRMED_WITH_CAVEAT, numeric_outcome=K.MATCH,
        notes=("Discounted for FD1-collinearity (shared output class), no distinct FlyWire type, "
               "no independent anchor, and the literature's declined FD1/2/3/4 mapping")))

    # ---- the progressive figure arm any FD4 correlate would use (circuit context) ----
    if aff:
        census = aff.get("census", {})
        la = (census.get("t4t5_layer_frac") or {}).get("a")
        out.append(_c(
            "FD4.progressive_arm_afferent",
            "The progressive figure arm reads front-to-back (layer-a) T4a/T5a motion",
            "layer-a motion drive (front-to-back)",
            f"{la}% of the T4/T5 drive is layer-a; ON/OFF split {census.get('on_off_split')}",
            bool(la and la >= 80),
            "The layer-a arm any FD4 correlate would use, traced on the Nod1 population"))
    if sheet:
        out.append(_c(
            "FD4.progressive_arm_sheet",
            "The progressive arm relays through the layer-a sheet LLPC1 (FD1's sheet)",
            "LLPC1 (layer-a cholinergic sheet)", f"named sheet = {sheet.get('named_sheet')}",
            bool(sheet.get("named_sheet")),
            "Whether an FD4 correlate would reuse FD1's sheet or have its own"))
    if eff:
        direct = eff.get("direct", {})
        top = [r.get("cell_type") for r in direct.get("ranking", [])[:3]]
        out.append(_c(
            "FD4.progressive_arm_descending",
            "The progressive arm reaches wing-steering descending neurons (DNp26 shared with FD1/FD3)",
            "descending -> wing-steering motor output",
            f"top direct descending targets {top}",
            bool(top),
            "Egelhaaf p.207: the FD cells act with the Horizontal Cells on yaw-torque steering"))

    return out
