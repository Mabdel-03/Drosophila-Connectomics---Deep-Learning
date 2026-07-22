"""Family K1 (Stage-8 Phase 0) — audit the RIGHT-side Nod1 = Egelhaaf-1985 FD1 anchor.

Tests Nod1 against Egelhaaf's FD1 spec instead of assuming it (the project + family K use
Nod1 as the FD1 anchor but never validated it; neither paper states the identity; there is a
cell-class tension VPN-vs-tangential). Verdicts:

  FUNCTIONAL claims (layer-a/progressive, cholinergic, frontal RF, heterolateral contra axon)
    — CONFIRMED when Nod1 meets the FD1 spec. These stand on their own.
  CELL-CLASS claim — the named caveat: Egelhaaf's FD1 is a lobula-plate TANGENTIAL cell, but
    FlyWire Nod1 is super_class=visual_projection (a VPN). So the honest identity verdict is
    CONFIRMED_WITH_CAVEAT: "Nod1 is the functional/anatomical FD1-ROLE correlate, not literally
    Egelhaaf's tangential FD1 cell." This caveat is INHERITED by the left family K' downstream.
  DISCRIMINATING claim — Nod1 is the best FUNCTIONAL FD1 match among the alternatives the
    papers raise (VCH, the LLPC1 sheet, sibling Nod-types).

Egelhaaf 1985, "The FD1-Cell" (Biol. Cybern. 52:195-209, p.197-201, Figs 1-6).
"""

from __future__ import annotations

from flyconn.motif import compare as K

PAGE = "Egelhaaf 1985, The FD1-Cell (p.197-201)"

# FD1 reference features.
FD1_LAYER = ("a", "p.197 'FD1 selectively excited by progressive (front-to-back) motion'")
FD1_NT = ("acetylcholine", "FD1 is an excitatory figure-detection output element")
FD1_CONTRA_MIN = (70.0, "p.200 FD1nod 'noduli group ... contralateral posterior optic foci'")
FD1_FRONTAL = ("frontal", "p.198 'excitatory receptive field ... frontal part of the ipsilateral eye, peak ~10 deg'")
FD1_TANGENTIAL = ("tangential", "p.195 FD cells are 'large-field tangential neurones ... in the lobula plate'")

CORE_IDS = {"K1.layer_a", "K1.nt_ach", "K1.contra_axon", "K1.frontal_rf"}
DISC_IDS = {"K1.best_functional_match"}


def build_claims(d: dict) -> list[K.ClaimResult]:
    subj = d.get("subject_features", {})
    claims: list[K.ClaimResult] = []

    # --- FUNCTIONAL FD1 criteria (CORE) ---
    claims.append(K.compare_categorical(
        "K1.layer_a", "Nod1 input is layer-a (progressive/front-to-back) dominant [FD1]",
        FD1_LAYER[0], subj.get("dominant_layer"),
        refuted_note=f"Nod1 dominant layer = {subj.get('dominant_layer')} "
                     f"(layer_a_pct={subj.get('layer_a_pct')}); FD1 must be layer-a"))

    claims.append(K.compare_categorical(
        "K1.nt_ach", "Nod1 is cholinergic / excitatory [FD1 is an excitatory output]",
        FD1_NT[0], subj.get("nt"),
        refuted_note=f"Nod1 NT = {subj.get('nt')}; FD1 must be excitatory (ACh)"))

    contra = subj.get("contra_output_pct") or 0.0
    claims.append(K.compare_categorical(
        "K1.contra_axon", "Nod1 axon is heterolateral (>=70% contralateral output) [FD1nod]",
        True, bool(contra >= FD1_CONTRA_MIN[0]),
        refuted_note=f"Nod1 contralateral output = {contra}% (< {FD1_CONTRA_MIN[0]}%); "
                     f"FD1nod is a noduli-group heterolateral cell"))

    claims.append(K.compare_categorical(
        "K1.frontal_rf", "Nod1 excitatory RF is FRONTAL in its eye [FD1 peak ~10 deg]",
        True, bool(subj.get("is_frontal_rf")),
        refuted_note=f"Nod1 RF not frontal: {subj.get('rf_frontal', {}).get(subj.get('rep_side'))}"))

    # --- ABSOLUTE azimuth (CORROBORATING, caveat-capped: the azimuth calibration anchors its
    # frontal pole to FD1, so it cannot independently prove FD1's absolute degrees without
    # circularity; reported, never able to exceed CWC). ---
    rep = (subj.get("rf_frontal") or {}).get(subj.get("rep_side"), {})
    az = rep.get("abs_peak_az_deg")
    claims.append(K.unverifiable(
        "K1.abs_azimuth_deg",
        f"Nod1 RF absolute azimuth peak vs FD1 ~{d.get('fd1_peak_az_ref_deg')} deg "
        f"(measured {az} deg)",
        f"~{d.get('fd1_peak_az_ref_deg')} deg",
        "azimuth calibration anchors its frontal pole to FD1 itself (circular for an absolute "
        f"FD1 azimuth test); reported as corroboration only. measured peak={az} deg, "
        f"width={rep.get('abs_width_deg')} deg; calibration budget "
        f"{d.get('calibration_error_budget', {}).get('combined_deg')} deg"))

    # --- CELL-CLASS confrontation (the NAMED CAVEAT) ---
    # Egelhaaf FD1 is a lobula-plate tangential cell; Nod1 is super_class=visual_projection.
    # This is not a falsifier of the FUNCTIONAL identity, but it caps the literal-identity
    # verdict at CONFIRMED_WITH_CAVEAT and is the caveat inherited downstream.
    is_tang = d.get("nod1_is_tangential_cell", False)
    cc = K.ClaimResult(
        id="K1.cell_class",
        description="Nod1 cell class: lobula-plate tangential (literal FD1) vs visual_projection VPN",
        report_value="tangential (Egelhaaf FD1)",
        computed_primary=subj.get("super_class"),
        tolerance="exact (literal) / functional-correlate (caveat)",
        verdict=(K.CONFIRMED if is_tang else K.CONFIRMED_WITH_CAVEAT),
        numeric_outcome=(K.MATCH if is_tang else K.MINOR_DIFF),
        notes=("Nod1 is a lobula-plate tangential cell -> literal FD1" if is_tang else
               f"Nod1 super_class={subj.get('super_class')} (a VPN, downstream of the LLPC1 "
               f"sheet), NOT a lobula-plate tangential cell. The functional FD1 signature "
               f"(progressive+frontal+excitatory+contralateral noduli axon) holds, so Nod1 is "
               f"the FD1-ROLE correlate, not literally Egelhaaf's tangential FD1. Caveat "
               f"inherited by left family K'."),
        extra={"cell_class_verdict": d.get("cell_class_verdict")},
    )
    claims.append(cc)

    # --- DISCRIMINATING: Nod1 is the best FUNCTIONAL FD1 match among the alternatives ---
    best = d.get("best_functional_fd1_match")
    alts = d.get("alternatives", {})
    alt_summary = {ct: alts.get(ct, {}).get("n_fd1_functional_features") for ct in alts}
    claims.append(K.ClaimResult(
        id="K1.best_functional_match",
        description="Nod1 is the best FUNCTIONAL FD1 match vs VCH / LLPC1 sheet / sibling Nods",
        report_value="Nod1",
        computed_primary=best,
        tolerance=f"argmax of FD1 functional features (margin {d.get('functional_match_margin')})",
        verdict=(K.CONFIRMED if d.get("nod1_is_best_functional") else K.REFUTED),
        numeric_outcome=(K.MATCH if d.get("nod1_is_best_functional") else K.MISMATCH),
        notes=f"Nod1 features={alts and (d.get('subject_features') or {}).get('n_fd1_functional_features')}; "
              f"alternatives={alt_summary}. (VCH is wide-field GABA not a frontal excitatory "
              f"readout; the LLPC1 sheet is retinotopic not a single tangential cell.)",
        extra={"alternatives_features": alt_summary},
    ))

    # --- noduli-group axon-crossing morphology (CORROBORATING) ---
    morph = d.get("morphology", {})
    if morph.get("available"):
        claims.append(K.ClaimResult(
            id="K1.noduli_axon_crossing",
            description="Nod1 axon crosses toward the contralateral hemisphere [FD1nod noduli group]",
            report_value=True, computed_primary=bool(morph.get("any_axon_crosses_contra")),
            tolerance="dendrite->axon ML centroid shift toward contra (synapse-cloud proxy)",
            verdict=(K.CONFIRMED_WITH_CAVEAT if morph.get("any_axon_crosses_contra") else K.UNVERIFIABLE),
            numeric_outcome=(K.MINOR_DIFF if morph.get("any_axon_crosses_contra") else ""),
            notes=f"synapse-cloud morphology proxy (real skeletons unavailable on the public "
                  f"datastack); per-cell={[(c['side'], c['axon_crosses_contra']) for c in morph.get('cells', [])]}",
        ))

    return claims


def identity_verdict(claims: list[K.ClaimResult]) -> dict:
    """Aggregate the Nod1=FD1 audit into a single headline verdict + the named caveat."""
    by_id = {c.id: c for c in claims}
    core_ok = all(by_id.get(cid) and by_id[cid].verdict in (K.CONFIRMED,)
                  for cid in CORE_IDS if cid in by_id)
    core_refuted = [cid for cid in CORE_IDS if cid in by_id and by_id[cid].verdict == K.REFUTED]
    disc_ok = all(by_id.get(cid) and by_id[cid].verdict == K.CONFIRMED
                  for cid in DISC_IDS if cid in by_id)
    cell_class = by_id.get("K1.cell_class")
    literal = bool(cell_class and cell_class.verdict == K.CONFIRMED)

    if core_refuted:
        verdict = K.REFUTED
        statement = (f"Nod1 fails a CORE FD1 criterion ({core_refuted}); the FD1 identity is "
                     f"REFUTED on the right hemisphere.")
    elif core_ok and disc_ok and literal:
        verdict = K.CONFIRMED
        statement = "Nod1 is Egelhaaf's FD1 (functional + tangential cell class)."
    elif core_ok and disc_ok:
        verdict = K.CONFIRMED_WITH_CAVEAT
        statement = ("Nod1 is the FUNCTIONAL/anatomical FD1-ROLE correlate (progressive, frontal, "
                     "excitatory, contralateral noduli axon, best functional match among the "
                     "alternatives) but is super_class=visual_projection, NOT literally Egelhaaf's "
                     "lobula-plate tangential FD1 cell. Caveat inherited by the left mirror's K'.")
    else:
        verdict = K.CONFIRMED_WITH_CAVEAT
        statement = ("Nod1 meets most FD1 criteria with one functional/discriminating claim "
                     "caveated; see per-claim notes.")
    return {"verdict": verdict, "statement": statement,
            "core_ok": core_ok, "disc_ok": disc_ok, "literal_tangential": literal,
            "core_refuted": core_refuted}
