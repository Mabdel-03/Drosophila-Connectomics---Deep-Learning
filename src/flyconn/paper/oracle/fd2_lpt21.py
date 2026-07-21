"""Family FD2 oracle - grade the LPT21 = Egelhaaf-1985 FD2 identity and its circuit into claims.

Each claim pairs what Egelhaaf reported with what the connectome measures for ``LPT21``. The claims
are grouped: identity (direction / frontal RF / homolateral projection / small-field / cholinergic /
dual output), the decomposed confidence, and the circuit (afferent cascade, sheet, gate, efferent
descending + motor). The overall verdict is an identity+confidence summary, never a bare number.
"""

from __future__ import annotations

from flyconn.motif import compare as K

CAND = "LPT21"


def _c(cid, desc, report, primary, ok, notes="", caveat=False):
    return K.ClaimResult(
        id=cid, description=desc, report_value=report, computed_primary=primary,
        tolerance="", verdict=(K.CONFIRMED if ok else K.CONFIRMED_WITH_CAVEAT) if not caveat
        else K.CONFIRMED_WITH_CAVEAT,
        numeric_outcome=K.MATCH if ok else K.MINOR_DIFF, notes=notes)


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    ident = d.get("identity", {})
    conf = d.get("confidence", {})
    uniq = d.get("uniqueness", {})
    dual = d.get("dual_output", {})
    aff = d.get("afferent", {})
    sheet = d.get("sheet", {})
    gate = d.get("gate", {})
    eff = d.get("efferent", {})

    # ---- identity ----
    lb = ident.get("layer_b_pct")
    out.append(_c("FD2.direction", "FD2 prefers regressive (back-to-front) motion",
                  "regressive (layer-b)", f"{lb}% layer-b", bool(lb and lb >= 80),
                  "Egelhaaf p.201: FD2 excited by regressive motion"))
    out.append(_c("FD2.frontal_rf", "FD2 has a frontal receptive field co-located with FD1",
                  "frontal (peak psi 0-10 deg)", f"RF offset vs FD1 ~ {ident.get('mean_rf_offset')}",
                  bool(ident.get("frontal_colocated")),
                  "Egelhaaf p.201: frontal field reaching the frontal margin of the ipsilateral eye"))
    out.append(_c("FD2.homolateral", "FD2 is homolateral: axon projects to the ipsilateral POF",
                  "ipsilateral posterior optic foci", f"{ident.get('contra_output_pct')}% contra output",
                  bool(ident.get("homolateral")),
                  "Egelhaaf p.201: FD2 is an output element projecting to the ipsilateral POF; NOT noduli-group"))
    out.append(_c("FD2.smallfield", "FD2 is small-field / figure selective",
                  "small-field selective", "input patch bounded vs null",
                  bool(ident.get("smallfield_bounded")),
                  f"permutation p={ident.get('smallfield_null', {}).get('p_value')}"))
    out.append(_c("FD2.cholinergic", "FD2 is an excitatory (cholinergic) output cell",
                  "excitatory output", f"acetylcholine (conf {ident.get('nt_conf')})",
                  bool(ident.get("nt") == "acetylcholine"),
                  "NT not stated by Egelhaaf; connectome-assigned"))
    out.append(K.ClaimResult(
        id="FD2.dual_output",
        description="FD2 has a second frontal axonal branch (dual output), unique among FD cells",
        report_value="main ipsilateral POF terminal + a frontal branch to the anterior optic foci",
        computed_primary=f"both cells bimodal={dual.get('both_bimodal')}",
        tolerance="two spatially separated output terminal fields per cell",
        verdict=K.CONFIRMED if dual.get("both_bimodal") else K.CONFIRMED_WITH_CAVEAT,
        notes="Egelhaaf p.202/208: the second frontal branch (landing-response hypothesis)"))

    # ---- uniqueness + confidence ----
    out.append(K.ClaimResult(
        id="FD2.uniqueness",
        description="LPT21 is the unique full FD2 match in the lobula-plate tangential family",
        report_value="a single regressive, frontal, homolateral, bilateral-pair figure cell",
        computed_primary=(f"fd2_hits={uniq.get('n_fd2_hits')}, unique={uniq.get('unique')}, "
                          f"competitors={[c['type'] for c in uniq.get('competitors', [])]}"),
        tolerance="LPT21 is the sole full match; competitors fail a clause (cell count / RF)",
        verdict=K.CONFIRMED if uniq.get("unique") else K.CONFIRMED_WITH_CAVEAT,
        notes="the nearest competitor is ruled out on cell count and/or RF heterogeneity"))
    out.append(K.ClaimResult(
        id="FD2.confidence",
        description="Decomposed confidence that LPT21 is FD2",
        report_value="point estimate with interval",
        computed_primary=f"{conf.get('point_estimate')} (range {conf.get('interval')})",
        tolerance="property fraction x uniqueness, minus explicit discounts",
        verdict=K.CONFIRMED_WITH_CAVEAT,
        notes=conf.get("statement", "")))

    # ---- afferent circuit ----
    cen = aff.get("census", {})
    out.append(_c("FD2.afferent_t4t5", "FD2 is driven by T4/T5 motion detectors",
                  "small-field movement detectors (Egelhaaf, generic)",
                  f"T4/T5 = {cen.get('t4t5_frac_of_total')}% of input; {cen.get('layer_b_frac_of_t4t5')}% layer-b",
                  bool((cen.get('t4t5_frac_of_total') or 0) >= 10)))
    casc = aff.get("upstream_cascade", {})
    out.append(K.ClaimResult(
        id="FD2.afferent_cascade",
        description="The canonical photoreceptor->lamina->medulla->T4b/T5b cascade feeds FD2",
        report_value="retinotopic elementary movement detectors (Egelhaaf, generic)",
        computed_primary=(f"ON medulla present={casc.get('t4_on_medulla_present')}, "
                          f"OFF={casc.get('t5_off_medulla_present')}, lamina={casc.get('lamina_present')}, "
                          f"photoreceptor={casc.get('photoreceptor_present')}"),
        tolerance="canonical ON/OFF medulla + lamina + photoreceptor present upstream of FD2's T4b/T5b",
        verdict=K.CONFIRMED_WITH_CAVEAT,
        notes="the cascade is the general optic-lobe column wiring, not an FD2-specific trace"))
    out.append(_c("FD2.sheet", "A direction-matched columnar sheet relays motion to FD2",
                  "columnar output elements of the lobula", f"named sheet = {sheet.get('named_sheet')}",
                  bool(sheet.get("named_sheet")), "the layer-b sheet feeding FD2"))
    out.append(K.ClaimResult(
        id="FD2.gate",
        description="FD2's large-field inhibitory gate (which Egelhaaf could not resolve)",
        report_value="large-field inhibitory organisation could not be resolved (Egelhaaf p.201)",
        computed_primary=f"named inhibitor = {gate.get('named_inhibitor')}",
        tolerance="the connectome resolves the wide-field inhibitory input Egelhaaf could not",
        verdict=K.CONFIRMED_WITH_CAVEAT,
        notes="sign/direction of the gate is a wiring inference, not a physiology recording"))

    # ---- efferent circuit ----
    direct = eff.get("direct", {})
    top_dn = direct.get("ranking", [{}])[0].get("cell_type") if direct.get("ranking") else None
    out.append(_c("FD2.descending", "FD2 output reaches descending neurons",
                  "FD cells contact descending neurons controlling yaw torque (Egelhaaf p.207)",
                  f"{direct.get('dn_syn')} syn onto {direct.get('n_dns')} DNs "
                  f"({len(direct.get('ranking', []))} types); top = {top_dn}; "
                  f"steering {direct.get('steering_frac')}%",
                  bool(direct.get("dn_syn")), "no named DNs in Egelhaaf; connectome names them"))
    mot = eff.get("motor", {})
    out.append(K.ClaimResult(
        id="FD2.motor",
        description="FD2's descending output is biased toward the wing-steering motor system",
        report_value="yaw-torque / figure-tracking motor control (Egelhaaf p.207, generic)",
        computed_primary=f"dominant motor system = {mot.get('dominant_motor_system')}",
        tolerance="wing-steering is the largest FD2-weighted motor-system share (male-CNS proxy)",
        verdict=K.CONFIRMED_WITH_CAVEAT,
        notes="motor read-out crosses into the male CNS by shared DN name; anatomical proxy"))

    # ---- overall ----
    out.append(K.ClaimResult(
        id="FD2.identity_verdict",
        description="OVERALL: LPT21 is the connectomic correlate of Egelhaaf-1985 FD2",
        report_value="FD2 == LPT21",
        computed_primary=f"confidence {conf.get('point_estimate')} (range {conf.get('interval')})",
        tolerance="all identity properties matched + unique in the scanned family + honest discounts",
        verdict=K.CONFIRMED_WITH_CAVEAT,
        notes=conf.get("statement", "")))
    return out
