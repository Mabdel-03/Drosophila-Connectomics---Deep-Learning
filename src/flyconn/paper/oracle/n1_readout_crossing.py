"""Stage-9 Channel N1 oracle: the readout-crossing / command-convergence link.

Verifies that each side's Nod1 crosses to drive the opposite hemisphere's DNp26, reproduces the
per-direction synapse counts as a positive control, and records the asymmetry and convergence.
"""

from __future__ import annotations

from flyconn.motif import compare as K

# Positive-control anchors (live v783, measured this session).
EXPECT_LEFT_NOD1_TO_RIGHT_DNP26 = 159
EXPECT_RIGHT_NOD1_TO_LEFT_DNP26 = 289


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []

    # Each side's Nod1 reaches the CONTRALATERAL DNp26 (the link exists, both directions).
    l2r = d.get("left_nod1_to_right_dnp26", 0)
    r2l = d.get("right_nod1_to_left_dnp26", 0)
    out.append(K.compare_categorical(
        "N1.left_nod1_crosses", "Left-soma Nod1 reaches the RIGHT DNp26 (readout crosses midline)",
        True, bool(l2r > 0),
        refuted_note=f"left Nod1 -> right DNp26 = {l2r} synapses"))
    out.append(K.compare_categorical(
        "N1.right_nod1_crosses", "Right-soma Nod1 reaches the LEFT DNp26 (readout crosses midline)",
        True, bool(r2l > 0),
        refuted_note=f"right Nod1 -> left DNp26 = {r2l} synapses"))

    # Positive control: reproduce the measured per-direction counts.
    out.append(K.compare_count(
        "N1.left_nod1_to_right_dnp26_count", "Left Nod1 -> right DNp26 synapse count",
        EXPECT_LEFT_NOD1_TO_RIGHT_DNP26, l2r, rel=0.10, abs_floor=20, drift_dir="down"))
    out.append(K.compare_count(
        "N1.right_nod1_to_left_dnp26_count", "Right Nod1 -> left DNp26 synapse count",
        EXPECT_RIGHT_NOD1_TO_LEFT_DNP26, r2l, rel=0.10, abs_floor=20, drift_dir="down"))

    # The mechanism: Nod1 output is majority-crossing (position-based).
    cross = d.get("nod1_output_crossing_pct")
    out.append(K.compare_categorical(
        "N1.nod1_output_crosses", f"Nod1 output is majority cross-midline (position-based; {cross}%)",
        True, bool(cross is not None and cross >= 70.0),
        refuted_note=f"Nod1 position-based output crossing = {cross}%"))

    # Each DNp26's Nod1 input is dominated by the CONTRALATERAL Nod1 (the steering command is
    # driven by the opposite hemisphere's figure readout).
    comp = d.get("dnp26_input_composition", {})
    for dn_side, c in comp.items():
        frac = c.get("contra_frac_of_nod1_input")
        out.append(K.compare_categorical(
            "N1.dnp26_%s_contra_driven" % dn_side,
            f"DNp26_{dn_side} Nod1 input is dominantly contralateral ({frac}%)",
            True, bool(frac is not None and frac >= 80.0),
            refuted_note=f"DNp26_{dn_side} contralateral-Nod1 fraction = {frac}%"))

    # The two circuits converge on shared steering targets (a bilateral integration locus exists).
    shared = d.get("shared_steering_targets", {})
    out.append(K.compare_categorical(
        "N1.shared_convergence",
        f"Left and right Nod1 converge on shared downstream cells ({shared.get('n_shared')} cells)",
        True, bool((shared.get("n_shared") or 0) > 0),
        refuted_note=f"shared targets = {shared.get('n_shared')}"))

    # The link is ASYMMETRIC (reported, not a pass/fail): record as an unverifiable-style note.
    out.append(K.unverifiable(
        "N1.asymmetry",
        f"Readout-crossing asymmetry left vs right (ratio {d.get('asymmetry_ratio')})",
        f"L->R {r2l} vs R->L {l2r}",
        f"the two crossing directions differ in strength (ratio {d.get('asymmetry_ratio')}); "
        f"reported as a structural asymmetry of the inter-hemispheric readout link, consistent "
        f"with the per-hemisphere proofreading-completeness differences seen in Stage 8."))

    return out
