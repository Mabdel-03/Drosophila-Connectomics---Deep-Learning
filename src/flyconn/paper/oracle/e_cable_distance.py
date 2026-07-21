"""Family E — LLPC1 integrates lobula-plate motion with lobula form on two separate
dendritic fields, and the inhibition is spatially compartmentalized (Fig 4, Fig S4,
Table S8, main text p.5-6).

Claims:
  * Input composition of the 100 LLPC1: 29% direct T4/T5 motion vs 31% lobula form
    (Tm/TmY/Y/Tlp/Li); the rest shared inhibition.
  * The motion (T4/T5) and form (Tm/TmY) synapses occupy two spatially distinct fields,
    median 20 um apart (~twice the ~10 um field radius).
  * On the T4a, the VCH->T4a synapse sits a median 2.3 um from the T4a->LLPC1 output
    terminal, vs 51 um for the T4a's other inputs; VCH is the closer input in 99% of
    286 T4a (Wilcoxon p ~ 7e-48). [presynaptic terminal gating]
  * On the LLPC1, the inhibitory classes sit at distinct distances from the nearest
    T4a-excitation synapse: LPi15 1.3 um, VCH 2.0 um (interdigitated, dendritic), but
    PVLP011 185 um (a separate distal output compartment).

Primary distances are Euclidean (the paper states the result is "sign-identical under a
Euclidean metric"); a geodesic upgrade is optional.

Source: Fig 4 (p.5), Fig S4 (p.13), Table S8 (p.18), p.5-6.
"""

from __future__ import annotations

from flyconn.motif import compare as K

MOTION_PCT = (29.0, "Fig 4a / p.5 '29% direct T4/T5 motion'")
FORM_PCT = (31.0, "Fig 4a / p.5 '31% lobula form'")
FIELD_SEP_UM = (20.0, "Fig 4c / p.5 'two fields median 20 um apart'")

# T4a-side terminal gating (Fig S4a / Table S8).
VCH_TO_TERMINAL_UM = (2.3, "Table S8 'VCH->T4a input -> T4a->LLPC1 output terminal: 2.3 um'")
OTHER_TO_TERMINAL_UM = (51.5, "Table S8 'other T4a inputs -> same output terminal: 51.5 um'")
VCH_CLOSEST_FRAC = (99.0, "Fig S4a 'VCH is the closer input in 99% of 286 T4a'")
N_T4A_SKELETONS = (286, "Table S8 'n = 286 T4a'")

# LLPC1-side compartmentalization (Table S8).
LPI15_TO_EXC_UM = (1.3, "Table S8 'LPi15 -> nearest T4a-excitation on LLPC1: 1.3 um'")
VCH_TO_EXC_UM = (2.0, "Table S8 'VCH -> nearest T4a-excitation on LLPC1: 2.0 um'")
PVLP011_TO_EXC_UM = (184.7, "Table S8 'PVLP011 -> nearest T4a-excitation on LLPC1: 184.7 um'")
PAGE = "Fig 4 / Fig S4 / Table S8"


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    out.append(K.compare_pct(
        "E.motion_pct", "LLPC1 direct T4/T5 motion input (%)", MOTION_PCT[0],
        d["motion_pct"], pp=5.0))
    out.append(K.compare_pct(
        "E.form_pct", "LLPC1 lobula form input (%)", FORM_PCT[0], d["form_pct"], pp=5.0))

    # Two-field separation (Euclidean centroid distance, median over LLPC1).
    out.append(K.compare_count(
        "E.field_sep", "Motion vs form dendritic-field separation (um, median)",
        FIELD_SEP_UM[0], round(d["field_sep_um"], 1), rel=0.40, abs_floor=5, drift_dir="down"))

    # T4a terminal gating: VCH input is far closer to the relayed terminal than other inputs.
    out.append(K.compare_categorical(
        "E.vch_terminal_closest", "VCH->T4a input is closer to the T4a->LLPC1 terminal than other inputs",
        True, d["vch_to_terminal_um"] < d["other_to_terminal_um"]))
    out.append(K.compare_pct(
        "E.vch_closest_frac", "VCH is the closest input in ~99% of T4a (%)",
        VCH_CLOSEST_FRAC[0], d["vch_closest_frac"], pp=8.0))
    # Wilcoxon: VCH distances significantly smaller than other-input distances.
    out.append(K.ClaimResult(
        id="E.wilcoxon", description="VCH-vs-other terminal-distance Wilcoxon (VCH smaller)",
        report_value="p ~ 7e-48", computed_primary=f"p={d['wilcoxon_p']:.2e}",
        tolerance="p < 1e-10 and VCH median smaller",
        verdict=K.CONFIRMED if (d["wilcoxon_p"] < 1e-10 and d["vch_to_terminal_um"] < d["other_to_terminal_um"]) else K.REFUTED,
        numeric_outcome=K.MATCH if d["wilcoxon_p"] < 1e-10 else K.MISMATCH,
        notes=f"VCH median {d['vch_to_terminal_um']:.1f} um vs other {d['other_to_terminal_um']:.1f} um, n={d['n_t4a']}"))

    # LLPC1-side compartments: LPi15/VCH dendritic (near excitation), PVLP011 distal (far).
    out.append(K.compare_count(
        "E.lpi15_to_exc", "LPi15 -> nearest T4a-excitation on LLPC1 (um)", LPI15_TO_EXC_UM[0],
        round(d["lpi15_to_exc_um"], 1), rel=0.60, abs_floor=2, drift_dir="down"))
    out.append(K.compare_count(
        "E.vch_to_exc", "VCH -> nearest T4a-excitation on LLPC1 (um)", VCH_TO_EXC_UM[0],
        round(d["vch_to_exc_um"], 1), rel=0.60, abs_floor=2, drift_dir="down"))
    # The discriminating qualitative claim: PVLP011 is on a DISTINCT distal compartment
    # (far from excitation), unlike LPi15/VCH.
    out.append(K.compare_categorical(
        "E.pvlp011_distal", "PVLP011 sits on a distinct distal compartment (>>LPi15/VCH)",
        True, d["pvlp011_to_exc_um"] > 10 * max(d["lpi15_to_exc_um"], d["vch_to_exc_um"])))
    return out
