"""Internal-consistency audit of the paper's own arithmetic — computed from the oracle
ALONE (no connectome data), so any failure is unambiguously a paper-internal error.

This catches the kind of mistake the predecessor audit found (a table whose subtype rows
did not sum to its stated total). Each check recomputes a stated total or percentage from
the paper's own component numbers.
"""

from __future__ import annotations

from flyconn.motif import compare as K

from . import a_vch_loop as A
from . import b_inhibitor_screen as B
from . import g_output_census as G


def build_claims(_derived=None) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []

    # 1. VCH T4/T5 input %: 12,301 / 27,576 should equal the stated 44.6%.
    pct = 100.0 * A.CLAIMS["t4t5_in_syn"][0] / A.CLAIMS["vch_input_syn"][0]
    out.append(K.compare_pct(
        "Z9.vch_in_pct", "Paper-internal: 12,301/27,576 == 44.6%",
        A.CLAIMS["t4t5_in_pct_of_input"][0], pct, pp=1.0))

    # 2. Reciprocal % of inputs: 912 / 1,022 == 89%.
    rpct = 100.0 * A.CLAIMS["reciprocal_n"][0] / A.CLAIMS["t4t5_in_neurons"][0]
    out.append(K.compare_pct(
        "Z9.recip_pct_in", "Paper-internal: 912/1,022 == 89%",
        A.CLAIMS["reciprocal_pct_of_inputs"][0], rpct, pp=1.5))

    # 3. Reciprocal % of outputs: 912 / 1,476 == 62%.
    rpo = 100.0 * A.CLAIMS["reciprocal_n"][0] / A.CLAIMS["t4t5_out_neurons"][0]
    out.append(K.compare_pct(
        "Z9.recip_pct_out", "Paper-internal: 912/1,476 == 62%",
        A.CLAIMS["reciprocal_pct_of_outputs"][0], rpo, pp=1.5))

    # 4. exc/inhib mean ratio: (12,301/1,022)/(7,273/1,476) ~ 2.4.
    ratio = (A.CLAIMS["t4t5_in_syn"][0] / A.CLAIMS["t4t5_in_neurons"][0]) / \
            (A.CLAIMS["t4t5_out_syn"][0] / A.CLAIMS["t4t5_out_neurons"][0])
    out.append(K.compare_ratio(
        "Z9.exc_inhib_ratio", "Paper-internal: mean-syn ratio in [2,3] (~2.4x)",
        2.0, 3.0, ratio))

    # 5. layer-a reciprocal fraction is < total reciprocal (882 <= 912).
    out.append(K.compare_categorical(
        "Z9.layer_a_le_total", "Paper-internal: 882 layer-a reciprocal <= 912 total",
        True, A.CLAIMS["reciprocal_layer_a_n"][0] <= A.CLAIMS["reciprocal_n"][0]))

    # 6. Inhibitor screen: recip_frac consistent with reciprocal count being <= field.
    #    Each row's recip_frac should be in (0,1].
    bad = [t for t, r in B.TABLE1.items() if not (0 < r["recip_frac"] <= 1.0)]
    out.append(K.compare_categorical(
        "Z9.b_recip_frac_range", "Paper-internal: Table 1 reciprocal fractions in (0,1]",
        [], bad, refuted_note=f"out-of-range rows: {bad}"))

    # 7. Output census: Nod1 driver count (91) <= sheet size (100).
    out.append(K.compare_categorical(
        "Z9.nod1_drivers_le_100", "Paper-internal: Nod1 drivers (91) <= 100 LLPC1",
        True, G.TOP_TARGETS["Nod1"]["drivers"] <= 100))

    return out
