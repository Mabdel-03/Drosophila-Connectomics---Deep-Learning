"""Family A — the VCH-T4/T5 reciprocal loop is the circuit's entry point.

Sources: main text "The VCH-T4/T5 reciprocal loop..." (p.2), Fig 2 (p.4),
S5/S7 "Directional opponency..." (p.16), S9 Table S9 (p.19).
"""

from __future__ import annotations

from flyconn.motif import compare as K

# --- the claims (value, page) ---
CLAIMS = {
    # Table S9 overview.
    "vch_nt": ("gaba", "S9"),
    "vch_input_syn": (27_576, "S9 / p.2"),
    "vch_output_syn": (32_363, "S9"),
    "vch_upstream_partners": (3_619, "derived from S9 / p.2 '1,022 ... 12,301'"),

    # T4/T5 -> VCH input (p.2 'of its 27,576 input synapses, 12,301 (44.6%) arrive from
    # 1,022 T4/T5 neurons'; mean 12.0 syn/partner).
    "t4t5_in_neurons": (1_022, "p.2 / Fig2a"),
    "t4t5_in_syn": (12_301, "p.2 / Fig2a"),
    "t4t5_in_pct_of_input": (44.6, "p.2"),
    "t4t5_in_mean_syn": (12.0, "p.2 'mean 12.0 syn per partner'"),

    # VCH -> T4/T5 output (p.2 'VCH ... synapses back onto 1,476 T4/T5 neurons (7,273
    # synapses)'; mean 4.9).
    "t4t5_out_neurons": (1_476, "p.2 / Fig2a"),
    "t4t5_out_syn": (7_273, "p.2 / Fig2a"),
    "t4t5_out_mean_syn": (4.9, "p.2 'mean 4.9'"),

    # Reciprocal (p.2 '912 are reciprocal ... 89% of VCH's T4/T5 inputs; 62% of its
    # T4/T5 outputs').
    "reciprocal_n": (912, "p.2 / Fig2a"),
    "reciprocal_pct_of_inputs": (89.0, "p.2 '89% of VCH's T4/T5 inputs'"),
    "reciprocal_pct_of_outputs": (62.0, "p.2 '62% of its T4/T5 outputs'"),

    # Excitation vs inhibition (p.2 'about 2.4 times the inhibitory feedback'; means 12.0/4.9).
    "exc_inhib_ratio": ((2.0, 3.0), "p.2 '~2.4x' (band 2-3x)"),

    # Same-direction opponency (S5 p.16): 99.1% of VCH's T4/T5 input syn are layer-a;
    # 882 of 912 reciprocal partners are layer-a.
    "vch_layer_a_pct": (99.1, "S5 p.16 '99.1% come from layer-a'"),
    "reciprocal_layer_a_n": (882, "p.5 / S5 '882 of the 912 reciprocal partners are layer-a'"),
}


def build_claims(d: dict) -> list[K.ClaimResult]:
    """d = derived dict from derive.a_vch_loop.run(). Live track is primary."""
    out: list[K.ClaimResult] = []
    sec = d.get("secondary", {})

    out.append(K.compare_categorical(
        "A.vch_nt", "VCH neurotransmitter is GABA", CLAIMS["vch_nt"][0], d["vch_nt"]))

    for cid, key, dkey, desc in [
        ("A.vch_in_syn", "vch_input_syn", "vch_input_syn", "VCH total input synapses"),
        ("A.vch_out_syn", "vch_output_syn", "vch_output_syn", "VCH total output synapses"),
        ("A.vch_up_partners", "vch_upstream_partners", "vch_upstream_partners", "VCH upstream partners"),
        ("A.t4t5_in_neurons", "t4t5_in_neurons", "t4t5_in_neurons", "T4/T5 input neurons to VCH"),
        ("A.t4t5_in_syn", "t4t5_in_syn", "t4t5_in_syn", "T4/T5 input synapses to VCH"),
        ("A.t4t5_out_neurons", "t4t5_out_neurons", "t4t5_out_neurons", "T4/T5 output neurons from VCH"),
        ("A.t4t5_out_syn", "t4t5_out_syn", "t4t5_out_syn", "T4/T5 output synapses from VCH"),
        ("A.reciprocal_n", "reciprocal_n", "reciprocal_n", "Reciprocal T4/T5 partners"),
        ("A.reciprocal_layer_a_n", "reciprocal_layer_a_n", "reciprocal_layer_a_n",
         "Reciprocal partners that are layer-a (same direction)"),
    ]:
        out.append(K.compare_count(
            cid, desc, CLAIMS[key][0], d[dkey], sec.get(dkey), rel=0.02, drift_dir="down"))

    for cid, key, dkey, desc in [
        ("A.t4t5_in_pct", "t4t5_in_pct_of_input", "t4t5_in_pct_of_input", "T4/T5 = % of VCH input syn"),
        ("A.reciprocal_pct_in", "reciprocal_pct_of_inputs", "reciprocal_pct_of_inputs",
         "Reciprocal = % of T4/T5 inputs"),
        ("A.reciprocal_pct_out", "reciprocal_pct_of_outputs", "reciprocal_pct_of_outputs",
         "Reciprocal = % of T4/T5 outputs"),
        ("A.vch_layer_a_pct", "vch_layer_a_pct", "vch_layer_a_pct",
         "VCH T4/T5 input syn that are layer-a (front-to-back)"),
    ]:
        out.append(K.compare_pct(cid, desc, CLAIMS[key][0], d[dkey], pp=3.0))

    lo, hi = CLAIMS["exc_inhib_ratio"][0]
    out.append(K.compare_ratio(
        "A.exc_inhib_ratio", "Excitatory drive ~2.4x inhibitory feedback (mean syn/partner)",
        lo, hi, d["exc_inhib_ratio"]))

    # Internal consistency: the paper's own mean-syn-per-partner figures.
    out.append(K.compare_count(
        "A.mean_in_consistency", "VCH T4/T5 input mean syn/partner (12,301/1,022 ~= 12.0)",
        CLAIMS["t4t5_in_mean_syn"][0],
        round(CLAIMS["t4t5_in_syn"][0] / CLAIMS["t4t5_in_neurons"][0], 1),
        rel=0.05, abs_floor=1, drift_dir="down"))
    return out
