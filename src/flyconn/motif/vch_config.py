"""The verification *oracle*: every quantitative claim in vch_t4t5_report.pdf,
encoded as data, plus a few shared constants and the scratch output dir helper.

Keeping the report's numbers in one place (rather than scattered through the
verification code) means the comparison engine reads the claim and the computed
value side by side, and a future re-run against a new report only edits this file.

Source: vch_t4t5_report.pdf, "Connectomic Analysis of VCH-T4/T5 Circuitry",
Left VCH (720575940627706398) in the FlyWire FAFB connectome, materialization v783.
"""

from __future__ import annotations

from pathlib import Path

from ..paths import data_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VCH_ROOT = 720_575_940_627_706_398  # Left VCH neuron the report studies
VERSION = "783"

# The 8 defensible T4/T5 subtype labels (exact cell_type strings in neurons.parquet).
CANONICAL_T4T5 = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")

# Per-synapse neurotransmitter probability columns in flywire_synapses_783.feather,
# in a fixed order so argmax -> NT name is reproducible.
SYN_NT_COLS = ("gaba", "ach", "glut", "oct", "ser", "da")
# Map the raw-table NT column name -> canonical vocabulary (schemas.NT_CANONICAL keys).
SYN_NT_TO_CANONICAL = {
    "gaba": "gaba",
    "ach": "acetylcholine",
    "glut": "glutamate",
    "oct": "octopamine",
    "ser": "serotonin",
    "da": "dopamine",
}


def motif_dir(version: str = VERSION) -> Path:
    """Scratch dir for large/regenerable motif intermediates; created on demand."""
    d = data_root() / f"v{version}" / "motif"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# The report's claims (the oracle)
# ---------------------------------------------------------------------------

# Table 1 — Left VCH overview.
OVERVIEW = {
    "nt": "gaba",
    "total_input_syn": 27_576,
    "total_output_syn": 32_363,
    "upstream_partners": 3_619,
    "downstream_partners": 10_948,
}

# Table 2 — T4/T5 INPUTS to Left VCH. (neurons, synapses) per subtype.
T4T5_INPUTS = {
    "total_neurons": 1_030,
    "total_syn": 13_018,
    "pct_of_input": 47.2,          # % of VCH's total input synapses
    "mean_syn_per_neuron": 12.6,
    "subtypes": {                  # subtype -> (neurons, synapses)
        "T4a": (471, 5_942),
        "T5a": (470, 6_225),
        "T4b": (28, 99),
        "T5b": (23, 97),
        "T4c": (13, 20),
        "T5c": (12, 21),
        "T4d": (12, 15),
        "T5d": (6, 7),
    },
}

# Table 3 — T4/T5 OUTPUTS from Left VCH. Synapse counts in the report are "~" approx.
T4T5_OUTPUTS = {
    "total_neurons": 1_487,
    "total_syn": 7_328,
    "pct_of_output": 22.6,
    "mean_syn_per_neuron": 4.9,
    "subtypes": {                  # subtype -> (neurons, approx synapses)
        "T4a": (435, 2_500),
        "T5a": (434, 2_550),
        "T4c": (161, 500),
        "T4d": (152, 450),
        "T4b": (110, 350),
        "T5d": (59, 200),
        "T5c": (48, 160),
        "T5b": (47, 200),
        "Other/unclear": (41, 400),
    },
}

# Section 6 — reciprocal set (both input to and output from VCH).
RECIPROCAL = {
    "n": 918,
    "pct_of_outputs": 61.7,   # 918 / 1487
    "pct_of_inputs": 89.1,    # 918 / 1030
}

# Section 5/6 — excitation vs inhibition (mean syn/neuron, in vs out).
GAIN = {
    "input_mean": 12.6,
    "output_mean": 4.9,
    "ratio_min": 2.0,         # report: "2-3x stronger"
    "ratio_max": 3.0,
}

# Section 5 — laterality claim.
HEMISPHERE = {
    "input": "ipsilateral",     # report: right-hem T4/T5 -> left VCH ("ipsilateral")
    "output": "contralateral",  # report: VCH -> right-hem targets
    # The falsifiable consequence: input and output synapses lie in DIFFERENT hemispheres.
    "input_output_differ": True,
}

# Section 7.1 — downstream of the 918 reciprocal T4/T5.
DOWNSTREAM_GLOBAL = {
    "total_syn": 578_238,
    "unique_targets": 144_796,
}

# Table 4 — top-20 individual downstream targets, by synapse count.
# (label, synapses, n_t4t5_drivers, nt, class). "copy N" labels are cosmetic: each row
# is a distinct neuron (root_id) of the given cell_type. We match on (cell_type, syn, drivers).
TOP20_TARGETS = [
    ("LPi14", 19_521, 802, "GABA", "optic/LOP"),
    ("LPi14", 17_685, 850, "GABA", "optic/LOP"),
    ("VCH", 12_579, 918, "GABA", "visual_centrifugal"),
    ("CT1", 7_803, 906, "GABA", "optic/ME>LO"),
    ("HSE", 5_564, 532, "ACh", "visual_projection"),
    ("DCH", 4_523, 367, "GABA", "visual_centrifugal"),
    ("HSN", 4_387, 354, "ACh", "visual_projection"),
    ("LPi15", 2_637, 685, "GABA", "optic/LOP"),
    ("HSS", 1_817, 501, "ACh", "visual_projection"),
    ("LT33", 1_481, 388, "GABA", "optic/LO"),
    ("LPT26", 1_177, 225, "ACh", "visual_projection"),
    ("LPT04_HST", 869, 317, "ACh", "visual_projection"),
    ("Nod2", 792, 276, "GABA", "visual_projection"),
    ("Li14", 723, 111, "GABA", "optic/LO"),
    ("Nod1", 712, 232, "ACh", "visual_projection"),
    ("Li14", 712, 119, "GABA", "optic/LO"),
    ("SAD043", 698, 3, "GABA", "central"),
    ("Li14", 643, 105, "GABA", "optic/LO"),
    ("Li14", 628, 92, "GABA", "optic/LO"),
    ("Nod1", 615, 224, "ACh", "visual_projection"),
]

# Section 7.4 — major cell-type populations among downstream targets.
# cell_type -> (copies = distinct neurons, total synapses).
DOWNSTREAM_POPULATIONS = {
    "TmY20": (141, 19_714),
    "LLPC1": (110, 17_705),
    "Y1": (79, 16_649),
    "TmY16": (76, 13_305),
    "TmY14": (177, 8_316),
    "Tlp5": (27, 6_774),
    "C3": (495, 5_691),
    "LPLC2": (97, 5_320),
    # "T4a/T5a (529/534 copies, ~20800 syn)" — lateral excitation; checked separately.
}

# Table 6 — Nodulus (Nod) neurons among downstream targets.
# cell_type -> (copies, total_syn, n_drivers, nt).
NODULUS = {
    "Nod2": (1, 792, 276, "GABA"),
    "Nod1": (3, 1_330, 457, "ACh"),
    "Nod3": (1, 49, 15, "ACh"),
    "Nod4": (2, 21, 10, "ACh"),
    "Nod5": (1, 29, 14, "ACh"),
}

# Section 9 — annotation-completeness asymmetry (qualitative / hard to verify).
ANNOTATION = {
    "left_hem_tagged_pct": 72,
    "right_hem_tagged_pct": 97,
}
