"""The muscular-projection ORACLE: the paper's motor-side tables encoded as data.

Mirrors ``motif.vch_config`` but for Stage 5 (the bilateral muscular completion). Keeping
the paper's numbers in one place lets the verifier read claim and computed value side by
side, and lets the agent self-check its DN->MN->muscle trace against the published
exemplars BEFORE extending the map to both wings.

Source: Figure_Ground_Circuit.pdf, Supplementary Tables S11-S19. The motor read-out is the
male whole-CNS connectome (male-cns:v1.0); each motor neuron is annotated with the body
side it innervates (``somaSide``). Wing laterality is defined relative to the DN's own
side: ipsi if the MN's somaSide == the DN's somaSide, else contra (S10.2).
"""

from __future__ import annotations

from pathlib import Path

from ..paths import data_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VERSION = "783"  # cache lives under v783/ alongside the FAFB data; MCNS is keyed inside

# The two named functional endpoints the campaign seeds from (S10.4).
SEED_DNS = ("DNbe001", "DNp26")

# The seven figure-driven wing-steering DN types (S14 / Fig 6).
FIGURE_DNS = ("DNa04", "DNbe001", "DNge107", "DNbe005", "DNp26", "DNg32", "DNge094")

# Motor systems (MCNS motor-neuron subclasses); wing-steering is the focus.
MOTOR_SYSTEMS = (
    "jump_ttm", "wing_steering", "wing_power", "leg", "neck_gaze", "haltere", "abdominal",
)
# The MCNS subclass label for wing-steering motor neurons (S10.2 "wm").
WING_STEERING_SUBCLASS = "wm"
# Power muscles excluded from the steering-laterality denominator (DLM/DVM; S10.2).
POWER_MUSCLES = ("DLM", "DVM", "DLMn", "DVMn")

# Descending channels (S11).
CHANNELS = ("direct", "nodtype", "broadcast")

# ---------------------------------------------------------------------------
# S14 - per-DN wing laterality (figure_input channel+syn, steering syn, ipsi frac, wing)
# ---------------------------------------------------------------------------
DN_LATERALITY = {
    "DNa04":   {"figure_channel": "direct", "figure_syn": 51,  "steering_syn": 908,  "ipsi_frac": 0.99, "wing": "ipsilateral"},
    "DNbe001": {"figure_channel": "direct", "figure_syn": 727, "steering_syn": 1181, "ipsi_frac": 0.52, "wing": "bilateral"},
    "DNge107": {"figure_channel": "nodtype", "figure_syn": 22,  "steering_syn": 1044, "ipsi_frac": 0.58, "wing": "bilateral"},
    "DNbe005": {"figure_channel": "nodtype", "figure_syn": 30,  "steering_syn": 459,  "ipsi_frac": 0.42, "wing": "bilateral"},
    "DNp26":   {"figure_channel": "nodtype", "figure_syn": 508, "steering_syn": 386,  "ipsi_frac": 0.23, "wing": "contralateral"},
    "DNg32":   {"figure_channel": "nodtype", "figure_syn": 54,  "steering_syn": 313,  "ipsi_frac": 0.03, "wing": "contralateral"},
    "DNge094": {"figure_channel": "nodtype", "figure_syn": 83,  "steering_syn": 37,   "ipsi_frac": 0.00, "wing": "contralateral"},
}

# ---------------------------------------------------------------------------
# S15 - DNp26 -> steering muscle MN, by wing (rel. to DN), summed over both DNp26 cells
# wing: "contra" | "ipsi" | "mixed"
# ---------------------------------------------------------------------------
DNP26_MUSCLES = {
    "hg1": {"wing": "contra", "syn": 122},
    "i1":  {"wing": "contra", "syn": 105},
    "hg2": {"wing": "mixed",  "syn": 117},  # 87 ipsi, 30 contra
    "b3":  {"wing": "contra", "syn": 28},
    "hg3": {"wing": "contra", "syn": 5},
}

# ---------------------------------------------------------------------------
# S16 - direct LLPC1->DN channel: leading MCNS-resolved targets
#   (DN, circuit_syn, top_motor_system, [top muscles])
# ---------------------------------------------------------------------------
DIRECT_CHANNEL = [
    ("DNbe001", 736, "wing_steering", ["hg3", "DLMn c-f", "hg1"]),
    ("DNae002", 533, "haltere",       ["MNhm42", "MNnm03", "b3"]),
    ("DNpe056", 282, "abdominal",     ["MNad01", "ps2", "MNad34"]),
    ("DNp26",   219, "wing_steering", ["hg1", "hg2", "i1"]),
    ("DNg97",   126, "leg",           ["tibia flexor", "sternal anterior rotator"]),
    ("DNg82",   124, "wing_power",    ["DLMn c-f", "DLMn a,b"]),
    ("DNa04",   53,  "wing_steering", ["hg1", "MNnm08", "b3"]),
    ("DNa02",   50,  "leg",           ["sternal anterior rotator", "femur reductor"]),
    ("DNp31",   43,  "wing_power",    ["DLMn c-f", "MNwm36", "ps1"]),
    ("DNp07",   35,  "leg",           ["tarsus levator", "i2", "MNml81"]),
]

# ---------------------------------------------------------------------------
# S17 - Nod-type relay channel: leading MCNS-resolved targets
# ---------------------------------------------------------------------------
NODTYPE_CHANNEL = [
    ("DNp26",   289, "wing_steering", ["hg1", "hg2", "i1"]),
    ("DNb03",   87,  "neck_gaze",     ["ADNM1", "ADNM2", "b3"]),
    ("DNge094", 83,  "wing_steering", ["b3", "MNnm08", "MNnm07/MNnm12"]),
    ("DNg32",   54,  "wing_steering", ["MNhm43", "tp1", "b2"]),
    ("DNbe005", 30,  "wing_steering", ["hg2", "hg1", "hg3"]),
    ("DNge107", 22,  "wing_steering", ["b1", "b2", "hg3"]),
    ("DNb02",   6,   "neck_gaze",     ["MNhm03", "MNnm11", "ADNM1"]),
]

# ---------------------------------------------------------------------------
# S18 - PLP/PVLP broadcast route: leading MCNS-resolved targets
# ---------------------------------------------------------------------------
BROADCAST_CHANNEL = [
    ("DNp103", 85, "leg",       ["sternal anterior rotator MN", "TTMn"]),
    ("DNp06",  74, "jump_ttm",  ["TTMn", "MNad42", "sternal anterior rotator"]),
    ("DNg40",  63, "jump_ttm",  ["TTMn", "tergopleural/pleural promotor MN"]),
    ("DNa07",  30, "wing_steering", ["hg1", "b2", "DLMn c-f"]),
    ("DNp27",  28, "abdominal", ["ps1 MN", "hg2 MN", "MNad42"]),
    ("DNp03",  19, "wing_power", ["DLMn c-f", "ps1 MN", "MNwm36"]),
    ("DNa16",  17, "haltere",   ["MNhm03", "ADNM1 MN", "hg4 MN"]),
    ("DNp01",  16, "jump_ttm",  ["TTMn", "DVMn 1a-c"]),
]

# ---------------------------------------------------------------------------
# S11 - channel summary + MCNS coverage (circuit-to-DN syn, n DNs, coverage %)
# ---------------------------------------------------------------------------
CHANNEL_SUMMARY = {
    "direct":    {"circuit_to_dn_syn": 2174, "n_dns": 25, "mcns_coverage_pct": 99.0},
    "nodtype":   {"circuit_to_dn_syn": 712,  "n_dns": 53, "mcns_coverage_pct": 98.0},
    "broadcast": {"circuit_to_dn_syn": 628,  "n_dns": 54, "mcns_coverage_pct": 79.0},
}

# ---------------------------------------------------------------------------
# S12 - MCNS motor-system distribution by channel (% circuit-weighted over resolvable)
# ---------------------------------------------------------------------------
MOTOR_SYSTEM_PCT = {
    "direct":    {"jump_ttm": 0,  "wing_steering": 36, "wing_power": 11, "leg": 12, "neck_gaze": 7,  "haltere": 16, "abdominal": 17},
    "nodtype":   {"jump_ttm": 2,  "wing_steering": 55, "wing_power": 10, "leg": 8,  "neck_gaze": 14, "haltere": 4,  "abdominal": 7},
    "broadcast": {"jump_ttm": 17, "wing_steering": 17, "wing_power": 9,  "leg": 32, "neck_gaze": 5,  "haltere": 3,  "abdominal": 17},
}


def muscular_dir(version: str = VERSION) -> Path:
    """Scratch dir for large/regenerable muscular intermediates; created on demand."""
    d = data_root() / f"v{version}" / "muscular"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_power_muscle(name) -> bool:
    """True if a muscle/MN name is a (bilateral) power muscle, excluded from steering."""
    if name is None:
        return False
    s = str(name).upper()
    return any(p.upper() in s for p in POWER_MUSCLES)
