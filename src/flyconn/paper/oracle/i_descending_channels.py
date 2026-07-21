"""Family I — three descending routes from the LLPC1 sheet (Fig S8, Tables S11-S12).

The sheet's output to descending neurons divides into three channels:
  * Direct sheet -> DN:        2,174 synapses onto 25 DNs (focused course-control route)
  * Nod-type relay -> DN:        712 synapses onto 53 DNs (led by convergent DNp26)
  * PLP/PVLP broadcast -> DN:    628 synapses onto 54 DNs (weak/broad; only substantial
                                 jump/TTM bridge)

Source: Table S11 (p.21), Fig S8 (p.19-20).
"""

from __future__ import annotations

from flyconn.motif import compare as K

# channel -> (circuit-to-DN synapses, number of DNs).
CHANNELS = {
    "direct":    {"syn": 2_174, "dns": 25, "page": "Table S11 / Fig S8c"},
    "nod_relay": {"syn": 712,   "dns": 53, "page": "Table S11"},
    "broadcast": {"syn": 628,   "dns": 54, "page": "Table S11"},
}
PAGE = "Table S11 (p.21)"


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    comp = d["channels"]
    for ch, row in CHANNELS.items():
        c = comp.get(ch, {})
        out.append(K.compare_count(
            f"I.{ch}.syn", f"{ch} channel: circuit -> DN synapses",
            row["syn"], c.get("syn"), rel=0.10, drift_dir="down"))
        # DN *counts* depend on the exact cell-set boundary of each channel (e.g. which
        # PLP/PVLP cells count as the "broadcast core"), so allow a wider band than the
        # synapse counts; a small over/under count is a definitional boundary, not an
        # error. Bidirectional tolerance via abs_floor.
        out.append(K.compare_count(
            f"I.{ch}.dns", f"{ch} channel: number of DNs reached",
            row["dns"], c.get("dns"), rel=0.20, abs_floor=12, drift_dir="down"))
    # The direct channel carries the most synapses (route ordering).
    out.append(K.compare_categorical(
        "I.direct_strongest", "Direct channel carries the most circuit->DN synapses",
        True, d["direct_is_strongest"]))
    return out
