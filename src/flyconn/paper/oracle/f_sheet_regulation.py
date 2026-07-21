"""Family F — sheet-regulating inhibition: LPi15 (feed-forward opponent), PVLP011
(recurrent output gain), and VCH's direct contact onto the sheet.

Claims:
  * LPi15 contacts all 100 LLPC1 with 6,472 GABA synapses (Table S7) and is driven
    94.8% by opposite-direction (layer-b) T4/T5 (S5 / Fig S7a).
  * PVLP011 reads all 100 LLPC1 and feeds GABA back to 99 of them (Table S2/S5).
  * VCH directly inhibits 77 of the 100 LLPC1 (Table 1 / S2: 588 GABA synapses to 77/100).

Source: Table S7 (p.18), Table S2 (p.15), S5 (p.16), Fig 2/3.
"""

from __future__ import annotations

from flyconn.motif import compare as K

LPI15_LLPC1_REACHED = (100, "Table S7 / S2 'LPi15 ... all 100 LLPC1'")
LPI15_SYN = (6_472, "Table S7 'LPi15 ... 6,472 GABA synapses'")
LPI15_LAYER_B_PCT = (94.8, "S5 p.16 'LPi15 ... 94.8% ... layer-b (back-to-front)'")

PVLP011_READS = (100, "Table S2 'PVLP011 receives from all 100 LLPC1'")
PVLP011_FEEDS_BACK = (99, "Table S2 / S5 'feeds GABA back to 99 of them'")

VCH_DIRECT_LLPC1 = (77, "Table 1 / S2 '588 GABA synapses to 77/100'")
VCH_DIRECT_SYN = (588, "Table S2 'VCH direct input to sheet: 588 GABA synapses to 77/100'")
PAGE = "Table S7 / S2 / S5"


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    out.append(K.compare_count(
        "F.lpi15_reach", "LPi15 reaches all 100 LLPC1", LPI15_LLPC1_REACHED[0],
        d["lpi15_llpc1_reached"], rel=0.05, abs_floor=3, drift_dir="down"))
    out.append(K.compare_count(
        "F.lpi15_syn", "LPi15 -> LLPC1 synapses", LPI15_SYN[0],
        d["lpi15_syn"], rel=0.10, drift_dir="down"))
    out.append(K.compare_pct(
        "F.lpi15_layer_b", "LPi15 driven by opposite-direction (layer-b) T4/T5 (%)",
        LPI15_LAYER_B_PCT[0], d["lpi15_layer_b_pct"], pp=4.0))

    out.append(K.compare_count(
        "F.pvlp011_reads", "PVLP011 reads all 100 LLPC1", PVLP011_READS[0],
        d["pvlp011_reads"], rel=0.05, abs_floor=3, drift_dir="down"))
    out.append(K.compare_count(
        "F.pvlp011_feedsback", "PVLP011 feeds GABA back to 99/100 LLPC1", PVLP011_FEEDS_BACK[0],
        d["pvlp011_feeds_back"], rel=0.05, abs_floor=3, drift_dir="down"))

    out.append(K.compare_count(
        "F.vch_direct_llpc1", "VCH directly inhibits 77/100 LLPC1", VCH_DIRECT_LLPC1[0],
        d["vch_direct_llpc1"], rel=0.08, abs_floor=4, drift_dir="down"))
    out.append(K.compare_count(
        "F.vch_direct_syn", "VCH -> LLPC1 direct synapses", VCH_DIRECT_SYN[0],
        d["vch_direct_syn"], rel=0.12, drift_dir="down"))
    return out
