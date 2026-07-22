"""Family G — LLPC1's output is dominated by a single excitatory readout, Nod1
(Fig 5a, Table S3).

The 100-cell sheet makes 106,269 output synapses. Top shared targets (Table S3,
"Drivers/100" = number of the 100 LLPC1 contacting the target):

  PVLP011  100/100  3,168 syn  GABA  (recurrent output gain control)
  PLP163   100/100  2,534 syn  ACh   (excitatory PLP readout)
  PLP249    97/100  4,389 syn  GABA  (GABAergic selector; largest syn count)
  Nod1      91/100  4,228 syn  ACh   (Nod-type relay; dominant excitatory readout)   [Fig5a/text]
  Nod2      70/100    826 syn  GABA
  DNbe001   70/100    736 syn  ACh   (strongest direct descending target)
  LPi15     71/100    ...  GABA  (universal input)
  VCH       51/100    111 syn  GABA  (closes context loop)

Source: Table S3 (p.15), Fig 5a (p.7), main text p.6 'Nod1 ... 4,228 synapses from 91
of the 100 LLPC1'.
"""

from __future__ import annotations

from flyconn.motif import compare as K

OUTPUT_TOTAL_SYN = (106_269, "p.6 '106,269 output synapses'")

# target -> (drivers/100, total syn, NT). syn counts are from Fig5a / Table S3.
TOP_TARGETS = {
    "PLP249":  {"drivers": 97,  "syn": 4_389, "nt": "gaba"},
    "Nod1":    {"drivers": 91,  "syn": 4_228, "nt": "acetylcholine"},
    "PVLP011": {"drivers": 100, "syn": 3_168, "nt": "gaba"},
    "PLP163":  {"drivers": 100, "syn": 2_534, "nt": "acetylcholine"},
    "Nod2":    {"drivers": 70,  "syn": 826,   "nt": "gaba"},
    "DNbe001": {"drivers": 70,  "syn": 736,   "nt": "acetylcholine"},
}
# Nod1 is the dominant *excitatory* readout (PLP249/PVLP011 are GABAergic).
NOD1_IS_TOP_EXCITATORY = (True, "p.6 / Fig5a 'Nod1 ... dominates ... principal excitatory forward channel'")
PAGE = "Table S3 (p.15) / Fig 5a"


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    out.append(K.compare_count(
        "G.output_total", "LLPC1 sheet total output synapses",
        OUTPUT_TOTAL_SYN[0], d["output_total_syn"], rel=0.05, drift_dir="down"))

    comp = d["targets"]  # {type: {drivers, syn, nt}}
    for t, row in TOP_TARGETS.items():
        c = comp.get(t, {})
        out.append(K.compare_count(
            f"G.{t}.drivers", f"{t}: number of LLPC1 (of 100) contacting it",
            row["drivers"], c.get("drivers"), rel=0.05, abs_floor=3, drift_dir="down"))
        out.append(K.compare_count(
            f"G.{t}.syn", f"{t}: synapses from the LLPC1 sheet",
            row["syn"], c.get("syn"), rel=0.08, drift_dir="down"))
        out.append(K.compare_categorical(
            f"G.{t}.nt", f"{t}: neurotransmitter", row["nt"], c.get("nt")))

    out.append(K.compare_categorical(
        "G.nod1_top_excitatory", "Nod1 is the dominant excitatory readout of the sheet",
        NOD1_IS_TOP_EXCITATORY[0], d["nod1_is_top_excitatory"],
        refuted_note=f"top excitatory cholinergic target computed = {d.get('top_excitatory')}"))
    return out
