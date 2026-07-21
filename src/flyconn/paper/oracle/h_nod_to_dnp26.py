"""Family H — Nod1 relays the figure signal to the steering command neuron DNp26
(Fig 5b, Table S10, main text p.6-7).

Claims:
  * The two Nod1 cells make 662 synapses onto 27 descending neurons.
  * 97% (644/662) of that descending output targets known steering DNs.
  * DNp26 receives 448 synapses from Nod1 — ~3x the 149 it receives directly from the
    sheet, and the strongest convergent course-control target.
  * Among Nod types, only Nod1 relays steering: Nod2 is GABAergic; Nod5 sends its
    descending output to the neck/gaze DN DNb03 (247 syn); Nod3 is weak.

Source: Table S10 (p.21), Fig 5b (p.7), p.6-7.
"""

from __future__ import annotations

from flyconn.motif import compare as K

NOD1_DN_SYN = (662, "p.6 'The two Nod1 cells make 662 synapses onto 27 descending neurons'")
NOD1_N_DNS = (27, "p.6")
NOD1_STEERING_FRAC = (97.0, "p.6 '97% (644 of 662) targets known steering DNs'")
NOD1_TO_DNP26 = (448, "p.7 / Table S10 'DNp26 receives 448 synapses from Nod1'")
DIRECT_SHEET_TO_DNP26 = (149, "p.7 'three-fold more than the 149 it receives directly'")

# Table S10: Nod type -> (LLPC1 syn it receives, NT, ->DNp26 syn).
NOD_TABLE = {
    "Nod1": {"llpc1_syn": 4_228, "nt": "acetylcholine", "to_dnp26": 448},
    "Nod2": {"llpc1_syn": 826,   "nt": "gaba",          "to_dnp26": 1},
    "Nod3": {"llpc1_syn": 21,    "nt": "acetylcholine", "to_dnp26": 0},
    "Nod5": {"llpc1_syn": 13,    "nt": "acetylcholine", "to_dnp26": 0},
}
PAGE = "Table S10 (p.21)"


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    out.append(K.compare_count(
        "H.nod1_dn_syn", "Nod1 -> descending-neuron synapses",
        NOD1_DN_SYN[0], d["nod1_dn_syn"], rel=0.08, drift_dir="down"))
    out.append(K.compare_count(
        "H.nod1_n_dns", "Nod1 -> number of descending neurons",
        NOD1_N_DNS[0], d["nod1_n_dns"], rel=0.10, abs_floor=3, drift_dir="down"))
    out.append(K.compare_pct(
        "H.nod1_steering_frac", "Nod1 descending output to known steering DNs (%)",
        NOD1_STEERING_FRAC[0], d["nod1_steering_frac"], pp=5.0))
    out.append(K.compare_count(
        "H.nod1_to_dnp26", "Nod1 -> DNp26 synapses",
        NOD1_TO_DNP26[0], d["nod1_to_dnp26"], rel=0.08, drift_dir="down"))
    out.append(K.compare_count(
        "H.direct_to_dnp26", "Direct sheet -> DNp26 synapses (Nod1 route is larger)",
        DIRECT_SHEET_TO_DNP26[0], d["direct_to_dnp26"], rel=0.12, abs_floor=10, drift_dir="down"))
    # Nod1 route to DNp26 exceeds the direct route (the relay-dominance claim).
    out.append(K.compare_categorical(
        "H.nod1_route_dominates", "Nod1->DNp26 exceeds direct sheet->DNp26",
        True, d["nod1_to_dnp26"] > d["direct_to_dnp26"]))
    # Nod2 is GABAergic (does not relay an excitatory steering signal).
    out.append(K.compare_categorical(
        "H.nod2_gaba", "Nod2 is GABAergic (not a steering relay)",
        "gaba", d["nod2_nt"]))
    return out
