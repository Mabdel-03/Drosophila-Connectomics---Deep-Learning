"""Family C — among the T4/T5 targets, only LLPC1 can be the figure-output sheet
(Table S1, Fig S6, main text p.3).

Classifying every annotated target of the 912 reciprocal T4/T5 by functional class
(Table S1):
  optic intrinsic feedback (TmY/Tm/Y/C3/Mi/Tlp)  48%  recurrent; stays in optic lobe
  optic inhibitory (LPi/Li/LT/CT1)               24%  GABAergic; not an excitatory readout
  wide-field tangential/centrifugal (HS/VCH/...)  13%  collapse retinotopy
  columnar projection sheet (LLPC/LPC)             7%  excitatory, retinotopic, projecting
  other visual-projection (LPT/Nod)                5%  relays, not the first sheet
  looming/object VPN (LPLC/LC)                     2%  feature/escape pathway

Within the columnar projection class (Fig S6b), LLPC1 receives ~40x more T4/T5 input
than its direction-sibling sheets (LLPC2/3, LPC1/2). LPLC2 is the looming pathway.

Source: Table S1 (p.12), Fig S6 (p.14), p.3.
"""

from __future__ import annotations

from flyconn.motif import compare as K

# functional class -> % of the annotated reciprocal-T4/T5 broadcast (Table S1).
CLASS_BROADCAST_PCT = {
    "optic_intrinsic_feedback": 48,
    "optic_inhibitory": 24,
    "widefield_tangential_centrifugal": 13,
    "columnar_projection_sheet": 7,
    "other_visual_projection": 5,
    "looming_object_vpn": 2,
}
COLUMNAR_SHEET_PCT = (7, "Table S1 'columnar projection sheet (LLPC/LPC) 7%'")
# LLPC1 dominates its sibling sheets by ~40x in T4/T5 input (Fig S6b).
LLPC1_VS_SIBLINGS_FOLD = (40, "Fig S6b '~40x more T4/T5 input than its direction-sibling sheets'")
# LLPC1 is the unique excitatory + retinotopic + central-projecting columnar sheet.
LLPC1_IS_UNIQUE = (True, "p.3 'LLPC1 is therefore the unique candidate'")
PAGE = "Table S1 (p.12) / Fig S6"


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    # Columnar projection sheets are a small minority (~7%) of the broadcast.
    out.append(K.compare_pct(
        "C.columnar_pct", "Columnar projection sheets = % of reciprocal-T4/T5 broadcast",
        COLUMNAR_SHEET_PCT[0], d["columnar_pct"], pp=4.0))
    # LLPC1 dominates its sibling sheets (the ~40x fold; verify it is >= ~10x and the max).
    out.append(K.compare_ratio(
        "C.llpc1_vs_siblings", "LLPC1 gets ~40x more T4/T5 than sibling sheets",
        10.0, 80.0, d["llpc1_vs_siblings_fold"]))
    out.append(K.compare_categorical(
        "C.llpc1_top_sheet", "LLPC1 is the top columnar projection sheet by T4/T5 input",
        "LLPC1", d["top_columnar_sheet"]))
    # LPLC2 (looming) gets far less T4/T5 than LLPC1 (parallel pathway, not the sheet).
    out.append(K.compare_categorical(
        "C.lplc2_not_sheet", "LPLC2 (looming) is not the figure sheet (< LLPC1 T4/T5 input)",
        True, d["lplc2_less_than_llpc1"]))
    return out
