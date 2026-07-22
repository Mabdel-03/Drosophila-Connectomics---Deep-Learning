"""Family B — VCH is the only centrifugal, T4/T5-reciprocal inhibitor that gates the
LLPC1 sheet (Table 1, Fig 3).

The eight largest inhibitory T4/T5 targets are scored on three criteria:
  (1) T4/T5-reciprocal  (count of right-hemi T4/T5 that both drive and receive from it)
  (2) LLPC1 sheet contact (number of the 100 LLPC1 it inhibits)
  (3) centrifugal identity (super_class == visual_centrifugal)
Only VCH and DCH pass all three.

Source: Table 1 (p.3), Fig 3 (p.5).
"""

from __future__ import annotations

from flyconn.motif import compare as K

# Table 1: per-neuron (super_class, T4/T5 reciprocal, recip fraction, LLPC1 contact /100).
TABLE1 = {
    "VCH":   {"super_class": "visual_centrifugal", "reciprocal": 912,  "recip_frac": 0.89, "llpc1": 77},
    "DCH":   {"super_class": "visual_centrifugal", "reciprocal": 942,  "recip_frac": 0.86, "llpc1": 52},
    "CT1":   {"super_class": "optic",              "reciprocal": 5972, "recip_frac": 1.00, "llpc1": 3},
    "LPi15": {"super_class": "optic",              "reciprocal": 1771, "recip_frac": 0.64, "llpc1": 100},
    "LPi14": {"super_class": "optic",              "reciprocal": 1736, "recip_frac": 0.59, "llpc1": 90},
    "Am1":   {"super_class": "optic",              "reciprocal": 1193, "recip_frac": 0.56, "llpc1": 99},
    "LT33":  {"super_class": "optic",              "reciprocal": 2630, "recip_frac": 0.94, "llpc1": 7},
    "Li14":  {"super_class": "optic",              "reciprocal": 1316, "recip_frac": 0.96, "llpc1": 14},
}
PAGE = "Table 1 (p.3)"

# The screen's verdict: which types pass all three criteria.
PASS_ALL_THREE = {"VCH", "DCH"}


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    comp = d["table1"]  # {type: {super_class, reciprocal, llpc1, is_centrifugal}}

    for t, row in TABLE1.items():
        c = comp.get(t, {})
        # (2) LLPC1 sheet contact — the discriminating, retinotopic criterion.
        out.append(K.compare_count(
            f"B.{t}.llpc1", f"{t}: LLPC1 sheet contact (/100)",
            row["llpc1"], c.get("llpc1"), rel=0.10, abs_floor=3, drift_dir="down"))
        # (3) centrifugal identity (categorical).
        out.append(K.compare_categorical(
            f"B.{t}.centrifugal", f"{t}: super_class",
            row["super_class"], c.get("super_class")))

    # The headline categorical: exactly {VCH, DCH} pass all three criteria.
    out.append(K.compare_categorical(
        "B.pass_all_three", "Only VCH & DCH pass all 3 figure-circuit criteria",
        sorted(PASS_ALL_THREE), sorted(d["pass_all_three"]),
        refuted_note=f"computed pass-set = {sorted(d['pass_all_three'])}"))
    return out
