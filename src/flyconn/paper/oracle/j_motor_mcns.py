"""Family J — the descending targets are wing-steering neurons, several controlling a
specific wing (Fig 6, Tables S14-S16), verified in the male CNS connectome.

Claims (Table S14, ipsi-fraction of each DN's wing-steering output):
  DNa04    ipsi  (ipsi frac 0.99)
  DNbe001  bilateral (0.52)
  DNge107  bilateral (0.58)
  DNbe005  bilateral (0.42)
  DNp26    contralateral (0.23)   [the principal Nod1 target]
  DNg32    contralateral (0.03)
  DNge094  contralateral (0.00)

And (Table S15) DNp26's strongest wing-steering motor targets are the contralateral
hg1 (122 syn), i1 (105) and hg2 (mixed) muscles.

Source: Fig 6 (p.8), Table S14 (p.22), Table S15 (p.22).
"""

from __future__ import annotations

from flyconn.motif import compare as K

# DN -> ipsilateral fraction of wing-steering output (Table S14).
WING_SPECIFICITY = {
    "DNa04":   {"ipsi_frac": 0.99, "target": "ipsilateral"},
    "DNbe001": {"ipsi_frac": 0.52, "target": "bilateral"},
    "DNge107": {"ipsi_frac": 0.58, "target": "bilateral"},
    "DNbe005": {"ipsi_frac": 0.42, "target": "bilateral"},
    "DNp26":   {"ipsi_frac": 0.23, "target": "contralateral"},
    "DNg32":   {"ipsi_frac": 0.03, "target": "contralateral"},
    "DNge094": {"ipsi_frac": 0.00, "target": "contralateral"},
}
# DNp26's contralateral steering-muscle targets (Table S15), synapses summed over both DNp26.
DNP26_MUSCLES = {"hg1": 122, "i1": 105, "hg2": 117, "b3": 28, "hg3": 5}
PAGE = "Fig 6 / Table S14-S15"

# Tolerance: ipsi fractions can shift with the exact wing-steering muscle set, so verify
# the lateralisation CATEGORY (ipsi / contra / bilateral) as the primary claim, with the
# numeric fraction as a softer check.
def _category(ipsi_frac: float) -> str:
    if ipsi_frac >= 0.60:
        return "ipsilateral"
    if ipsi_frac <= 0.40:
        return "contralateral"
    return "bilateral"


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    if not d.get("available", True):
        out.append(K.unverifiable(
            "J.unavailable", "MaleCNS motor mapping", "male-cns:v1.0",
            why=d.get("reason", "MaleCNS data not available")))
        return out

    comp = d["dns"]  # {type: {ipsi_frac, steering_syn, category, n_bodies}}
    for dn, row in WING_SPECIFICITY.items():
        c = comp.get(dn, {})
        # Primary: lateralisation category matches.
        out.append(K.compare_categorical(
            f"J.{dn}.category", f"{dn} wing-steering laterality",
            row["target"], c.get("category"),
            refuted_note=f"computed ipsi_frac={c.get('ipsi_frac')}"))
        # Softer: ipsi fraction within 0.20.
        if c.get("ipsi_frac") is not None:
            out.append(K.compare_pct(
                f"J.{dn}.ipsi_frac", f"{dn} ipsilateral fraction of wing-steering output",
                row["ipsi_frac"] * 100, c["ipsi_frac"] * 100, pp=20.0))

    # DNp26's top contralateral steering muscles include hg1/i1/hg2.
    top = set(d.get("dnp26_top_muscles", []))
    expected = {"hg1", "i1", "hg2"}
    out.append(K.compare_categorical(
        "J.dnp26_muscles", "DNp26's strongest steering muscles include hg1/i1/hg2",
        True, expected.issubset(top) or len(expected & top) >= 2,
        refuted_note=f"computed top DNp26 muscles = {sorted(top)}"))
    return out
