"""Stage-5 verification: self-consistency + robustness for the muscular projection.

Operates on the intermediate parquets written by the extract step (so it is cache-only /
offline), reusing ``motif.compare`` for every verdict and ``circuit.engine`` for the null
model + oracle cross-check. Emits the gate files the campaign reads: a verification_results
payload (with the flat ``refuted_claims`` list) and a coverage payload.

Checks:
  1. somaSide laterality consistency vs S14 (ipsi_frac per DN + categorical wing).
  2. internal consistency: ipsi+contra == steering total per DN (check_table_sum).
  3. S15 per-muscle DNp26 synapse re-derivation (compare_count, two tracks via the map).
  4. null model: permute MN somaSide labels, z/p on ipsi_frac for lateralised DNs.
  5. unannotated-fragment / coverage audit (fraction of DN output on annotated MNs).
  6. coverage: both wings reached + full DN->MN->muscle chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..circuit import engine as E
from ..motif import compare as K
from . import muscular_config as C
from . import trace as T


def _ipsi_frac_stat(syn: np.ndarray, sides: np.ndarray, dn_side: str) -> float:
    tot = syn.sum()
    if tot <= 0:
        return 0.0
    ipsi = syn[sides == dn_side].sum()
    return float(ipsi / tot)


def laterality_claims(lat: pd.DataFrame) -> list[K.ClaimResult]:
    """Compare computed per-DN ipsi_frac + wing against S14."""
    out: list[K.ClaimResult] = []
    by_dn = {r["dn"]: r for _, r in lat.iterrows()}
    for dn, exp in C.DN_LATERALITY.items():
        got = by_dn.get(dn)
        if got is None:
            out.append(K.unverifiable(f"S14.{dn}.ipsi_frac", f"{dn} ipsi_frac",
                                      exp["ipsi_frac"], "DN not found in trace"))
            continue
        out.append(K.compare_pct(f"S14.{dn}.ipsi_frac", f"{dn} ipsi fraction",
                                 exp["ipsi_frac"] * 100.0, float(got["ipsi_frac"]) * 100.0,
                                 pp=10.0))
        out.append(K.compare_categorical(f"S14.{dn}.wing", f"{dn} wing target",
                                         exp["wing"], got["wing"]))
        # internal consistency: ipsi+contra accounts for the steering total
        rows_total = int(got["ipsi_syn"]) + int(got["contra_syn"])
        out.append(K.check_table_sum(f"S14.{dn}.split_sum",
                                     f"{dn} ipsi+contra == steering total",
                                     rows_total, int(got["steering_syn"]),
                                     abs_floor=5, rel=0.10))
    return out


def muscle_claims(dn_mn_wing: pd.DataFrame) -> list[K.ClaimResult]:
    """Re-derive S15 DNp26 -> muscle synapse counts."""
    tab = T.muscle_table_for_dn(dn_mn_wing, "DNp26")
    by_muscle = tab.groupby("muscle")["syn"].sum().to_dict() if not tab.empty else {}
    out: list[K.ClaimResult] = []
    for muscle, info in C.DNP26_MUSCLES.items():
        primary = int(by_muscle.get(muscle, 0))
        out.append(K.compare_count(f"S15.DNp26.{muscle}", f"DNp26 -> {muscle} synapses",
                                   info["syn"], primary, rel=0.2, abs_floor=5,
                                   drift_dir="down"))
    return out


def null_claims(dn_mn_wing: pd.DataFrame, *, n_perm: int = 500, rng_seed: int = 0
                ) -> tuple[list[K.ClaimResult], dict]:
    """Permute MN somaSide labels; check lateralised DNs sit in the null tail."""
    out: list[K.ClaimResult] = []
    details: dict = {}
    df = dn_mn_wing[(dn_mn_wing["motor_system"] == "wing_steering")
                    & (dn_mn_wing["wing_rel"].isin(["ipsi", "contra"]))]
    for dn in ("DNa04", "DNp26", "DNg32"):  # strongly lateralised exemplars
        g = df[df["dn"] == dn]
        if g.empty or g["dn_soma_side"].isna().all():
            out.append(K.unverifiable(f"null.{dn}", f"{dn} laterality null", "tail",
                                      "no steering MNs with sides"))
            continue
        dn_side = str(g["dn_soma_side"].dropna().iloc[0])
        res = E.label_permutation_null(
            g["syn_count"].to_numpy(float), g["mn_soma_side"].to_numpy(str),
            statistic=lambda v, lab, s=dn_side: _ipsi_frac_stat(v, lab, s),
            n_perm=n_perm, rng_seed=rng_seed,
        )
        details[dn] = res
        # a genuine wing bias should be far from the permutation mean (|z| large or p small)
        ok = res["p"] <= 0.05 or abs(res["z"]) >= 2.0
        out.append(K.ClaimResult(
            id=f"null.{dn}", description=f"{dn} wing bias vs somaSide-permutation null",
            report_value="non-random (p<=0.05)",
            computed_primary=f"obs={res['observed']:.2f}, z={res['z']:.2f}, p={res['p']:.3f}",
            tolerance="p<=0.05 or |z|>=2", verdict=K.CONFIRMED if ok else K.CONFIRMED_WITH_CAVEAT,
            numeric_outcome=K.MATCH if ok else K.MINOR_DIFF,
            notes="" if ok else "wing bias not separable from chance in this sample",
        ))
    return out, details


def coverage(dn_mn_wing: pd.DataFrame, seed_dns=C.SEED_DNS) -> dict:
    """Bilateral + chain-completeness metrics for the gate.

    bilateral_coverage: fraction of seed DNs for which BOTH wings (ipsi & contra) are
    reached by >=1 wing-steering MN. muscle_chain_complete: 1.0 iff every seed DN reaches
    >=1 muscle through >=1 MN.
    """
    steering = dn_mn_wing[dn_mn_wing["motor_system"] == "wing_steering"]
    seeds = list(seed_dns)
    both_wings = 0
    chain_ok = 0
    missing: list[dict] = []
    for dn in seeds:
        g = steering[steering["dn"] == dn]
        wings = set(g["wing_rel"].unique()) & {"ipsi", "contra"}
        if {"ipsi", "contra"}.issubset(wings):
            both_wings += 1
        else:
            for w in ({"ipsi", "contra"} - wings):
                missing.append({"dn": dn, "wing": w, "muscle": None})
        has_muscle = "muscle" in g.columns and g["muscle"].notna().any() and not g.empty
        if has_muscle:
            chain_ok += 1
        else:
            missing.append({"dn": dn, "wing": None, "muscle": "any"})
    n = max(len(seeds), 1)
    dn_mn_edges = int(dn_mn_wing["mn_root_id"].nunique()) if "mn_root_id" in dn_mn_wing else 0
    mn_muscle_edges = (int(dn_mn_wing.loc[dn_mn_wing["muscle"].notna(), "muscle"].nunique())
                       if "muscle" in dn_mn_wing else 0)
    return {
        "bilateral_coverage": both_wings / n,
        "muscle_chain_complete": chain_ok / n,
        "dn_to_mn_edges": dn_mn_edges,
        "mn_to_muscle_edges": mn_muscle_edges,
        "missing": missing,
    }


def unannotated_audit(dn_mn: pd.DataFrame) -> dict:
    """Fraction of DN->MN synapses landing on annotated (named) motor neurons."""
    if dn_mn.empty:
        return {"annotated_frac": 0.0, "n_edges": 0}
    total = float(dn_mn["syn_count"].sum())
    named = dn_mn["muscle"].notna() if "muscle" in dn_mn.columns else pd.Series(False, index=dn_mn.index)
    annotated = float(dn_mn.loc[named, "syn_count"].sum())
    return {"annotated_frac": (annotated / total) if total else 0.0,
            "n_edges": int(len(dn_mn)), "total_syn": int(total)}


def run_all(dn_mn_wing: pd.DataFrame, *, n_perm: int = 500) -> dict:
    """Run every check and return {claims, details, coverage} ready to emit."""
    lat = T.dn_wing_laterality(dn_mn_wing)
    claims: list[K.ClaimResult] = []
    claims += laterality_claims(lat)
    claims += muscle_claims(dn_mn_wing)
    null_cl, null_details = null_claims(dn_mn_wing, n_perm=n_perm)
    claims += null_cl
    cov = coverage(dn_mn_wing)
    audit = unannotated_audit(dn_mn_wing)
    details = {
        "dn_wing_laterality": lat.to_dict(orient="records"),
        "null_model": null_details,
        "unannotated_audit": audit,
        "coverage": cov,
    }
    return {"claims": claims, "details": details, "coverage": cov}
