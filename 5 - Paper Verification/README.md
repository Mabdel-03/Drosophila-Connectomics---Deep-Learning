# 5 — Paper Verification

End-to-end, no-shortcuts confirmation of **every** quantitative claim in
`Figure_Ground_Circuit.pdf` ("Finding Vision-Behavior Circuit through Connectomics") —
the VCH → T4/T5 → LLPC1 → Nod1 → DNp26 → wing-steering figure-ground circuit.

Unlike Stage 4 (`flyconn.motif`, which audited the *predecessor* VCH–T4/T5 report from
the offline dump), this stage verifies the **current** paper across **both** connectomes
and uses the **live FlyWire CAVE API** so the absolute counts match the paper to the digit.

## Result

```
111 CONFIRMED · 6 CONFIRMED_WITH_CAVEAT · 0 REFUTED · 5 UNVERIFIABLE   (122 claims)
```

No quantitative claim is contradicted. The 6 caveats are down-drift counts on the
Nod-relay arm; the 5 UNVERIFIABLE are physiology predictions / mechanism interpretations
(recorded with their anatomical proxy). See **[REPORT.md](REPORT.md)** for the narrative
and **[VERIFICATION.md](VERIFICATION.md)** for the full per-claim ledger.

## Method

`oracle → derive → compare → ledger`, reusing `flyconn.motif.compare` unchanged.

- **Oracle** (`src/flyconn/paper/oracle/`): every paper claim encoded as data, one module
  per family (A–J), each number citing its page/table. Family **Y** audits the paper's own
  arithmetic from the oracle alone; family **Z** holds the interpretive/prediction claims.
- **Derive** (`src/flyconn/paper/derive/`): re-computes each claim from the connectome.
  - **FlyWire** (families A–I): live CAVE `flywire_fafb_public`, materialization v783,
    `synapses_nt_v1`, no cleft threshold (`fw_access.LiveCaveFlyWire`). This reproduces the
    paper's absolute counts exactly (the offline v783 dump is a ~65% snapshot; kept as a
    cross-check / no-token fallback via `OfflineFlyWire`). Live queries are cached to
    scratch parquet, so re-runs need no token and are fast.
  - **MaleCNS** (family J): the public `male-cns:v1.0` bulk flat files (no auth) —
    `subclass == 'wm'` wing-steering motor neurons, ipsi/contra from `somaSide`.
- **Verdicts**: CONFIRMED · CONFIRMED_WITH_CAVEAT · REFUTED · UNVERIFIABLE.

## Claim families

| | Family | Headline checks |
|---|---|---|
| A | VCH–T4/T5 loop | 27,576/32,363 syn; 1,022 in (12,301, 44.6%); 912 reciprocal; 2.4× |
| B | Inhibitor screen | only VCH & DCH pass all 3 criteria; LLPC1 contact /100 |
| C | LLPC1 = figure sheet | 17,499 reciprocal-T4/T5 vs siblings 318–401 (~48×) |
| D | Retinotopy null | 454 T4a → 9,223 syn → 100 LLPC1; observed ≪ in-degree null (z, p) |
| E | Dual dendrite + cable | VCH→T4a ~1.8 µm from terminal vs ~29 µm; Wilcoxon p~6e-49 |
| F | Sheet regulation | LPi15 100/6,472/94.8% layer-b; PVLP011 100/99; VCH 77/588 |
| G | Output census | 106,269 syn; Nod1 dominant excitatory readout |
| H | Nod1 → DNp26 | Nod1 → 662 syn / 27 DNs; DNp26 ← 448 (> 149 direct) |
| I | Descending channels | direct 2,174 / Nod 712 / broadcast 628 |
| J | Wing steering (MaleCNS) | DNa04 ipsi 0.99; DNp26/DNg32 contra 0.23/0.03; DNp26→hg1/i1/hg2 |
| Y | Internal arithmetic | the paper's own percentages/ratios are self-consistent |
| Z | Interpretive | divisive-norm, readout, silencing predictions → proxies, UNVERIFIABLE |

## How to run

```bash
# One-time: download the public MaleCNS bulk files (no auth).
python scripts/malecns_prep.py        # or: sbatch slurm/malecns_prep.sbatch

# Verify everything (needs the CAVE token at ~/.cloudvolume/secrets/cave-secret.json).
python scripts/paper_verify.py        # or: sbatch slurm/paper_verify.sbatch
python scripts/paper_verify.py --only A,E,J     # a subset
python scripts/paper_verify.py --offline        # offline FlyWire (relationships only)

# Tests (pure logic, no token / big files):
python -m pytest tests/test_paper_verify.py -q
```

### Skeletons (FD3 / Family K morphology)

The CAVE skeleton service does not work on the public datastack (no L2 cache). To get **real**
skeletons we use `fafbseg.flywire.skeletonize_neuron` (meshes via CloudVolume + local
skeletonization). This needs the `flyconn_cave` env (has `cloudvolume`/`skeletor`/`navis`) and a
**second token file** that CloudVolume's graphene auth reads:

```bash
# 1. one-time: add the FlyWire token where CloudVolume looks for it (same 32-char token as
#    cave-secret.json). Either copy it:
cp ~/.cloudvolume/secrets/cave-secret.json ~/.cloudvolume/secrets/chunkedgraph-secret.json
#    or let fafbseg write it:
#    python -c "from fafbseg import flywire; flywire.set_chunkedgraph_secret('<TOKEN>')"

# 2. one-time: install fafbseg into the flyconn_cave env:
/orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/pip install fafbseg

# 3. prefetch skeletons for the FD3 pair + FD1 anchor (caches to .flyconn_cache, ~33s/cell):
/orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave/bin/python scripts/fetch_fd3_skeletons.py
```

Once cached, `paper_verify.py --only K` (any env) computes **skeleton-backed** morphology claims;
without a cached skeleton it falls back to the synapse-cloud proxy automatically. Live API caches
(synapses, skeletons) write to the **project tree** `.flyconn_cache/` (set `FLYCONN_CACHE_ROOT` to
override) — never scratch, which is over quota.

## Code

`src/flyconn/paper/`: `fw_access.py` (live/offline source), `geometry.py` (nm→µm,
patch radius, nearest-terminal distance), `oracle/` + `derive/` (one module per family),
`malecns/` (MaleCNS access + motor classification), `ledger.py` (assembly + emitters),
`figures.py`. Runners: `scripts/paper_verify.py`, `scripts/malecns_prep.py`.
