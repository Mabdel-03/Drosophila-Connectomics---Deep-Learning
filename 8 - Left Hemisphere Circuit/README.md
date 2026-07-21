# 8 — Left-Hemisphere Figure-Ground Circuit

**Question (existence, not assumed):** does the LEFT optic lobe of FlyWire FAFB v783 contain a
faithful **mirror / complement** of the right-hemisphere figure-ground circuit that
`Figure_Ground_Circuit.pdf` (Ziyin … Poggio) derived? If yes, name the precise left-side cells,
show how they connect, and demonstrate the mirror; if no/partial, say exactly where it breaks.

The right circuit (the paper's scope, "the right-side front-to-back exemplar"):

```
LEFT-soma VCH (GABAergic centrifugal, crosses hemispheres)
  → gates layer-a (front-to-back) T4a terminals  → 100-cell cholinergic LLPC1 sheet
  → Nod1 (dominant excitatory readout)  → DNp26 (steering command)  → CONTRA wing (hg1/i1)
```

Centrifugal cells cross, so the **left sheet is gated by the RIGHT-soma VCH**
(`720575940627502338`). The left circuit, if it exists, should be the enantiomer: same cell
types / NT / wiring motifs, body-side labels flipped, and the command steering the **opposite
wing**, so the two hemispheres form the bilateral figure-ground steering system.

## Method

A direct flyconn stage that runs the proven Stage-5 family code at **both** hemispheres through
one parametrized code path (`SideConfig`), so every left↔right difference is biological /
proofreading, not a code artifact, and a genuine absence shows up as a stage that fails to
populate. The right run is the **positive control** and must still reproduce the paper to the
digit.

- **Phase 0** — re-examine the right-side **Nod1 = Egelhaaf FD1** anchor (it was assumed, never
  tested; neither paper states it; Nod1 is a `visual_projection` VPN, not a lobula-plate
  tangential cell). Tests the four functional FD1 criteria + the cell-class question + the
  alternatives (VCH, the LLPC1 sheet, sibling Nods).
- **Phase 0.5** — a cheap **7-stage existence screen** (PRESENT / WEAK / ABSENT) as a go/no-go,
  with absolute floors so the proofreading-completeness deficit can't silently upgrade "absent"
  to "present".
- **Families A–I + K** for both sides (entry loop, inhibitor screen, sheet uniqueness,
  retinotopy null, cable-distance terminal gating, sheet regulation, output census, Nod1→DNp26,
  descending channels, Egelhaaf FD homologues).
- **Family M** — mirror symmetry + negative controls + the **wing-flip** (the complement crux,
  in MaleCNS): the two DNp26 bodies must steer opposite physical wings via the same muscles.
- **Cell-identity manifest** — every left+right circuit root id, with provenance (the artifact
  the prior cheap pass never produced).

Verdict policy is tuned for an existence test: scale-robust statistics (fractions, z-scores,
ranks, wing category) carry the call; count claims get down-drift tolerance only above an
absolute floor; categorical/sign/layer/wing claims are exact; negative controls must fail to
gate (a passing wrong-gate REFUTES the mirror).

## Outputs

- `left_hemisphere_results.json` — full ledger: Phase-0 audit, existence screen, families A–K
  (both sides), family M (mirror + controls + wing-flip), with the headline verdict.
- `cell_identity_manifest.json` — every left+right circuit root id with provenance.
- `VERIFICATION.md` — per-claim ledger + the 7-stage PRESENT/WEAK/ABSENT roll-up.
- `REPORT.md` — narrative: existence verdict, the named left cells, the wing-flip, caveats.
- `figures/` — `stage_existence.png` (headline), `mirror_comparison.png`, `wing_flip.png`,
  `per_celltype_bars.png`.

## How to run

```bash
# SLURM (cluster), torch-free CAVE env, 48G / 2h
sbatch slurm/left_hemisphere_verify.sbatch

# or directly in the flyconn_cave env
python scripts/left_hemisphere_verify.py            # all, live CAVE v783 primary (cache-backed)
python scripts/left_hemisphere_verify.py --only A,M  # a subset
python scripts/left_hemisphere_verify.py --offline   # offline FlyWire cross-check
```

Needs the CAVE token (`~/.cloudvolume/secrets/cave-secret.json` or `$CAVE_TOKEN`) and, for the
wing-flip, the MaleCNS bulk files (`scripts/malecns_prep.py`). Live queries are cached under
`.flyconn_cache/v783/paper/`, so re-runs are fast and survive CAVE outages.

## Code map

- `src/flyconn/paper/oracle/consts.py` — `SideConfig` + the VCH/DCH root maps (additive; RIGHT = the paper)
- `src/flyconn/paper/derive/sheet.py` — `get_sheet(src, meta, cfg)`, cache keyed by (track, side)
- `src/flyconn/paper/derive/{a..k}.py` + `oracle/{a..k}.py` — the families (threaded by `cfg`)
- `src/flyconn/paper/derive/k1_nod1_fd1.py` + `oracle/k1_nod1_fd1.py` — Phase-0 FD1 audit
- `src/flyconn/paper/derive/existence_screen.py` — Phase-0.5 go/no-go screen
- `src/flyconn/paper/derive/identities.py` — the cell-identity manifest
- `src/flyconn/paper/derive/m_mirror.py` + `oracle/m_mirror.py` — mirror + negative controls
- `src/flyconn/muscular/mirror.py` — the wing-flip (MaleCNS)
- `src/flyconn/paper/left_hemisphere.py` — orchestrator + emitters; `figures_left.py` — plots
- `scripts/left_hemisphere_verify.py`, `slurm/left_hemisphere_verify.sbatch`, `tests/test_left_hemisphere.py`
