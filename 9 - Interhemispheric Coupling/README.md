# 9 — Inter-Hemispheric Coupling of the Figure-Ground Circuits

**Question:** how do the left and right figure-ground course-control circuits (derived in Stage 8)
connect and communicate with each other? Stage 8 showed the two circuits are mirror images that
steer opposite wings; this stage identifies, at synapse resolution, the wiring through which the
two hemispheres are coupled, and validates it against Egelhaaf (1985), who showed the
figure-detection cells receive contralateral inhibition.

## The coupling, in three structural routes plus a gating subtlety

- **Channel N1 — readout crossing / command convergence.** Each side's Nod1 (the FD1-role
  readout, a noduli-group cell whose axon crosses the midline) drives the OPPOSITE hemisphere's
  steering command DNp26. This is the structural basis of the Stage-8 wing-flip.
- **Channel N2 — heterolateral input bridges, and the Egelhaaf test.** Cells whose axon crosses
  the midline onto the contralateral figure cells (LLPC1 sheet, Nod1) carry contralateral
  inhibition. The central biological result: the inhibitory bridges (H1, H2) read regressive
  (back-to-front, lobula-plate layer-b) motion, exactly the contralateral regressive inhibition
  Egelhaaf inferred for FD1.
- **Channel N3 — centrifugal gating (soma vs arbor).** The gating cells VCH/DCH have their soma
  on one side but their entire arbor in the opposite lobe. By soma-side they look ~99%
  contralateral; by synapse position they do NOT cross. They are soma-displaced local gaters, not
  axonal bridges, and they do not couple to each other directly. This distinction is why the
  crossing definition is position-based, not soma-based.
- **Channel N4 — shared downstream convergence.** The two readouts converge on a common pool of
  premotor, central, and neuromodulatory cells (including octopaminergic OA-VUM cells), and
  cross-talk directly (Nod1 onto Nod1), with feedback onto the heterolateral bridges.
- **Channel N5 — systematic discovery.** A completeness sweep over all cross-midline edges of the
  circuit, ranking the genuine axonal bridges and explicitly separating them from the soma-side
  T4/T5 artifact.

## Method note (load-bearing)

A true inter-hemispheric crossing is defined by where a synapse physically sits relative to the
optic-lobe midline on the synapse position-x axis (~530 um), NOT by the soma-side annotation. The
soma-side label conflates arbor geometry with axonal crossing: VCH/DCH have a displaced soma and
read 99% "contralateral" by soma but 0% by position. The midline is derived from the synapse-x
distribution itself (the left and right LLPC1 dendrite bands separate cleanly), with a fail-fast
self-test that the two lobes are well separated before any crossing is counted.

## Outputs

- `interhemispheric_results.json` — per-channel ledger (N1-N5) with verdicts.
- `bridge_manifest.json` — every inter-hemispheric bridge cell, named, with root id, NT,
  super_class, soma side, arbor side, input direction.
- `VERIFICATION.md` — per-finding ledger grouped by channel.
- `REPORT.md` — narrative of the coupling architecture and the Egelhaaf validation.
- `interhemispheric_circuit.tex` / `.pdf` — paper-style writeup.
- `figures/` — readout crossing, heterolateral bridges by direction, shared convergence,
  soma-vs-position (the methodological figure).

## How to run

```bash
sbatch slurm/interhemispheric_verify.sbatch            # cluster, CAVE env, 48G / 2h
python scripts/interhemispheric_verify.py              # all channels, live CAVE v783 (cached)
python scripts/interhemispheric_verify.py --only N1,N3  # a subset
```

Needs the CAVE token (`~/.cloudvolume/secrets/cave-secret.json` or `$CAVE_TOKEN`). Live queries
are cached under `.flyconn_cache/v783/paper/`, so reruns are fast. The heaviest pulls are the N2
bridge discovery and the N5 sweep (inputs onto the figure cells), so the runner is resilient to
CAVE outages and can be scoped with `--only`.

## Code map

- `src/flyconn/paper/derive/interhemi_common.py` — synapse-space midline + position-based crossing
- `src/flyconn/paper/derive/n1_readout_crossing.py` ... `n5_discovery.py` + matching `oracle/n*.py`
- `src/flyconn/paper/derive/bridge_manifest.py` — the named bridge-cell manifest
- `src/flyconn/paper/interhemispheric.py` — orchestrator + emit; `figures_inter.py` — plots
- `scripts/interhemispheric_verify.py`, `slurm/interhemispheric_verify.sbatch`, `tests/test_interhemispheric.py`
