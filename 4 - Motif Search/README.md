# 4 — Motif Search

Independent verification of the **VCH–T4/T5 motion-detection motif** described in
`vch_t4t5_report.pdf` (repo root). The report analyzes the Left VCH neuron
(root `720575940627706398`), a GABAergic lobula-plate tangential cell, and the T4/T5
elementary motion detectors in the FlyWire FAFB connectome (materialization v783),
claiming a densely reciprocal T4/T5 ↔ VCH feedback loop that broadcasts to ~145k
downstream targets.

The report was generated against the **live FlyWire CAVE API** and wrote its raw data
to `/tmp/*.json` (now gone), so it is not reproducible from its own outputs. This stage
re-derives **every quantitative claim** from the **offline v783 data** already on disk
and renders a per-claim verdict.

## Outputs (committed here)
- **[VERIFICATION.md](VERIFICATION.md)** — human-readable pass/fail tables + findings narrative.
- **verification_results.json** — machine-readable per-claim results (`meta`, `claims`, `details`).
- **figures/** — diagnostic plots (subtype bars, laterality histogram, top-20 targets, cross-track counts).

## Method (dual-track + attribution)
Each numeric claim is computed on two tracks and the gap is attributed to a cause
before a verdict is assigned:

- **Primary (API analog):** the raw per-synapse table `flywire_synapses_783.feather`,
  filtered to VCH — the closest offline analog to the live `synapse_query`. Includes
  non-proofread partners, per-synapse NT probabilities, neuropil, and 3D positions.
- **Secondary (proofread graph):** the repo's derived `edges_full.parquet`
  (inner-joined to the 139,255 proofread neurons) — systematically lower counts.

Verdicts: **CONFIRMED** (within tolerance) · **CONFIRMED_WITH_CAVEAT** (matches after
a demonstrated data-source difference) · **REFUTED** (report appears wrong) ·
**UNVERIFIABLE** (no ground truth in the offline dump).

## Headline findings
1. **Absolute counts are not reproducible from the offline dump** (even the raw table is
   ~60–65% of the report's synapse/partner counts). The live API likely includes
   post-snapshot proofreading edits and/or a looser synapse-confidence threshold; a
   `cleft_score` sweep is reported as evidence. → counts CONFIRMED_WITH_CAVEAT.
2. **The "input ipsilateral / output contralateral" laterality claim is REFUTED** —
   VCH's input *and* output synapses both lie in the right optic lobe (LOP_R). VCH is a
   left-soma centrifugal cell whose dendrite and axon both sit in the right optic lobe:
   a within-(right)-optic-lobe recurrent loop, not a hemispheric relay.
3. **The report's Table 2 is internally inconsistent** (subtype synapses sum to 12,426,
   not the stated 13,018) → REFUTED, independent of any data source.
4. **Sign claims CONFIRMED** at both neuron and per-synapse level (VCH GABAergic −,
   T4/T5 cholinergic +); reciprocity (~89%) and the dominant-input / LPi14-top-target
   claims hold, with caveats noted (LPi14 is an extreme hub).

## How to run
```bash
# Heavy step: two memory-mapped streaming passes over the 9.5 GB synapse table (~10 s).
sbatch slurm/vch_extract.sbatch
# Light step: re-derive claims + emit JSON/MD/figures (depends on the extract output).
sbatch slurm/vch_verify.sbatch
# Chained:
jid=$(sbatch --parsable slurm/vch_extract.sbatch)
sbatch --dependency=afterok:$jid slurm/vch_verify.sbatch
```
The scan is fast enough to run directly on a compute node too:
```bash
python -u scripts/vch_extract.py && python -u scripts/vch_verify.py
```
Tests: `python -m pytest tests/test_motif_verify.py -q`.

## Code
Logic lives in the `flyconn.motif` package (importable, unit-tested):
`vch_config.py` (the report's claims as data), `synapse_extract.py` (the chunked
Arrow-IPC reader), `vch_verify.py` (Stage B re-derivation), `downstream.py` (Stage C
aggregation), `compare.py` (tolerance/verdict engine), `figures.py`. Thin runners are
`scripts/vch_extract.py` and `scripts/vch_verify.py`. Large intermediates are written to
scratch under `$FLYCONN_DATA_ROOT/v783/motif/` (not committed).
