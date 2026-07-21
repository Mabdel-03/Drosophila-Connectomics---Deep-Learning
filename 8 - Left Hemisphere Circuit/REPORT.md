# Stage 8 — Does a faithful LEFT mirror of the figure-ground circuit exist?

## Headline

**MIRROR_PRESENT** — The left hemisphere contains a faithful mirror of the right figure-ground circuit: every core stage mirrors the right, and the DNp26 command steers the OPPOSITE wing (the complement). FD1 label carries the Phase-0 caveat.

The investigation tested existence rather than assuming it: a cheap 7-stage screen (go/no-go), then the full A-K mirror at the same code path as the right positive control, the wing-flip complement in MaleCNS, and falsifying negative controls.

## What the left circuit IS (named cells)

- Gating VCH (right-soma, crosses): `720575940627502338` (gaba)
- VCH-gated left T4a: 411 cells  
- Left LLPC1 sheet: 86 cells (of 106 annotated)
- Left Nod1 (FD1-role correlate): `[720575940625528556, 720575940628438427]`
- Left FD3 (LPT42_Nod4): `[720575940625992781]`
- Left DNp26: `[720575940619432261]`

## The complement (wing-flip)

- DNp26 target wings by soma side: {'L': 'R', 'R': 'L'}  
- Opposite wings (complement): **True**  
- Shared steering muscles: ['b3', 'hg1', 'hg2', 'i1']

The left brain's Nod1 drives the somaSide-L DNp26 (steering the RIGHT wing); the right brain's Nod1 drives the somaSide-R DNp26 (steering the LEFT wing). The two circuits steer OPPOSITE wings through the SAME muscles — a genuine mirror complement.

## Honest left-right differences (the 2 REFUTED claims)

These are real, interpretable quantitative differences — reported, not smoothed. Neither breaks
the mirror (the core stages, the complement, and the controls all hold):

- **`left.B.pass_all_three`** — on the left, only **VCH** passes all three figure-circuit
  criteria, not {VCH, DCH} as on the right. The left (right-soma) **DCH contacts 35/86 LLPC1**,
  below the paper's ≥50 sheet-contact threshold (right DCH = 52/100). The *secondary* gater is
  weaker on the left; the *primary* gater VCH (on which the paper's argument rests) mirrors
  cleanly (61/86 contact). Likely a proofreading/completeness effect landing across a hard cut.
- **`left.H.direct_to_dnp26`** — the left **direct** sheet→DNp26 input is **272 synapses** vs the
  right paper's 149. The left has a *stronger* direct DNp26 channel — but the Nod1 relay still
  dominates (left Nod1→DNp26 = 448 > direct 272), so the circuit logic (relay-led steering) is
  preserved. A genuine left-right weighting difference, not a wiring failure.

## Nod1 = FD1 anchor (Phase 0)

**CONFIRMED_WITH_CAVEAT** — Nod1 is the FUNCTIONAL/anatomical FD1-ROLE correlate (progressive, frontal, excitatory, contralateral noduli axon, best functional match among the alternatives) but is super_class=visual_projection, NOT literally Egelhaaf's lobula-plate tangential FD1 cell. Caveat inherited by the left mirror's K'.

## Existence screen

Decision: **GO**. Left stages: {'1_entry_gate': 'PRESENT', '2_gated_sheet': 'PRESENT', '3_readout': 'PRESENT', '4_direction': 'PRESENT', '5_downstream': 'PRESENT', '6_crossing': 'PRESENT', '7_retinotopy_proxy': 'PRESENT'}

## Coverage note

Families A, B, D, E, F, G, H, I, K were run for both hemispheres, plus Phase 0 (FD1 audit),
Phase 0.5 (existence screen), Family M (mirror + controls + wing-flip), and the cell-identity
manifest. **Family C** (LLPC1-is-the-unique-sheet, a sheet-elimination sanity check) requires
the 912-reciprocal-T4/T5 broadcast query (~500k synapses onto ~112k segments), which stalls on
the live CAVE API; it is **deferred** and does not affect the existence/mirror/complement verdict
(C is downstream-independent of the mirror tests). It can be backfilled with
`python scripts/left_hemisphere_verify.py --only C` when the API is responsive, or via the SLURM
job (2 h wall).

## Caveats

- The left sheet (86 LLPC1) is smaller than the right (100): a per-hemisphere proofreading-completeness deficit (~7-14% across all counts), NOT circuit absence — the scale-robust statistics (reciprocal fraction, Nod1 rank/share, retinotopy z, contra %, wing category) are preserved.

- Nod1 is the FD1-ROLE correlate (visual_projection VPN), not literally Egelhaaf's lobula-plate tangential FD1 cell; the connectivity mirror + wing-flip do not depend on the FD1 label.

- MaleCNS (male) vs FlyWire FAFB (female): the wing-flip is a cross-animal homology bridge. NT labels are predictions; synapse counts are not weights.
