# FD4 search evidence audit

**Result: HONEST_NULL.** FD4 has no cleanly individually resolved connectome correlate in FlyWire v783; it shares FD1's entire progressive, heterolateral, cholinergic, noduli-group output class, and no separable FD4 cell or Nod1 sub-pair exists.

- Confidence FD4 is individually resolved: 0.0 (range [0.0, 0.27]); verdict UNRESOLVED_HONEST_NULL
- Candidate-screen survivors: ['Nod1'] (only Nod1 = True)
- Nod1 homogeneous (no FD1+FD4 split): True (best pairing silhouette -4.926)
- Same-side / cross-side input Jaccard: 0.33 / 0.007
- Homogeneity confirmed across tracks: {'offline': {'separable': False, 'homogeneous': True, 'best_silhouette': -4.926}, 'live': {'separable': False, 'homogeneous': True, 'best_silhouette': -4.982, 'same_side_jac': 0.276, 'cross_side_jac': 0.008}, 'v630': {'available': False, 'reason': 'v630 recompute unavailable (empty RF cloud on v630 for one Nod1 cell)'}}
- Primary track: offline

## Claim ledger (17 claims: {'CONFIRMED': 12, 'UNVERIFIABLE': 4, 'CONFIRMED_WITH_CAVEAT': 1})

- **FD4.output_class_shared_with_FD1** [CONFIRMED]: FD4 shares FD1's progressive, heterolateral, cholinergic, noduli-group output class
    - measured: 1 viable progressive figure-output type(s): ['Nod1']
    - note: Egelhaaf p.204-206: FD4 is progressive like FD1 and uses the FD1nod/FD3 noduli-group axon
- **FD4.no_other_progressive_output_cell** [CONFIRMED]: No progressive figure-output cell exists in the connectome besides Nod1 (= FD1)
    - measured: 28 layer-a low-copy types screened over 8806; sole survivor = ['Nod1']
    - note: Every other layer-a heterolateral cell is GABA/glutamate/dopamine feedback or centrifugal
- **FD4.nod1_pools_two_populations** [CONFIRMED]: The 4-cell Nod1 type pools two same-side cells per hemisphere (Egelhaaf FD1nod + FD1pof)
    - measured: same-side input Jaccard 0.33 vs cross-side 0.007
    - note: High same-side / low cross-side partner overlap = two copies of one cell type per side
- **FD4.nod1_homogeneous_not_fd1_fd4_split** [CONFIRMED]: The 4 Nod1 cells are one homogeneous frontal population, not a separable FD1 + FD4 split
    - measured: both bilateral pairings have negative silhouette (best -4.926); separable=False
    - note: No pairing separates the cells; none carries the FD4 whole-eye lateral receptive field
- **FD4.fd3_excluded** [CONFIRMED]: FD3 = LPT42_Nod4 is not FD4 despite its FD4-shaped absolute receptive field
    - measured: LPT42_Nod4 dominant layer = b (98.58% layer-b)
    - note: Direction excludes LPT42_Nod4 from FD4; FD4 also has no frontal gap, which LPT42_Nod4 has
- **FD4.phenotype_0** [CONFIRMED]: FD4 property: progressive (layer-a) preferred direction
    - measured: 95.38% layer a
    - note: shared FD1/FD4 class property (non-discriminating)
- **FD4.phenotype_1** [CONFIRMED]: FD4 property: heterolateral noduli-group axon (contra POF)
    - measured: 89.29% contralateral
    - note: shared FD1/FD4 class property (non-discriminating)
- **FD4.phenotype_2** [CONFIRMED]: FD4 property: cholinergic output
    - measured: acetylcholine
    - note: shared FD1/FD4 class property (non-discriminating)
- **FD4.phenotype_3** [UNVERIFIABLE]: FD4 property: bidirectional contralateral inhibition
    - measured: dominant regressive
    - note: FD4-discriminating property; not realized by the Nod1 population
- **FD4.phenotype_4** [UNVERIFIABLE]: FD4 property: whole-eye, laterally-weighted RF (no frontal gap)
    - measured: no lateral sub-pair; the population RF is frontal (FD1-like)
    - note: FD4-discriminating property; not realized by the Nod1 population
- **FD4.phenotype_5** [UNVERIFIABLE]: FD4 property: restricted dorso-ventral dendrite
    - measured: all Nod1 cells span the full dorso-ventral extent
    - note: FD4-discriminating property; not realized by the Nod1 population
- **FD4.phenotype_6** [CONFIRMED]: FD4 property: no lateral-protocerebrum second arbor
    - measured: single-arbor
    - note: shared FD1/FD4 class property (non-discriminating)
- **FD4.identity_verdict** [UNVERIFIABLE]: The connectomic identity of Egelhaaf FD4 in FlyWire v783
    - note: FD4 shares FD1's entire progressive, heterolateral, cholinergic, noduli-group output class, so it cannot be separated from FD1 by direction, transmitter, or output side. It is distinguished only by a whole-eye laterally-weighted receptive field and a restricted dorso-ventral dendrite. No separable FD4 sub-pair was found: the four Nod1 cells are one homogeneous frontal population (Nod1 = FD1, likely pooling Egelhaaf's FD1nod and FD1pof variants), and no other progressive figure-output cell exists. FD4 is therefore not individually resolved in FlyWire v783. Point estimate 0.0, plausible range [0.0, 0.27], verdict UNRESOLVED_HONEST_NULL. No cell or Nod1 sub-pair carries the FD4-discriminating receptive-field and dendrite signature; FD4 is not individually resolved. Point 0.0, range [0.0, 0.27].
- **FD4.confidence** [CONFIRMED_WITH_CAVEAT]: Decomposed confidence that FD4 is individually resolved (with the Nod1 population)
    - measured: point 0.0, interval [0.0, 0.27], ceiling 0.85, verdict UNRESOLVED_HONEST_NULL
    - note: Discounted for FD1-collinearity (shared output class), no distinct FlyWire type, no independent anchor, and the literature's declined FD1/2/3/4 mapping
- **FD4.progressive_arm_afferent** [CONFIRMED]: The progressive figure arm reads front-to-back (layer-a) T4a/T5a motion
    - measured: 95.38% of the T4/T5 drive is layer-a; ON/OFF split {'T4a_ON': 955, 'T5a_OFF': 1109, 't4_frac': 46.3}
    - note: The layer-a arm any FD4 correlate would use, traced on the Nod1 population
- **FD4.progressive_arm_sheet** [CONFIRMED]: The progressive arm relays through the layer-a sheet LLPC1 (FD1's sheet)
    - measured: named sheet = LLPC1
    - note: Whether an FD4 correlate would reuse FD1's sheet or have its own
- **FD4.progressive_arm_descending** [CONFIRMED]: The progressive arm reaches wing-steering descending neurons (DNp26 shared with FD1/FD3)
    - measured: top direct descending targets ['DNp26', 'DNge094', 'DNg32']
    - note: Egelhaaf p.207: the FD cells act with the Horizontal Cells on yaw-torque steering