# Stage 9 — Inter-Hemispheric Coupling of the Figure-Ground Circuits: VERIFICATION

**FlyWire track:** live · v783 · no cleft threshold  
**Synapse-space midline:** 529.0 um (self-test True)  
**Verdict counts:** {'CONFIRMED': 21, 'UNVERIFIABLE': 6} (n=27)

## Channel N1 — Readout crossing: Nod1 -> contralateral DNp26 command

Verdicts: {'CONFIRMED': 8, 'UNVERIFIABLE': 1}

| Finding | Computed | Verdict |
|---|---|---|
| Left-soma Nod1 reaches the RIGHT DNp26 (readout crosses midline) | True | PASS CONFIRMED |
| Right-soma Nod1 reaches the LEFT DNp26 (readout crosses midline) | True | PASS CONFIRMED |
| Left Nod1 -> right DNp26 synapse count | 159 | PASS CONFIRMED |
| Right Nod1 -> left DNp26 synapse count | 289 | PASS CONFIRMED |
| Nod1 output is majority cross-midline (position-based; 86.9%) | True | PASS CONFIRMED |
| DNp26_left Nod1 input is dominantly contralateral (100.0%) | True | PASS CONFIRMED |
| DNp26_right Nod1 input is dominantly contralateral (100.0%) | True | PASS CONFIRMED |
| Left and right Nod1 converge on shared downstream cells (95 cells) | True | PASS CONFIRMED |
| Readout-crossing asymmetry left vs right (ratio 1.82) | None | N/A UNVERIFIABLE |

## Channel N2 — Heterolateral input bridges (Egelhaaf regressive inhibition)

Verdicts: {'CONFIRMED': 3, 'UNVERIFIABLE': 2}

| Finding | Computed | Verdict |
|---|---|---|
| Heterolateral bridges onto the figure pathway are present (['H1']) | True | PASS CONFIRMED |
| H1 (the contralateral bridge) reads REGRESSIVE motion (layer-b) | b | PASS CONFIRMED |
| The dominant horizontal-motion inhibitory bridge onto the FD1/Nod1 pathway is REGRESSIVE (Egelhaaf 1985): H1 (regressive, 2185 synapses) | True | PASS CONFIRMED |
| Additional contralateral inhibitory bridges reading VERTICAL motion (['MeLp2', 'cLP05']) | None | N/A UNVERIFIABLE |
| Inventory of heterolateral bridges onto the figure pathway | None | N/A UNVERIFIABLE |

## Channel N3 — Centrifugal gating: soma-vs-arbor laterality

Verdicts: {'CONFIRMED': 5}

| Finding | Computed | Verdict |
|---|---|---|
| VCH is NOT an inter-hemispheric axonal bridge (soma 99.2%% contra but arbor does not cross; position 0.1%%) | False | PASS CONFIRMED |
| VCH soma and arbor are on opposite sides (displaced soma, contralateral arbor) | True | PASS CONFIRMED |
| DCH is NOT an inter-hemispheric axonal bridge (soma 99.2%% contra but arbor does not cross; position 0.1%%) | False | PASS CONFIRMED |
| DCH soma and arbor are on opposite sides (displaced soma, contralateral arbor) | True | PASS CONFIRMED |
| The left and right gating cells do not synapse on each other directly | False | PASS CONFIRMED |

## Channel N4 — Shared downstream convergence (bilateral integration)

Verdicts: {'CONFIRMED': 3, 'UNVERIFIABLE': 2}

| Finding | Computed | Verdict |
|---|---|---|
| Left and right figure readouts converge on shared downstream cells (95) | True | PASS CONFIRMED |
| The two readouts cross-talk (Nod1 among shared targets: ['Nod3', 'Nod3', 'Nod1', 'Nod2', 'Nod1', 'Nod1', 'Nod5', 'Nod1']) | True | PASS CONFIRMED |
| A heterolateral bridge (H1) receives from both readouts (feedback loop) | True | PASS CONFIRMED |
| Neuromodulatory convergence onto both readouts' shared pool (['OA-VUMa4', 'OA-VUMa1', 'OA-VUMa4']) | None | N/A UNVERIFIABLE |
| Roles of the shared convergence cells | None | N/A UNVERIFIABLE |

## Channel N5 — Systematic discovery of inter-hemispheric edges

Verdicts: {'CONFIRMED': 2, 'UNVERIFIABLE': 1}

| Finding | Computed | Verdict |
|---|---|---|
| Systematic discovery recovers the dominant axonal bridge H1 among 20 genuine bridges | True | PASS CONFIRMED |
| The centrifugal gaters (VCH/DCH) are a soma-side artifact removed by the position test (['DCH', 'VCH']) | True | PASS CONFIRMED |
| Position-based inter-hemispheric input fraction onto the circuit (2.8%) | None | N/A UNVERIFIABLE |
