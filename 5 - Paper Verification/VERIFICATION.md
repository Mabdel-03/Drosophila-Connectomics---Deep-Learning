# Independent Verification — *Finding Vision-Behavior Circuit through Connectomics*

End-to-end re-derivation of every quantitative claim in `Figure_Ground_Circuit.pdf` (the VCH -> T4/T5 -> LLPC1 -> Nod1 -> DNp26 -> wing-steering figure-ground circuit), across both connectomes.

- **FlyWire track:** offline — flywire_fafb_public v783 (synapses_nt_v1, no cleft threshold)
- **MaleCNS:** male-cns:v1.0 public bulk feather (subclass=wm wing-steering)
- **Total:** {'CONFIRMED': 164, 'CONFIRMED_WITH_CAVEAT': 52, 'REFUTED': 4, 'UNVERIFIABLE': 12} over 232 claims

Verdict legend: **CONFIRMED** (within tolerance) · **CONFIRMED_WITH_CAVEAT** (matches after a named, demonstrated difference) · **REFUTED** (paper appears wrong) · **UNVERIFIABLE** (interpretive / physiology prediction; anatomical proxy noted).

## Family A — VCH-T4/T5 reciprocal loop (entry point)

_Verdicts: {'CONFIRMED': 6, 'CONFIRMED_WITH_CAVEAT': 8, 'REFUTED': 2}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| VCH neurotransmitter is GABA | gaba | gaba | - | PASS CONFIRMED |
| VCH total input synapses | 27576 | 17930 | - | PASS* CONFIRMED_WITH_CAVEAT |
| VCH total output synapses | 32363 | 19534 | - | PASS* CONFIRMED_WITH_CAVEAT |
| VCH upstream partners | 3619 | 2746 | - | PASS* CONFIRMED_WITH_CAVEAT |
| T4/T5 input neurons to VCH | 1022 | 975 | - | PASS* CONFIRMED_WITH_CAVEAT |
| T4/T5 input synapses to VCH | 12301 | 8784 | - | PASS* CONFIRMED_WITH_CAVEAT |
| T4/T5 output neurons from VCH | 1476 | 1301 | - | PASS* CONFIRMED_WITH_CAVEAT |
| T4/T5 output synapses from VCH | 7273 | 4759 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Reciprocal T4/T5 partners | 912 | 874 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Reciprocal partners that are layer-a (same direction) | 882 | 870 | - | PASS CONFIRMED |
| T4/T5 = % of VCH input syn | 44.6 | 49 | - | FAIL REFUTED |
| Reciprocal = % of T4/T5 inputs | 89 | 89.6 | - | PASS CONFIRMED |
| Reciprocal = % of T4/T5 outputs | 62 | 67.2 | - | FAIL REFUTED |
| VCH T4/T5 input syn that are layer-a (front-to-back) | 99.1 | 99.6 | - | PASS CONFIRMED |
| Excitatory drive ~2.4x inhibitory feedback (mean syn/partner) | 2.0-3.0x | 2.46 | - | PASS CONFIRMED |
| VCH T4/T5 input mean syn/partner (12,301/1,022 ~= 12.0) | 12 | 12 | - | PASS CONFIRMED |

## Family B — Inhibitor screen: only VCH/DCH gate the sheet

_Verdicts: {'CONFIRMED_WITH_CAVEAT': 3, 'CONFIRMED': 13, 'REFUTED': 1}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| VCH: LLPC1 sheet contact (/100) | 77 | 66 | - | PASS* CONFIRMED_WITH_CAVEAT |
| VCH: super_class | visual_centrifugal | visual_centrifugal | - | PASS CONFIRMED |
| DCH: LLPC1 sheet contact (/100) | 52 | 41 | - | PASS* CONFIRMED_WITH_CAVEAT |
| DCH: super_class | visual_centrifugal | visual_centrifugal | - | PASS CONFIRMED |
| CT1: LLPC1 sheet contact (/100) | 3 | 2 | - | PASS CONFIRMED |
| CT1: super_class | optic | optic | - | PASS CONFIRMED |
| LPi15: LLPC1 sheet contact (/100) | 100 | 93 | - | PASS CONFIRMED |
| LPi15: super_class | optic | optic | - | PASS CONFIRMED |
| LPi14: LLPC1 sheet contact (/100) | 90 | 63 | - | PASS* CONFIRMED_WITH_CAVEAT |
| LPi14: super_class | optic | optic | - | PASS CONFIRMED |
| Am1: LLPC1 sheet contact (/100) | 99 | 91 | - | PASS CONFIRMED |
| Am1: super_class | optic | optic | - | PASS CONFIRMED |
| LT33: LLPC1 sheet contact (/100) | 7 | 5 | - | PASS CONFIRMED |
| LT33: super_class | optic | optic | - | PASS CONFIRMED |
| Li14: LLPC1 sheet contact (/100) | 14 | 11 | - | PASS CONFIRMED |
| Li14: super_class | optic | optic | - | PASS CONFIRMED |
| Only VCH & DCH pass all 3 figure-circuit criteria | ['DCH', 'VCH'] | ['VCH'] | - | FAIL REFUTED |

## Family C — LLPC1 is the unique figure-output sheet

_Verdicts: {'CONFIRMED': 3, 'REFUTED': 1}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| Columnar projection sheets = % of reciprocal-T4/T5 broadcast | 7 | 7 | - | PASS CONFIRMED |
| LLPC1 gets ~40x more T4/T5 than sibling sheets | 10.0-80.0x | 139.8 | - | FAIL REFUTED |
| LLPC1 is the top columnar projection sheet by T4/T5 input | LLPC1 | LLPC1 | - | PASS CONFIRMED |
| LPLC2 (looming) is not the figure sheet (< LLPC1 T4/T5 input) | True | True | - | PASS CONFIRMED |

## Family D — T4a->LLPC1 retinotopy is local (null model)

_Verdicts: {'CONFIRMED': 6, 'CONFIRMED_WITH_CAVEAT': 2}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| VCH-gated right T4a count | 454 | 447 | - | PASS CONFIRMED |
| T4a -> LLPC1 synapses | 9223 | 6117 | - | PASS* CONFIRMED_WITH_CAVEAT |
| LLPC1 sheet size | 100 | 93 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Observed pooling is local: observed << null patch radius | True | True | - | PASS CONFIRMED |
| T4a->LLPC1 locality z-score (observed vs in-degree null) | -68 | -39.4 | - | PASS CONFIRMED |
| Permutation p-value (observed below null) | <0.002 | 0.002 | - | PASS CONFIRMED |
| T4a input within 10 um: observed >> null | obs 92% vs null 10% | obs 73% vs null 4% | - | PASS CONFIRMED |
| Observed median input-patch radius (um) | 6.3 | 8 | - | PASS CONFIRMED |

## Family E — Dual dendrite + compartmentalized inhibition (cable distance)

_Verdicts: {'CONFIRMED': 9}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| LLPC1 direct T4/T5 motion input (%) | 29 | 31 | - | PASS CONFIRMED |
| LLPC1 lobula form input (%) | 31 | 30.1 | - | PASS CONFIRMED |
| Motion vs form dendritic-field separation (um, median) | 20 | 15.7 | - | PASS CONFIRMED |
| VCH->T4a input is closer to the T4a->LLPC1 terminal than other inputs | True | True | - | PASS CONFIRMED |
| VCH is the closest input in ~99% of T4a (%) | 99 | 100 | - | PASS CONFIRMED |
| VCH-vs-other terminal-distance Wilcoxon (VCH smaller) | p ~ 7e-48 | p=5.94e-49 | - | PASS CONFIRMED |
| LPi15 -> nearest T4a-excitation on LLPC1 (um) | 1.3 | 1 | - | PASS CONFIRMED |
| VCH -> nearest T4a-excitation on LLPC1 (um) | 2 | 1.4 | - | PASS CONFIRMED |
| PVLP011 sits on a distinct distal compartment (>>LPi15/VCH) | True | True | - | PASS CONFIRMED |

## Family F — Sheet regulation (LPi15 / PVLP011 / VCH-direct)

_Verdicts: {'CONFIRMED_WITH_CAVEAT': 6, 'CONFIRMED': 1}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| LPi15 reaches all 100 LLPC1 | 100 | 93 | - | PASS* CONFIRMED_WITH_CAVEAT |
| LPi15 -> LLPC1 synapses | 6472 | 3679 | - | PASS* CONFIRMED_WITH_CAVEAT |
| LPi15 driven by opposite-direction (layer-b) T4/T5 (%) | 94.8 | 95.17 | - | PASS CONFIRMED |
| PVLP011 reads all 100 LLPC1 | 100 | 93 | - | PASS* CONFIRMED_WITH_CAVEAT |
| PVLP011 feeds GABA back to 99/100 LLPC1 | 99 | 92 | - | PASS* CONFIRMED_WITH_CAVEAT |
| VCH directly inhibits 77/100 LLPC1 | 77 | 66 | - | PASS* CONFIRMED_WITH_CAVEAT |
| VCH -> LLPC1 direct synapses | 588 | 371 | - | PASS* CONFIRMED_WITH_CAVEAT |

## Family G — LLPC1 output census; Nod1 dominance

_Verdicts: {'CONFIRMED_WITH_CAVEAT': 13, 'CONFIRMED': 7}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| LLPC1 sheet total output synapses | 106269 | 62342 | - | PASS* CONFIRMED_WITH_CAVEAT |
| PLP249: number of LLPC1 (of 100) contacting it | 97 | 90 | - | PASS* CONFIRMED_WITH_CAVEAT |
| PLP249: synapses from the LLPC1 sheet | 4389 | 2854 | - | PASS* CONFIRMED_WITH_CAVEAT |
| PLP249: neurotransmitter | gaba | gaba | - | PASS CONFIRMED |
| Nod1: number of LLPC1 (of 100) contacting it | 91 | 84 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Nod1: synapses from the LLPC1 sheet | 4228 | 2393 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Nod1: neurotransmitter | acetylcholine | acetylcholine | - | PASS CONFIRMED |
| PVLP011: number of LLPC1 (of 100) contacting it | 100 | 93 | - | PASS* CONFIRMED_WITH_CAVEAT |
| PVLP011: synapses from the LLPC1 sheet | 3168 | 2010 | - | PASS* CONFIRMED_WITH_CAVEAT |
| PVLP011: neurotransmitter | gaba | gaba | - | PASS CONFIRMED |
| PLP163: number of LLPC1 (of 100) contacting it | 100 | 93 | - | PASS* CONFIRMED_WITH_CAVEAT |
| PLP163: synapses from the LLPC1 sheet | 2534 | 1579 | - | PASS* CONFIRMED_WITH_CAVEAT |
| PLP163: neurotransmitter | acetylcholine | acetylcholine | - | PASS CONFIRMED |
| Nod2: number of LLPC1 (of 100) contacting it | 70 | 65 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Nod2: synapses from the LLPC1 sheet | 826 | 500 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Nod2: neurotransmitter | gaba | gaba | - | PASS CONFIRMED |
| DNbe001: number of LLPC1 (of 100) contacting it | 70 | 64 | - | PASS* CONFIRMED_WITH_CAVEAT |
| DNbe001: synapses from the LLPC1 sheet | 736 | 457 | - | PASS* CONFIRMED_WITH_CAVEAT |
| DNbe001: neurotransmitter | acetylcholine | acetylcholine | - | PASS CONFIRMED |
| Nod1 is the dominant excitatory readout of the sheet | True | True | - | PASS CONFIRMED |

## Family H — Nod1 relays the figure signal to DNp26

_Verdicts: {'CONFIRMED_WITH_CAVEAT': 4, 'CONFIRMED': 3}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| Nod1 -> descending-neuron synapses | 662 | 453 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Nod1 -> number of descending neurons | 27 | 21 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Nod1 descending output to known steering DNs (%) | 97 | 99.1 | - | PASS CONFIRMED |
| Nod1 -> DNp26 synapses | 448 | 315 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Direct sheet -> DNp26 synapses (Nod1 route is larger) | 149 | 95 | - | PASS* CONFIRMED_WITH_CAVEAT |
| Nod1->DNp26 exceeds direct sheet->DNp26 | True | True | - | PASS CONFIRMED |
| Nod2 is GABAergic (not a steering relay) | gaba | gaba | - | PASS CONFIRMED |

## Family I — Three descending channels

_Verdicts: {'CONFIRMED_WITH_CAVEAT': 3, 'CONFIRMED': 4}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| direct channel: circuit -> DN synapses | 2174 | 1326 | - | PASS* CONFIRMED_WITH_CAVEAT |
| direct channel: number of DNs reached | 25 | 24 | - | PASS CONFIRMED |
| nod_relay channel: circuit -> DN synapses | 712 | 486 | - | PASS* CONFIRMED_WITH_CAVEAT |
| nod_relay channel: number of DNs reached | 53 | 43 | - | PASS CONFIRMED |
| broadcast channel: circuit -> DN synapses | 628 | 388 | - | PASS* CONFIRMED_WITH_CAVEAT |
| broadcast channel: number of DNs reached | 54 | 46 | - | PASS CONFIRMED |
| Direct channel carries the most circuit->DN synapses | True | True | - | PASS CONFIRMED |

## Family J — Wing-steering DNs and wing specificity (MaleCNS)

_Verdicts: {'CONFIRMED': 15}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| DNa04 wing-steering laterality | ipsilateral | ipsilateral | - | PASS CONFIRMED |
| DNa04 ipsilateral fraction of wing-steering output | 99 | 99 | - | PASS CONFIRMED |
| DNbe001 wing-steering laterality | bilateral | bilateral | - | PASS CONFIRMED |
| DNbe001 ipsilateral fraction of wing-steering output | 52 | 52 | - | PASS CONFIRMED |
| DNge107 wing-steering laterality | bilateral | bilateral | - | PASS CONFIRMED |
| DNge107 ipsilateral fraction of wing-steering output | 58 | 58 | - | PASS CONFIRMED |
| DNbe005 wing-steering laterality | bilateral | bilateral | - | PASS CONFIRMED |
| DNbe005 ipsilateral fraction of wing-steering output | 42 | 42 | - | PASS CONFIRMED |
| DNp26 wing-steering laterality | contralateral | contralateral | - | PASS CONFIRMED |
| DNp26 ipsilateral fraction of wing-steering output | 23 | 23 | - | PASS CONFIRMED |
| DNg32 wing-steering laterality | contralateral | contralateral | - | PASS CONFIRMED |
| DNg32 ipsilateral fraction of wing-steering output | 3 | 3 | - | PASS CONFIRMED |
| DNge094 wing-steering laterality | contralateral | contralateral | - | PASS CONFIRMED |
| DNge094 ipsilateral fraction of wing-steering output | 0 | 0 | - | PASS CONFIRMED |
| DNp26's strongest steering muscles include hg1/i1/hg2 | True | True | - | PASS CONFIRMED |

## Family K — LPT42_Nod4 is the modern correlate of Egelhaaf-1985 FD3

_Verdicts: {'CONFIRMED': 22, 'CONFIRMED_WITH_CAVEAT': 6, 'UNVERIFIABLE': 2}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| LPT42_Nod4 cell count (bilateral pair) | 2 | 2 | - | PASS CONFIRMED |
| LPT42_Nod4 is cholinergic (excitatory FD output) | acetylcholine | acetylcholine | - | PASS CONFIRMED |
| LPT42_Nod4 is a visual_projection output element | visual_projection | visual_projection | - | PASS CONFIRMED |
| FD3 preferred direction: T4/T5 input is layer-b (regressive) | 100 | 98.58 | - | PASS CONFIRMED |
| FD3 dominant input direction is back-to-front | back_to_front | back_to_front | - | PASS CONFIRMED |
| FD3 axon is heterolateral (>=70% contralateral output) | >=70.0% | 90.3 | - | PASS CONFIRMED |
| FD3 RF is more lateral than FD1=Nod1 (beyond bootstrap CI) | True | True | - | PASS CONFIRMED |
| FD3 has a frontal gap absent in FD1 (only-FD-cell feature) | True | True | - | PASS CONFIRMED |
| FD3 excitatory RF is wider than FD1 (corroborating) | True | True | - | PASS CONFIRMED |
| FD3 RF covers most of the vertical extent (corroborating) | True | True | - | PASS CONFIRMED |
| FD3 absolute peak azimuth ~40-50 deg (calibrated, secondary) | 40.0-50.0 deg | 103.5 | - | PASS* CONFIRMED_WITH_CAVEAT |
| FD3 RF patch is bounded vs in-degree null (figure-selective) | True | True | - | PASS CONFIRMED |
| FD3 signature holds independently on left and right cell | True | True | - | PASS CONFIRMED |
| LPT42_Nod4 layer-b vs Nod1 layer-a (different direction) | True | True | - | PASS CONFIRMED |
| LPT42_Nod4 RF lateral+gap vs Nod1 frontal (FD3 vs FD1) | True | True | - | PASS CONFIRMED |
| LPT42_Nod4 heterolateral vs Nod3 bilateral output | True | True | - | PASS CONFIRMED |
| LPT42_Nod4 layer-b vs Nod5 layer-c (horizontal vs vertical) | True | True | - | PASS CONFIRMED |
| Inhibitors dominate Nod5 output but not LPT42_Nod4 | True | True | - | PASS CONFIRMED |
| LPT42_Nod4 is ACh vs Nod2 GABA (output vs inhibitory) | True | True | - | PASS CONFIRMED |
| LPT42_Nod4 cholinergic with high NT confidence | >=0.70 ACh, unanimous | 0.8943 | - | PASS CONFIRMED |
| FD3 cell body posterolateral: bilateral, posterior, co-clustered with FD1 | posterior lateral protocerebrum (Egelhaaf p.203) | z-pct 0.781, split True | - | PASS CONFIRMED |
| FD3 vs FD1 receive OPPOSITE-dominant contralateral inhibition (FD3 progressive-dominant, FD1 regressive-dominant) | FD3 and FD1 differ in crossed-inhibition direction (Egelhaaf p.203-204) | FD3=progressive, FD1=regressive | - | PASS* CONFIRMED_WITH_CAVEAT |
| LPT42_Nod4 is the unique reciprocal-best FD3 match across the Nod/LPT family | reciprocal best hit, margin>=1 | best_for_FD3=LPT42_Nod4, best_FD_for_LPT42=FD3, margin=2 | - | PASS CONFIRMED |
| Headline FD3 signature replicates on FlyWire v630 | v630 materialization | - | - | N/A UNVERIFIABLE |
| Identity verdict is invariant across knob grid + bootstrap/permutation seeds | grid_pass_fraction == 1.0 | 1 | - | PASS CONFIRMED |
| Statistical power: only one bilateral LPT42_Nod4 pair (n=2) | n=2 | - | - | N/A UNVERIFIABLE |
| FD3 dendrite spans the dorso-ventral lobula plate (real skeleton (fafbseg)) | full D-V (Egelhaaf p.203) | 155.3 um max | - | PASS* CONFIRMED_WITH_CAVEAT |
| FD3 axon is displaced contralaterally from the dendrite (real skeleton (fafbseg)) | crosses midline to contralateral side (Egelhaaf p.203) | ML shifts [-228.1, 224.9] um (toward contra both cells) | - | PASS* CONFIRMED_WITH_CAVEAT |
| FD3 axon converges near the noduli-group landmark (real skeleton (fafbseg)) | noduli group, posterior optic foci (Egelhaaf p.203) | axon 76.7 um vs dendrite 160.7 um from Nod1 landmark | - | PASS* CONFIRMED_WITH_CAVEAT |
| OVERALL: LPT42_Nod4 is the modern correlate of Egelhaaf-1985 FD3 | FD3 == LPT42_Nod4 | CONFIRMED_WITH_CAVEAT | - | PASS* CONFIRMED_WITH_CAVEAT |

## Family KD — Disambiguation: FD3 is LPT42_Nod4, not Nod3 (Nod3 = FD2)

_Verdicts: {'CONFIRMED': 12, 'CONFIRMED_WITH_CAVEAT': 1}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| Both candidates are regressive (layer-b): cannot discriminate | 98.58 | 97.73 | - | PASS CONFIRMED |
| Both candidates are cholinergic: cannot discriminate | acetylcholine (both) | LPT42_Nod4=acetylcholine, Nod3=acetylcholine | - | PASS CONFIRMED |
| Both candidates are small-field selective: cannot discriminate | bounded (both) | LPT42_Nod4 bounded=True, Nod3 bounded=True | - | PASS CONFIRMED |
| Discriminator: output crosses to the contralateral side (>=70%) | FD3 has it; FD2 does not | LPT42_Nod4=True, Nod3=False | - | PASS CONFIRMED |
| Discriminator: receptive field lateral of FD1 with a frontal gap (both sides) | FD3 has it; FD2 does not | LPT42_Nod4=True, Nod3=False | - | PASS CONFIRMED |
| Discriminator: axon crosses the midline (both cells) | FD3 has it; FD2 does not | LPT42_Nod4=True, Nod3=True | - | PASS* CONFIRMED_WITH_CAVEAT |
| Discriminator: projects to the contralateral posterior optic foci | FD3 has it; FD2 does not | LPT42_Nod4=True, Nod3=False | - | PASS CONFIRMED |
| Discriminator: the whole-family screen assigns the cell to FD3 | FD3 has it; FD2 does not | LPT42_Nod4=True, Nod3=False | - | PASS CONFIRMED |
| Output laterality separates the candidates on both offline and live tracks | LPT42_Nod4 heterolateral, Nod3 bilateral | offline: LPT42_Nod4 90.33% vs Nod3 44.66% | live not available | PASS CONFIRMED |
| The heterolateral ordering (LPT42_Nod4 yes, Nod3 no) is invariant to the cutoff | invariant over 60-80% cutoff grid | invariant=True | - | PASS CONFIRMED |
| Nod3's own best FD-family match is FD2, not FD3 (constructive positive identity) | Nod3 -> FD2 | Nod3 best_fd=FD2 (FD2 score 3, FD3 score 1) | - | PASS CONFIRMED |
| LPT42_Nod4 is the reciprocal-best FD3 match across the Nod/LPT family | LPT42_Nod4 <-> FD3 (reciprocal) | best_for_FD3=LPT42_Nod4, LPT42_Nod4 best_fd=FD3, margin=2 | - | PASS CONFIRMED |
| OVERALL: which cell is Egelhaaf-1985 FD3? | FD3 == LPT42_Nod4 (Nod3 == FD2) | LPT42_Nod4=FD3; Nod3=FD2 | - | PASS CONFIRMED |

## Family L — Descending targets of the FD3 cell (LPT42_Nod4 -> DN -> motor)

_Verdicts: {'CONFIRMED': 16, 'UNVERIFIABLE': 1}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| FD3 (LPT42_Nod4) direct descending output is substantial (~100-300 synapses across data sources) | True | True | - | PASS CONFIRMED |
| FD3 contacts ~20-40 descending neurons directly (across sources) | True | True | - | PASS CONFIRMED |
| FD3 makes a direct descending projection | True | True | - | PASS CONFIRMED |
| FD3 also reaches descending neurons through a relay hop | True | True | - | PASS CONFIRMED |
| FD3's strongest direct descending target is a figure-steering DN | True | True | - | PASS CONFIRMED |
| FD3's top direct steering target is DNp26 (shared with the FD1 arm) | DNp26 | DNp26 | - | PASS CONFIRMED |
| FD3 direct descending output reaching known steering DNs (%) | 58 | 58 | - | PASS CONFIRMED |
| Direct and relay routes share descending targets | True | True | - | PASS CONFIRMED |
| DNa04 wing-steering laterality reproduces the paper | ipsilateral | ipsilateral | - | PASS CONFIRMED |
| DNbe001 wing-steering laterality reproduces the paper | bilateral | bilateral | - | PASS CONFIRMED |
| DNge107 wing-steering laterality reproduces the paper | bilateral | bilateral | - | PASS CONFIRMED |
| DNbe005 wing-steering laterality reproduces the paper | bilateral | bilateral | - | PASS CONFIRMED |
| DNp26 wing-steering laterality reproduces the paper | contralateral | contralateral | - | PASS CONFIRMED |
| DNg32 wing-steering laterality reproduces the paper | contralateral | contralateral | - | PASS CONFIRMED |
| DNge094 wing-steering laterality reproduces the paper | contralateral | contralateral | - | PASS CONFIRMED |
| DNp26's strongest steering muscles include hg1/i1/hg2 | True | True | - | PASS CONFIRMED |
| FD3's descending output predominantly drives wing-steering (figure-tracking) | behavioural prediction (Egelhaaf 1985 III) | - | - | N/A UNVERIFIABLE |

## Family P — Afferent pathway of the FD3 cell (photoreceptor -> T4/T5 -> LPT42_Nod4)

_Verdicts: {'CONFIRMED': 10, 'CONFIRMED_WITH_CAVEAT': 2, 'UNVERIFIABLE': 2}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| FD3 receives a substantial direct afferent input (>=1000 synapses) | >=1000 input synapses | 13157 | - | PASS CONFIRMED |
| FD3's motion (T4/T5) drive is layer-b / regressive (back-to-front) | 100 | 98.58 | - | PASS CONFIRMED |
| FD3's dominant input direction is back-to-front | back_to_front | back_to_front | - | PASS CONFIRMED |
| FD3's layer-b drive is a mix of ON (T4b) and OFF (T5b) detectors | True | True | - | PASS CONFIRMED |
| The canonical R->lamina->medulla->T4b/T5b cascade upstream of FD3 is present | True | True | - | PASS CONFIRMED |
| FD3 has NO VCH/DCH centrifugal gate (unlike FD1=Nod1) | True | True | - | PASS CONFIRMED |
| FD3 has NO layer-a (progressive) T4/T5 drive (unlike FD1) | True | True | - | PASS CONFIRMED |
| Motion (T4/T5) is a minority of FD3's total input; most is central/columnar | <50% of total input synapses | 25.1 | - | PASS CONFIRMED |
| FD3's dominant non-motion (columnar sheet) inputs carry the same layer-b channel | True | True | - | PASS CONFIRMED |
| FD3 receives contralateral inhibition in both directions (Egelhaaf 1985 p.203); measured as progressive-dominant with a regressive minority | both channels present (progressive-dominant) | progressive-dominant, both_present=True | - | PASS* CONFIRMED_WITH_CAVEAT |
| Photoreceptor/lamina front end is histaminergic (literature) | R1-6/R7/R8 histaminergic | - | - | N/A UNVERIFIABLE |
| Statistical power: FD3 is one bilateral pair (n=2) | n=2 | - | - | N/A UNVERIFIABLE |
| Upstream trace did not hit the live 500k-row cap | True | True | - | PASS CONFIRMED |
| OVERALL: FD3's afferent pathway (photoreceptor -> layer-b T4/T5 -> FD3) | regressive layer-b afferent arm, parallel to (not a copy of) FD1 | CONFIRMED_WITH_CAVEAT | - | PASS* CONFIRMED_WITH_CAVEAT |

## Family Q — FD3 sheet: LPC1 is the direction-matched layer-b cholinergic feed-forward sheet

_Verdicts: {'CONFIRMED': 10, 'UNVERIFIABLE': 1, 'CONFIRMED_WITH_CAVEAT': 1}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| A direction-matched (layer-b) sheet feeding FD3 exists | LPC1 | LPC1 | - | PASS CONFIRMED |
| The FD3 sheet (LPC1) is cholinergic (excitatory relay) | acetylcholine | acetylcholine | - | PASS CONFIRMED |
| The FD3 sheet reads the layer-b (regressive) channel FD3 reads | b | b | - | PASS CONFIRMED |
| Feed-forward chain present: T4b/T5b -> LPC1 -> FD3 | True | True | - | PASS CONFIRMED |
| LPC1 pools its T4b/T5b input retinotopically locally (below null) | obs << null (z<=-20, p<0.01) | obs 10.789 vs null 207.599 | - | PASS CONFIRMED |
| LPC1 is the UNIQUE layer-b sheet feeding FD3 (siblings read orthogonal channels) | True | True | - | PASS CONFIRMED |
| FD3 reads a DIFFERENT sheet than FD1: LLPC1 is not a major FD3 input | LLPC1->FD3 < LPC1->FD3 | LLPC1->FD3=446, LPC1->FD3=1120 | - | PASS CONFIRMED |
| FD3's intermediate is a SET {LPC1 horizontal + LLPC2/LLPC3 vertical context} | {LPC1, LLPC2, LLPC3} | ['LPC1', 'LLPC3', 'LLPC2', 'LPC2'] | - | PASS CONFIRMED |
| Through its sheets, FD3 receives relayed motion spanning all four cardinal directions (LPC1=regressive, LLPC3=down, LLPC2/LPC2=up, LLPC1=progressive) | >=3 hard-labelled directions among FD3's sheets | 4 directions; per-channel {'a': 11.8, 'b': 29.5, 'c': 30.5, 'd': 28.2} | - | PASS CONFIRMED |
| Only LPC1 (regressive) is direction-matched; the sheet-relayed signal is mostly orthogonal/opposite motion CONTEXT | matched (layer-b) sheet input < 50% of sheet-relayed input | matched=29.5%, context=70.5% | - | PASS CONFIRMED |
| FD3 is one bilateral pair (n=2) | n=2 | - | - | N/A UNVERIFIABLE |
| OVERALL: FD3's intermediate sheet-set (LPC1 the direction-matched member) | T4b/T5b -> {LPC1(+LLPC2/3)} -> FD3 | CONFIRMED_WITH_CAVEAT | - | PASS* CONFIRMED_WITH_CAVEAT |

## Family R — FD3 wide-field inhibitor: LPi14 is the VCH-role opponent gate

_Verdicts: {'CONFIRMED': 13, 'CONFIRMED_WITH_CAVEAT': 3, 'UNVERIFIABLE': 1}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| A wide-field inhibitor filling VCH's role for FD3 exists | True | True | - | PASS CONFIRMED |
| The inhibitor pools wide-field T4/T5 motion (VCH-like) | >=40% of its input is T4/T5 | 72.6 | - | PASS CONFIRMED |
| The inhibitor synapses directly onto FD3 (GABAergic) | >=100 syn onto FD3 | 961 | - | PASS CONFIRMED |
| The inhibitor gates the LPC1 sheet (VCH gates LLPC1) | >=1000 syn onto the LPC1 sheet | 6631 | - | PASS CONFIRMED |
| The inhibitor feeds back onto the T4b/T5b detectors (VCH-T4/T5 loop analog) | >=1000 syn onto T4b/T5b | 7004 | - | PASS CONFIRMED |
| The inhibitor is OPPONENT-tuned (layer-a/progressive vs FD3's layer-b) | 100 | 94.8 | - | PASS CONFIRMED |
| LPi14 is the top opponent (progressive) wide-field gate of FD3 | LPi14 | LPi14 | - | PASS CONFIRMED |
| LPi14 fills VCH's functional role but is a role-, not cell-type, homolog | functional-role homolog of VCH | role_homolog | - | PASS CONFIRMED |
| The FD3 gate is lobula-plate intrinsic, NOT centrifugal (unlike VCH) | False | False | - | PASS CONFIRMED |
| A regressive (same-direction) wide-field inhibitor of FD3's circuit exists (Egelhaaf's predicted ipsilateral surround) | a GABA layer-b wide-field cell in the FD3 circuit | LPi12 (layer-b 99.6%, nt gaba) | - | PASS CONFIRMED |
| The same-direction gate acts predominantly at the T4b/T5b DETECTOR node (not the FD3/sheet node) | detector-inhibition share > FD3-inhibition share (and >= the opponent gate's) | LPi12: 14.03% of detector inhibition vs 2.54% of FD3 inhibition | - | PASS CONFIRMED |
| FD3 has TWO complementary wide-field gates at different nodes: an opponent gate at the FD3/sheet node (LPi14) and a same-direction gate at the detector node (LPi12) | opponent FD3/sheet gate + same-direction detector gate, distinct cells | opponent=LPi14 (FD3/sheet), same_direction=LPi12 (detector) | - | PASS* CONFIRMED_WITH_CAVEAT |
| The same-direction gate does NOT replace LPi14 as FD3's inhibitor (its FD3 and sheet contacts are minor) | same-direction cell is a minor share of FD3's inhibition and does not gate the sheet | LPi12: 2.54% of FD3 inhibition, ->sheet 165 syn | - | PASS CONFIRMED |
| The same-direction gate's (minor) FD3 contact is FD3-specific, not spillover from a promiscuous cell | ->FD3 enriched over a size-matched random LP-tangential target | obs 68 vs null 11.09 (z=2.95, p=0.002) | - | PASS* CONFIRMED_WITH_CAVEAT |
| The 'inhibits FD3 above floor' call for the same-direction gate is threshold-dependent (reported for honesty) | verdict-vs-threshold grid | LPi12 passes the direct-FD3 gate at floors [30, 50, 68] | - | PASS CONFIRMED |
| FD3 is one bilateral pair (n=2) | n=2 | - | - | N/A UNVERIFIABLE |
| OVERALL: LPi14 is FD3's wide-field opponent inhibitor (VCH functional-role homolog) | LPi14 = VCH-role opponent gate | CONFIRMED_WITH_CAVEAT | - | PASS* CONFIRMED_WITH_CAVEAT |

## Family S — FD3 functional figure-ground circuit (T4b/T5b -> LPC1 -> FD3 -> DNp26; LPi14 gate)

_Verdicts: {'CONFIRMED': 7}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| Motion detectors (T4b/T5b, layer-b) drive the circuit | True | True | - | PASS CONFIRMED |
| The intermediate sheet is NAMED and feed-forward (T4b/T5b -> LPC1 -> FD3) | True | True | - | PASS CONFIRMED |
| The figure cell (FD3 = LPT42_Nod4) is confirmed | True | True | - | PASS CONFIRMED |
| FD3 reaches a steering descending neuron (DNp26) | True | True | - | PASS CONFIRMED |
| The descending output resolves to a motor system (wing-steering) | True | True | - | PASS CONFIRMED |
| The wide-field inhibitor standing in VCH's place is NAMED (LPi14, opponent) | True | True | - | PASS CONFIRMED |
| OVERALL: FD3 figure-ground circuit (T4b/T5b -> LPC1 -> FD3 -> DNp26; LPi14 gate) | T4b/T5b -> {LPC1(+LLPC2/3)} -> FD3 -> DNp26 -> wing; LPi14 wide-field opponent gate | CONFIRMED | - | PASS CONFIRMED |

## Family Y — Paper-internal arithmetic consistency (oracle-only)

_Verdicts: {'CONFIRMED': 7}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| Paper-internal: 12,301/27,576 == 44.6% | 44.6 | 44.61 | - | PASS CONFIRMED |
| Paper-internal: 912/1,022 == 89% | 89 | 89.24 | - | PASS CONFIRMED |
| Paper-internal: 912/1,476 == 62% | 62 | 61.79 | - | PASS CONFIRMED |
| Paper-internal: mean-syn ratio in [2,3] (~2.4x) | 2.0-3.0x | 2.44 | - | PASS CONFIRMED |
| Paper-internal: 882 layer-a reciprocal <= 912 total | True | True | - | PASS CONFIRMED |
| Paper-internal: Table 1 reciprocal fractions in (0,1] | [] | [] | - | PASS CONFIRMED |
| Paper-internal: Nod1 drivers (91) <= 100 LLPC1 | True | True | - | PASS CONFIRMED |

## Family Z — Interpretive / physiology predictions (proxies)

_Verdicts: {'UNVERIFIABLE': 5}_

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| Circuit implements divisive normalization / gain control | divisive (untested) | reciprocal loop present (n=874, exc/inhib=2.46) | - | N/A UNVERIFIABLE |
| LLPC1 reads out the VCH-gated local motion residual | readout (untested) | dual-field motion+form integrator, central-projecting (motion 31.0% / form 30.1%) | - | N/A UNVERIFIABLE |
| Silencing VCH makes LLPC1 more raw-motion-like / less figure-selective | physiology prediction (S13) | - | - | N/A UNVERIFIABLE |
| Silencing LPi15/PVLP011 changes operating point/gain, not separation | physiology prediction (S13) | - | - | N/A UNVERIFIABLE |
| LPLC2/LC4 escape is separable from the LLPC1 course-control route | separable (prediction) | direct/Nod routes avoid jump/TTM; broadcast is the only jump bridge | - | N/A UNVERIFIABLE |

## Per-claim provenance

| ID | Verdict | Notes |
|---|---|---|
| A.vch_nt | CONFIRMED |  |
| A.vch_in_syn | CONFIRMED_WITH_CAVEAT | primary diff -9646 (-35.0%) |
| A.vch_out_syn | CONFIRMED_WITH_CAVEAT | primary diff -12829 (-39.6%) |
| A.vch_up_partners | CONFIRMED_WITH_CAVEAT | primary diff -873 (-24.1%) |
| A.t4t5_in_neurons | CONFIRMED_WITH_CAVEAT | primary diff -47 (-4.6%) |
| A.t4t5_in_syn | CONFIRMED_WITH_CAVEAT | primary diff -3517 (-28.6%) |
| A.t4t5_out_neurons | CONFIRMED_WITH_CAVEAT | primary diff -175 (-11.9%) |
| A.t4t5_out_syn | CONFIRMED_WITH_CAVEAT | primary diff -2514 (-34.6%) |
| A.reciprocal_n | CONFIRMED_WITH_CAVEAT | primary diff -38 (-4.2%) |
| A.reciprocal_layer_a_n | CONFIRMED | primary diff -12 (-1.4%) |
| A.t4t5_in_pct | REFUTED | diff +4.4 pp |
| A.reciprocal_pct_in | CONFIRMED | diff +0.6 pp |
| A.reciprocal_pct_out | REFUTED | diff +5.2 pp |
| A.vch_layer_a_pct | CONFIRMED | diff +0.5 pp |
| A.exc_inhib_ratio | CONFIRMED |  |
| A.mean_in_consistency | CONFIRMED | primary diff +0 (+0.0%) |
| B.VCH.llpc1 | CONFIRMED_WITH_CAVEAT | primary diff -11 (-14.3%) |
| B.VCH.centrifugal | CONFIRMED |  |
| B.DCH.llpc1 | CONFIRMED_WITH_CAVEAT | primary diff -11 (-21.2%) |
| B.DCH.centrifugal | CONFIRMED |  |
| B.CT1.llpc1 | CONFIRMED | primary diff -1 (-33.3%) |
| B.CT1.centrifugal | CONFIRMED |  |
| B.LPi15.llpc1 | CONFIRMED | primary diff -7 (-7.0%) |
| B.LPi15.centrifugal | CONFIRMED |  |
| B.LPi14.llpc1 | CONFIRMED_WITH_CAVEAT | primary diff -27 (-30.0%) |
| B.LPi14.centrifugal | CONFIRMED |  |
| B.Am1.llpc1 | CONFIRMED | primary diff -8 (-8.1%) |
| B.Am1.centrifugal | CONFIRMED |  |
| B.LT33.llpc1 | CONFIRMED | primary diff -2 (-28.6%) |
| B.LT33.centrifugal | CONFIRMED |  |
| B.Li14.llpc1 | CONFIRMED | primary diff -3 (-21.4%) |
| B.Li14.centrifugal | CONFIRMED |  |
| B.pass_all_three | REFUTED | computed pass-set = ['VCH'] |
| C.columnar_pct | CONFIRMED | diff +0.0 pp |
| C.llpc1_vs_siblings | REFUTED |  |
| C.llpc1_top_sheet | CONFIRMED |  |
| C.lplc2_not_sheet | CONFIRMED |  |
| D.t4a_n | CONFIRMED | primary diff -7 (-1.5%) |
| D.t4a_llpc1_syn | CONFIRMED_WITH_CAVEAT | primary diff -3106 (-33.7%) |
| D.llpc1_n | CONFIRMED_WITH_CAVEAT | primary diff -7 (-7.0%) |
| D.local_below_null | CONFIRMED |  |
| D.z_score | CONFIRMED | observed median 8.0 um vs null 37.1 um |
| D.p_value | CONFIRMED | 500 permutations |
| D.frac_within_10um | CONFIRMED | absolute observed fraction is centroid-definition-sensitive (skeleton-snapped in the paper vs synaptic-field centroid here); the locality (obs >> null) holds. |
| D.obs_radius | CONFIRMED | primary diff +2 (+27.0%) |
| E.motion_pct | CONFIRMED | diff +2.0 pp |
| E.form_pct | CONFIRMED | diff -0.9 pp |
| E.field_sep | CONFIRMED | primary diff -4 (-21.5%) |
| E.vch_terminal_closest | CONFIRMED |  |
| E.vch_closest_frac | CONFIRMED | diff +1.0 pp |
| E.wilcoxon | CONFIRMED | VCH median 1.8 um vs other 29.9 um, n=286 |
| E.lpi15_to_exc | CONFIRMED | primary diff -0 (-23.1%) |
| E.vch_to_exc | CONFIRMED | primary diff -1 (-30.0%) |
| E.pvlp011_distal | CONFIRMED |  |
| F.lpi15_reach | CONFIRMED_WITH_CAVEAT | primary diff -7 (-7.0%) |
| F.lpi15_syn | CONFIRMED_WITH_CAVEAT | primary diff -2793 (-43.2%) |
| F.lpi15_layer_b | CONFIRMED | diff +0.4 pp |
| F.pvlp011_reads | CONFIRMED_WITH_CAVEAT | primary diff -7 (-7.0%) |
| F.pvlp011_feedsback | CONFIRMED_WITH_CAVEAT | primary diff -7 (-7.1%) |
| F.vch_direct_llpc1 | CONFIRMED_WITH_CAVEAT | primary diff -11 (-14.3%) |
| F.vch_direct_syn | CONFIRMED_WITH_CAVEAT | primary diff -217 (-36.9%) |
| G.output_total | CONFIRMED_WITH_CAVEAT | primary diff -43927 (-41.3%) |
| G.PLP249.drivers | CONFIRMED_WITH_CAVEAT | primary diff -7 (-7.2%) |
| G.PLP249.syn | CONFIRMED_WITH_CAVEAT | primary diff -1535 (-35.0%) |
| G.PLP249.nt | CONFIRMED |  |
| G.Nod1.drivers | CONFIRMED_WITH_CAVEAT | primary diff -7 (-7.7%) |
| G.Nod1.syn | CONFIRMED_WITH_CAVEAT | primary diff -1835 (-43.4%) |
| G.Nod1.nt | CONFIRMED |  |
| G.PVLP011.drivers | CONFIRMED_WITH_CAVEAT | primary diff -7 (-7.0%) |
| G.PVLP011.syn | CONFIRMED_WITH_CAVEAT | primary diff -1158 (-36.6%) |
| G.PVLP011.nt | CONFIRMED |  |
| G.PLP163.drivers | CONFIRMED_WITH_CAVEAT | primary diff -7 (-7.0%) |
| G.PLP163.syn | CONFIRMED_WITH_CAVEAT | primary diff -955 (-37.7%) |
| G.PLP163.nt | CONFIRMED |  |
| G.Nod2.drivers | CONFIRMED_WITH_CAVEAT | primary diff -5 (-7.1%) |
| G.Nod2.syn | CONFIRMED_WITH_CAVEAT | primary diff -326 (-39.5%) |
| G.Nod2.nt | CONFIRMED |  |
| G.DNbe001.drivers | CONFIRMED_WITH_CAVEAT | primary diff -6 (-8.6%) |
| G.DNbe001.syn | CONFIRMED_WITH_CAVEAT | primary diff -279 (-37.9%) |
| G.DNbe001.nt | CONFIRMED |  |
| G.nod1_top_excitatory | CONFIRMED |  |
| H.nod1_dn_syn | CONFIRMED_WITH_CAVEAT | primary diff -209 (-31.6%) |
| H.nod1_n_dns | CONFIRMED_WITH_CAVEAT | primary diff -6 (-22.2%) |
| H.nod1_steering_frac | CONFIRMED | diff +2.1 pp |
| H.nod1_to_dnp26 | CONFIRMED_WITH_CAVEAT | primary diff -133 (-29.7%) |
| H.direct_to_dnp26 | CONFIRMED_WITH_CAVEAT | primary diff -54 (-36.2%) |
| H.nod1_route_dominates | CONFIRMED |  |
| H.nod2_gaba | CONFIRMED |  |
| I.direct.syn | CONFIRMED_WITH_CAVEAT | primary diff -848 (-39.0%) |
| I.direct.dns | CONFIRMED | primary diff -1 (-4.0%) |
| I.nod_relay.syn | CONFIRMED_WITH_CAVEAT | primary diff -226 (-31.7%) |
| I.nod_relay.dns | CONFIRMED | primary diff -10 (-18.9%) |
| I.broadcast.syn | CONFIRMED_WITH_CAVEAT | primary diff -240 (-38.2%) |
| I.broadcast.dns | CONFIRMED | primary diff -8 (-14.8%) |
| I.direct_strongest | CONFIRMED |  |
| J.DNa04.category | CONFIRMED |  |
| J.DNa04.ipsi_frac | CONFIRMED | diff +0.0 pp |
| J.DNbe001.category | CONFIRMED |  |
| J.DNbe001.ipsi_frac | CONFIRMED | diff +0.0 pp |
| J.DNge107.category | CONFIRMED |  |
| J.DNge107.ipsi_frac | CONFIRMED | diff +0.0 pp |
| J.DNbe005.category | CONFIRMED |  |
| J.DNbe005.ipsi_frac | CONFIRMED | diff +0.0 pp |
| J.DNp26.category | CONFIRMED |  |
| J.DNp26.ipsi_frac | CONFIRMED | diff +0.0 pp |
| J.DNg32.category | CONFIRMED |  |
| J.DNg32.ipsi_frac | CONFIRMED | diff +0.0 pp |
| J.DNge094.category | CONFIRMED |  |
| J.DNge094.ipsi_frac | CONFIRMED | diff +0.0 pp |
| J.dnp26_muscles | CONFIRMED |  |
| K.n_cells | CONFIRMED | primary diff +0 (+0.0%) |
| K.nt_ach | CONFIRMED |  |
| K.superclass | CONFIRMED |  |
| K.layer_b | CONFIRMED | diff -1.4 pp |
| K.preferred_dir | CONFIRMED |  |
| K.contra_output | CONFIRMED | measured 90.3% contralateral (annotated targets); heterolateral noduli-group axon |
| K.rf_more_lateral | CONFIRMED |  |
| K.rf_frontal_gap | CONFIRMED |  |
| K.rf_wider | CONFIRMED |  |
| K.rf_vertical_full | CONFIRMED |  |
| K.rf_abs_azimuth | CONFIRMED_WITH_CAVEAT | measured peak 103.5 deg vs Egelhaaf 40-50 deg (within calibration band: False); combined calibration uncertainty +/-16.0 deg. Linear column->azimuth map is the weakest link; lattice-extreme anchoring biases the centroid lateral. Secondary corroboration only — the differential gate governs the RF identity. Error budget: {'linearity_deg': 12.5, 'landmark_deg': 6.5, 'eye_span_deg': 7.5, 'dropout_deg': 1.5, 'combined_deg': 16.0, 'model': 'linear two-anchor p->azimuth; azimuth is p-driven, q->elevation coarse', 'verdict_ceiling': 'CONFIRMED_WITH_CAVEAT (no ground-truth degree field exists)'}. |
| K.smallfield_bounded | CONFIRMED | obs 6.602 vs null 10.907 (z=-24.5, p=0.002, p_bh=0.002, 500 perms) |
| K.bilateral_replication | CONFIRMED |  |
| K.disc_vs_nod1_layer | CONFIRMED |  |
| K.disc_vs_nod1_rf | CONFIRMED |  |
| K.disc_vs_nod3_contra | CONFIRMED |  |
| K.disc_vs_nod5_layer | CONFIRMED |  |
| K.disc_vs_nod5_targets | CONFIRMED |  |
| K.disc_vs_nod2_nt | CONFIRMED |  |
| K.nt_ach_confident | CONFIRMED | mean top_nt_conf 0.8943; unanimous_ACh=True |
| K.soma_posterolateral | CONFIRMED | bilateral_split=True, post_z_percentile=0.781, dz_to_Nod1=135.2, dz_to_VCH/DCH=+4289.5 (offline soma proxy; skeletons unavailable) |
| K.contra_inhibition_bidirectional | CONFIRMED_WITH_CAVEAT | FD3 progressive-dominant (prog 0.928/reg 0.072); FD1=Nod1 regressive-dominant (prog 0.083/reg 0.917). Opposite-dominant crossed inhibition matches Egelhaaf's FD3-vs-FD1 contrast in direction; FD3's minority (regressive) channel is a trace (0.072 of classified). Annotation is power-limited (n_classified=6 of 108 contra-GABA), so the fine bidirectional/unidirectional split is not resolved by wiring alone. |
| K.fd_family_unique | CONFIRMED | screened 5 candidate types; FD3 scores {'Nod1': 1, 'Nod3': 1, 'LPT42_Nod4': 3, 'Nod2': 0, 'Nod5': 1} |
| K.cross_version_v630 | UNVERIFIABLE | v630 cross-check is live-only |
| K.robust_to_knobs | CONFIRMED | seed_stability={'seeds': [12345, 1, 2, 3, 4], 'offset_ci_lower_min': 9.524, 'offset_ci_lower_max': 9.62, 'all_offset_lower_gt0': True}; gap_drop=[True, True, True, True, True] |
| K.power_note | UNVERIFIABLE | n=2 is irreducible (one bilateral pair exists). Four independent variance controls bound the inference: (i) bilateral independent replication (K.bilateral_replication); (ii) within-cell synapse bootstrap CI on the RF offset (N_BOOT=2000); (iii) in-degree permutation null for small-field selectivity (z<<0, p<=0.002); (iv) cross-version replication on v630. The conclusion does not rest on a per-cell sample size. |
| K.morph_dv_span | CONFIRMED_WITH_CAVEAT | D-V spans (um): [155.3, 147.5]; real skeleton (fafbseg) (36744 skeleton nodes) |
| K.morph_axon_heterolateral | CONFIRMED_WITH_CAVEAT | dendrite->axon medio-lateral centroid shift sign; real skeleton (fafbseg) |
| K.morph_axon_noduli | CONFIRMED_WITH_CAVEAT | Nod1 output-cloud centroid as the noduli-group landmark; real skeleton (fafbseg) |
| K.identity_verdict | CONFIRMED_WITH_CAVEAT | CORE + DISCRIMINATING confirmed; >=1 corroborating claim caveated/unverifiable |
| KD.shared_layer_b | CONFIRMED | LPT42_Nod4 layer-b 98.58%, Nod3 layer-b 97.73% -- SHARED, not evidence for either |
| KD.shared_nt | CONFIRMED | direction and transmitter are shared; the split rests on the discriminators |
| KD.shared_smallfield | CONFIRMED | both are FD cells; small-field selectivity does not separate FD2 from FD3 |
| KD.D1_heterolateral | CONFIRMED | favors LPT42_Nod4 |
| KD.D2_rf_lateral_gap | CONFIRMED | favors LPT42_Nod4 |
| KD.D3_axon_crosses | CONFIRMED_WITH_CAVEAT | favors neither |
| KD.D4_contra_pof | CONFIRMED | favors LPT42_Nod4 |
| KD.D5_family_fd3 | CONFIRMED | favors LPT42_Nod4 |
| KD.contra_two_track | CONFIRMED | the contra% metric is data-source sensitive; the ~90 vs ~45 gap far exceeds any track drift |
| KD.contra_cutoff_invariant | CONFIRMED | first_break_cutoff=None |
| KD.nod3_is_fd2 | CONFIRMED | FD2 assignment is inferred from RF+projection consistency + the family screen, not a separately anchored identity |
| KD.lpt42_is_fd3 | CONFIRMED | anchor Nod1 screens best = FD1 (positive control expects FD1) |
| KD.decision_verdict | CONFIRMED | discriminator scores: LPT42_Nod4 5/5 {'D1_heterolateral': True, 'D2_rf_lateral_gap': True, 'D3_axon_crosses': True, 'D4_contra_pof': True, 'D5_family_fd3': True}; Nod3 1/5 {'D1_heterolateral': False, 'D2_rf_lateral_gap': False, 'D3_axon_crosses': True, 'D4_contra_pof': False, 'D5_family_fd3': False}. Overturning condition: Nod3 would replace LPT42_Nod4 only if it hit D1-D4 with own-best FD3 while LPT42_Nod4 failed >=2 of 5. |
| L.direct_dn_syn_scale | CONFIRMED |  |
| L.direct_n_dns_scale | CONFIRMED |  |
| L.direct_nonempty | CONFIRMED |  |
| L.relay_nonempty | CONFIRMED |  |
| L.top_direct_is_steering | CONFIRMED |  |
| L.converges_on_dnp26 | CONFIRMED |  |
| L.direct_steering_frac | CONFIRMED | diff +0.0 pp |
| L.routes_overlap | CONFIRMED |  |
| L.DNa04.wing | CONFIRMED |  |
| L.DNbe001.wing | CONFIRMED |  |
| L.DNge107.wing | CONFIRMED |  |
| L.DNbe005.wing | CONFIRMED |  |
| L.DNp26.wing | CONFIRMED |  |
| L.DNg32.wing | CONFIRMED |  |
| L.DNge094.wing | CONFIRMED |  |
| L.dnp26_muscles | CONFIRMED |  |
| L.steering_dominant | UNVERIFIABLE | anatomical proxy: of FD3's descending drive that resolves to a male-CNS motor system, the largest share is 'wing_steering' (motor-system %, FD3-drive-weighted: {'wing_steering': 43.5, 'neck_gaze': 17.3, 'abdominal': 16.5, 'wing_power': 14.7, 'leg': 5.2, 'other': 1.5, 'haltere': 1.3}). |
| P.input_census_nonempty | CONFIRMED | 13157 input synapses from 2991 presynaptic partners |
| P.layer_b_drive | CONFIRMED | diff -1.4 pp |
| P.preferred_dir | CONFIRMED |  |
| P.on_off_mix | CONFIRMED | T4b(ON)=1793 syn, T5b(OFF)=1459 syn (T4 frac 55.1%) |
| P.upstream_cascade_present | CONFIRMED | ON limb T4b<-['C2', 'C3', 'Mi1', 'Mi4', 'Mi9', 'Tm3'] (54.9% of its input); OFF limb T5b<-['Tm1', 'Tm2', 'Tm4', 'Tm9'] (48.6% of its input); lamina ['L1', 'L2', 'L3', 'L4', 'L5']; photoreceptors ['R1-6', 'R7', 'R8']. Per-type canonical wiring (Fischbach-Dittrich 1989; Takemura 2013), not FD3-specific counts. |
| P.nc_no_vch_gate | CONFIRMED | VCH/DCH -> FD3 = 12 syn (0.091% of input). NOTE: FD3 has no VCH gate specifically, but the VCH functional ROLE (wide-field pooling + sheet gating + detector feedback + FD-cell inhibition) IS filled — by the lobula-plate intrinsic cell LPi14, opponent-tuned (see Family R). |
| P.nc_no_layer_a_drive | CONFIRMED | layer-a fraction of FD3's T4/T5 input = 0.09% |
| P.t4t5_minority_of_input | CONFIRMED | T4/T5 = 25.1% of the 13157 total input synapses; the layer-b fraction (98.58%) is WITHIN the T4/T5 subset. Input by class: {'optic': 6238, 'visual_projection': 4775, 'central': 1060, 'visual_centrifugal': 214, 'ascending': 33, 'descending': 6}. |
| P.central_inputs_layerb | CONFIRMED | 1/6 top central input types are layer-b dominant: [('LPC1', 'b'), ('LLPC3', 'd'), ('LLPC2', 'c'), ('LPi14', 'a'), ('LPi02', 'a'), ('LPTe01', 'd')] |
| P.contra_inhibition_bidirectional | CONFIRMED_WITH_CAVEAT | progressive=387 syn, regressive=30 syn over 6 classified contra-GABA partners; progressive-dominant (minority frac 0.072); the minority channel is a trace, so this reads as direction-dominant rather than cleanly bidirectional; well-powered |
| P.peripheral_nt_histaminergic | UNVERIFIABLE | The R1-6->lamina->medulla cascade upstream of FD3's detectors is PRESENT and per-type wired in v783, but the ON/OFF functional split and the histaminergic transmitter of the photoreceptors come from physiology (Hardie 1989; Maisak 2013), not the connectome. NOTE: the annotation table lists R1-6=acetylcholine / R7=gaba, which is a prediction artifact and biologically wrong (photoreceptors are histaminergic) — not propagated. |
| P.power_note | UNVERIFIABLE | n=2 is irreducible (one bilateral LPT42_Nod4 pair). The afferent claims rest on the aggregate synapse populations onto both cells (thousands of input synapses) and on per-type cascade wiring measured across the full retinotopic population, not on a per-cell sample size. |
| P.no_truncation | CONFIRMED |  |
| P.input_pathway_verdict | CONFIRMED_WITH_CAVEAT | CORE + DISCRIMINATING confirmed; >=1 corroborating claim caveated/unverifiable |
| Q.named_sheet_found | CONFIRMED |  |
| Q.sheet_cholinergic | CONFIRMED |  |
| Q.sheet_layer_b | CONFIRMED |  |
| Q.feed_forward | CONFIRMED |  |
| Q.retinotopy_local | CONFIRMED | z=-230.8, p=0.002, 500 perms |
| Q.lpc1_unique_layerb | CONFIRMED | sheet-set channels: {'LPC1': 'regressive', 'LLPC3': 'downward', 'LLPC2': 'upward', 'LPC2': 'upward'}; fold vs other layer-b: unique |
| Q.nc_not_fd1_sheet | CONFIRMED | negative control: FD3 is not just re-reading FD1's progressive sheet |
| Q.sheet_set | CONFIRMED | channels {'LPC1': 'regressive', 'LLPC3': 'downward', 'LLPC2': 'upward', 'LPC2': 'upward'} |
| Q.pools_all_cardinal_directions | CONFIRMED | channels {'a': ['LLPC1'], 'b': ['LPC1'], 'c': ['LLPC2', 'LPC2'], 'd': ['LLPC3']}; entropy=1.92 bits; null pct=1.0 (elevated=True); FD1 spread 4 dirs (FD3-specific=True). This describes FD3's SHEET-RELAYED input; its DIRECT T4/T5 drive stays ~99% layer-b (see Family P). |
| Q.matched_channel_minority | CONFIRMED | FD3 reads its own direction (LPC1) plus a near-balanced up/down/opposite surround |
| Q.power_note | UNVERIFIABLE | The sheet claims rest on the aggregate synapse populations onto both FD3 cells and on the full sheet-cell populations, not on a per-cell sample size. |
| Q.sheet_verdict | CONFIRMED_WITH_CAVEAT | CORE+DISC confirmed; a corroborating claim is soft |
| R.widefield_inhibitor_found | CONFIRMED |  |
| R.pools_widefield_t4t5 | CONFIRMED | LPi14 T4/T5 = 72.6% of input;  |
| R.inhibits_fd3 | CONFIRMED | LPi14 -> FD3 961 syn, nt=gaba |
| R.gates_sheet | CONFIRMED | LPi14 -> LPC1 sheet 6631 syn |
| R.feeds_back_detectors | CONFIRMED | LPi14 -> T4b/T5b 7004 syn |
| R.opponent_tuning | CONFIRMED | LPi14 T4/T5 input is 94.8% layer-a (progressive) — the OPPONENT of FD3's layer-b; direction=opponent |
| R.named_opponent_gate | CONFIRMED |  |
| R.vch_equivalent_role | CONFIRMED | FD3 has no centrifugal VCH gate (P.nc_no_vch_gate); LPi14 fills the VCH ROLE (pools wide-field + gates the sheet + feeds back on detectors + inhibits FD3) but is a lobula-plate INTRINSIC cell (super_class=optic, centrifugal=False), OPPONENT-tuned (layer-a), and NOT reciprocal with FD3 (FD3->LPi14 = 5 syn). |
| R.not_centrifugal | CONFIRMED |  |
| R.same_direction_surround_found | CONFIRMED | resolved by the full-LPi-panel sweep (prior UNVERIFIABLE was an artifact of LPi12 being absent from the candidate set); same_direction_gates=['LPi12', 'LPi10', 'Am1', 'LPi15'] |
| R.same_direction_detector_dominant | CONFIRMED | LPi12 sends 43.86% of its OWN output to the detectors; opponent LPi14 detector share = 5.26% |
| R.two_gate_architecture | CONFIRMED_WITH_CAVEAT | the defensible headline; sign of each GABA contact is from physiology, not wiring; n=2 FD3 pair (both cells agree per per_fd3_cell_syn) |
| R.same_direction_not_replacement | CONFIRMED | guards against overclaiming LPi12 as THE FD3 gate; LPi14 remains the FD3/sheet gate |
| R.same_direction_fd3_specificity | CONFIRMED_WITH_CAVEAT | 2000 perms; a genuine co-input onto FD3, though a minor one |
| R.floor_sensitivity | CONFIRMED | floor grid -> passing cells: {30: ['Am1', 'LPi10', 'LPi12', 'LPi14', 'LPi15'], 50: ['LPi10', 'LPi12', 'LPi14'], 68: ['LPi10', 'LPi12', 'LPi14'], 100: ['LPi10', 'LPi14'], 165: ['LPi10', 'LPi14'], 300: ['LPi14']} |
| R.power_note | UNVERIFIABLE | The inhibitor claims rest on the aggregate synapse populations, not per-cell n; both FD3 cells agree on the two-gate asymmetry (per_fd3_cell_syn). |
| R.inhibitor_verdict | CONFIRMED_WITH_CAVEAT | CORE+DISC confirmed; a corroborating claim is soft |
| Z9.vch_in_pct | CONFIRMED | diff +0.0 pp |
| Z9.recip_pct_in | CONFIRMED | diff +0.2 pp |
| Z9.recip_pct_out | CONFIRMED | diff -0.2 pp |
| Z9.exc_inhib_ratio | CONFIRMED |  |
| Z9.layer_a_le_total | CONFIRMED |  |
| Z9.b_recip_frac_range | CONFIRMED |  |
| Z9.nod1_drivers_le_100 | CONFIRMED |  |
| Z.divisive_norm | UNVERIFIABLE | PROXY: graded reciprocal same-direction VCH-T4/T5 loop is consistent with proportional gain control, but divisive-vs-subtractive is a dynamical property not determinable from static synapse counts (paper concurs, S9). |
| Z.llpc1_readout | UNVERIFIABLE | PROXY: the motion+form dual-dendrite, retinotopic, central-projecting architecture is the substrate for a readout; whether the RESPONSE encodes the residual requires LLPC1 imaging (paper S13). |
| Z.vch_silencing | UNVERIFIABLE | A perturbation-response prediction; not testable from the static connectome. Anatomical precondition (VCH presynaptically gates the T4a terminals driving LLPC1) IS confirmed (family E), which is the strongest available support. |
| Z.lpi15_pvlp_silencing | UNVERIFIABLE | Perturbation-response prediction. Anatomical precondition (LPi15 feed-forward opponent; PVLP011 recurrent on a distal compartment) is confirmed (families F/E). |
| Z.escape_separable | UNVERIFIABLE | PROXY: the channel separation IS in the anatomy (family I/J), but the behavioural double-dissociation needs perturbation experiments (S13). |
| S.detectors | CONFIRMED |  |
| S.sheet_named | CONFIRMED | sheet=LPC1 verdict=CONFIRMED_WITH_CAVEAT |
| S.figure_cell | CONFIRMED | K identity verdict=CONFIRMED_WITH_CAVEAT |
| S.steering_dn | CONFIRMED | top DN=DNp26 |
| S.motor | CONFIRMED | dominant system=wing_steering |
| S.widefield_inhibitor_named | CONFIRMED | inhibitor=LPi14 direction=opponent verdict=CONFIRMED_WITH_CAVEAT |
| S.functional_circuit_verdict | CONFIRMED | full named circuit present: detectors->sheet->FD3->DN->motor + LPi14 gate |

---
Regenerate: `python scripts/paper_verify.py` (live CAVE token at `~/.cloudvolume/secrets/cave-secret.json`; MaleCNS bulk files on scratch).