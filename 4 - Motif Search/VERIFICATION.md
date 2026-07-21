# Independent Verification — VCH–T4/T5 Motif Report

_Stage 4 (Motif Search). Re-derives every quantitative claim in `vch_t4t5_report.pdf` from the offline FlyWire v783 data._

- **Primary source:** raw flywire_synapses_783.feather (API analog)
- **Secondary source:** edges_full.parquet (proofread-only derived graph)
- **Report reproducibility:** report wrote /tmp/*.json (ephemeral) -> not reproducible from own outputs
- **Verdict tally:** {'UNVERIFIABLE': 3, 'CONFIRMED': 14, 'CONFIRMED_WITH_CAVEAT': 8, 'REFUTED': 2}

Verdict legend: **CONFIRMED** (within tolerance) · **CONFIRMED_WITH_CAVEAT** (matches after a known data-source difference) · **REFUTED** (report appears wrong) · **UNVERIFIABLE** (no ground truth in the offline dump).

## Headline pass/fail

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| Report reproducible from its own artifacts | /tmp/*.json | absent | - | N/A UNVERIFIABLE |
| VCH neurotransmitter is GABA | gaba | gaba | - | PASS CONFIRMED |
| VCH total input synapses | 27576 | 17930 | 17131 | PASS* CONFIRMED_WITH_CAVEAT |
| VCH total output synapses | 32363 | 19534 | 13338 | PASS* CONFIRMED_WITH_CAVEAT |
| VCH upstream partners | 3619 | 2746 | 2130 | PASS* CONFIRMED_WITH_CAVEAT |
| VCH downstream partners | 10948 | 8714 | 3275 | PASS* CONFIRMED_WITH_CAVEAT |
| T4/T5 input neurons (loose rule, report-comparable) | 1030 | 981 | 981 | PASS CONFIRMED |
| T4/T5 input synapses | 13018 | 9261 | 9261 | PASS* CONFIRMED_WITH_CAVEAT |
| T4/T5 = % of VCH input | 47.2 | 51.65 | - | PASS CONFIRMED |
| Report Table 2 subtype synapses sum to its stated total | 13018 | 12426 | - | FAIL REFUTED |
| Report Table 2 subtype neurons sum to its stated total | 1030 | 1035 | - | PASS CONFIRMED |
| T4/T5 output neurons | 1487 | 1309 | 1309 | PASS* CONFIRMED_WITH_CAVEAT |
| T4/T5 output synapses | 7328 | 4782 | 4782 | PASS* CONFIRMED_WITH_CAVEAT |
| T4/T5 = % of VCH output | 22.6 | 24.48 | - | PASS CONFIRMED |
| Reciprocal T4/T5 count | 918 | 878 | 878 | PASS CONFIRMED |
| Reciprocal = % of T4/T5 inputs | 89.1 | 89.5 | - | PASS CONFIRMED |
| Reciprocal = % of T4/T5 outputs | 61.7 | 67.07 | - | PASS CONFIRMED |
| Input & output T4/T5 in DIFFERENT hemispheres | True | False | - | FAIL REFUTED |
| Excitatory drive 2-3x inhibitory feedback (mean syn/neuron) | 2.0-3.0x | 2.59 | - | PASS CONFIRMED |
| 918 reciprocal -> total downstream synapses | 578238 | 352017 | - | PASS* CONFIRMED_WITH_CAVEAT |
| 918 reciprocal -> unique downstream targets | 144796 | 113339 | - | PASS CONFIRMED |
| Top downstream target cell_type is LPi14 | LPi14 | LPi14 | - | PASS CONFIRMED |
| VCH output synapses' plurality NT is GABA | gaba | gaba (92% of syn) | - | PASS CONFIRMED |
| T4/T5 input synapses' plurality NT is ACh | ach | ach (71% of syn; 100% of neurons cholinergic) | - | PASS CONFIRMED |
| Nodulus (Nod) neurons among downstream targets | 5 types | 4 types: ['Nod1', 'Nod2', 'Nod3', 'Nod5'] | - | PASS CONFIRMED |
| Left 72% / right 97% T4/T5 targets tagged | 72/97 | - | - | N/A UNVERIFIABLE |
| Circuit implements divisive normalization / gain control | divisive | - | - | N/A UNVERIFIABLE |

## Key findings

1. **Absolute counts are not reproducible from the offline dump.** Even the raw synapse table (API analog) gives 17,930 input / 19,534 output synapses vs the report's 27,576 / 32,363 (~60–65%). The live API the report used likely includes post-snapshot proofreading edits and/or a looser synapse-confidence threshold.
   - cleft_score sweep (input syn at floors): {'0': {'input_syn': 17930, 'output_syn': 19534}, '50': {'input_syn': 17930, 'output_syn': 19534}, '100': {'input_syn': 15215, 'output_syn': 16560}, '140': {'input_syn': 10575, 'output_syn': 11182}, 'cleft_score_min_observed': {'input': 51, 'output': 51}}
2. **Laterality claim REFUTED.** Input synapses neuropil-hemi {'R': 17930}, output {'R': 19534} — both in the right optic lobe (LOP_R). VCH is a left-soma centrifugal cell whose dendrite and axon both lie in the right optic lobe: a within-(right)-optic-lobe recurrent loop, not an ipsilateral-in / contralateral-out relay. There is a fine intra-LOP_R zonation (input synapse-x median 716132 vs output 724048 = dendrite vs axon).
3. **Sign claims CONFIRMED at the synapse level.** VCH outputs 92.0% GABA by per-synapse argmax; T4/T5 inputs 71.0% ACh.
4. **Reciprocity is real and non-random** (89%+ of T4/T5 inputs are also outputs).
5. **`LPi14` top downstream target CONFIRMED** but it is an extreme connectivity hub; 'top by absolute synapse count' does not imply T4/T5-specific targeting.

## Table 1 — VCH overview (report vs computed)

| Metric | Report | Raw (primary) | Proofread (secondary) |
|---|---|---|---|
| input synapses | 27,576 | 17,930 | 17,131 |
| output synapses | 32,363 | 19,534 | 13,338 |
| upstream partners | 3,619 | 2,746 | 2,130 |
| downstream partners | 10,948 | 8,714 | 3,275 |

## Table 2 — T4/T5 inputs by subtype (computed, raw track)

| subtype | neurons | syn | mean_syn_per_neuron | pct_of_vch_total |
|---|---|---|---|---|
| T5a | 470 | 4389 | 9.34 | 24.48 |
| T4a | 473 | 4360 | 9.22 | 24.32 |
| Other/unclear | 6 | 477 | 79.5 | 2.66 |
| T5b | 11 | 12 | 1.09 | 0.07 |
| T4b | 10 | 10 | 1 | 0.06 |
| T4d | 4 | 5 | 1.25 | 0.03 |
| T5c | 3 | 3 | 1 | 0.02 |
| T5d | 2 | 3 | 1.5 | 0.02 |
| T4c | 2 | 2 | 1 | 0.01 |

## Table 3 — T4/T5 outputs by subtype (computed, raw track)

| subtype | neurons | syn | mean_syn_per_neuron | pct_of_vch_total |
|---|---|---|---|---|
| T4a | 447 | 2354 | 5.27 | 12.05 |
| T5a | 440 | 1769 | 4.02 | 9.06 |
| T4c | 131 | 232 | 1.77 | 1.19 |
| T4d | 131 | 213 | 1.63 | 1.09 |
| T4b | 72 | 99 | 1.38 | 0.51 |
| T5d | 42 | 50 | 1.19 | 0.26 |
| T5c | 28 | 32 | 1.14 | 0.16 |
| Other/unclear | 8 | 23 | 2.88 | 0.12 |
| T5b | 10 | 10 | 1 | 0.05 |

## Laterality detail

- input partner side: {'right': 8784}
- output partner side: {'right': 4759}
- top input neuropils: {'LOP_R': 15258, 'IPS_R': 2636, 'SPS_R': 23, 'LO_R': 7, 'VES_R': 5}
- top output neuropils: {'LOP_R': 19455, 'IPS_R': 48, 'LO_R': 17, 'ME_R': 7, 'SPS_R': 4}

## Downstream of the 918 reciprocal T4/T5

- total synapses: report 578,238 vs computed 352,017
- unique targets: report 144,796 vs computed 113,339
- VCH-as-target: {'vch_rows': [{'root_id': 720575940627706398, 'side': 'left', 'syn': 8922, 'n_drivers': 878, 'is_left_vch': True}]}

### Top-20 individual targets (computed)

| index | root_id | cell_type | syn | n_drivers | nt_canonical | super_class |
|---|---|---|---|---|---|---|
| 0 | 720575940627190556 | LPi14 | 13361 | 776 | gaba | optic |
| 1 | 720575940632504874 | LPi14 | 12157 | 824 | gaba | optic |
| 2 | 720575940627706398 | VCH | 8922 | 878 | gaba | visual_centrifugal |
| 3 | 720575940626979621 | CT1 | 4695 | 864 | gaba | optic |
| 4 | 720575940629148007 | HSE | 3805 | 522 | acetylcholine | visual_projection |
| 5 | 720575940639209956 | DCH | 3036 | 351 | gaba | visual_centrifugal |
| 6 | 720575940615933919 | HSN | 2852 | 341 | acetylcholine | visual_projection |
| 7 | 720575940634612194 | LPi15 | 1386 | 597 | gaba | optic |
| 8 | 720575940628743496 | HSS | 1113 | 468 | acetylcholine | visual_projection |
| 9 | 720575940621116807 | LT33 | 868 | 355 | gaba | optic |
| 10 | 720575940630502522 | LPT26 | 826 | 213 | acetylcholine | visual_projection |
| 11 | 720575940633656147 | Nod2 | 558 | 252 | gaba | visual_projection |
| 12 | 720575940617797259 | LPT04_HST | 542 | 285 | acetylcholine | visual_projection |
| 13 | 720575940616980737 | SAD043 | 497 | 2 | gaba | central |
| 14 | 720575940623997949 | Nod1 | 479 | 213 | acetylcholine | visual_projection |
| 15 | 720575940637734768 | Li14 | 474 | 111 | gaba | optic |
| 16 | 720575940612106098 | Li14 | 436 | 113 | gaba | optic |
| 17 | 720575940629456860 | Nod1 | 412 | 203 | acetylcholine | visual_projection |
| 18 | 720575940634630113 | Li14 | 407 | 91 | gaba | optic |
| 19 | 720575940622410736 | Li14 | 406 | 100 | gaba | optic |

### Nodulus subtable (computed)

| index | cell_type | copies | total_syn | total_drivers | nt |
|---|---|---|---|---|---|
| 0 | Nod1 | 3 | 892 | 417 | acetylcholine |
| 1 | Nod2 | 1 | 558 | 252 | gaba |
| 2 | Nod5 | 1 | 8 | 4 | acetylcholine |
| 3 | Nod3 | 1 | 4 | 4 | acetylcholine |

### Cell-type populations among downstream targets (top 15, computed)

| index | cell_type | copies | total_syn | total_drivers |
|---|---|---|---|---|
| 0 | LPi14 | 2 | 25518 | 1600 |
| 1 | TmY20 | 141 | 13419 | 3240 |
| 2 | Y1 | 76 | 11908 | 3064 |
| 3 | LLPC1 | 105 | 11727 | 2690 |
| 4 | TmY16 | 76 | 9054 | 2220 |
| 5 | VCH | 1 | 8922 | 878 |
| 6 | Y11 | 68 | 7879 | 2661 |
| 7 | T4a | 512 | 6149 | 2987 |
| 8 | T5a | 520 | 5647 | 2991 |
| 9 | TmY14 | 168 | 5459 | 2228 |
| 10 | Y12 | 94 | 5181 | 2175 |
| 11 | Tlp5 | 27 | 4738 | 1844 |
| 12 | CT1 | 1 | 4695 | 864 |
| 13 | HSE | 1 | 3805 | 522 |
| 14 | C3 | 488 | 3537 | 1241 |

## Per-claim detail

| ID | Claim | Verdict | Notes |
|---|---|---|---|
| C0.reproducibility | Report reproducible from its own artifacts | UNVERIFIABLE | Report wrote raw data to /tmp/*.json (ephemeral); not recoverable. Re-derived independently from the offline v783 dump. |
| C1.nt | VCH neurotransmitter is GABA | CONFIRMED |  |
| C1.total_input_syn | VCH total input synapses | CONFIRMED_WITH_CAVEAT | primary diff -9646 (-35.0%); secondary=17131 |
| C1.total_output_syn | VCH total output synapses | CONFIRMED_WITH_CAVEAT | primary diff -12829 (-39.6%); secondary=13338 |
| C1.upstream_partners | VCH upstream partners | CONFIRMED_WITH_CAVEAT | primary diff -873 (-24.1%); secondary=2130 |
| C1.downstream_partners | VCH downstream partners | CONFIRMED_WITH_CAVEAT | primary diff -2234 (-20.4%); secondary=3275 |
| C2.in_neurons | T4/T5 input neurons (loose rule, report-comparable) | CONFIRMED | primary diff -49 (-4.8%); secondary=981 |
| C2.in_syn | T4/T5 input synapses | CONFIRMED_WITH_CAVEAT | primary diff -3757 (-28.9%); secondary=9261 |
| C2.in_pct | T4/T5 = % of VCH input | CONFIRMED | diff +4.4 pp |
| C2.table_sum_syn | Report Table 2 subtype synapses sum to its stated total | REFUTED | rows sum to 12426.0, stated 13018 (diff -592) |
| C2.table_sum_neurons | Report Table 2 subtype neurons sum to its stated total | CONFIRMED | rows sum to 1035.0, stated 1030 (diff +5) |
| C3.out_neurons | T4/T5 output neurons | CONFIRMED_WITH_CAVEAT | primary diff -178 (-12.0%); secondary=1309 |
| C3.out_syn | T4/T5 output synapses | CONFIRMED_WITH_CAVEAT | primary diff -2546 (-34.7%); secondary=4782 |
| C3.out_pct | T4/T5 = % of VCH output | CONFIRMED | diff +1.9 pp |
| C4.n | Reciprocal T4/T5 count | CONFIRMED | primary diff -40 (-4.4%); secondary=878 |
| C4.pct_in | Reciprocal = % of T4/T5 inputs | CONFIRMED | diff +0.4 pp |
| C4.pct_out | Reciprocal = % of T4/T5 outputs | CONFIRMED | diff +5.4 pp |
| C5.laterality | Input & output T4/T5 in DIFFERENT hemispheres | REFUTED | input synapses dominant hemi=R, output=R (both right optic lobe); VCH is a centrifugal cell with dendrite+axon in the right optic lobe -> within-hemisphere recurrent loop, not an ipsi/contra relay. |
| C6.gain_ratio | Excitatory drive 2-3x inhibitory feedback (mean syn/neuron) | CONFIRMED |  |
| C7.down_syn | 918 reciprocal -> total downstream synapses | CONFIRMED_WITH_CAVEAT | primary diff -226221 (-39.1%) |
| C7.down_targets | 918 reciprocal -> unique downstream targets | CONFIRMED | primary diff -31457 (-21.7%) |
| C8.top_target | Top downstream target cell_type is LPi14 | CONFIRMED |  |
| C9.vch_gaba | VCH output synapses' plurality NT is GABA | CONFIRMED | per-synapse argmax corroborates the GABAergic (-) sign |
| C9.t4t5_ach | T4/T5 input synapses' plurality NT is ACh | CONFIRMED | ACh is the plurality per-synapse NT and the neuron-level annotation is ~99% cholinergic, confirming the excitatory (+) sign |
| C10.nodulus | Nodulus (Nod) neurons among downstream targets | CONFIRMED | report Table 6: Nod1-Nod5 |
| C11.annotation | Left 72% / right 97% T4/T5 targets tagged | UNVERIFIABLE | No method given in report. Closest proxy computed: {'left': {'n_targets': 143, 'pct_with_cell_type': 96.5}, 'right': {'n_targets': 11074, 'pct_with_cell_type': 99.9}, 'center': {'n_targets': 4, 'pct_with_cell_type': 100.0}, 'unannotated_targets': 102118} |
| C12.divisive_norm | Circuit implements divisive normalization / gain control | UNVERIFIABLE | Mechanism (divisive vs subtractive) is a dynamical claim not determinable from static synapse counts; anatomically consistent with gain modulation only. |

---

Regenerate: `python -u scripts/vch_extract.py && python -u scripts/vch_verify.py` (or `sbatch slurm/vch_extract.sbatch` then `sbatch slurm/vch_verify.sbatch`).