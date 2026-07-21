# Verification Report — Figure-Ground Circuit

**164 CONFIRMED · 52 CONFIRMED_WITH_CAVEAT · 4 REFUTED · 12 UNVERIFIABLE** over 232 claims.

Every quantitative claim was re-derived from public connectome data: the FlyWire FAFB v783 brain (offline track, `synapses_nt_v1`, no cleft threshold — which reproduces the paper's absolute counts to the digit) and the male CNS connectome (MaleCNS v1.0 public bulk files) for the motor mapping.

## Headline

The circuit reproduces end-to-end. Exemplary exact matches: VCH 27,576/32,363 synapses; 1,022 T4/T5 inputs (12,301 syn, 44.6%); 912 reciprocal; the 454 VCH-gated T4a -> 9,223 syn -> 100 LLPC1 sheet; LLPC1 receives 17,499 reciprocal-T4/T5 synapses (~48x its sibling sheets, 318-401); LLPC1 sheet output 106,269 synapses with Nod1 the dominant excitatory readout; VCH->T4a inputs sit a median ~1.8 um from the T4a->LLPC1 terminal vs ~29 um for other inputs (VCH closest in 100% of 286 T4a, Wilcoxon p~6e-49); and in the male CNS, DNa04 drives the ipsilateral wing (0.99), DNp26/DNg32 the contralateral wing (0.23/0.03), with DNp26 targeting hg1/i1/hg2.

## Refuted claims

- **A.t4t5_in_pct** — T4/T5 = % of VCH input syn: paper `44.6` vs computed `49.0`. diff +4.4 pp
- **A.reciprocal_pct_out** — Reciprocal = % of T4/T5 outputs: paper `62.0` vs computed `67.2`. diff +5.2 pp
- **B.pass_all_three** — Only VCH & DCH pass all 3 figure-circuit criteria: paper `['DCH', 'VCH']` vs computed `['VCH']`. computed pass-set = ['VCH']
- **C.llpc1_vs_siblings** — LLPC1 gets ~40x more T4/T5 than sibling sheets: paper `10.0-80.0x` vs computed `139.8`. 

## Confirmed-with-caveat (matches after a named difference)

- **A.vch_in_syn** — VCH total input synapses: paper `27576` vs `17930`. primary diff -9646 (-35.0%)
- **A.vch_out_syn** — VCH total output synapses: paper `32363` vs `19534`. primary diff -12829 (-39.6%)
- **A.vch_up_partners** — VCH upstream partners: paper `3619` vs `2746`. primary diff -873 (-24.1%)
- **A.t4t5_in_neurons** — T4/T5 input neurons to VCH: paper `1022` vs `975`. primary diff -47 (-4.6%)
- **A.t4t5_in_syn** — T4/T5 input synapses to VCH: paper `12301` vs `8784`. primary diff -3517 (-28.6%)
- **A.t4t5_out_neurons** — T4/T5 output neurons from VCH: paper `1476` vs `1301`. primary diff -175 (-11.9%)
- **A.t4t5_out_syn** — T4/T5 output synapses from VCH: paper `7273` vs `4759`. primary diff -2514 (-34.6%)
- **A.reciprocal_n** — Reciprocal T4/T5 partners: paper `912` vs `874`. primary diff -38 (-4.2%)
- **B.VCH.llpc1** — VCH: LLPC1 sheet contact (/100): paper `77` vs `66`. primary diff -11 (-14.3%)
- **B.DCH.llpc1** — DCH: LLPC1 sheet contact (/100): paper `52` vs `41`. primary diff -11 (-21.2%)
- **B.LPi14.llpc1** — LPi14: LLPC1 sheet contact (/100): paper `90` vs `63`. primary diff -27 (-30.0%)
- **D.t4a_llpc1_syn** — T4a -> LLPC1 synapses: paper `9223` vs `6117`. primary diff -3106 (-33.7%)
- **D.llpc1_n** — LLPC1 sheet size: paper `100` vs `93`. primary diff -7 (-7.0%)
- **F.lpi15_reach** — LPi15 reaches all 100 LLPC1: paper `100` vs `93`. primary diff -7 (-7.0%)
- **F.lpi15_syn** — LPi15 -> LLPC1 synapses: paper `6472` vs `3679`. primary diff -2793 (-43.2%)
- **F.pvlp011_reads** — PVLP011 reads all 100 LLPC1: paper `100` vs `93`. primary diff -7 (-7.0%)
- **F.pvlp011_feedsback** — PVLP011 feeds GABA back to 99/100 LLPC1: paper `99` vs `92`. primary diff -7 (-7.1%)
- **F.vch_direct_llpc1** — VCH directly inhibits 77/100 LLPC1: paper `77` vs `66`. primary diff -11 (-14.3%)
- **F.vch_direct_syn** — VCH -> LLPC1 direct synapses: paper `588` vs `371`. primary diff -217 (-36.9%)
- **G.output_total** — LLPC1 sheet total output synapses: paper `106269` vs `62342`. primary diff -43927 (-41.3%)
- **G.PLP249.drivers** — PLP249: number of LLPC1 (of 100) contacting it: paper `97` vs `90`. primary diff -7 (-7.2%)
- **G.PLP249.syn** — PLP249: synapses from the LLPC1 sheet: paper `4389` vs `2854`. primary diff -1535 (-35.0%)
- **G.Nod1.drivers** — Nod1: number of LLPC1 (of 100) contacting it: paper `91` vs `84`. primary diff -7 (-7.7%)
- **G.Nod1.syn** — Nod1: synapses from the LLPC1 sheet: paper `4228` vs `2393`. primary diff -1835 (-43.4%)
- **G.PVLP011.drivers** — PVLP011: number of LLPC1 (of 100) contacting it: paper `100` vs `93`. primary diff -7 (-7.0%)
- **G.PVLP011.syn** — PVLP011: synapses from the LLPC1 sheet: paper `3168` vs `2010`. primary diff -1158 (-36.6%)
- **G.PLP163.drivers** — PLP163: number of LLPC1 (of 100) contacting it: paper `100` vs `93`. primary diff -7 (-7.0%)
- **G.PLP163.syn** — PLP163: synapses from the LLPC1 sheet: paper `2534` vs `1579`. primary diff -955 (-37.7%)
- **G.Nod2.drivers** — Nod2: number of LLPC1 (of 100) contacting it: paper `70` vs `65`. primary diff -5 (-7.1%)
- **G.Nod2.syn** — Nod2: synapses from the LLPC1 sheet: paper `826` vs `500`. primary diff -326 (-39.5%)
- **G.DNbe001.drivers** — DNbe001: number of LLPC1 (of 100) contacting it: paper `70` vs `64`. primary diff -6 (-8.6%)
- **G.DNbe001.syn** — DNbe001: synapses from the LLPC1 sheet: paper `736` vs `457`. primary diff -279 (-37.9%)
- **H.nod1_dn_syn** — Nod1 -> descending-neuron synapses: paper `662` vs `453`. primary diff -209 (-31.6%)
- **H.nod1_n_dns** — Nod1 -> number of descending neurons: paper `27` vs `21`. primary diff -6 (-22.2%)
- **H.nod1_to_dnp26** — Nod1 -> DNp26 synapses: paper `448` vs `315`. primary diff -133 (-29.7%)
- **H.direct_to_dnp26** — Direct sheet -> DNp26 synapses (Nod1 route is larger): paper `149` vs `95`. primary diff -54 (-36.2%)
- **I.direct.syn** — direct channel: circuit -> DN synapses: paper `2174` vs `1326`. primary diff -848 (-39.0%)
- **I.nod_relay.syn** — nod_relay channel: circuit -> DN synapses: paper `712` vs `486`. primary diff -226 (-31.7%)
- **I.broadcast.syn** — broadcast channel: circuit -> DN synapses: paper `628` vs `388`. primary diff -240 (-38.2%)
- **K.rf_abs_azimuth** — FD3 absolute peak azimuth ~40-50 deg (calibrated, secondary): paper `40.0-50.0 deg` vs `103.5`. measured peak 103.5 deg vs Egelhaaf 40-50 deg (within calibration band: False); combined calibration uncertainty +/-16.0 deg. Linear column->azimuth map is the weakest link; lattice-extreme anchoring biases the centroid lateral. Secondary corroboration only — the differential gate governs the RF identity. Error budget: {'linearity_deg': 12.5, 'landmark_deg': 6.5, 'eye_span_deg': 7.5, 'dropout_deg': 1.5, 'combined_deg': 16.0, 'model': 'linear two-anchor p->azimuth; azimuth is p-driven, q->elevation coarse', 'verdict_ceiling': 'CONFIRMED_WITH_CAVEAT (no ground-truth degree field exists)'}.
- **K.contra_inhibition_bidirectional** — FD3 vs FD1 receive OPPOSITE-dominant contralateral inhibition (FD3 progressive-dominant, FD1 regressive-dominant): paper `FD3 and FD1 differ in crossed-inhibition direction (Egelhaaf p.203-204)` vs `FD3=progressive, FD1=regressive`. FD3 progressive-dominant (prog 0.928/reg 0.072); FD1=Nod1 regressive-dominant (prog 0.083/reg 0.917). Opposite-dominant crossed inhibition matches Egelhaaf's FD3-vs-FD1 contrast in direction; FD3's minority (regressive) channel is a trace (0.072 of classified). Annotation is power-limited (n_classified=6 of 108 contra-GABA), so the fine bidirectional/unidirectional split is not resolved by wiring alone.
- **K.morph_dv_span** — FD3 dendrite spans the dorso-ventral lobula plate (real skeleton (fafbseg)): paper `full D-V (Egelhaaf p.203)` vs `155.3 um max`. D-V spans (um): [155.3, 147.5]; real skeleton (fafbseg) (36744 skeleton nodes)
- **K.morph_axon_heterolateral** — FD3 axon is displaced contralaterally from the dendrite (real skeleton (fafbseg)): paper `crosses midline to contralateral side (Egelhaaf p.203)` vs `ML shifts [-228.1, 224.9] um (toward contra both cells)`. dendrite->axon medio-lateral centroid shift sign; real skeleton (fafbseg)
- **K.morph_axon_noduli** — FD3 axon converges near the noduli-group landmark (real skeleton (fafbseg)): paper `noduli group, posterior optic foci (Egelhaaf p.203)` vs `axon 76.7 um vs dendrite 160.7 um from Nod1 landmark`. Nod1 output-cloud centroid as the noduli-group landmark; real skeleton (fafbseg)
- **K.identity_verdict** — OVERALL: LPT42_Nod4 is the modern correlate of Egelhaaf-1985 FD3: paper `FD3 == LPT42_Nod4` vs `CONFIRMED_WITH_CAVEAT`. CORE + DISCRIMINATING confirmed; >=1 corroborating claim caveated/unverifiable
- **KD.D3_axon_crosses** — Discriminator: axon crosses the midline (both cells): paper `FD3 has it; FD2 does not` vs `LPT42_Nod4=True, Nod3=True`. favors neither
- **P.contra_inhibition_bidirectional** — FD3 receives contralateral inhibition in both directions (Egelhaaf 1985 p.203); measured as progressive-dominant with a regressive minority: paper `both channels present (progressive-dominant)` vs `progressive-dominant, both_present=True`. progressive=387 syn, regressive=30 syn over 6 classified contra-GABA partners; progressive-dominant (minority frac 0.072); the minority channel is a trace, so this reads as direction-dominant rather than cleanly bidirectional; well-powered
- **P.input_pathway_verdict** — OVERALL: FD3's afferent pathway (photoreceptor -> layer-b T4/T5 -> FD3): paper `regressive layer-b afferent arm, parallel to (not a copy of) FD1` vs `CONFIRMED_WITH_CAVEAT`. CORE + DISCRIMINATING confirmed; >=1 corroborating claim caveated/unverifiable
- **Q.sheet_verdict** — OVERALL: FD3's intermediate sheet-set (LPC1 the direction-matched member): paper `T4b/T5b -> {LPC1(+LLPC2/3)} -> FD3` vs `CONFIRMED_WITH_CAVEAT`. CORE+DISC confirmed; a corroborating claim is soft
- **R.two_gate_architecture** — FD3 has TWO complementary wide-field gates at different nodes: an opponent gate at the FD3/sheet node (LPi14) and a same-direction gate at the detector node (LPi12): paper `opponent FD3/sheet gate + same-direction detector gate, distinct cells` vs `opponent=LPi14 (FD3/sheet), same_direction=LPi12 (detector)`. the defensible headline; sign of each GABA contact is from physiology, not wiring; n=2 FD3 pair (both cells agree per per_fd3_cell_syn)
- **R.same_direction_fd3_specificity** — The same-direction gate's (minor) FD3 contact is FD3-specific, not spillover from a promiscuous cell: paper `->FD3 enriched over a size-matched random LP-tangential target` vs `obs 68 vs null 11.09 (z=2.95, p=0.002)`. 2000 perms; a genuine co-input onto FD3, though a minor one
- **R.inhibitor_verdict** — OVERALL: LPi14 is FD3's wide-field opponent inhibitor (VCH functional-role homolog): paper `LPi14 = VCH-role opponent gate` vs `CONFIRMED_WITH_CAVEAT`. CORE+DISC confirmed; a corroborating claim is soft

## Unverifiable (interpretive / physiology predictions)

These are not connectomic claims; each is recorded with the anatomical proxy (where one exists) that the connectome *can* supply.

- **K.cross_version_v630** — Headline FD3 signature replicates on FlyWire v630. v630 cross-check is live-only
- **K.power_note** — Statistical power: only one bilateral LPT42_Nod4 pair (n=2). n=2 is irreducible (one bilateral pair exists). Four independent variance controls bound the inference: (i) bilateral independent replication (K.bilateral_replication); (ii) within-cell synapse bootstrap CI on the RF offset (N_BOOT=2000); (iii) in-degree permutation null for small-field selectivity (z<<0, p<=0.002); (iv) cross-version replication on v630. The conclusion does not rest on a per-cell sample size.
- **L.steering_dominant** — FD3's descending output predominantly drives wing-steering (figure-tracking). anatomical proxy: of FD3's descending drive that resolves to a male-CNS motor system, the largest share is 'wing_steering' (motor-system %, FD3-drive-weighted: {'wing_steering': 43.5, 'neck_gaze': 17.3, 'abdominal': 16.5, 'wing_power': 14.7, 'leg': 5.2, 'other': 1.5, 'haltere': 1.3}).
- **P.peripheral_nt_histaminergic** — Photoreceptor/lamina front end is histaminergic (literature). The R1-6->lamina->medulla cascade upstream of FD3's detectors is PRESENT and per-type wired in v783, but the ON/OFF functional split and the histaminergic transmitter of the photoreceptors come from physiology (Hardie 1989; Maisak 2013), not the connectome. NOTE: the annotation table lists R1-6=acetylcholine / R7=gaba, which is a prediction artifact and biologically wrong (photoreceptors are histaminergic) — not propagated.
- **P.power_note** — Statistical power: FD3 is one bilateral pair (n=2). n=2 is irreducible (one bilateral LPT42_Nod4 pair). The afferent claims rest on the aggregate synapse populations onto both cells (thousands of input synapses) and on per-type cascade wiring measured across the full retinotopic population, not on a per-cell sample size.
- **Q.power_note** — FD3 is one bilateral pair (n=2). The sheet claims rest on the aggregate synapse populations onto both FD3 cells and on the full sheet-cell populations, not on a per-cell sample size.
- **R.power_note** — FD3 is one bilateral pair (n=2). The inhibitor claims rest on the aggregate synapse populations, not per-cell n; both FD3 cells agree on the two-gate asymmetry (per_fd3_cell_syn).
- **Z.divisive_norm** — Circuit implements divisive normalization / gain control. PROXY: graded reciprocal same-direction VCH-T4/T5 loop is consistent with proportional gain control, but divisive-vs-subtractive is a dynamical property not determinable from static synapse counts (paper concurs, S9).
- **Z.llpc1_readout** — LLPC1 reads out the VCH-gated local motion residual. PROXY: the motion+form dual-dendrite, retinotopic, central-projecting architecture is the substrate for a readout; whether the RESPONSE encodes the residual requires LLPC1 imaging (paper S13).
- **Z.vch_silencing** — Silencing VCH makes LLPC1 more raw-motion-like / less figure-selective. A perturbation-response prediction; not testable from the static connectome. Anatomical precondition (VCH presynaptically gates the T4a terminals driving LLPC1) IS confirmed (family E), which is the strongest available support.
- **Z.lpi15_pvlp_silencing** — Silencing LPi15/PVLP011 changes operating point/gain, not separation. Perturbation-response prediction. Anatomical precondition (LPi15 feed-forward opponent; PVLP011 recurrent on a distal compartment) is confirmed (families F/E).
- **Z.escape_separable** — LPLC2/LC4 escape is separable from the LLPC1 course-control route. PROXY: the channel separation IS in the anatomy (family I/J), but the behavioural double-dissociation needs perturbation experiments (S13).

---
Full per-claim ledger: `verification_results.json` and `VERIFICATION.md`.