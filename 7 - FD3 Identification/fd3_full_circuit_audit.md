# FD3 Full-Circuit Evidence Audit

Primary track: `offline`

| Claim ID | Section | Evidence | Status | Claim | Caveat |
|---|---|---|---|---|---|
| K.identity | Identity | direct_connectome | confirmed | The strongest connectomic FD3 candidate is LPT42_Nod4. | Candidate screen has margin 2. |
| K.direction | Identity | direct_connectome | confirmed | LPT42_Nod4 draws almost all measured T4/T5 motion input from layer-b. | Layer-b fraction: 98.58%. |
| K.rf_relative | Identity | direct_connectome | confirmed | LPT42_Nod4 is lateral to FD1 and has a frontal gap on both sides. | Absolute visual angle remains calibration-limited; the core comparison is relative to FD1. |
| K.small_field | Identity | direct_connectome | confirmed | The FD3 candidate pools a spatially bounded retinotopic input patch. | Permutation p=0.002; this is wiring support, not a direct physiology recording. |
| K.output_side | Identity | direct_connectome | confirmed | Most LPT42_Nod4 output is contralateral. | Contralateral output: 90.33%. |
| K.morphology | Identity | direct_connectome | confirmed_with_caveat | Both reconstructed skeletons have a dorsoventral dendritic span and crossed axonal displacement consistent with FD3. | Skeleton supports overall geometry; fine branch-level identity is not overinterpreted. |
| K.cross_version | Provenance | provenance | unavailable | Live or v630 replication is included only if available in the current run. | v630 cross-check is live-only |
| P.t4t5_fraction | Inputs | direct_connectome | confirmed | T4/T5 motion detectors provide a minority of all FD3 input. | T4/T5 fraction of all input: 25.1%. |
| P.layer_b_motion | Inputs | direct_connectome | confirmed | Within the measured T4/T5 subset, FD3 is dominated by layer-b back-to-front input. | Layer-b fraction of T4/T5 input: 98.58%. |
| P.upstream_cascade | Inputs | direct_connectome_and_literature | confirmed_with_caveat | Canonical photoreceptor, lamina, medulla, T4/T5 pathways feeding the FD3 detector types are present. | The upstream cascade is measured at type level, not as a unique per-object trace. |
| P.central_inputs | Inputs | direct_connectome | confirmed | FD3 receives diverse non-T4/T5 input, including cholinergic projection sheets and inhibitory inputs. | Only one leading central input type is marked as a layer-b carrier in the current derived table. |
| P.contra_inhibition | Inputs | direct_connectome_and_literature | confirmed_with_caveat | Contralateral inhibitory input is present and compatible with FD3 physiology. |  |
| L.direct_dns | Outputs | direct_connectome | confirmed | FD3 directly contacts descending neurons, led by DNp26. | Direct DN synapses: 138; DN types: 24. |
| L.relay_dns | Outputs | direct_connectome | confirmed_with_caveat | FD3 reaches a broader descending set through strong central partners. | Relay membership depends on the reported intermediary synapse threshold. |
| L.motor_proxy | Outputs | cross_connectome_proxy | confirmed_with_caveat | FD3-weighted descending output is anatomically biased toward wing-steering motor systems. | This is an anatomical motor-system proxy, not a behavioral perturbation result. |
| L.dnp26_muscles | Outputs | cross_connectome_proxy | confirmed_with_caveat | DNp26 links FD3 to contralateral wing-steering muscles in the male CNS mapping. | Neuron matching is by shared descending-neuron type name across connectomes. |
