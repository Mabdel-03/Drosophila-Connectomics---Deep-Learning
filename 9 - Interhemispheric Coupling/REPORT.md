# Stage 9 — How the left and right figure-ground circuits connect and communicate

## Headline

The two hemispheric figure circuits are coupled by three structural routes, identified and quantified at synapse resolution: a readout crossing in which each side's Nod1 drives the opposite hemisphere's steering command DNp26; heterolateral input bridges (H1, H2) that carry regressive-tuned contralateral inhibition onto the figure cells, matching Egelhaaf (1985); and a shared downstream convergence where the two readouts meet on common premotor, central, and neuromodulatory cells. The centrifugal gaters VCH and DCH, though their somata are registered to the opposite hemisphere, are shown NOT to be axonal bridges.

## Channel N1 — readout crossing / command convergence

- Left Nod1 -> right DNp26: 159 synapses  
- Right Nod1 -> left DNp26: 289 synapses  
- Nod1 output crossing (position-based): 86.9%  
- DNp26_left: 100.0% of its Nod1 input is contralateral  
- DNp26_right: 100.0% of its Nod1 input is contralateral  
- Asymmetry ratio: 1.82

Each steering command is driven by the opposite hemisphere's figure readout, which is the structural basis of the established wing-flip.

## Channel N2 — heterolateral input bridges (Egelhaaf regressive inhibition)

- H1 (glutamate, optic): 2185 crossing synapses onto the figure machinery, input layer b (regressive)  
- MeLp2 (gaba, optic): 469 crossing synapses onto the figure machinery, input layer c (upward)  
- LC14a1 (acetylcholine, optic): 433 crossing synapses onto the figure machinery, input layer c (upward)  
- PLP248 (glutamate, central): 200 crossing synapses onto the figure machinery, input layer None (None)  
- LC14b (acetylcholine, optic): 138 crossing synapses onto the figure machinery, input layer a (progressive)  
- OA-VUMa4 (octopamine, central): 120 crossing synapses onto the figure machinery, input layer None (None)  
- CB0053 (dopamine, central): 116 crossing synapses onto the figure machinery, input layer None (None)  
- cLP05 (gaba, visual_centrifugal): 112 crossing synapses onto the figure machinery, input layer c (upward)  

Inhibitory bridges onto the FD1/Nod1 pathway: ['H1', 'MeLp2', 'cLP05', 'MeMe_e08']; regressive-tuned: ['H1']. Egelhaaf regressive-inhibition pattern holds: True.

## Channel N3 — centrifugal gating (soma vs arbor)

- VCH: soma 99.2% contralateral, but position-based crossing 0.1% (soma and arbor swapped: True)  
- DCH: soma 99.2% contralateral, but position-based crossing 0.1% (soma and arbor swapped: True)  

The gaters are soma-displaced cells local to the lobe they gate, not axonal bridges, and they do not couple to each other directly.

## Channel N4 — shared downstream convergence

- 95 cells receive from both the left and the right figure readout  
- Roles: {'other': 30, 'central': 25, 'premotor_central': 26, 'bridge_feedback': 3, 'readout_crosstalk': 8, 'neuromodulatory': 3}  
- Readout cross-talk: ['Nod3', 'Nod3', 'Nod1', 'Nod2', 'Nod1', 'Nod1', 'Nod5', 'Nod1']  
- Bridge feedback: ['H2', 'H1', 'H1']  
- Neuromodulatory convergence: ['OA-VUMa4', 'OA-VUMa1', 'OA-VUMa4']

## Channel N5 — systematic discovery

- Of 255579 input synapses onto the circuit, 7094 (2.8%) are genuine position-based inter-hemispheric crossings  
- Genuine bridge types: ['AN_multi_11', 'AN_multi_28', 'CB0025', 'CB0053', 'CB0237', 'CB0432', 'CB1138', 'CL131', 'DNp27', 'H1', 'LC14a1', 'LC14b', 'LCe07', 'LPT50', 'MeLp2', 'MeMe_e13', 'OA-VUMa4', 'PLP078', 'PLP248', 'vCal1']  
- Soma-side artifact types removed by the position test: ['CB0399', 'DCH', 'Nod1', 'VCH', 'cLP05', 'mALC5']

## Caveats

- A true crossing is defined by synapse position relative to the synapse-space midline (529.0 um), not by the soma-side annotation, because the soma side conflates arbor geometry with axonal crossing (the VCH/DCH case).  

- Predicted neurotransmitter is used as an anatomical sign annotation, not a measured transmitter. Synapse counts are anatomical evidence, not efficacy.  

- The left-right asymmetries are reported with the per-hemisphere proofreading-completeness caveat established in Stage 8.
