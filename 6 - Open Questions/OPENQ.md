# Verification — Open Questions

- Datasets: {'fafb': {'materialization': 783}, 'mcns': {'materialization': 'v1.0'}}
- Seeds: ['LLPC1', 'Nod1', 'DNp26', 'LPLC2', 'LC4', 'DNp01', 'DNp03', 'DNp04', 'DNp06']
- Tracks: {'q1': 'single-animal MCNS', 'q2': 'bilateral FlyWire (left vs right control)', 'q3': 'escape census FlyWire -> MCNS muscle'}

## Verdict tally

CONFIRMED: 19 · UNVERIFIABLE: 1

## Headline claims

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| LLPC1 present in MCNS (>=1 body) | >=1 body (~285) | True | - | PASS CONFIRMED |
| Nod1 present in MCNS (>=1 body) | >=1 body (~4) | True | - | PASS CONFIRMED |
| DNp26 present in MCNS (>=1 body) | >=1 body (~2) | True | - | PASS CONFIRMED |
| LLPC1 -> Nod1 synaptic edge exists in MCNS (syn>0) | syn>0 | True | - | PASS CONFIRMED |
| Nod1 -> DNp26 synaptic edge exists in MCNS (syn>0) | syn>0 | True | - | PASS CONFIRMED |
| Nod1 among LLPC1's top excitatory targets in MCNS | top excitatory readout | True | - | PASS CONFIRMED |
| Fraction of LLPC1->Nod1->DNp26 chain reproduced within MCNS alone | 1.0 (full chain, single animal) | 1 | - | PASS CONFIRMED |
| RIGHT sheet reciprocal T4/T5 fraction (control) | 89.2 | 89.24 | - | PASS CONFIRMED |
| RIGHT sheet: VCH presynaptically gates driving terminals (control) | present | True | - | PASS CONFIRMED |
| RIGHT sheet: Nod1 dominant excitatory readout (control) | dominant | True | - | PASS CONFIRMED |
| LEFT sheet: VCH (right-soma) presynaptically gates driving terminals | present | True | - | PASS CONFIRMED |
| LEFT sheet: Nod1 dominates the readout (rank parallels right) | dominant | True | - | PASS CONFIRMED |
| LEFT sheet: pooling is spatially local / retinotopic | local | - | - | N/A UNVERIFIABLE |
| LEFT reciprocal T4/T5 fraction parallels RIGHT (within tolerance) | ~89.2% (right) | 86.28 | - | PASS CONFIRMED |
| Escape-route DN census downstream of LPLC2/LC4 is non-empty | >=1 DN | True | - | PASS CONFIRMED |
| Looming command cluster (DNp01/03/04/06) downstream of LPLC2/LC4 | present (giant fibre DNp01 + DNp03/04/06) | True | - | PASS CONFIRMED |
| Escape route reaches jump/TTM (tergotrochanter) muscle in MCNS | reaches jump/TTM | True | - | PASS CONFIRMED |
| Escape DN-set disjoint from steering (LLPC1) DN-set | overlap=0 (separable) | 0 | - | PASS CONFIRMED |
| Escape muscle-targets disjoint from wing-steering muscles | overlap=0 (separable) | 0 | - | PASS CONFIRMED |
| Escape output anatomically separable from LLPC1 steering output | separable (distinct DNs + muscles) | True | - | PASS CONFIRMED |

## Q1 — single-animal closure (MCNS)

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| LLPC1 present in MCNS (>=1 body) | >=1 body (~285) | True | - | PASS CONFIRMED |
| Nod1 present in MCNS (>=1 body) | >=1 body (~4) | True | - | PASS CONFIRMED |
| DNp26 present in MCNS (>=1 body) | >=1 body (~2) | True | - | PASS CONFIRMED |
| LLPC1 -> Nod1 synaptic edge exists in MCNS (syn>0) | syn>0 | True | - | PASS CONFIRMED |
| Nod1 -> DNp26 synaptic edge exists in MCNS (syn>0) | syn>0 | True | - | PASS CONFIRMED |
| Nod1 among LLPC1's top excitatory targets in MCNS | top excitatory readout | True | - | PASS CONFIRMED |
| Fraction of LLPC1->Nod1->DNp26 chain reproduced within MCNS alone | 1.0 (full chain, single animal) | 1 | - | PASS CONFIRMED |


Raw fields:

- `question` = Q1_single_animal_closure
- `existence` = {LLPC1=285, Nod1=4, DNp26=2, DNa04=2, DNbe001=2}
- `wiring` = {llpc1_to_nod1_syn=10871, llpc1_contacting_nod1=272, nod1_to_dnp26_syn=504, nod1_contacting_dnp26=4}
- `nod1_readout_rank` = {nod1_rank=2, nod1_syn=10871, n_target_types=673, top5_targets=[{'type': 'PLP249', 'weight': 13433, 'rank': 1}, {'type': 'Nod1', 'weight': 10871, 'rank': 2}, {'type': 'PVLP011', 'weight': 9547, 'rank': 3}, {'type': 'PLP163', 'weight': 7807, 'rank': 4}, {'type': 'Y13', 'weight': 4082, 'rank': 5}], nod1_is_dominant_readout=True}
- `chain_reproduced` = 1.0
- `verdict` = CONFIRMED
- `claims` = [{'id': 'Q1.exist', 'description': 'LLPC1/Nod1/DNp26 readout cells exist in MCNS', 'report_value': 'present', 'computed_primary': {'LLPC1': 285, 'Nod1': 4, 'DNp26': 2, 'DNa04': 2, 'DNbe001': 2}, 'computed_secondary': None, 'tolerance': 'all >0', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': 'recon positive control', 'extra': {}}, {'id': 'Q1.llpc1_nod1', 'description': 'LLPC1 -> Nod1 synaptic edge present in MCNS', 'report_value': '~4228 (FlyWire)', 'computed_primary': 10871, 'computed_secondary': None, 'tolerance': '>0 syn; >0 LLPC1 contacting Nod1', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': '272 LLPC1 bodies contact Nod1 (10871 syn)', 'extra': {}}, {'id': 'Q1.nod1_dnp26', 'description': 'Nod1 -> DNp26 relay present in MCNS', 'report_value': '~448 (FlyWire)', 'computed_primary': 504, 'computed_secondary': None, 'tolerance': '>0 syn; >0 Nod1 contacting DNp26', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': '4 Nod1 bodies contact DNp26 (504 syn)', 'extra': {}}, {'id': 'Q1.nod1_dominant', 'description': 'Nod1 is a dominant excitatory LLPC1 readout in MCNS', 'report_value': 'rank 1 (FlyWire)', 'computed_primary': 'rank 2 of 673', 'computed_secondary': None, 'tolerance': 'Nod1 in top-3 LLPC1 target types', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': 'Nod1 syn from LLPC1 = 10871', 'extra': {}}]
- `track` = mcns

## Q2 — bilateral / 4-direction generalization

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| RIGHT sheet reciprocal T4/T5 fraction (control) | 89.2 | 89.24 | - | PASS CONFIRMED |
| RIGHT sheet: VCH presynaptically gates driving terminals (control) | present | True | - | PASS CONFIRMED |
| RIGHT sheet: Nod1 dominant excitatory readout (control) | dominant | True | - | PASS CONFIRMED |
| LEFT sheet: VCH (right-soma) presynaptically gates driving terminals | present | True | - | PASS CONFIRMED |
| LEFT sheet: Nod1 dominates the readout (rank parallels right) | dominant | True | - | PASS CONFIRMED |
| LEFT sheet: pooling is spatially local / retinotopic | local | - | - | N/A UNVERIFIABLE |
| LEFT reciprocal T4/T5 fraction parallels RIGHT (within tolerance) | ~89.2% (right) | 86.28 | - | PASS CONFIRMED |


Raw fields:

- `question` = Q2_bilateral_generalization
- `right` = {sheet_side=right, gating_soma_side=left, gating_present=True, n_vch=1, n_dch=1, vch_in_syn=27576, vch_out_syn=32363, t4t5_partners=1022, t4t5_syn=12301, reciprocal_partners=912, n_t4a=454, n_llpc1_sheet=100, nod1_rank=2, nod1_share_pct=7.89, nod1_is_top_excitatory=True}
- `left` = {sheet_side=left, gating_soma_side=right, gating_present=True, n_vch=1, n_dch=1, vch_in_syn=25630, vch_out_syn=29987, t4t5_partners=984, t4t5_syn=11924, reciprocal_partners=849, n_t4a=411, n_llpc1_sheet=86, nod1_rank=2, nod1_share_pct=7.11, nod1_is_top_excitatory=True}
- `claims` = [{'id': 'Q2.gating_present', 'description': 'VCH gating present for left sheet (as for right)', 'report_value': True, 'computed_primary': True, 'computed_secondary': None, 'tolerance': 'exact', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': '', 'extra': {}}, {'id': 'Q2.reciprocal_frac', 'description': 'T4/T5->VCH reciprocal fraction: left vs right', 'report_value': 89.2, 'computed_primary': 86.3, 'computed_secondary': None, 'tolerance': '+/-10.0 pp', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': 'diff -2.9 pp', 'extra': {}}, {'id': 'Q2.sheet_size', 'description': 'LLPC1 sheet size: left vs right (T4a-driven)', 'report_value': 100, 'computed_primary': 86, 'computed_secondary': None, 'tolerance': '+/-30% or +/-10 (drift down)', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': True, 'notes': 'primary diff -14 (-14.0%)', 'extra': {}}, {'id': 'Q2.nod1_dominant_left', 'description': 'Nod1 is the top excitatory readout of the LEFT sheet', 'report_value': True, 'computed_primary': True, 'computed_secondary': None, 'tolerance': 'exact', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': '', 'extra': {}}, {'id': 'Q2.retinotopy_left', 'description': 'Left-sheet pooling is spatially local/retinotopic', 'report_value': 'local', 'computed_primary': None, 'computed_secondary': None, 'tolerance': 'n/a', 'verdict': 'UNVERIFIABLE', 'numeric_outcome': '', 'drift_explains': False, 'notes': 'needs per-side synapse-position retinotopy null (paper.geometry / derive.d_retinotopy_null); deferred as too heavy for the cheap structural parallel', 'extra': {}}]
- `track` = live

## Q3 — escape-route census to muscle

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| Escape-route DN census downstream of LPLC2/LC4 is non-empty | >=1 DN | True | - | PASS CONFIRMED |
| Looming command cluster (DNp01/03/04/06) downstream of LPLC2/LC4 | present (giant fibre DNp01 + DNp03/04/06) | True | - | PASS CONFIRMED |
| Escape route reaches jump/TTM (tergotrochanter) muscle in MCNS | reaches jump/TTM | True | - | PASS CONFIRMED |
| Escape DN-set disjoint from steering (LLPC1) DN-set | overlap=0 (separable) | 0 | - | PASS CONFIRMED |
| Escape muscle-targets disjoint from wing-steering muscles | overlap=0 (separable) | 0 | - | PASS CONFIRMED |
| Escape output anatomically separable from LLPC1 steering output | separable (distinct DNs + muscles) | True | - | PASS CONFIRMED |


Raw fields:

- `question` = Q3_escape_route_census
- `flywire` = {n_lplc2=210, n_lc4=104, n_dn_types=28, command_dn_syn={'DNp01': 3875, 'DNp03': 741, 'DNp04': 6634, 'DNp06': 1711}, census_top15={'DNp04': 6634, 'DNp103': 3904, 'DNp01': 3875, 'DNp02': 2365, 'DNp11': 1979, 'DNp06': 1711, 'DNg40': 1424, 'DNp35': 1325, 'DNp05': 1243, 'DNp03': 741, 'DNp71': 487, 'DNpe025': 461, 'DNpe056': 425, 'DNp55': 230, 'DNpe042': 157}}
- `mcns` = {escape_dns_in_mcns=['DNa07', 'DNae007', 'DNb05', 'DNc02', 'DNg40', 'DNge054', 'DNp01', 'DNp02', 'DNp03', 'DNp04', 'DNp05', 'DNp06', 'DNp09', 'DNp102', 'DNp103', 'DNp11', 'DNp27', 'DNp30', 'DNp35', 'DNp55', 'DNp69', 'DNp70', 'DNp71', 'DNpe021', 'DNpe025', 'DNpe042', 'DNpe045', 'DNpe056'], n_escape_motor_edges=13, n_steering_motor_edges=131, escape_muscles=['STTMm', 'TTMn'], steering_muscles=['MNwm35', 'MNwm36', 'b1', 'b2', 'b3', 'hg1', 'hg2', 'hg3', 'hg4', 'i1', 'i2', 'iii1', 'ps1', 'ps2', 'tp1', 'tp2', 'tpn']}
- `separability` = {dn_overlap=[], dn_separable=True, muscle_overlap=[], muscle_separable=True, separable=True}
- `claims` = [{'id': 'Q3.census', 'description': 'Escape DN census downstream of LPLC2/LC4 (looming cluster present)', 'report_value': 'weaker route (untraced)', 'computed_primary': {'DNp01': 3875, 'DNp03': 741, 'DNp04': 6634, 'DNp06': 1711}, 'computed_secondary': None, 'tolerance': '>=1 command DN with syn>=floor', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': '28 DN types downstream of LPLC2/LC4', 'extra': {}}, {'id': 'Q3.dn_separable', 'description': 'Escape DNs are disjoint from LLPC1/Nod1 steering DNs', 'report_value': True, 'computed_primary': True, 'computed_secondary': None, 'tolerance': 'exact', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': '', 'extra': {}}, {'id': 'Q3.muscle_separable', 'description': 'Escape muscles (jump/TTM) are disjoint from wing-steering muscles', 'report_value': True, 'computed_primary': True, 'computed_secondary': None, 'tolerance': 'exact', 'verdict': 'CONFIRMED', 'numeric_outcome': 'MATCH', 'drift_explains': False, 'notes': '', 'extra': {}}]
- `track` = live
