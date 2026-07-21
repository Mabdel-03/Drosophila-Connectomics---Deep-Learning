# Verification — Muscular Projection

- Datasets: {'mcns': {'materialization': 'v1.0'}, 'fafb': {'materialization': 783}}
- Seeds: ['DNbe001', 'DNp26']
- Tracks: {'primary': 'all-synapse CAVE', 'secondary': 'min_syn>=5'}

## Verdict tally

CONFIRMED: 26 · CONFIRMED_WITH_CAVEAT: 3

## Headline claims

| Claim | Report | Computed (primary) | Secondary | Verdict |
|---|---|---|---|---|
| DNa04 ipsi fraction | 99 | 99.4 | - | PASS CONFIRMED |
| DNa04 wing target | ipsilateral | ipsilateral | - | PASS CONFIRMED |
| DNa04 ipsi+contra == steering total | 908 | 908 | - | PASS CONFIRMED |
| DNbe001 ipsi fraction | 52 | 52.4 | - | PASS CONFIRMED |
| DNbe001 wing target | bilateral | bilateral | - | PASS CONFIRMED |
| DNbe001 ipsi+contra == steering total | 1181 | 1181 | - | PASS CONFIRMED |
| DNge107 ipsi fraction | 58 | 58.3 | - | PASS CONFIRMED |
| DNge107 wing target | bilateral | bilateral | - | PASS CONFIRMED |
| DNge107 ipsi+contra == steering total | 1044 | 1044 | - | PASS CONFIRMED |
| DNbe005 ipsi fraction | 42 | 42.5 | - | PASS CONFIRMED |
| DNbe005 wing target | bilateral | bilateral | - | PASS CONFIRMED |
| DNbe005 ipsi+contra == steering total | 459 | 459 | - | PASS CONFIRMED |
| DNp26 ipsi fraction | 23 | 22.5 | - | PASS CONFIRMED |
| DNp26 wing target | contralateral | contralateral | - | PASS CONFIRMED |
| DNp26 ipsi+contra == steering total | 386 | 386 | - | PASS CONFIRMED |
| DNg32 ipsi fraction | 3 | 3.2 | - | PASS CONFIRMED |
| DNg32 wing target | contralateral | contralateral | - | PASS CONFIRMED |
| DNg32 ipsi+contra == steering total | 313 | 313 | - | PASS CONFIRMED |
| DNge094 ipsi fraction | 0 | 0 | - | PASS CONFIRMED |
| DNge094 wing target | contralateral | contralateral | - | PASS CONFIRMED |
| DNge094 ipsi+contra == steering total | 37 | 37 | - | PASS CONFIRMED |
| DNp26 -> hg1 synapses | 122 | 122 | - | PASS CONFIRMED |
| DNp26 -> i1 synapses | 105 | 105 | - | PASS CONFIRMED |
| DNp26 -> hg2 synapses | 117 | 117 | - | PASS CONFIRMED |
| DNp26 -> b3 synapses | 28 | 28 | - | PASS CONFIRMED |
| DNp26 -> hg3 synapses | 5 | 5 | - | PASS CONFIRMED |
| DNa04 wing bias vs somaSide-permutation null | non-random (p<=0.05) | obs=0.50, z=-0.08, p=0.938 | - | PASS* CONFIRMED_WITH_CAVEAT |
| DNp26 wing bias vs somaSide-permutation null | non-random (p<=0.05) | obs=0.52, z=-0.49, p=0.638 | - | PASS* CONFIRMED_WITH_CAVEAT |
| DNg32 wing bias vs somaSide-permutation null | non-random (p<=0.05) | obs=0.47, z=-0.19, p=0.790 | - | PASS* CONFIRMED_WITH_CAVEAT |

## Per-DN wing laterality (S14)

| dn | steering_syn | ipsi_syn | contra_syn | ipsi_frac | wing |
|---|---|---|---|---|---|
| DNbe001 | 1181 | 619 | 562 | 0.524 | bilateral |
| DNge107 | 1044 | 609 | 435 | 0.583 | bilateral |
| DNa04 | 908 | 903 | 5 | 0.994 | ipsilateral |
| DNbe005 | 459 | 195 | 264 | 0.425 | bilateral |
| DNp26 | 386 | 87 | 299 | 0.225 | contralateral |
| DNg32 | 313 | 10 | 303 | 0.032 | contralateral |
| DNge094 | 37 | 0 | 37 | 0.0 | contralateral |
