# Evidence seed

`seed_graph.v1.json` is a small, reviewable bootstrap graph for the flight
pathway. It is not a new connectome export and it does not claim complete
coverage.

Observed seed data were read on 2026-07-17 from the existing deployed
`/fly-fgs` artifacts:

- `index.html` SHA-256
  `f19434ee872984035fafbfb492538609cf458111ddb64ad9f998e6c371187932`
  supplies the selected FAFB v783 LLPC1/NOD1-to-DN structural counts.
- `wing_dns.json` SHA-256
  `b3ae85e6151228919617a87907730705ec3acb8d8b441465ce182926d6c32390`
  supplies MANC `male-cns:v1.0` type-level DN-to-muscle summaries. Those
  summaries are explicitly represented as aggregates *via* motor neurons;
  the bundled data do not contain the individual MN edges.

The FAFB-to-MANC bridge uses cell-type labels only. Root IDs and MANC body IDs
must never be joined. The male/female and specimen mismatch remains visible in
the crosswalk confidence.

This bootstrap graph contains no coordinate-valued records, so its dataset
references intentionally omit `coordinate_units`. Native FlyWire skeleton and
synapse coordinates are nanometres (scale to SI metres by `1e-9`) and must be
preserved as such at the ingestion boundary.

Two whole-pathway coverage records (LLPC1 618/2174 and NOD1 545/712 resolved)
preserve the audit totals in the approved implementation plan. They are marked
provisional and low confidence because the raw audit CSV was not present in
the cloned repository or deployed `/fly-fgs` directory. The two 100% records
refer only to the seven hand-selected display targets and are explicitly not
whole-pathway coverage.

Every count is an anatomical structural count. No count in this directory is
a physiological synaptic weight, firing-rate gain, muscle activation, or
mechanical parameter.

## Registered female downstream snapshot

The seed above remains the topology consumed by the exploratory simulator. A
separate content-addressed fixture at
`../reference/banc_fanc_wing_pathway_evidence.v1.json` now supports the
`evidence.banc_fanc_wing_pathway` structural gate. It preserves female BANC
v888 NOD1→DNp26 edges, 62 proofread wing MNs, and 18 DNp26→wing-MN edges whose
corrected total is **129** raw structural synapses. It also preserves the
independent published FANC v840 one-sided premotor-to-wing-MN matrix.

That reference fixture does not merge identifier spaces or replace this seed.
BANC/FANC premotor matching was not performed, the premotor identity crosswalk
is incomplete, and no candidate was promoted to an identity match. It therefore
does not yet provide the simulator with a complete
DN→identified-premotor→MN→muscle causal route or physiological parameters.
