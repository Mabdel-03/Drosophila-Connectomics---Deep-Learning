# 4 — Motif Search

_Placeholder for the motif-analysis stage._

Search the prepared connectome for recurring connectivity motifs (feedforward loops,
reciprocal pairs, triads, rich-club structure, cell-type-level motif enrichment vs null
models). Consumes the stage-1 edge list / sparse adjacency and the node labels:

```python
from flyconn.config import load_config
from flyconn.io import read_parquet, load_csr

cfg = load_config()
edges = read_parquet(cfg.paths().edges)            # pre_idx, post_idx, syn_count, pre_nt
A = load_csr(cfg.paths().adjacency_counts)         # unsigned raw counts
```

To be filled in after stages 1–2.

## Mesh proximity pilot

The package also includes an optional mesh-based experiment for asking:

> Among neuron pairs whose synapse-derived dendrite mesh samples are within 2 μm,
> what fraction have any real synaptic connection?

This is separate from the core data-prep/modeling environments because it fetches
public FlyWire meshes with `cloud-volume`. The default path uses the public
`precomputed://gs://flywire_v141_m783` mesh source and does not require a
FlyWire/CAVE token.

```bash
pip install -e ".[mesh]"
```

Run the deterministic 1,000-neuron `optic_left` pilot:

```bash
python -m flyconn.experiments.proximity \
  --subgraph optic_left \
  --n 1000 \
  --seed 0 \
  --threshold-um 2 \
  --connection any \
  --mesh-source cloudvolume-public \
  --mesh-path precomputed://gs://flywire_v141_m783 \
  --lod 1 \
  --lod-fallback 0 \
  --site-radius-nm 500 \
  --sample-spacing-nm 250
```

Outputs are written under
`$FLYCONN_DATA_ROOT/v783/experiments/proximity/optic_left_n1000_seed0/`:
`sample_neurons.parquet`, `near_pairs.parquet`, `mesh_failures.parquet`,
`summary.json`, and `proximity_report.md`. Mesh arrays are cached under
`$FLYCONN_DATA_ROOT/v783/mesh_cache/` so interrupted or repeated runs can resume.

The `--mesh-source fafbseg` path remains available for authenticated FlyWire mesh
access, but it is not needed for the public v783 pilot.

## Whole-connectome proximity batch

The full-connectome version is split into restartable Slurm phases: prepare neuron
metadata and post-synaptic site shards, sample dendrite mesh points by neuron shard,
build spatial tiles, search exact 2 μm near pairs by tile, then reduce/enrich the
final pair table.

Submit the default all-neuron run:

```bash
bash slurm/proximity_submit_full.sh
```

Run a small batch smoke test:

```bash
MAX_NEURONS=50 RUN_NAME=proximity_smoke_n50 bash slurm/proximity_submit_full.sh
```

Outputs land under
`$FLYCONN_DATA_ROOT/v783/experiments/proximity/<RUN_NAME>/`, with the default run
name `whole_connectome_lod1_sp250_r500_t2_any`. Key outputs are:

- `neurons_manifest.parquet`: neuron metadata, names/type labels, neurotransmitter,
  side/flow, positions, ontology IDs, and dominant pre/post neuropils.
- `sample_stats/` and `mesh_failures/`: per-shard sampling diagnostics.
- `near_pairs_enriched.parquet`: final exact near-pair table with endpoint metadata,
  closest sampled points, distance, synapse counts, and connected flags.
- `summaries/`: overall, by-neuron, distance-bin, cell-class/type, neurotransmitter,
  side/flow, and dominant-neuropil aggregate tables.
