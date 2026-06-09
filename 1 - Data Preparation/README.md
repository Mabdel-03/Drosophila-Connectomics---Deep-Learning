# 1 — Data Preparation

Pulls the **full adult Drosophila brain connectome** (FlyWire **FAFB v783**) — the
entire connectivity matrix plus all neuron labels/annotations — and turns it into
clean, **model-ready** artifacts for the downstream stages.

All real logic lives in the importable package [`flyconn.data_prep`](../src/flyconn/data_prep/);
this directory holds only thin `sbatch` entrypoints and notebooks (dir names with
spaces/leading digits can't be Python modules). **Everything runs as SLURM batch
jobs** (`account=mabdel03`, `partition=pi_tpoggio`) — never on the login node.

## Pipeline

```
download  ->  build (neurons + edges/adjacency)  ->  validate
```

| stage | script | outputs (on scratch) |
|---|---|---|
| download | [`slurm/download.sbatch`](../slurm/download.sbatch) | `raw/` (5 Zenodo + 3 GitHub files) + `_download_manifest.json` |
| build | [`slurm/build.sbatch`](../slurm/build.sbatch) | `processed/neurons.parquet`, `node_index_map.parquet`, `edges.parquet`, `adjacency_*csr.npz`, `adjacency.pt`, `_schema.json` |
| validate | [`slurm/validate.sbatch`](../slurm/validate.sbatch) | `reports/data_card_v783.md` + `validation_report_v783.json` |

Data lands under `$FLYCONN_DATA_ROOT/v783/` (default
`/orcd/scratch/orcd/012/mabdel03/connectome_data/v783/`).

## Run it

```bash
cd "/orcd/data/tpoggio/001/mabdel03/Connectomics/1 - Data Preparation"

# 0) smoke-test the downloader on the tiny root-ids file first
ONLY=proofread_root_ids_783.npy sbatch ../slurm/download.sbatch

# 1) full download (~10.5 GB, resumable, md5-verified)
sbatch ../slurm/download.sbatch        # or: ./00_download.sh

# 2) build all artifacts, then validate (one job)
sbatch ../slurm/build.sbatch           # or: ./01_build.sh

# 3) (re)validate + regenerate the data card from existing artifacts
sbatch ../slurm/validate.sbatch        # or: ./02_validate.sh

# watch
squeue -u mabdel03
tail -f ../slurm/logs/flyconn-*_*.out
```

## Key conventions

- **Node set:** the ~139,255 proofread neurons in `proofread_root_ids_783.npy`. Each
  gets a contiguous integer `idx` (sorted by `root_id`) — the ANN node id.
- **Adjacency orientation:** source-major `A[i, j]` = weight of edge `i → j`
  (row = presynaptic). For an RNN weight matrix `W` with `r_next = W @ r`, use `A.T`.
- **Neurotransmitter signs:** taken from the presynaptic neuron (Dale's principle).
  Multiple policies are built so the choice is deferred to model time:
  `flyvis_standard` (ACh +1, GABA −1, **Glut −1 — inhibitory in fly**, monoamines 0),
  `glut_excitatory` (Glut +1), and an unsigned raw-count matrix.

## Provenance & citation (CC-BY-4.0 — attribution required)

- Connectivity: **Dorkenwald et al. 2024, _Nature_** (FlyWire) · Zenodo
  [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866)
- Annotations: **Schlegel et al. 2024, _Nature_** ·
  [`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations)
