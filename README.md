# Drosophila Connectomics — Deep Learning

Examining how deep-learning models can replicate the fly brain, and how better
deep-learning models can be built from it. The near-term goal: turn the **FlyWire
adult _Drosophila_ brain connectome** into a **functional artificial neural network**
that can be trained / run inference on GPU.

## Structure

The project is organized into four stages. A clean importable package
[`src/flyconn/`](src/flyconn/) holds all code; the numbered stage directories hold
notebooks, SLURM entrypoints, and stage READMEs.

| stage | directory | status |
|---|---|---|
| 1 · Data Preparation | [`1 - Data Preparation/`](1%20-%20Data%20Preparation/) | **implemented** — pulls FAFB v783, builds model-ready artifacts |
| 2 · Initial Exploration | [`2 - Initial Exploration/`](2%20-%20Initial%20Exploration/) | placeholder |
| 3 · Modeling | [`3 - Modeling/`](3%20-%20Modeling/) | placeholder |
| 4 · Motif Search | [`4 - Motif Search/`](4%20-%20Motif%20Search/) | placeholder |

```
src/flyconn/            importable package (config, paths, io, data_prep/)
configs/data_v783.yaml  single source of truth (URLs, md5s, NT policies)
slurm/                  batch scripts (account=mabdel03, partition=pi_tpoggio)
tests/                  pytest unit tests
```

## Quickstart (ORCD cluster)

Everything runs as SLURM batch jobs; bulk data lives on scratch
(`$FLYCONN_DATA_ROOT`, default `/orcd/scratch/orcd/012/mabdel03/connectome_data`),
never in git.

```bash
cd "/orcd/data/tpoggio/001/mabdel03/Connectomics"
mamba activate /orcd/home/002/mabdel03/conda_envs/consortium
pip install -e .                       # one-time

sbatch slurm/download.sbatch           # pull FAFB v783 (~10.5 GB, md5-verified)
sbatch slurm/build.sbatch              # build neurons + edges + adjacency, then validate
```

Outputs (on scratch under `v783/processed/`): `neurons.parquet`, `edges.parquet`,
`adjacency_<policy>_csr.npz`, `adjacency.pt`, plus a data card under `v783/reports/`.
See [`1 - Data Preparation/README.md`](1%20-%20Data%20Preparation/README.md) for details.

## Data, license & citation

Built from public, **CC-BY-4.0** FlyWire data — attribution required:

- **Connectivity:** Dorkenwald et al. 2024, _Nature_ — _Neuronal wiring diagram of an
  adult brain_ (FlyWire). Zenodo [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866).
- **Annotations:** Schlegel et al. 2024, _Nature_ — _Whole-brain annotation and
  multi-connectome cell typing of Drosophila_.
  [`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations).

FAFB v783: **139,255 neurons**; **2,700,513 directed connections** at the standard
FlyWire ≥5-synapse threshold (the canonical connectome), plus a **15,091,983-edge**
no-threshold full graph. (The Codex landing page shows "3,732,460 connections"; that
display figure isn't reproducible from the public Zenodo v783 file at any integer
threshold, so this pipeline pins to the value it actually derives — see the data card.)
