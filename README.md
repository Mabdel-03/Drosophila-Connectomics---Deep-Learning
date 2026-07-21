# Drosophila Connectomics: Deep Learning

This project studies how computational (artificial neural network) models can be built
directly from the fruit fly brain connectome, and how those models behave on machine
learning tasks. The near-term goal is to turn the **FlyWire adult _Drosophila_ brain
connectome** into a **functional artificial neural network** that can be trained and run
for inference on a GPU, then to experiment with using that network for different tasks.

The connectome supplies a fixed wiring diagram: which neurons connect to which, and whether
each connection is excitatory or inhibitory (derived from the presynaptic neurotransmitter,
following Dale's principle). The modeling stages keep that structure fixed and learn only
the connection magnitudes, so the resulting networks are constrained by real biology rather
than free-form.

## Tasks and scope

The project is intended as a platform for experimenting with the connectome on several
downstream tasks. The current state is:

- **Visual classification (implemented).** Stage 3 trains connectome-constrained
  feedforward and recurrent networks on MNIST, injecting the images through the real
  photoreceptor neurons and reading out from biologically defined output neurons.
- **Language modeling (planned).** A natural-language task is a stated future direction and
  is not yet implemented.
- **Motif analysis (planned).** Stage 4 (connectivity motif search) is a placeholder; the
  data layer it depends on is ready.

## Structure

The project is organized into four stages. A clean importable package
[`src/flyconn/`](src/flyconn/) holds all code; the numbered stage directories hold
notebooks, SLURM entrypoints, and stage READMEs.

| stage | directory | status |
|---|---|---|
| 1 · Data Preparation | [`1 - Data Preparation/`](1%20-%20Data%20Preparation/) | implemented: pulls FAFB v783, builds model-ready artifacts |
| 2 · Initial Exploration | [`2 - Initial Exploration/`](2%20-%20Initial%20Exploration/) | implemented: `eda.ipynb` exploratory analysis |
| 3 · Modeling | [`3 - Modeling/`](3%20-%20Modeling/) | implemented: connectome-constrained MNIST classifiers |
| 4 · Motif Search | [`4 - Motif Search/`](4%20-%20Motif%20Search/) | placeholder |

```
src/flyconn/            importable package (config, paths, io, data_prep/, models/)
configs/data_v783.yaml  single source of truth (URLs, md5s, NT policies)
configs/model_base.yaml shared stage-3 hyperparameters
configs/experiments/    24 leaf configs (6 subgraphs x 2 archs x 2 inits)
slurm/                  batch scripts (account=mit_general, partition=pi_tpoggio)
tests/                  pytest unit tests
```

## Environments and setup

Bulk data is never stored in git. It lives on scratch under `$FLYCONN_DATA_ROOT` (default
`/orcd/scratch/orcd/012/mabdel03/connectome_data`), and the package appends `/v783`. Set
`FLYCONN_DATA_ROOT` to relocate all artifacts and the MNIST cache.

There are four working environments, matched to the stages. The package itself is
installed editable with `pip install -e .` in every case.

### Option A: conda (recommended on the ORCD cluster)

The cluster already provides a `consortium` conda env
(`/orcd/home/002/mabdel03/conda_envs/consortium`) that satisfies stages 1 and 3, including a
CUDA build of torch. The SLURM scripts activate it automatically through
[`slurm/common.sh`](slurm/common.sh), so no setup is required to submit jobs there.

To build the stage 1 / 3 env from scratch on another machine:

```bash
cd "/orcd/data/tpoggio/001/mabdel03/Connectomics"
mamba env create -f environment.yml      # or: conda env create -f environment.yml
mamba activate flyconn
pip install -e .
```

The stage 2 notebook env (`flyconn_eda`) is separate because it adds JupyterLab and the
plotting stack. Create it and register its Jupyter kernel with the helper script:

```bash
bash scripts/setup_eda_env.sh
```

This builds the env on scratch (the `/home` per-user inode quota is exhausted), installs
`flyconn` into it, and registers the kernel **`Python (flyconn_eda)`**. See
[`2 - Initial Exploration/README.md`](2%20-%20Initial%20Exploration/README.md) for detail.

The stage 5 environment (`flyconn_cave`) is separate too: it adds the live-connectome
stack (`caveclient`, `navis`, `cloud-volume`) for querying FlyWire FAFB and the male-CNS
connectome via the CAVE API, and is deliberately **torch-free** (stage 5 only reads/writes
parquet + queries CAVE). Build it with:

```bash
bash scripts/setup_cave_env.sh
```

It lives on the tpoggio group volume
(`/orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave`) — `/home` is out of inodes and
personal scratch is over quota. The stage-5 SLURM scripts
([`slurm/muscular_extract.sbatch`](slurm/muscular_extract.sbatch),
[`slurm/muscular_verify.sbatch`](slurm/muscular_verify.sbatch),
[`slurm/cave_probe.sbatch`](slurm/cave_probe.sbatch)) activate it via `$FLYCONN_CAVE_ENV`
(set in [`slurm/common.sh`](slurm/common.sh)), while stages 1–4 keep using `$FLYCONN_ENV`
(the torch-capable `consortium` env). See [`4 - Motif Search/README.md`](4%20-%20Motif%20Search/README.md)
and the stage 5 directory for detail. The stage-5 unit tests are torch-free and run in this
env; the torch-dependent stage 1/3 tests do not (run those under `$FLYCONN_ENV`).

### Option B: pip

Three pip requirements files mirror the conda environments:

```bash
cd "/orcd/data/tpoggio/001/mabdel03/Connectomics"
python -m venv .venv && source .venv/bin/activate   # or any virtual environment

# stages 1 and 3 (data preparation, modeling)
pip install -r requirements.txt
pip install -e .

# stage 2 (notebooks); superset of requirements.txt
pip install -r requirements-eda.txt
pip install -e .
python -m ipykernel install --user --name flyconn_eda --display-name "Python (flyconn_eda)"

# stage 5 (live connectome / CAVE); torch-free
pip install -e ".[cave]"

# running the tests
pip install -r requirements-dev.txt
```

A plain `pip install` of torch yields a CPU build, which is sufficient to write the
model-ready `adjacency.pt` but not for GPU training in stage 3. For GPU training, install
the CUDA torch wheel for your platform from <https://pytorch.org/get-started/locally/>.

## End-to-end run guide

On the ORCD cluster every stage runs as a SLURM batch job
(`account=mit_general`, `partition=pi_tpoggio`), never on the login node. The same logic is
also exposed as `python -m` module entrypoints for local or interactive use.

### Stage 1 · Data Preparation

Pulls FAFB v783 (about 10.5 GB, md5-verified) and builds neurons, edges, and adjacency
matrices, then validates them.

```bash
cd "/orcd/data/tpoggio/001/mabdel03/Connectomics"

# SLURM (cluster)
sbatch slurm/download.sbatch     # download; ONLY=<key> restricts, FORCE=1 re-downloads
sbatch slurm/build.sbatch        # build neurons + edges + adjacency, then validate
sbatch slurm/validate.sbatch     # re-validate and regenerate the data card

# or directly, inside an activated env
python -m flyconn.data_prep.run all --config configs/data_v783.yaml
# individual stages: download | build | validate | neurons | edges
```

Outputs land under `$FLYCONN_DATA_ROOT/v783/`: `processed/neurons.parquet`,
`processed/edges.parquet`, `processed/adjacency_<policy>_csr.npz`, `processed/adjacency.pt`,
and a data card under `reports/`. See
[`1 - Data Preparation/README.md`](1%20-%20Data%20Preparation/README.md).

### Stage 2 · Initial Exploration

Open [`2 - Initial Exploration/eda.ipynb`](2%20-%20Initial%20Exploration/eda.ipynb) and
select the **`Python (flyconn_eda)`** kernel (in VS Code, the kernel picker at the top
right; in JupyterLab, the kernel menu). The notebook reads the stage-1 artifacts and
explores composition, neurotransmitter balance, degree distributions, and the visual-system
substrate used in stage 3. Heavy interactive cells should run on a compute node rather than
the login node.

### Stage 3 · Modeling

Trains connectome-constrained MNIST classifiers across a 24-run grid (6 subgraphs x 2
architectures x 2 initializations). Training uses the `consortium` env (CUDA torch +
torchvision) on a GPU node.

```bash
cd "/orcd/data/tpoggio/001/mabdel03/Connectomics"

# generate the 24 leaf configs (one-time, or after editing the grid)
python -m flyconn.models.run grid

# single run on the cluster
EXP=optic_left_ff_unroll_initA sbatch slurm/train.sbatch
#   smoke test (1 epoch): STAGE=smoke EXP=<name> sbatch slurm/train.sbatch

# full 24-run grid (up to 4 concurrent A100s)
cd "3 - Modeling" && ./03_train.sh        # submits slurm/train_array.sbatch (--array=0-23%4)

# or directly, inside an activated env
python -m flyconn.models.run build --config configs/experiments/optic_left_ff_unroll_initA.yaml
python -m flyconn.models.run train --config configs/experiments/optic_left_ff_unroll_initA.yaml
python -m flyconn.models.run aggregate    # collect per-run summaries into results/summary.csv
```

Per-run outputs go to `$FLYCONN_DATA_ROOT/v783/models/<run_name>/` (`ckpt_best.pt`,
`metrics.jsonl`, `summary.json`, `config_resolved.json`). For interactive build and
inspection, open
[`3 - Modeling/modeling.ipynb`](3%20-%20Modeling/modeling.ipynb) with the
**`Python (consortium)`** kernel on a GPU node. See
[`3 - Modeling/README.md`](3%20-%20Modeling/README.md).

### Tests

```bash
python -m pytest tests/ -q
```

The tests run on small synthetic graphs with no network access or large files, covering
the data-preparation logic ([`tests/test_data_prep.py`](tests/test_data_prep.py)) and the
connectome-constrained model core ([`tests/test_models.py`](tests/test_models.py)).

## Data, license, and citation

Built from public, **CC-BY-4.0** FlyWire data, which requires attribution:

- **Connectivity:** Dorkenwald et al. 2024, _Nature_, _Neuronal wiring diagram of an adult
  brain_ (FlyWire). Zenodo
  [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866).
- **Annotations:** Schlegel et al. 2024, _Nature_, _Whole-brain annotation and
  multi-connectome cell typing of Drosophila_.
  [`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations).

FAFB v783 contains **139,255 neurons** and **2,700,513 directed connections** at the
standard FlyWire threshold of at least 5 synapses per pair (the canonical connectome), plus
a **15,091,983-edge** no-threshold full graph. The Codex landing page shows a figure of
"3,732,460 connections"; that display value is not reproducible from the public Zenodo v783
file at any integer threshold, so this pipeline pins to the value it derives directly. See
the generated data card for the full provenance and hashes.
