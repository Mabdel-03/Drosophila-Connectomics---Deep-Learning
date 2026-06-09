# 2 — Initial Exploration

Exploratory analysis of the prepared connectome (stage-1 artifacts on scratch).

## Notebook

[`eda.ipynb`](eda.ipynb) — two parts:

**Part I (whole brain):** dataset sanity, composition (super_class / cell_class / side /
flow), neurotransmitter breakdown, in/out degree distributions, hub neurons (CT1, APL, …),
signed-adjacency structure (E:I balance), and synaptic flow between super-classes.

**Part II (toward a cell-type-level visual ANN):** narrows to the visual system
(~97k neurons: optic + visual-projection/centrifugal + photoreceptors) and reshapes it into
the substrate for a trainable network — **one node per `cell_type`**, edges = aggregated
type→type synapse weights. Sections: (A) visual subset + helpers `visual_mask` /
`collapse_to_celltype`, (B) the type×type connectivity matrix (raw / per-source-normalized /
signed — the future weight matrix), (C) feedforward layering (retina→lamina→medulla→T4/T5→
lobula→VPN), (D) input→output reachability (photoreceptors → LC/VPN readouts), (E) per-type
signal-property table, (F) biology sanity checks (R→L→Mi/Tm→T4/T5 pathways), and (G) a
modeling-readiness summary with the v1 model spec and open decisions for stage 3.

Verified by headless execution on the `flyconn_eda` kernel (18 code cells, 0 errors).

## Environment / kernel

EDA uses a dedicated conda env **`flyconn_eda`** (separate from `consortium`, which has
no Jupyter kernel) with the plotting/analysis stack + a registered Jupyter kernel.

**One-time setup:**

```bash
cd /orcd/data/tpoggio/001/mabdel03/Connectomics
bash scripts/setup_eda_env.sh
```

This creates the env at `/orcd/scratch/orcd/012/mabdel03/conda_envs/flyconn_eda` (on
scratch — the `/home` per-user inode quota is exhausted; from
[`environment-eda.yml`](../environment-eda.yml)), `pip install -e .`'s the `flyconn`
package into it, and registers the kernel **`Python (flyconn_eda)`**.

**Run the notebook:**

- **VS Code:** open `eda.ipynb`, pick the `Python (flyconn_eda)` kernel (top-right).
- **JupyterLab:** `mamba run -p /orcd/scratch/orcd/012/mabdel03/conda_envs/flyconn_eda jupyter lab`

The notebook resolves artifact paths via the `flyconn` package; data lives at
`$FLYCONN_DATA_ROOT/v783/processed/` (default
`/orcd/scratch/orcd/012/mabdel03/connectome_data/...`). If the kernel doesn't see that
env var, the package falls back to the default scratch root automatically.

> Heavy interactive work (large graph algorithms, embeddings) should run on a compute
> node, not the login node — launch JupyterLab inside an `srun`/`sbatch` allocation on
> `pi_tpoggio` if a cell gets expensive.

## Loading artifacts (quick reference)

```python
from flyconn.config import load_config
from flyconn.io import read_parquet, load_csr

cfg = load_config(); P = cfg.paths()
neurons = read_parquet(P.neurons)                       # 139,255 nodes; row i == idx i
edges   = read_parquet(P.edges)                         # canonical >=5-syn connections
A       = load_csr(P.adjacency_signed(cfg.default_nt_policy))  # signed CSR, A[i,j] = i->j
A_counts = load_csr(P.adjacency_counts)                 # unsigned raw synapse counts
```
