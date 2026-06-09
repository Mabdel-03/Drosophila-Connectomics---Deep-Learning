# 3 — Modeling

Connectome-constrained **trainable** feedforward / RNN networks that classify **MNIST**.
The real **connectivity (which edges exist) and synaptic signs (excitatory/inhibitory from
neurotransmitter) are FIXED from the data**; only the **edge magnitudes** are learned (plus
a small MNIST input encoder and a linear readout). Code lives in
[`flyconn.models`](../src/flyconn/models/); this dir holds the notebook + launchers.

## The experiment grid (6 × 2 × 2 = 24 runs)

| axis | values |
|---|---|
| subgraph | `whole`, `whole_right`, `whole_left`, `optic`, `optic_right`, `optic_left` |
| flavor | `ff_unroll` (inject at t=0, leak≈1), `rnn` (persistent inject, leaky, BPTT) |
| init | `from_data` (magnitudes = scaled real synapse counts), `random` (log-normal matched amplitude) |

The `optic*` subgraphs use the stage-2 `visual_mask` union (optic + visual_projection +
visual_centrifugal + photoreceptors), so MNIST enters through real photoreceptors (R1-6/R7/R8).

**Headline question:** does the real wiring (`from_data`) beat amplitude-matched random
wiring (`random`), and across which subgraphs/flavors?

## The model (`connectome_net.py`)

One `ConnectomeNet`. Per-edge weight `W_value = sign · softplus(theta)` — `sign` is a fixed
buffer (Dale's law hard-enforced; gradients can only rescale magnitude, never flip sign),
`theta` is the only core parameter. Forward = a leaky-integrator unroll for `T` steps using a
gather→`index_add` scatter (sparse `W @ h`; avoids the `torch.sparse_csr_tensor` autograd
detach, pytorch #98929). Stability with no learnable node params comes from **spectral-radius
≈0.9 init** (sparse power iteration, never densified) + `tanh` + fixed leak + grad-clip.
Orientation: artifacts are source-major `A[i,j]=i→j`, transposed to `W[post,pre]` in
`build_subgraph`.

Memory: a dense 139k² matrix is 78 GB — infeasible — so everything is an **edge vector**
(≤2.6M floats). Input = photoreceptors; readout = VPN (optic) / descending+VPN (whole-brain),
mean-pooled → `Linear(·,10)`. `T` defaults to `max(10, BFS hops input→readout)`.

## Environment / run

Training uses the **`consortium`** env (torch 2.3.1+cu121 + torchvision + flyconn). Everything
runs via `sbatch` on `pi_tpoggio` (1× A100) — **never the login node**. Artifacts go to scratch
(`$FLYCONN_DATA_ROOT/v783/{mnist,models,results}`).

```bash
cd "/orcd/data/tpoggio/001/mabdel03/Connectomics"

# single run
EXP=optic_left_ff_unroll_initA sbatch slurm/train.sbatch
# smoke (1 epoch):  STAGE=smoke EXP=... sbatch slurm/train.sbatch

# full 24-run grid (<=4 concurrent A100s)
cd "3 - Modeling" && ./03_train.sh        # = sbatch ../slurm/train_array.sbatch

# interactive build + inspection + short smoke train
#   open modeling.ipynb with the Python (consortium) kernel (use a GPU node for the smoke train)
```

Per-run outputs: `…/models/<subgraph>_<arch>_<init>/{ckpt_best.pt, metrics.jsonl, summary.json,
config_resolved.json}`.

## CLI

```bash
python -m flyconn.models.run grid                       # (re)generate the 24 leaf configs
python -m flyconn.models.run build --config <leaf.yaml> # construct + print model info, no train
python -m flyconn.models.run train --config <leaf.yaml> # full training run
```

Reference: flyvis (Lappalainen 2024, connectome-constrained DMN), Song et al. 2016
(excitatory-inhibitory RNN sign constraints).
