#!/bin/bash
# Sourced by every Connectomics sbatch job. Modules + conda + scratch data root.
# Keep idempotent and side-effect-free beyond exports + module loads.

module load miniforge/25.11.0-0 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true

# Repo root (this file lives in <repo>/slurm/).
export FLYCONN_REPO="${FLYCONN_REPO:-/orcd/data/tpoggio/001/mabdel03/Connectomics}"

# Reuse the existing `consortium` env (full prefix; it is not on the default
# envs_dirs, so activate by path). It already has pandas/pyarrow/scipy/torch/etc.
export FLYCONN_ENV="${FLYCONN_ENV:-/orcd/home/002/mabdel03/conda_envs/consortium}"

# All bulk data on scratch (group disk is ~87% full). paths.py appends /v<version>.
export FLYCONN_DATA_ROOT="${FLYCONN_DATA_ROOT:-/orcd/scratch/orcd/012/mabdel03/connectome_data}"
mkdir -p "$FLYCONN_DATA_ROOT"

export TOKENIZERS_PARALLELISM=false
