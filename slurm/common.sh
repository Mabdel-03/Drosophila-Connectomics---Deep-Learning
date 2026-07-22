#!/bin/bash
# Sourced by every Connectomics sbatch job. Modules + conda + scratch data root.
# Keep idempotent and side-effect-free beyond exports + module loads.

module load miniforge/25.11.0-0 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true

# Repo root (this file lives in <repo>/slurm/).
export FLYCONN_REPO="${FLYCONN_REPO:-/orcd/data/tpoggio/001/mabdel03/Connectomics}"

# Stage 1-4 env (the heavy data-prep + torch modeling jobs). Reuse the existing
# `consortium` env (full prefix; not on the default envs_dirs, so activate by path). It
# has pandas/pyarrow/scipy/torch/etc.
export FLYCONN_ENV="${FLYCONN_ENV:-/orcd/home/002/mabdel03/conda_envs/consortium}"

# Stage 5 (Muscular Projection / live CAVE) env: a DEDICATED, torch-free env with the
# connectome stack (caveclient/navis/cloud-volume) + the offline data toolchain. Built by
# scripts/setup_cave_env.sh on the tpoggio group volume (personal scratch is over quota,
# /home inode quota is exhausted). Stage-5 sbatch files activate THIS env, not FLYCONN_ENV.
export FLYCONN_CAVE_ENV="${FLYCONN_CAVE_ENV:-/orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave}"

# All bulk data on scratch (group disk is ~87% full). paths.py appends /v<version>.
export FLYCONN_DATA_ROOT="${FLYCONN_DATA_ROOT:-/orcd/scratch/orcd/012/mabdel03/connectome_data}"
mkdir -p "$FLYCONN_DATA_ROOT"

export TOKENIZERS_PARALLELISM=false

# Stage 5 (live CAVE) only: if a repo-local .cave_secret exists, export it as CAVE_TOKEN
# so caveclient authenticates without a global secret file. Guarded + quiet; no-op for
# every offline job. The file is gitignored.
if [ -z "${CAVE_TOKEN:-}" ] && [ -f "$FLYCONN_REPO/.cave_secret" ]; then
  export CAVE_TOKEN="$(tr -d '[:space:]' < "$FLYCONN_REPO/.cave_secret")"
fi
