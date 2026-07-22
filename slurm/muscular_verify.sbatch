#!/bin/bash
#SBATCH --job-name=flyconn-muscular-verify
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
# Unconditional override: common.sh set this to scratch (over quota); Stage 5 uses the group volume.
export FLYCONN_DATA_ROOT="/orcd/data/tpoggio/001/mabdel03/connectome_data"
mamba activate "$FLYCONN_CAVE_ENV"; cd "$FLYCONN_REPO"
# Stage 5 (offline): re-derive S14/S15, run nulls + coverage, emit the
# '5 - Muscular Projection/' dataset + gate files. Cache-only; no network, no token.
# Depends on muscular_extract having populated the cache (chain with --dependency=afterok).
exec python -u scripts/muscular_verify.py
