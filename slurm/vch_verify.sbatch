#!/bin/bash
#SBATCH --job-name=flyconn-vch-verify
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"; cd "$FLYCONN_REPO"
# Stage 4: re-derive every report claim from the compact parquets + emit JSON/MD/figures.
# Depends on scripts/vch_extract.py having run first (chain with --dependency=afterok).
exec python -u scripts/vch_verify.py
