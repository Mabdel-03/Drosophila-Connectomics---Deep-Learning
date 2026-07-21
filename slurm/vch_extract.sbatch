#!/bin/bash
#SBATCH --job-name=flyconn-vch-extract
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"; cd "$FLYCONN_REPO"
# Stage 4: two memory-mapped streaming passes over the 9.5 GB raw synapse table.
exec python -u scripts/vch_extract.py
