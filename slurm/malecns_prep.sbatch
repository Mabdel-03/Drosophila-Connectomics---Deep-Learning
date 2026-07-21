#!/bin/bash
#SBATCH --job-name=flyconn-malecns-prep
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"; cd "$FLYCONN_REPO"
# Download the public MaleCNS v1.0 bulk flat files used by family J (no auth).
exec python -u scripts/malecns_prep.py
