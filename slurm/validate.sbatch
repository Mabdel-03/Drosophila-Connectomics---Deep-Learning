#!/bin/bash
#SBATCH --job-name=flyconn-validate
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out

# Re-run validation + regenerate the data card from existing processed artifacts.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

exec python -u -m flyconn.data_prep.run validate \
  --config "$FLYCONN_REPO/configs/data_v783.yaml"
