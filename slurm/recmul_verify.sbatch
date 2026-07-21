#!/bin/bash
#SBATCH --job-name=flyconn-recmul-verify
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out

# One-off verification for the Family-3 "recmul" variant:
#   (1) unit tests (GPU-free, data-free)
#   (2) 1-epoch smoke of the riskiest cell (pure-linear, trainable-W) — must stay finite.
# Throwaway; not part of the standing grid.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

echo "================ (1) pytest tests/test_models.py ================"
python -m pytest tests/test_models.py -q

echo "================ (2) smoke: optic_left_linear_trainW_initA (1 epoch) ================"
python -u -m flyconn.models.run smoke \
  --config configs/recmul_experiments/optic_left_linear_trainW_initA.yaml

echo "================ recmul-verify DONE ================"
