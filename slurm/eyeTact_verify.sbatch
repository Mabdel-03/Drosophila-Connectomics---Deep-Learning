#!/bin/bash
#SBATCH --job-name=flyconn-eyeTact-verify
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out

# Gating smoke (1 epoch each) for the two riskiest eyeTact cells on the FULL connectome at
# T=12: the faithful pure-linear learned cell (forward blow-up risk) and the rigid ReLU cell
# (rigid trajectory-readout + deep ReLU). Must fit memory and give a FINITE val_loss before
# the full 72-run array is launched. Throwaway.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

echo "================ (1) smoke: eyeTact_whole_learned_linear_T12 (faithful pure-linear) ================"
python -u -m flyconn.models.run smoke \
  --config configs/eyeTact_experiments/eyeTact_whole_learned_linear_T12.yaml

echo "================ (2) smoke: eyeTact_whole_rigid_relu_T12 ================"
python -u -m flyconn.models.run smoke \
  --config configs/eyeTact_experiments/eyeTact_whole_rigid_relu_T12.yaml

echo "================ eyeTact-verify DONE ================"
