#!/bin/bash
#SBATCH --job-name=flyconn-eyeTact-cifar-verify
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out

# Phase-0 gating smoke (1 epoch each) for the two riskiest CIFAR-10 eyeTact cells on the
# FULL connectome at T=12: the faithful pure-linear learned cell (forward blow-up risk) and
# the rigid ReLU untrained cell (rigid trajectory readout + deep ReLU on a frozen core).
# Both must fit memory and report a FINITE val_loss before the larger arrays are launched.
# Throwaway. Run the cifar_download warm-up first.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

echo "============ (1) smoke: eyeTact_cifar_whole_learned_linear_trained_T12 ============"
python -u -m flyconn.models.run smoke \
  --config configs/eyeTact_cifar_experiments/eyeTact_cifar_whole_learned_linear_trained_T12.yaml

echo "============ (2) smoke: eyeTact_cifar_whole_rigid_relu_untrained_T12 ============"
python -u -m flyconn.models.run smoke \
  --config configs/eyeTact_cifar_experiments/eyeTact_cifar_whole_rigid_relu_untrained_T12.yaml

echo "============ eyeTact-cifar-verify DONE ============"
