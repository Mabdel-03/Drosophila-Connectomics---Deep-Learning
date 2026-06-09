#!/bin/bash
#SBATCH --job-name=flyconn-train
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out

# Train ONE connectome-constrained MNIST model. Pass the experiment via:
#   EXP=optic_left_ff_unroll_initA sbatch slurm/train.sbatch
# or override with CONFIG=/abs/path.yaml.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

CONFIG="${CONFIG:-$FLYCONN_REPO/configs/experiments/${EXP}.yaml}"
STAGE="${STAGE:-train}"
exec python -u -m flyconn.models.run "$STAGE" --config "$CONFIG"
