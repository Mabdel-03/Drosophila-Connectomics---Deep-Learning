#!/bin/bash
#SBATCH --job-name=flyconn-eyeTact-cols-smoke
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
# Confirm a rigid eyeTact cell builds + trains with the FAITHFUL FlyWire columns retinotopy.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"
echo "=== smoke: eyeTact_whole_rigid_tanh_T05 (eye_source=columns) ==="
python -u -m flyconn.models.run smoke \
  --config configs/eyeTact_experiments/eyeTact_whole_rigid_tanh_T05.yaml
echo "=== eyeTact columns smoke DONE ==="
