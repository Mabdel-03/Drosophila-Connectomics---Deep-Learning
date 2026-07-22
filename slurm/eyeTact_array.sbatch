#!/bin/bash
#SBATCH --job-name=flyconn-eyeTact-grid
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-143%6
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out

# Train the Family-4 "eyeTact" grid on the FULL connectome (whole, N=139,255, E=2,630,012):
#   {learned eye, rigid eye} x {linear, tanh, relu} x T(1..12) x {untrained, trained} = 144.
# 'untrained' = connectome frozen as-measured (only I/O heads train); 'trained' = edge
# magnitudes also learn. whole is ~3x the edges of the optic* grids, so --mem=120G (vs 64G)
# and --time=12h (T=12 BPTT on 139k nodes is slow).
# Up to 6 A100s concurrently. Config selected by $SLURM_ARRAY_TASK_ID over the sorted glob.
#   python -m flyconn.models.run eyeTact_grid      # (re)generate the 72 configs
#   sbatch slurm/eyeTact_array.sbatch              # run them (override --array for a subset)
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/eyeTact_experiments/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
  echo "task $SLURM_ARRAY_TASK_ID >= $N configs; nothing to do"; exit 0
fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eyeTact-array] task $SLURM_ARRAY_TASK_ID -> $CONFIG"
exec python -u -m flyconn.models.run train --config "$CONFIG"
