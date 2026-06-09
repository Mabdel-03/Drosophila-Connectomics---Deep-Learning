#!/bin/bash
#SBATCH --job-name=flyconn-train-grid
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-23%4
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out

# Train the full 24-run grid (6 subgraphs x {ff_unroll,rnn} x {initA,initB}),
# at most 4 A100s concurrently. Config selected by $SLURM_ARRAY_TASK_ID over the
# sorted list of configs/experiments/*.yaml.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/experiments/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
  echo "task $SLURM_ARRAY_TASK_ID >= $N configs; nothing to do"; exit 0
fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[array] task $SLURM_ARRAY_TASK_ID -> $CONFIG"
exec python -u -m flyconn.models.run train --config "$CONFIG"
