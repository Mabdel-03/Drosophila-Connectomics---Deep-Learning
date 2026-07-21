#!/bin/bash
#SBATCH --job-name=flyconn-eyeTact-cifar-mitgpu
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --array=0-179%12
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out
# Overflow runner for the CIFAR eyeTact grid on the mit_normal_gpu partition (idle L40S
# nodes) to parallelize past the pi_tpoggio QOS cap. Same sorted config glob + index mapping
# + skip-if-done guard as eyeTact_cifar_array.sbatch, so it is safe to run concurrently with
# it (each cell is claimed by whichever task starts first; the other skips on summary.json).
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/eyeTact_cifar_experiments/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then echo "task >= $N; nothing"; exit 0; fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

RUN_NAME="$(basename "$CONFIG" .yaml)"
SUMMARY="$FLYCONN_DATA_ROOT/v783/models/$RUN_NAME/summary.json"
if [ -f "$SUMMARY" ]; then
  echo "[eyeTact-cifar-mitgpu] $RUN_NAME already done; skipping"; exit 0
fi

echo "[eyeTact-cifar-mitgpu] task $SLURM_ARRAY_TASK_ID -> $CONFIG"
exec python -u -m flyconn.models.run train --config "$CONFIG"
