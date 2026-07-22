#!/bin/bash
#SBATCH --job-name=flyconn-eyeTact-cifar-p1
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --array=0-47%8
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out

# Phase-1 CIFAR-10 eyeTact: the 48 representative cells (T in {1,4,8,12} x 2 eyes x 3 acts x
# {untrained,trained}) curated into configs/eyeTact_cifar_phase1/ by generate_eyeTact_cifar_grid.
# A CURATED SUBDIR (not an --array index range) is required: the sorted-glob order is
# eye -> act -> {trained,untrained} -> T, so the chosen T values are scattered, not contiguous.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/eyeTact_cifar_phase1/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
  echo "task $SLURM_ARRAY_TASK_ID >= $N configs; nothing to do"; exit 0
fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Skip-if-already-done so a re-launch (or overlap with Phase 2) doesn't redo finished cells.
RUN_NAME="$(basename "$CONFIG" .yaml)"
SUMMARY="$FLYCONN_DATA_ROOT/v783/models/$RUN_NAME/summary.json"
if [ -f "$SUMMARY" ]; then
  echo "[eyeTact-cifar-p1] $RUN_NAME already has summary.json; skipping"; exit 0
fi

echo "[eyeTact-cifar-p1] task $SLURM_ARRAY_TASK_ID -> $CONFIG"
exec python -u -m flyconn.models.run train --config "$CONFIG"
