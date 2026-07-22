#!/bin/bash
#SBATCH --job-name=flyconn-eyeTact-cifar
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --array=0-179%6
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out

# Train the CIFAR-10 eyeTact grid on the FULL connectome (whole, N=139,255, E=2,630,012):
#   {learned eye, rigid eye} x {linear, tanh, relu} x T(1..12) x {untrained, trained} = 144
#   luminance cells + 12 learned-eye RGB color-ablation controls + 24 tanh state_norm=rms
#   ablation cells = 180. 'untrained' = connectome frozen as-measured (only I/O heads train);
#   'trained' = edge magnitudes also learn. whole is ~3x the edges of the optic* grids, so
#   --mem=120G; --time=16h gives margin over the MNIST 12h (T=12 BPTT on 139k nodes is slow,
#   and CIFAR gives no reason to be faster). Config selected by $SLURM_ARRAY_TASK_ID over the
#   sorted glob; a skip-if-summary-exists guard means Phase-1 cells already run are not redone.
#   python -m flyconn.models.run eyeTact_cifar_grid   # (re)generate the 180 configs + phase1
#   sbatch slurm/cifar_download.sbatch                # warm the dataset cache FIRST
#   sbatch slurm/eyeTact_cifar_array.sbatch           # run them (override --array for a subset)
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/eyeTact_cifar_experiments/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
  echo "task $SLURM_ARRAY_TASK_ID >= $N configs; nothing to do"; exit 0
fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

RUN_NAME="$(basename "$CONFIG" .yaml)"
SUMMARY="$FLYCONN_DATA_ROOT/v783/models/$RUN_NAME/summary.json"
if [ -f "$SUMMARY" ]; then
  echo "[eyeTact-cifar] $RUN_NAME already has summary.json; skipping"; exit 0
fi

echo "[eyeTact-cifar] task $SLURM_ARRAY_TASK_ID -> $CONFIG"
exec python -u -m flyconn.models.run train --config "$CONFIG"
