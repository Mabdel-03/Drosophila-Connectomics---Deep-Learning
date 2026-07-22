#!/bin/bash
#SBATCH --job-name=flyconn-eyeTact-bcs
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out
# eyeTact overflow on ou_bcs_low (per-user cap gres/gpu=64 -> wide parallelism). Same sorted
# config glob + index mapping as the other eyeTact arrays.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"
mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/eyeTact_experiments/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then echo "task >= $N; nothing"; exit 0; fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eyeTact-bcs] task $SLURM_ARRAY_TASK_ID -> $CONFIG"
exec python -u -m flyconn.models.run train --config "$CONFIG"
