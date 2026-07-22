#!/bin/bash
#SBATCH --job-name=flyconn-eyeTact-mitgpu
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=43-143%12
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out
# Overflow runner: the not-yet-started eyeTact cells (indices 43-143) routed to the
# mit_normal_gpu partition (18+ idle L40S nodes) to parallelize past the pi_tpoggio QOS cap.
# Same sorted config glob + index mapping as eyeTact_array.sbatch.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"
mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/eyeTact_experiments/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then echo "task >= $N; nothing"; exit 0; fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eyeTact-mitgpu] task $SLURM_ARRAY_TASK_ID -> $CONFIG"
exec python -u -m flyconn.models.run train --config "$CONFIG"
