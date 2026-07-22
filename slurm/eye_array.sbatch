#!/bin/bash
#SBATCH --job-name=flyconn-eye-grid
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%4
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out

# Run the faithful "rigid eye" grid (the configs in configs/eye_experiments/).
# Each leaf config carries a `stage:` field (train for V1/V2, fit_rigid for V3); we
# dispatch to the matching CLI stage. Config selected by $SLURM_ARRAY_TASK_ID over the
# sorted glob, so the default --array range must match the config count (Phase 1 = 10).
#   python -m flyconn.models.run eye_grid          # regenerate Phase-1 configs (10)
#   sbatch slurm/eye_array.sbatch                  # run them (override --array for fewer)
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/eye_experiments/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
  echo "task $SLURM_ARRAY_TASK_ID >= $N configs; nothing to do"; exit 0
fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Read the `stage:` field from the leaf config (train | fit_rigid); default to train.
STAGE="$(grep -E '^stage:' "$CONFIG" | head -1 | awk '{print $2}')"
STAGE="${STAGE:-train}"
echo "[eye-array] task $SLURM_ARRAY_TASK_ID -> stage=$STAGE  $CONFIG"
exec python -u -m flyconn.models.run "$STAGE" --config "$CONFIG"
