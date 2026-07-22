#!/bin/bash
#SBATCH --job-name=flyconn-recmul-grid
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out

# Train the Family-3 "recmul" grid (recurrent-multiply, depth T=10):
#   {optic_left,optic_right} x {linear,linear_rms,tanh} x {frozenW,trainW} x {from_data}
# = 12 runs (Phase 1), up to 6 A100s concurrently. Every cell is gradient-trained (`train`)
# — even learn_core=false trains the encoder+readout while the connectome stays frozen — so,
# unlike eye_array.sbatch, there is no `stage:` dispatch; we always call `run train`.
# Config selected by $SLURM_ARRAY_TASK_ID over the sorted configs/recmul_experiments/*.yaml.
#   python -m flyconn.models.run recmul_grid       # (re)generate the 12 Phase-1 configs
#   sbatch slurm/recmul_array.sbatch               # run them (override --array for fewer)
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/recmul_experiments/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
  echo "task $SLURM_ARRAY_TASK_ID >= $N configs; nothing to do"; exit 0
fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[recmul-array] task $SLURM_ARRAY_TASK_ID -> $CONFIG"
exec python -u -m flyconn.models.run train --config "$CONFIG"
