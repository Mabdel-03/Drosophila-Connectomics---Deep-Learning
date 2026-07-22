#!/bin/bash
#SBATCH --job-name=flyconn-initablation
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --array=0-59%8
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out

# Family-5 "initablation": magnitude-vs-direction decomposition on the FULL connectome (whole,
# N=139,255) with the faithful rigid eye on MNIST. Re-initializes the SAME fixed mask 5 ways:
#   faithful (sign+counts), nomag (sign, constant mag), nodirmag (+1, constant mag),
#   shufsign (counts, permuted signs), randmag (lognormal mag, real signs).
# 5 variants x {relu, linearrms} x {untrained, trained} x {seed 0,1,2} = 60 cells, T=8.
# whole is ~3x the edges of the optic* grids -> --mem=120G; relu/linear are <= tanh runtime and
# whole-brain eyeTact cells finished <=3.7h (median ~1h), so 8h walltime is comfortable.
# Up to 8 A100s concurrently. Config selected by $SLURM_ARRAY_TASK_ID over the sorted glob.
#   python -m flyconn.models.run initablation_grid    # (re)generate the 60 configs
#   sbatch slurm/initablation_array.sbatch            # run them (override --array for a subset)
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

mapfile -t CONFIGS < <(ls "$FLYCONN_REPO"/configs/initablation_experiments/*.yaml | sort)
N=${#CONFIGS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
  echo "task $SLURM_ARRAY_TASK_ID >= $N configs; nothing to do"; exit 0
fi
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Skip-if-already-done so a re-launch doesn't redo finished cells.
RUN_NAME="$(basename "$CONFIG" .yaml)"
SUMMARY="$FLYCONN_DATA_ROOT/v783/models/$RUN_NAME/summary.json"
if [ -f "$SUMMARY" ]; then
  echo "[initablation] $RUN_NAME already has summary.json; skipping"; exit 0
fi

echo "[initablation] task $SLURM_ARRAY_TASK_ID -> $CONFIG"
exec python -u -m flyconn.models.run train --config "$CONFIG"
