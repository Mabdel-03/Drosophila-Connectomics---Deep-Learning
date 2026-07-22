#!/bin/bash
#SBATCH --job-name=flyconn-activity
#SBATCH --account=mit_general
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%A_%a.out
# Per-neuron firing-activity ranking for the 6 learned-eye eyeTact models
# ({linear,tanh,relu} x {untrained,trained}), full MNIST test set, T=8.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"; cd "$FLYCONN_REPO"
RUNS=(
  eyeTact_whole_learned_linear_untrained_T08
  eyeTact_whole_learned_linear_trained_T08
  eyeTact_whole_learned_tanh_untrained_T08
  eyeTact_whole_learned_tanh_trained_T08
  eyeTact_whole_learned_relu_untrained_T08
  eyeTact_whole_learned_relu_trained_T08
)
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "[activity-array] $SLURM_ARRAY_TASK_ID -> $RUN"
exec python -u -m flyconn.models.activity --run "$RUN" --top 50
