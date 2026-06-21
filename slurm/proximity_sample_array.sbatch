#!/bin/bash
#SBATCH --job-name=prox-sample
#SBATCH --account=mit_general
#SBATCH --partition=kellis
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/mabdel03/Drosophila-Connectomics---Deep-Learning/slurm/logs/%x_%A_%a.out

set -euo pipefail
source "/home/mabdel03/Drosophila-Connectomics---Deep-Learning/slurm/proximity_common.sh"
cd "$FLYCONN_REPO"
build_proximity_args
exec python -u -m flyconn.experiments.proximity_full sample-shard "${PROX_ARGS[@]}" --shard-id "$SLURM_ARRAY_TASK_ID"
