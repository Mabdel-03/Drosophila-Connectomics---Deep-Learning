#!/bin/bash
#SBATCH --job-name=prox-prepare
#SBATCH --account=mit_general
#SBATCH --partition=kellis
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/home/mabdel03/Drosophila-Connectomics---Deep-Learning/slurm/logs/%x_%j.out

set -euo pipefail
source "/home/mabdel03/Drosophila-Connectomics---Deep-Learning/slurm/proximity_common.sh"
cd "$FLYCONN_REPO"
build_proximity_args
exec python -u -m flyconn.experiments.proximity_full prepare "${PROX_ARGS[@]}"
