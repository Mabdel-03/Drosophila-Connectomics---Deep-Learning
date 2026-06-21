#!/bin/bash
#SBATCH --job-name=prox-tiles
#SBATCH --account=mit_general
#SBATCH --partition=kellis
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/mabdel03/Drosophila-Connectomics---Deep-Learning/slurm/logs/%x_%j.out

set -euo pipefail
source "/home/mabdel03/Drosophila-Connectomics---Deep-Learning/slurm/proximity_common.sh"
cd "$FLYCONN_REPO"
build_proximity_args
exec python -u -m flyconn.experiments.proximity_full build-tiles "${PROX_ARGS[@]}"
