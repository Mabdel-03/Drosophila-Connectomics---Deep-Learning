#!/bin/bash
#SBATCH --job-name=flyconn-left-hemisphere
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
# Stage 8 (live CAVE): the bilateral run doubles the family work and adds per-side 500-perm
# nulls, so use the torch-free CAVE env, 48G, 2h. Needs the CAVE token
# (~/.cloudvolume/secrets/cave-secret.json or $CAVE_TOKEN) and the MaleCNS bulk files
# (scripts/malecns_prep.py) for the wing-flip. Live queries are cached -> re-runs are fast.
mamba activate "$FLYCONN_CAVE_ENV"; cd "$FLYCONN_REPO"
exec python -u scripts/left_hemisphere_verify.py "$@"
