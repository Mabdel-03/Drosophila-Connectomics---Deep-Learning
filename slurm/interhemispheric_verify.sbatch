#!/bin/bash
#SBATCH --job-name=flyconn-interhemispheric
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
# Stage 9 (live CAVE): identify how the left and right figure circuits connect. The heaviest
# pulls are the heterolateral-bridge discovery (N2) and the systematic sweep (N5), which read
# every input onto the figure machinery, so use the torch-free CAVE env, 48G, 2h. Needs the CAVE
# token (~/.cloudvolume/secrets/cave-secret.json or $CAVE_TOKEN). Live queries are cached -> reruns
# are fast.
mamba activate "$FLYCONN_CAVE_ENV"; cd "$FLYCONN_REPO"
exec python -u scripts/interhemispheric_verify.py "$@"
