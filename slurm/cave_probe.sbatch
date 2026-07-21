#!/bin/bash
#SBATCH --job-name=flyconn-cave-probe
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
# Unconditional override: common.sh set this to scratch (over quota); Stage 5 uses the group volume.
export FLYCONN_DATA_ROOT="/orcd/data/tpoggio/001/mabdel03/connectome_data"
mamba activate "$FLYCONN_CAVE_ENV"; cd "$FLYCONN_REPO"
# Stage 5 pre-flight: does a compute node on pi_tpoggio have CAVE egress? Discover MCNS
# table names. If this prints UNREACHABLE, run scripts/muscular_extract.py on a login node
# instead (it is lightweight) and run only the cache-only verify in SLURM.
exec python -u scripts/cave_probe.py --write
