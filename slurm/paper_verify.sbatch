#!/bin/bash
#SBATCH --job-name=flyconn-paper-verify
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"; cd "$FLYCONN_REPO"
# Stage 5: re-derive EVERY claim in Figure_Ground_Circuit.pdf across FlyWire (live CAVE
# v783) + MaleCNS (offline bulk) and emit the per-claim ledger. Needs the CAVE token
# (~/.cloudvolume/secrets/cave-secret.json or $CAVE_TOKEN) and the MaleCNS bulk files
# (run scripts/malecns_prep.py first). Live queries are cached, so re-runs are fast.
exec python -u scripts/paper_verify.py "$@"
