#!/bin/bash
#SBATCH --job-name=flyconn-muscular-extract
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
# Stage-5 writes (CAVE cache + MCNS intermediates) go to the GROUP volume: personal
# scratch is over quota. common.sh already set FLYCONN_DATA_ROOT to scratch, so this MUST
# be an unconditional override (not :-), pointed at the writable group volume.
export FLYCONN_DATA_ROOT="/orcd/data/tpoggio/001/mabdel03/connectome_data"
mamba activate "$FLYCONN_CAVE_ENV"; cd "$FLYCONN_REPO"
# Stage 5 (network): query MCNS via live CAVE for DN->MN edges + annotations; populate the
# scratch cache + muscular/ intermediates. Requires the [cave] extra + a CAVE token.
# common.sh sources .cave_secret into CAVE_TOKEN if present; otherwise caveclient reads
# ~/.cloudvolume/secrets/cave-secret.json. If compute nodes lack egress (see cave_probe),
# run this script on a login node instead, then chain muscular_verify here.
exec python -u scripts/muscular_extract.py
