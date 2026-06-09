#!/bin/bash
#SBATCH --job-name=flyconn-build
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out

# Build node table + edge list + all adjacency matrices, then validate.
# --mem=120G headroom for the 9.5 GB synapse feather (peak ~30-60 GB); the core
# build (proofread_connections, 852 MB) needs far less.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

python -u -m flyconn.data_prep.run build    --config "$FLYCONN_REPO/configs/data_v783.yaml"
python -u -m flyconn.data_prep.run validate --config "$FLYCONN_REPO/configs/data_v783.yaml"
