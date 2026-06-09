#!/bin/bash
#SBATCH --job-name=flyconn-download
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out

# Pull all FlyWire FAFB v783 files (Zenodo connectivity ~10.5 GB + GitHub annotations)
# to scratch, md5-verified and resumable. CPU-only; runs as a batch job (not login).
#
# Optional env knobs:
#   ONLY="proofread_root_ids_783.npy"   # smoke-test on the small file first
#   FORCE=1                             # redownload even if already valid
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

ARGS=()
[ -n "${ONLY:-}" ] && ARGS+=(--only $ONLY)
[ -n "${FORCE:-}" ] && ARGS+=(--force)

exec python -u -m flyconn.data_prep.run download \
  --config "$FLYCONN_REPO/configs/data_v783.yaml" "${ARGS[@]}"
