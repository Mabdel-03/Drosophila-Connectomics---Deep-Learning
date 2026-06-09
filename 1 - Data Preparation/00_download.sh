#!/bin/bash
# Submit the download job. Pass ONLY=<key> / FORCE=1 through as needed, e.g.:
#   ONLY=proofread_root_ids_783.npy ./00_download.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch "$HERE/../slurm/download.sbatch"
