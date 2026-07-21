#!/bin/bash
#SBATCH --job-name=flyconn-cifar-download
#SBATCH --account=mit_general
#SBATCH --partition=pi_tpoggio
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out

# One-time CIFAR-10 warm-up: download + cache the dataset to scratch
# ($FLYCONN_DATA_ROOT/v783/cifar10) BEFORE launching any eyeTact_cifar array, so the
# array tasks don't race on a simultaneous first download over shared scratch. CPU-only.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

# Build both transform variants once so luma + rgb caches are present.
exec python -u -c "
from flyconn.models.data import get_loaders
for color in ('luma', 'rgb'):
    tr, va, te = get_loaders(batch_size=256, num_workers=2, download=True,
                             dataset='cifar10', color=color)
    print(f'[cifar-download] color={color}: train={len(tr.dataset)} '
          f'val={len(va.dataset)} test={len(te.dataset)} OK')
"
