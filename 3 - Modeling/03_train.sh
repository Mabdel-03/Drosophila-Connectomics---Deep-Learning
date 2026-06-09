#!/bin/bash
# Launch the full 24-run training grid as a SLURM array (<=4 concurrent A100s).
# To train a single run instead:  EXP=optic_left_ff_unroll_initA sbatch ../slurm/train.sbatch
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch "$HERE/../slurm/train_array.sbatch"
