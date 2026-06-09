#!/bin/bash
# Submit the validate job (regenerates the data card from existing artifacts).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch "$HERE/../slurm/validate.sbatch"
