#!/bin/bash
# Submit the build+validate job.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch "$HERE/../slurm/build.sbatch"
