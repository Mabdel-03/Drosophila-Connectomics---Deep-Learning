#!/bin/bash
#SBATCH --job-name=prox-dispatch
#SBATCH --account=mit_general
#SBATCH --partition=kellis
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/mabdel03/Drosophila-Connectomics---Deep-Learning/slurm/logs/%x_%j.out

set -euo pipefail
source "/home/mabdel03/Drosophila-Connectomics---Deep-Learning/slurm/proximity_common.sh"
cd "$FLYCONN_REPO"
build_proximity_args

N_TILES=$(python -u -m flyconn.experiments.proximity_full status-field "${PROX_ARGS[@]}" --field n_tiles)
if [ "$N_TILES" -le 0 ]; then
  echo "No tiles found; skipping pair/reduce jobs."
  exit 0
fi

PAIR_CONCURRENCY="${PAIR_CONCURRENCY:-80}"
PAIR_MAX=$((N_TILES - 1))
PAIR_JOB=$(sbatch --parsable --array=0-"$PAIR_MAX"%"$PAIR_CONCURRENCY" "$FLYCONN_REPO/slurm/proximity_pair_tiles_array.sbatch")
echo "Submitted pair tile array: $PAIR_JOB ($N_TILES tiles)"

REDUCE_BUCKETS="${REDUCE_BUCKETS:-64}"
REDUCE_CONCURRENCY="${REDUCE_CONCURRENCY:-8}"
REDUCE_MAX=$((REDUCE_BUCKETS - 1))
REDUCE_JOB=$(sbatch --parsable --dependency=afterok:"$PAIR_JOB" --array=0-"$REDUCE_MAX"%"$REDUCE_CONCURRENCY" "$FLYCONN_REPO/slurm/proximity_reduce_bucket_array.sbatch")
echo "Submitted reduce bucket array: $REDUCE_JOB ($REDUCE_BUCKETS buckets)"

FINAL_JOB=$(sbatch --parsable --dependency=afterok:"$REDUCE_JOB" "$FLYCONN_REPO/slurm/proximity_finalize_reduce.sbatch")
echo "Submitted finalize reduce job: $FINAL_JOB"
