#!/bin/bash
# Submit the whole-connectome proximity pipeline.
#
# Useful overrides:
#   MAX_NEURONS=50 bash slurm/proximity_submit_full.sh   # smoke test
#   SAMPLE_CONCURRENCY=16 PAIR_CONCURRENCY=40 bash slurm/proximity_submit_full.sh

set -euo pipefail

export FLYCONN_REPO="${FLYCONN_REPO:-/home/mabdel03/Drosophila-Connectomics---Deep-Learning}"
export FLYCONN_ENV="${FLYCONN_ENV:-/home/mabdel03/.conda/envs/flyconn}"
export FLYCONN_DATA_ROOT="${FLYCONN_DATA_ROOT:-/net/bmc-lab4/data/kellis/users/mabdel03/files/Connectomics/data}"
export RUN_NAME="${RUN_NAME:-whole_connectome_lod1_sp250_r500_t2_any}"
export SHARD_SIZE="${SHARD_SIZE:-500}"
export SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-32}"
export PAIR_CONCURRENCY="${PAIR_CONCURRENCY:-80}"

mkdir -p "$FLYCONN_REPO/slurm/logs"

if [ -n "${MAX_NEURONS:-}" ] && [ "${MAX_NEURONS}" != "0" ]; then
  N_NEURONS="$MAX_NEURONS"
else
  N_NEURONS=139255
fi
SAMPLE_MAX=$(((N_NEURONS + SHARD_SIZE - 1) / SHARD_SIZE - 1))

PREPARE_JOB=$(sbatch --parsable "$FLYCONN_REPO/slurm/proximity_prepare.sbatch")
echo "Submitted prepare job: $PREPARE_JOB"

SAMPLE_JOB=$(sbatch --parsable --dependency=afterok:"$PREPARE_JOB" --array=0-"$SAMPLE_MAX"%"$SAMPLE_CONCURRENCY" "$FLYCONN_REPO/slurm/proximity_sample_array.sbatch")
echo "Submitted sample array: $SAMPLE_JOB (0-$SAMPLE_MAX)"

TILES_JOB=$(sbatch --parsable --dependency=afterany:"$SAMPLE_JOB" "$FLYCONN_REPO/slurm/proximity_build_tiles.sbatch")
echo "Submitted build-tiles job: $TILES_JOB"

DISPATCH_JOB=$(sbatch --parsable --dependency=afterok:"$TILES_JOB" "$FLYCONN_REPO/slurm/proximity_dispatch_tiles.sbatch")
echo "Submitted tile dispatch job: $DISPATCH_JOB"
