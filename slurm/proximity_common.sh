#!/bin/bash
# Shared defaults for full-connectome proximity batch jobs.

module load miniforge/25.11.0-0 2>/dev/null || true

export FLYCONN_REPO="${FLYCONN_REPO:-/home/mabdel03/Drosophila-Connectomics---Deep-Learning}"
export FLYCONN_ENV="${FLYCONN_ENV:-/home/mabdel03/.conda/envs/flyconn}"
export FLYCONN_DATA_ROOT="${FLYCONN_DATA_ROOT:-/net/bmc-lab4/data/kellis/users/mabdel03/files/Connectomics/data}"
_FLYCONN_DATA_ROOT_REQUESTED="$FLYCONN_DATA_ROOT"

export RUN_NAME="${RUN_NAME:-whole_connectome_lod1_sp250_r500_t2_any}"
export SHARD_SIZE="${SHARD_SIZE:-500}"
export THRESHOLD_UM="${THRESHOLD_UM:-2}"
export MESH_SOURCE="${MESH_SOURCE:-cloudvolume-public}"
export MESH_PATH="${MESH_PATH:-precomputed://gs://flywire_v141_m783}"
export LOD="${LOD:-1}"
export LOD_FALLBACK="${LOD_FALLBACK:-0}"
export SITE_RADIUS_NM="${SITE_RADIUS_NM:-500}"
export SAMPLE_SPACING_NM="${SAMPLE_SPACING_NM:-250}"
export TILE_NM="${TILE_NM:-50000}"
export PAIR_CORE_CHUNK_SIZE="${PAIR_CORE_CHUNK_SIZE:-512}"
export PAIR_SPILL_ROWS="${PAIR_SPILL_ROWS:-500000}"
export REDUCE_BUCKETS="${REDUCE_BUCKETS:-64}"
export CONNECTION="${CONNECTION:-any}"
export CONFIG="${CONFIG:-$FLYCONN_REPO/configs/data_v783.yaml}"

mkdir -p "$FLYCONN_DATA_ROOT" "$FLYCONN_REPO/slurm/logs"

case "$-" in
  *u*) _FLYCONN_RESTORE_NOUNSET=1 ;;
  *) _FLYCONN_RESTORE_NOUNSET=0 ;;
esac
set +u

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$FLYCONN_ENV"
elif command -v mamba >/dev/null 2>&1; then
  mamba activate "$FLYCONN_ENV"
else
  echo "ERROR: conda/mamba not found" >&2
  exit 2
fi

if [ "$_FLYCONN_RESTORE_NOUNSET" = "1" ]; then
  set -u
fi

# The flyconn conda env has a configured FLYCONN_DATA_ROOT from earlier work.
# Re-export the requested value after activation so batch jobs target this workspace.
export FLYCONN_DATA_ROOT="$_FLYCONN_DATA_ROOT_REQUESTED"

build_proximity_args() {
  PROX_ARGS=(
    --config "$CONFIG"
    --run-name "$RUN_NAME"
    --shard-size "$SHARD_SIZE"
    --threshold-um "$THRESHOLD_UM"
    --connection "$CONNECTION"
    --mesh-source "$MESH_SOURCE"
    --mesh-path "$MESH_PATH"
    --lod "$LOD"
    --lod-fallback "$LOD_FALLBACK"
    --site-radius-nm "$SITE_RADIUS_NM"
    --sample-spacing-nm "$SAMPLE_SPACING_NM"
    --tile-nm "$TILE_NM"
    --pair-core-chunk-size "$PAIR_CORE_CHUNK_SIZE"
    --pair-spill-rows "$PAIR_SPILL_ROWS"
    --reduce-buckets "$REDUCE_BUCKETS"
  )
  if [ -n "${MAX_NEURONS:-}" ] && [ "${MAX_NEURONS}" != "0" ]; then
    PROX_ARGS+=(--max-neurons "$MAX_NEURONS")
  fi
  if [ "${FORCE:-0}" = "1" ]; then
    PROX_ARGS+=(--force)
  fi
}
