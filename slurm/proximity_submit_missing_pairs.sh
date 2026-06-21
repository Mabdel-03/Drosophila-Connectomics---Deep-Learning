#!/bin/bash
# Submit only missing pair tiles, then bucketed reduce/finalize.

set -euo pipefail

source "/home/mabdel03/Drosophila-Connectomics---Deep-Learning/slurm/proximity_common.sh"
cd "$FLYCONN_REPO"
build_proximity_args

PAIR_ARRAY=$(python - <<'PY'
import os

from flyconn.config import load_config
from flyconn.experiments import proximity_full

cfg = load_config(os.environ["CONFIG"])
params = proximity_full.FullRunParams(
    run_name=os.environ["RUN_NAME"],
    shard_size=int(os.environ["SHARD_SIZE"]),
    threshold_um=float(os.environ["THRESHOLD_UM"]),
    connection=os.environ["CONNECTION"],
    mesh_source=os.environ["MESH_SOURCE"],
    mesh_path=os.environ["MESH_PATH"],
    lod=int(os.environ["LOD"]),
    lod_fallback=int(os.environ["LOD_FALLBACK"]),
    site_radius_nm=float(os.environ["SITE_RADIUS_NM"]),
    sample_spacing_nm=float(os.environ["SAMPLE_SPACING_NM"]),
    tile_nm=float(os.environ["TILE_NM"]),
    pair_core_chunk_size=int(os.environ["PAIR_CORE_CHUNK_SIZE"]),
    pair_spill_rows=int(os.environ["PAIR_SPILL_ROWS"]),
    reduce_buckets=int(os.environ["REDUCE_BUCKETS"]),
)
missing = proximity_full.missing_pair_tiles(proximity_full.run_dir(cfg, params))
def ranges(values):
    if not values:
        return ""
    out = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = value
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(out)
print(ranges(missing))
PY
)

if [ -z "$PAIR_ARRAY" ]; then
  echo "No missing pair tiles. Submitting bucketed reduce only."
  PAIR_DEP=""
else
  PAIR_REPAIR_CONCURRENCY="${PAIR_REPAIR_CONCURRENCY:-8}"
  PAIR_JOB=$(sbatch --parsable --array="$PAIR_ARRAY%$PAIR_REPAIR_CONCURRENCY" "$FLYCONN_REPO/slurm/proximity_pair_tiles_array.sbatch")
  echo "Submitted missing pair repair array: $PAIR_JOB ($PAIR_ARRAY)"
  PAIR_DEP=":$PAIR_JOB"
fi

REDUCE_BUCKETS="${REDUCE_BUCKETS:-64}"
REDUCE_CONCURRENCY="${REDUCE_CONCURRENCY:-8}"
REDUCE_MAX=$((REDUCE_BUCKETS - 1))
if [ -n "$PAIR_DEP" ]; then
  REDUCE_JOB=$(sbatch --parsable --dependency=afterok"${PAIR_DEP}" --array=0-"$REDUCE_MAX"%"$REDUCE_CONCURRENCY" "$FLYCONN_REPO/slurm/proximity_reduce_bucket_array.sbatch")
else
  REDUCE_JOB=$(sbatch --parsable --array=0-"$REDUCE_MAX"%"$REDUCE_CONCURRENCY" "$FLYCONN_REPO/slurm/proximity_reduce_bucket_array.sbatch")
fi
echo "Submitted reduce bucket array: $REDUCE_JOB ($REDUCE_BUCKETS buckets)"

FINAL_JOB=$(sbatch --parsable --dependency=afterok:"$REDUCE_JOB" "$FLYCONN_REPO/slurm/proximity_finalize_reduce.sbatch")
echo "Submitted finalize reduce job: $FINAL_JOB"
