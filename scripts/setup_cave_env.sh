#!/bin/bash
# Create the dedicated Stage-5 connectome/CAVE env (flyconn_cave) and install the flyconn
# package into it. Idempotent-ish: re-running updates the env. No torch (stage 5 is
# parquet + CAVE only; flyconn.io imports torch lazily).
#
# Usage:  bash scripts/setup_cave_env.sh
set -euo pipefail

REPO="/orcd/data/tpoggio/001/mabdel03/Connectomics"
# Env lives on the tpoggio GROUP DATA volume, not /home and not personal scratch:
#   - /home per-user inode quota (1,000,000 files) is exhausted (~867k used).
#   - personal scratch (/orcd/scratch/orcd/012/mabdel03) is OVER QUOTA (mkdir fails).
#   - the group data volume (/orcd/data/tpoggio/001) has ~17B free inodes and ~8T free,
#     and already hosts this repo, so a ~70k-file env fits comfortably.
# Override with FLYCONN_CAVE_ENV to relocate.
ENV_PREFIX="${FLYCONN_CAVE_ENV:-/orcd/data/tpoggio/001/mabdel03/conda_envs/flyconn_cave}"
MAMBA="/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin/mamba"

mkdir -p "$(dirname "$ENV_PREFIX")"

echo "[setup] creating env at $ENV_PREFIX"
if [ -d "$ENV_PREFIX" ]; then
  "$MAMBA" env update -y -p "$ENV_PREFIX" -f "$REPO/environment-cave.yml"
else
  "$MAMBA" env create -y -p "$ENV_PREFIX" -f "$REPO/environment-cave.yml"
fi

# Use the env's own python directly (NOT `mamba run`, which on a loaded login node can
# hang on activation). --no-deps: every flyconn runtime dep is already provided above, so
# this is just a fast editable install of the package itself.
PY="$ENV_PREFIX/bin/python"

echo "[setup] installing flyconn (editable, --no-deps) into the env"
"$PY" -m pip install -e "$REPO" --no-build-isolation --no-deps -q

echo "[setup] verifying the connectome/CAVE stack imports"
"$PY" - <<'PYEOF'
import importlib
mods = ["numpy", "pandas", "pyarrow", "scipy", "yaml",
        "caveclient", "navis", "cloudvolume",
        "flyconn.connectome", "flyconn.circuit", "flyconn.muscular"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"  ok  {m:22} {getattr(mod, '__version__', '')}")
print("[setup] all imports OK")
PYEOF

echo "[setup] done."
echo "  python : $ENV_PREFIX/bin/python"
echo "  tests  : $PY -m pytest tests/ -q"
echo "  probe  : CAVE_TOKEN=... $PY scripts/cave_probe.py"
