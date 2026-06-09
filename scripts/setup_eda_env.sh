#!/bin/bash
# Create the dedicated EDA conda env (flyconn_eda), install the flyconn package into
# it, and register a Jupyter kernel. Idempotent-ish: re-running updates the env.
#
# Usage:  bash scripts/setup_eda_env.sh
set -euo pipefail

REPO="/orcd/data/tpoggio/001/mabdel03/Connectomics"
# Env lives on SCRATCH, not /home: the /home per-user inode quota (1,000,000 files)
# is exhausted, and a jupyterlab env adds ~60k+ small files. Scratch has ~6.3B inodes.
ENV_PREFIX="/orcd/scratch/orcd/012/mabdel03/conda_envs/flyconn_eda"
MAMBA="/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin/mamba"

mkdir -p "$(dirname "$ENV_PREFIX")"

echo "[setup] creating env at $ENV_PREFIX"
if [ -d "$ENV_PREFIX" ]; then
  "$MAMBA" env update -p "$ENV_PREFIX" -f "$REPO/environment-eda.yml"
else
  "$MAMBA" env create -p "$ENV_PREFIX" -f "$REPO/environment-eda.yml"
fi

# Use the env's own python directly (NOT `mamba run`, which on a loaded login node can
# hang on env activation). --no-deps keeps pip OFF the network: every dependency is
# already provided by conda above, so a plain editable install of flyconn is instant.
PY="$ENV_PREFIX/bin/python"

echo "[setup] installing flyconn (editable, --no-deps) into the env"
"$PY" -m pip install -e "$REPO" --no-build-isolation --no-deps -q

echo "[setup] registering Jupyter kernel 'flyconn_eda'"
"$PY" -m ipykernel install --user \
  --name flyconn_eda --display-name "Python (flyconn_eda)"

echo "[setup] done."
echo "  kernel : Python (flyconn_eda)"
echo "  python : $ENV_PREFIX/bin/python"
echo "  launch : mamba run -p $ENV_PREFIX jupyter lab   (or pick the kernel in VS Code)"
