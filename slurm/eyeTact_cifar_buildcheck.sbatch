#!/bin/bash
#SBATCH --job-name=flyconn-cifar-buildcheck
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/logs/%x_%j.out

# Construct (no training) one CIFAR cell of each input/color path on the FULL connectome to
# prove build_classifier wires the dataset spec correctly against the real graph: learned-luma,
# rigid-luma, and learned-rgb control. Also re-runs the data-free unit suite on the node. Cheap.
set -euo pipefail
source "/orcd/data/tpoggio/001/mabdel03/Connectomics/slurm/common.sh"
mamba activate "$FLYCONN_ENV"
cd "$FLYCONN_REPO"

echo "================ unit tests ================"
python -m pytest tests/test_data.py tests/test_eye.py tests/test_models.py -q

for CFG in \
  eyeTact_cifar_whole_learned_tanh_trained_T01 \
  eyeTact_cifar_whole_rigid_tanh_untrained_T01 \
  eyeTact_cifar_whole_learnedRGB_relu_trained_T04 ; do
  echo "================ build: $CFG ================"
  python -u -m flyconn.models.run build \
    --config "configs/eyeTact_cifar_experiments/$CFG.yaml"
done
echo "================ buildcheck DONE ================"
