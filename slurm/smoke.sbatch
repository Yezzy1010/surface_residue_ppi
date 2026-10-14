#!/bin/bash
#SBATCH --job-name=srppi_smoke
#SBATCH --partition=a40
#SBATCH --nodelist=node201
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.out
#SBATCH --error=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.err
set -euo pipefail
root=/datapool/home/zhangye/projects/surface-residue-ppi
source /datapool/home/zhangye/protein-models/miniconda3/etc/profile.d/conda.sh
conda activate surface-residue-dmasif
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/src"
export CUDA_HOME=/datapool/software/cuda/cuda_11.1.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="$root/data/cache/torch"
export TORCH_EXTENSIONS_DIR="$root/data/cache/torch_extensions"
export PYKEOPS_BUILD_FOLDER="$root/data/cache/keops"
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$PYKEOPS_BUILD_FOLDER"
cd "$root"
python -m unittest discover -s tests -v
python scripts/smoke.py "$@"
