#!/bin/bash
# Corrected continuation for records whose first full extraction used a
# case-collapsed SEQRES key.  The frozen protocol and all other outputs remain
# unchanged; this job writes replacement caches for only the listed indices.
#SBATCH --job-name=srppi_p2resume
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.out
#SBATCH --error=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.err
set -euo pipefail
root=/datapool/home/zhangye/projects/surface-residue-ppi
source /datapool/home/zhangye/protein-models/miniconda3/etc/profile.d/conda.sh
conda activate surface-residue-dmasif
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/src"
export CUDA_HOME=/datapool/software/cuda/cuda_11.1.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="$root/data/cache/torch"
export TORCH_EXTENSIONS_DIR="$root/data/cache/torch_extensions"
export PYKEOPS_BUILD_FOLDER="$root/data/cache/keops"
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$PYKEOPS_BUILD_FOLDER" "$root/data/cache"
cd "$root"
python -m py_compile scripts/extract_phase2_features.py
python scripts/extract_phase2_features.py --root "$root" \
  --protocol runs/phase2_protocol/20260925/local_20250925_phase2_protocol_01/report.json \
  --lmdb-indices 60866 79582 24076 24225
