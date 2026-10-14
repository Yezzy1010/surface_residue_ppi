#!/bin/bash
#SBATCH --job-name=srppi_p3feat
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%A_%a.out
#SBATCH --error=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%A_%a.err
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
cd "$root"
python scripts/extract_phase2_features.py --root "$root" \
  --protocol runs/phase3_protocol/20260927/expanded_v1/report.json \
  --exclude-report runs/phase2_features/275876_final_v2/report.json \
  --output-root runs/phase3_features --shard-count 4 --shard-index "$SLURM_ARRAY_TASK_ID" \
  --seed 20260927
