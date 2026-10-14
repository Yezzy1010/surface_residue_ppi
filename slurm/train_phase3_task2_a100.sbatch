#!/bin/bash
#SBATCH --job-name=srppi_p3t2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%A_%a.out
#SBATCH --error=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%A_%a.err
set -euo pipefail
root=/datapool/home/zhangye/projects/surface-residue-ppi
source /datapool/home/zhangye/protein-models/miniconda3/etc/profile.d/conda.sh
conda activate surface-residue-dmasif
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_THREADING_LAYER=GNU
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/src:$root/scripts"
export CUDA_HOME=/datapool/software/cuda/cuda_11.1.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$root"
python scripts/train_phase3_task2.py --seed-index "$SLURM_ARRAY_TASK_ID"
