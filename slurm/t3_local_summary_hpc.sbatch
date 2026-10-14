#!/bin/bash
#SBATCH --job-name=srppi_t3local_summary
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.out
#SBATCH --error=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.err
set -euo pipefail
root=/datapool/home/zhangye/projects/surface-residue-ppi
source /datapool/home/zhangye/protein-models/miniconda3/etc/profile.d/conda.sh
conda activate surface-residue-dmasif
export PYTHONPATH="$root/src:$root/scripts"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root"
python scripts/summarize_phase3_task3_local.py --details "$1"
