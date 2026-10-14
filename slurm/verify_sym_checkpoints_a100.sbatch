#!/bin/bash
#SBATCH --job-name=srppi_t2symld
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.out
#SBATCH --error=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.err
set -euo pipefail
root=/datapool/home/zhangye/projects/surface-residue-ppi
source /datapool/home/zhangye/protein-models/miniconda3/etc/profile.d/conda.sh
conda activate surface-residue-dmasif
export PYTHONUNBUFFERED=1 PYTHONPATH="$root/src:$root/scripts"
cd "$root"
python scripts/verify_sym_checkpoints.py
