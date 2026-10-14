#!/bin/bash
#SBATCH --job-name=srppi_p3prep
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.out
#SBATCH --error=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.err
set -euo pipefail
root=/datapool/home/zhangye/projects/surface-residue-ppi
python_env=/datapool/home/zhangye/protein-models/miniconda3/envs/surface-residue-dmasif
export PYTHONPATH="$root/src:$root/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$root"
"$python_env/bin/python" scripts/prepare_phase3_data.py --execution slurm --offline \
  --candidates runs/phase3_candidates/20260927/expanded_v1/report.json \
  --out runs/phase3_protocol/20260927/expanded_v1 --workers 8 "$@"
