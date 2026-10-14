#!/bin/bash
# Full train/val residue inventory. Read-only: no LMDB writes and no targets.
#SBATCH --job-name=srppi_residue_inventory
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
export PYTHONPATH="$root/src:$root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$root"
exec 9>"$root/data/atom3d/zenodo-4911102/.residue-inventory.lock"
flock -n 9 || { echo 'Another residue inventory already owns the dataset lock'; exit 1; }
"$python_env/bin/python" -m unittest discover -s tests -p 'test_residue_inventory.py' -v
"$python_env/bin/python" scripts/audit_residue_inventory.py --workers 8
