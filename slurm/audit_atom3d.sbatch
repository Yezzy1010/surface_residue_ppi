#!/bin/bash
#SBATCH --job-name=srppi_dips_audit
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.out
#SBATCH --error=/datapool/home/zhangye/projects/surface-residue-ppi/logs/%x_%j.err
set -euo pipefail
root=/datapool/home/zhangye/projects/surface-residue-ppi
python_env=/datapool/home/zhangye/protein-models/miniconda3/envs/surface-residue-dmasif
export PYTHONPATH="$root/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$root"
# Serialize extraction and audit; never run two writers on the same archive.
exec 9>"$root/data/atom3d/zenodo-4911102/.audit.lock"
flock -n 9 || { echo 'Another audit already owns the dataset lock'; exit 1; }
"$python_env/bin/python" -m unittest discover -s tests -p test_ppi_audit.py -v
"$python_env/bin/python" scripts/audit_atom3d.py "$@"
