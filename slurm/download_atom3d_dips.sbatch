#!/bin/bash
#SBATCH --job-name=srppi_dips_dl
#SBATCH --partition=hpc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/datapool/home/zhangye/projects/surface-residue-ppi/logs/srppi_dips_dl_%j.out
#SBATCH --error=/datapool/home/zhangye/projects/surface-residue-ppi/logs/srppi_dips_dl_%j.err

set -euo pipefail

BASE=/datapool/home/zhangye/projects/surface-residue-ppi/data/atom3d/zenodo-4911102
mkdir -p "$BASE"
URL='https://zenodo.org/api/records/4911102/files/PPI-DIPS-split.tar.gz/content'
FINAL="$BASE/PPI-DIPS-split.tar.gz"
PART="$FINAL.part"
EXPECTED_BYTES=13568121746
EXPECTED_MD5=b6095db528e4a2e1139fddc64b6965de
REPORT="$BASE/download_report_${SLURM_JOB_ID}.txt"
exec 9>"$BASE/.download.lock"
flock -n 9 || { echo 'A download already owns this destination'; exit 1; }
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "status=failed exit_code=$code" >> "$REPORT"; fi' EXIT

{
  echo "job_id=$SLURM_JOB_ID"
  echo "host=$(hostname)"
  echo "started=$(date --iso-8601=seconds)"
  echo "url=$URL"
  echo "expected_bytes=$EXPECTED_BYTES"
  echo "expected_md5=$EXPECTED_MD5"
  echo "part_before=$(stat -c %s "$PART" 2>/dev/null || echo 0)"
} | tee "$REPORT"

if [ -f "$FINAL" ]; then
  actual_bytes=$(stat -c %s "$FINAL")
  actual_md5=$(md5sum "$FINAL" | awk '{print $1}')
  if [ "$actual_bytes" = "$EXPECTED_BYTES" ] && [ "$actual_md5" = "$EXPECTED_MD5" ]; then
    echo "already_verified=true" | tee -a "$REPORT"
    exit 0
  fi
  echo "status=existing_final_mismatch; preserve for inspection" | tee -a "$REPORT"
  exit 4
fi

wget --continue --tries=30 --timeout=120 --waitretry=30 \
  --server-response --output-file="$BASE/wget_${SLURM_JOB_ID}.log" \
  -O "$PART" "$URL"

actual_bytes=$(stat -c %s "$PART")
actual_md5=$(md5sum "$PART" | awk '{print $1}')
{
  echo "part_after=$actual_bytes"
  echo "actual_md5=$actual_md5"
} | tee -a "$REPORT"

if [ "$actual_bytes" -ne "$EXPECTED_BYTES" ]; then
  echo "status=size_mismatch" | tee -a "$REPORT"
  exit 2
fi
if [ "$actual_md5" != "$EXPECTED_MD5" ]; then
  echo "status=md5_mismatch" | tee -a "$REPORT"
  exit 3
fi

mv "$PART" "$FINAL"
{
  echo "status=verified"
  echo "finished=$(date --iso-8601=seconds)"
} | tee -a "$REPORT"
