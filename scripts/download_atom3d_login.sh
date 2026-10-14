#!/bin/bash
set -euo pipefail
BASE=/datapool/home/zhangye/projects/surface-residue-ppi/data/atom3d/zenodo-4911102
URL='https://zenodo.org/api/records/4911102/files/PPI-DIPS-split.tar.gz/content'
FINAL="$BASE/PPI-DIPS-split.tar.gz"
PART="$FINAL.part"
EXPECTED_BYTES=13568121746
EXPECTED_MD5=b6095db528e4a2e1139fddc64b6965de
JOB_TAG="login_$(date +%Y%m%d_%H%M%S)"
LOG="$BASE/download_\${JOB_TAG}.log"
REPORT="$BASE/download_\${JOB_TAG}.txt"
mkdir -p "$BASE"
exec 9>"$BASE/.download.lock"
flock -n 9 || { echo "another downloader owns destination" >&2; exit 1; }
{
  echo "pid=$$"
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
  if [ "$actual_bytes" -eq "$EXPECTED_BYTES" ] && [ "$actual_md5" = "$EXPECTED_MD5" ]; then
    echo "status=already_verified" | tee -a "$REPORT"
    exit 0
  fi
  echo "status=existing_final_mismatch; refusing overwrite" | tee -a "$REPORT"
  exit 4
fi
wget --continue --tries=30 --timeout=120 --waitretry=30 \
  --progress=dot:giga -O "$PART" "$URL" 2>&1 | tee -a "$LOG"
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
echo "status=verified" | tee -a "$REPORT"
echo "finished=$(date --iso-8601=seconds)" | tee -a "$REPORT"
