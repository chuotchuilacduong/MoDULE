#!/usr/bin/env bash
# S13 / Table 16 — 5 mục tiêu phân hoá expert. Chờ S12 xong (marker trong log).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; L=$R/logs
until grep -q "### S12 XONG ###" "$R/s12.log" 2>/dev/null; do sleep 60; done
for obj in none cosine orthogonality cka output_decorrelation; do
  echo ""; echo "######## $(date '+%m-%d %T')  S13 / diversity_objective=$obj"
  $PY learn.py --config "config/learn/s13/s13_${obj}.yaml" > "$L/s13_${obj}.log" 2>&1
  echo "[exit $?]"
  grep -oE "\[Final Metrics\].*" "$L/s13_${obj}.log" | tail -1
  grep -oE "\[domain-mass @ epoch 30\].*" "$L/s13_${obj}.log" | tail -1
done
echo "### S13 XONG ###"
