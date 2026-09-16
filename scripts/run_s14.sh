#!/usr/bin/env bash
# S14 / Table 14 — architecture control dưới ngân sách tham số khớp nhau. Chờ S13 xong.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; L=$R/logs
until grep -q "### S13 XONG ###" "$R/s13.log" 2>/dev/null; do sleep 60; done
for a in dense_active dense_total standard_moe moe_no_modular module; do
  echo ""; echo "######## $(date '+%m-%d %T')  S14 / $a"
  $PY learn.py --config "config/learn/s14/s14_${a}.yaml" > "$L/s14_${a}.log" 2>&1
  echo "[exit $?]"
  grep -oE "bỏ qua [0-9]+ trọng số pretrained lệch shape" "$L/s14_${a}.log" | head -1
  grep -oE "TOTAL \(with MoE\) *: *[0-9,]+" "$L/s14_${a}.log" | head -1
  grep -oE "\[Final Metrics\].*" "$L/s14_${a}.log" | tail -1
done
echo "### S14 XONG ###"
