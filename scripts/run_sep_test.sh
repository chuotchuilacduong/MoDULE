#!/usr/bin/env bash
# A/B L_sep trên PACS: lambda_sep 0.0 vs 0.5, 30 epoch mỗi run (~1h/run trên card 16GB), rồi in bảng so sánh.
# Chạy từ gốc repo:  PY=python nohup bash scripts/run_sep_test.sh > sep_test.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/sep_test; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
[ -d dataset/data_folder/pacs ] || "$PY" -m dataset.downloader.pacs || exit 1

for tag in off on; do
  if [ -f "$DONE/$tag" ]; then echo "[skip] sep_$tag"; continue; fi
  echo ""; echo "######## $(date '+%m-%d %T')  sep_$tag"
  "$PY" -m learn --config "config/learn/sep_test/sep_${tag}.yaml" > "$LOG/sep_${tag}.log" 2>&1
  st=$?; [ $st -eq 0 ] && touch "$DONE/$tag" || { echo "[FAIL exit $st]"; grep -E "Traceback|Error" "$LOG/sep_${tag}.log" | tail -3; }
  grep -E "^epoch \[(1|10|20|30)/" "$LOG/sep_${tag}.log"
  grep -oE "\[Final Metrics\].*" "$LOG/sep_${tag}.log" | tail -1
done

echo ""; echo "### $(date '+%m-%d %T') BẢNG DOMAIN x EXPERT / CLASS x EXPERT (test split, gated mass) ###"
for v in 0.0 0.5; do
  ck=runs/_base_models/septest_lsep${v}/checkpoints/sep_$([ "$v" = "0.0" ] && echo off || echo on)_best.pt
  cp "config/learn/sep_test/sep_$([ "$v" = "0.0" ] && echo off || echo on).yaml" "runs/_base_models/septest_lsep${v}/learn.yaml"
  echo ""; echo "################ lambda_sep = $v ################"
  "$PY" scripts/print_mass_tables.py --checkpoint "$ck" --split test 2>&1 | grep -vE "Warning|param counts|MoE |├|└|─|TOTAL|Original|classifier|featurizer \(ViT|Experts per|Expert depth|Expert hidden|Router top" \
    | tee "$LOG/mass_tables_lsep${v}.txt"
done
echo ""; echo "### online_spec tổng hợp từ W&B ###"
"$PY" scripts/compare_sep_test.py
