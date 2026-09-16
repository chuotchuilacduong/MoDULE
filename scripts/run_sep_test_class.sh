#!/usr/bin/env bash
# L_sep theo trục CLASS trên PACS (đợt 2). 4 run x 30 epoch (~23 phút/run trên server):
#   Đợt 3 (cả hai trục):  RUNS="sep_on_both sep_on_layerwise sep_on_both_sumgate" bash scripts/run_sep_test_class.sh
#   Đợt 4 (ô class x domain): RUNS="sep_on_joint sep_on_joint_both sep_on_joint_both_sumgate" bash scripts/run_sep_test_class.sh
#   sep_on_class          lambda_sep=0.5, axis=class, gate softmax (mặc định)   -> so với sep_off (đợt 1)
#   sep_on_class_pi       như trên nhưng MI tính trên softmax pi (sep_use_gated=false, không bị cap 0.731)
#   sep_off_sumgate       lambda_sep=0,   gate_norm=sum  (đối chứng cho dòng dưới; đổi forward pass)
#   sep_on_class_sumgate  lambda_sep=0.5, axis=class, gate_norm=sum
# Chạy từ gốc repo:  PY=<python có torch> nohup bash scripts/run_sep_test_class.sh > sep_test_class.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/sep_test; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
[ -d dataset/data_folder/pacs ] || "$PY" -m dataset.downloader.pacs || exit 1

for name in ${RUNS:-sep_on_class sep_on_class_pi sep_off_sumgate sep_on_class_sumgate}; do
  if [ -f "$DONE/$name" ]; then echo "[skip] $name"; continue; fi
  echo ""; echo "######## $(date '+%m-%d %T')  $name"
  "$PY" -m learn --config "config/learn/sep_test/$name.yaml" > "$LOG/$name.log" 2>&1
  st=$?; [ $st -eq 0 ] && touch "$DONE/$name" || { echo "[FAIL exit $st]"; grep -E "Traceback|Error" "$LOG/$name.log" | tail -3; }
  grep -E "^epoch \[(1|10|20|30)/" "$LOG/$name.log"
  grep -oE "\[Final Metrics\].*" "$LOG/$name.log" | tail -1
  ck=$(ls -t "$(grep -oE 'runs/_base_models/septest_[^/]+/checkpoints' "config/learn/sep_test/$name.yaml")"/*_best.pt 2>/dev/null | head -1)
  outdir=$(dirname "$(dirname "$ck")"); cp "config/learn/sep_test/$name.yaml" "$outdir/learn.yaml"
  "$PY" scripts/print_mass_tables.py --checkpoint "$ck" --split test 2>&1 \
    | grep -vE "Warning|param counts|MoE |├|└|─|TOTAL|Original|classifier|featurizer \(ViT|Experts per|Expert depth|Expert hidden|Router top" \
    | tee "$LOG/mass_tables_$name.txt"
done
echo ""; echo "### $(date '+%m-%d %T') SEP-TEST CLASS XONG ###"
