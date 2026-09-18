#!/usr/bin/env bash
# Sensitivity to modularization loss weights (lambda_sp, lambda_bal, lambda_div) trên PACS.
# 7 tổ hợp x [base 30 epoch (M8/k2) -> ModULE unlearn class 0, k_u=4, 20 epoch]. ~50 phút / tổ hợp trên server.
# Clean TA = test_accuracy của base; RA/FA = sau unlearn. Bảng cuối: scripts/collect_lossw.py
#   PY=<python có torch> nohup bash scripts/run_lossw_ablation.sh > lossw.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/lossw; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
[ -d dataset/data_folder/pacs ] || "$PY" -m dataset.downloader.pacs || exit 1
TAGS=${TAGS:-"sp0_bal1_div1 sp1_bal0_div1 sp1_bal1_div0 sp0p1_bal1_div1 sp1_bal0p1_div1 sp1_bal1_div0p1 sp1_bal1_div1"}

for t in $TAGS; do
  ck=runs/_base_models/lossw_$t/checkpoints/base_${t}_best.pt
  if [ -f "$ck" ] && [ -f "$DONE/base_$t" ]; then echo "[skip] base_$t"; else
    echo ""; echo "######## $(date '+%m-%d %T')  base_$t"
    "$PY" -m learn --config "config/learn/lossw/base_$t.yaml" > "$LOG/base_$t.log" 2>&1
    st=$?; [ $st -eq 0 ] && touch "$DONE/base_$t" || { echo "[FAIL exit $st]"; grep -E "Traceback|Error" "$LOG/base_$t.log" | tail -2; continue; }
    grep -oE "\[Final Metrics\].*" "$LOG/base_$t.log" | tail -1
  fi
  if [ -f "$DONE/unl_$t" ]; then echo "[skip] unl_$t"; continue; fi
  echo "######## $(date '+%m-%d %T')  unl_$t"
  "$PY" -m unlearn --config "config/learn/lossw/unl_$t.yaml" > "$LOG/unl_$t.log" 2>&1
  st=$?; [ $st -eq 0 ] && touch "$DONE/unl_$t" || { echo "[FAIL exit $st]"; grep -E "Traceback|Error" "$LOG/unl_$t.log" | tail -2; }
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/unl_$t.log" | tail -1
done
echo ""; echo "### $(date '+%m-%d %T') LOSS-WEIGHT ABLATION XONG ###"
"$PY" scripts/collect_lossw.py
