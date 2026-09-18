#!/usr/bin/env bash
# Loss-weight ablation, lượt unlearn KHÔNG early-stop (fa_threshold=-1, đủ 20 epoch) trên 7 base đã có.
#   PY=<python có torch> nohup bash scripts/run_lossw_unlearn_full.sh > lossw_full.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/lossw; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
for t in sp0_bal1_div1 sp1_bal0_div1 sp1_bal1_div0 sp0p1_bal1_div1 sp1_bal0p1_div1 sp1_bal1_div0p1 sp1_bal1_div1; do
  [ -f "runs/_base_models/lossw_$t/checkpoints/base_${t}_best.pt" ] || { echo "[!] thiếu base $t"; continue; }
  if [ -f "$DONE/unlfull_$t" ]; then echo "[skip] unlfull_$t"; continue; fi
  echo ""; echo "######## $(date '+%m-%d %T')  unlfull_$t"
  "$PY" -m unlearn --config "config/learn/lossw/unlfull_$t.yaml" > "$LOG/unlfull_$t.log" 2>&1
  st=$?; [ $st -eq 0 ] && touch "$DONE/unlfull_$t" || { echo "[FAIL exit $st]"; grep -E "Traceback|Error" "$LOG/unlfull_$t.log" | tail -2; }
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/unlfull_$t.log" | tail -1
done
echo ""; echo "### $(date '+%m-%d %T') XONG ###"
"$PY" scripts/collect_lossw.py --full20
