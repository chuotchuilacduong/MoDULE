#!/usr/bin/env bash
# Chạy lại SPM (kèm chẩn đoán support bank) và SSD. Chờ lượt SSD hiện tại xong trước.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs; DONE=$R/done
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
until grep -q "### SSD XONG ###" "$R/ssd.log" 2>/dev/null; do sleep 30; done

for s in class domain; do
  echo ""; echo "######## $(date '+%m-%d %T')  SPM/$s (có chẩn đoán)"
  $PY unlearn.py --config "config/experiments/pacs_spm_unlearn_${s}.yaml" > "$LOG/SPM__${s}.log" 2>&1
  st=$?; echo "[exit $st]"; [ $st -eq 0 ] && touch "$DONE/SPM__${s}"
  grep -E "\[SPM-DIAG\]" "$LOG/SPM__${s}.log"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/SPM__${s}.log" | tail -1
  grep -oE "(Traceback|Error|error:).*" "$LOG/SPM__${s}.log" | head -2
done
echo "### SPM+SSD XONG ###"
