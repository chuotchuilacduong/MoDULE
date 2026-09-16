#!/usr/bin/env bash
# SSD (Selective Synaptic Dampening) cho bảng chính PACS, dùng cùng base aaedd1da4d
# như mọi baseline khác để so sánh được.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs; DONE=$R/done
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
for s in class domain; do
  echo ""; echo "######## $(date '+%m-%d %T')  SSD/$s"
  $PY config/experiments/run_moe_pipeline.py --config "config/experiments/pacs_ssd_${s}.yaml" \
      --stage unlearn --checkpoint "$BEST" --force > "$LOG/SSD__${s}.log" 2>&1
  st=$?; echo "[exit $st]"; [ $st -eq 0 ] && touch "$DONE/SSD__${s}"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/SSD__${s}.log" | tail -1
  grep -oE "(Traceback|Error|error:).*" "$LOG/SSD__${s}.log" | head -2
done
echo "### SSD XONG ###"
