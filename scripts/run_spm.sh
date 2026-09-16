#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs; DONE=$R/done
for s in class domain; do
  echo ""; echo "######## $(date '+%m-%d %T')  SPM/$s"
  $PY unlearn.py --config "config/experiments/pacs_spm_unlearn_${s}.yaml" > "$LOG/SPM__${s}.log" 2>&1
  st=$?; echo "[exit $st]"; [ $st -eq 0 ] && touch "$DONE/SPM__${s}"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/SPM__${s}.log" | tail -1
done
echo "### SPM XONG ###"
