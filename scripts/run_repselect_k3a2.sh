#!/usr/bin/env bash
# RepSelect với cấu hình chốt từ sweep2: k_pcs=3, rep_select_lr(alpha)=2.0, LoRA off.
# Chạy đủ 20 epoch + đúng study_name/W&B của main table. Domain trước (yêu cầu),
# rồi class để hai dòng trong bảng dùng chung siêu tham số.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs; DONE=$R/done
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
for scen in domain class; do
  echo ""; echo "######## $(date '+%m-%d %T')  RepSelect/$scen  (k_pcs=3, alpha=2.0)"
  $PY config/experiments/run_moe_pipeline.py \
      --config "config/experiments/pacs_repselect_${scen}.yaml" --stage unlearn \
      --checkpoint "$BEST" --force > "$LOG/RepSelect__${scen}.log" 2>&1
  st=$?; echo "[exit $st]"
  [ $st -eq 0 ] && touch "$DONE/RepSelect__${scen}"
  echo "--- epoch đầu / epoch cuối ---"
  grep -oE "\[RepSelect\] epoch .*" "$LOG/RepSelect__${scen}.log" | sed -n '1p;$p'
done
echo ""; echo "######## $(date '+%m-%d %T')  ### XONG ###"
