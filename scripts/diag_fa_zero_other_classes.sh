#!/usr/bin/env bash
# Retrain bỏ lớp 4 (horse) và lớp 5 (house), 15 epoch mỗi lượt, để kiểm tra FA=0.00
# có phải hiện tượng chung hay chỉ riêng lớp 0. Chờ bằng marker trong log, KHÔNG
# dùng pgrep (pgrep khớp vào cmdline của shell bọc -> deadlock như lần trước).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs
until grep -q "### XONG ###" "$R/repselect_k3a2.log" 2>/dev/null; do sleep 60; done
for c in 4 5; do
  echo ""; echo "######## $(date '+%m-%d %T')  Retrain bỏ lớp $c (15 epoch)"
  $PY retrain_baseline.py --config "config/main_table_pacs_seed42/retrain_pacs_class$c.yaml" \
      > "$LOG/DIAG_Retraining__class$c.log" 2>&1
  echo "[exit $?]"
  echo "--- FA từng epoch ---"
  grep -oE "eval @ epoch [0-9]+\] ra: [0-9.]+% \| fa: [0-9.]+% \| ta: [0-9.]+%" "$LOG/DIAG_Retraining__class$c.log"
done
echo ""; echo "######## $(date '+%m-%d %T')  ### DIAG XONG ###"
