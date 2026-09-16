#!/usr/bin/env bash
# Retraining/class chạy lại đủ 100 epoch. Lần trước bị kill ở epoch 88 nên trên đĩa
# chỉ còn _best.pt (chọn theo train-loss thấp nhất, KHÔNG phải theo TA) -> con số
# FA/RA/TA đang có không phải final-epoch như quy ước của main table.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42

while pgrep -f "run_sg_matched.sh|sweep_repselect.sh|run_two_fixes.sh|run_retrain_domain.sh" >/dev/null 2>&1; do
  sleep 180
done
echo "[*] $(date +%T) bắt đầu Retraining/class (100 epoch)"
$PY retrain_baseline.py --config config/main_table_pacs_seed42/retrain_pacs_class.yaml \
  > "$R/logs/Retraining__class.log" 2>&1
st=$?
[ "$st" -eq 0 ] && touch "$R/done/Retraining__class"
echo "[*] $(date +%T) kết thúc (exit $st)"
grep -oE "\[Final Metrics\].*" "$R/logs/Retraining__class.log" | tail -1
