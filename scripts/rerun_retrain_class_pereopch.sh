#!/usr/bin/env bash
# Chạy lại Retraining/class với đánh giá FA/RA/TA/MIA theo TỪNG epoch.
# Module.learn() vốn chỉ đánh giá một lần sau toàn bộ vòng lặp, nên lần chạy trước
# không để lại quỹ đạo nào — chỉ có [Final Metrics]. learn_eval_every=1 trong
# config/main_table_pacs_seed42/retrain_pacs_class.yaml bật phần đó.
# Chờ launcher chính xong để không tranh GPU.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42

while pgrep -f "run_retrain_and_rerun.sh" >/dev/null 2>&1; do
  echo "[*] $(date +%T) chờ launcher chính xong..."; sleep 300
done
echo "[*] $(date +%T) bắt đầu Retraining/class có đánh giá theo epoch"
$PYTHON retrain_baseline.py --config config/main_table_pacs_seed42/retrain_pacs_class.yaml \
  2>&1 | tee "$R/logs/Retraining__class__perepoch.log"
echo "[*] $(date +%T) xong"
echo "=== quỹ đạo FA ==="
grep -oE "\[eval @ epoch [0-9]+\].*" "$R/logs/Retraining__class__perepoch.log" | tail -20
