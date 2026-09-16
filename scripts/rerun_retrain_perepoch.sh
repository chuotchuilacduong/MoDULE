#!/usr/bin/env bash
# Chạy lại CẢ HAI Retraining (class + domain) với đánh giá FA/RA/TA/MIA theo từng epoch.
# Module.learn() vốn chỉ đánh giá một lần sau toàn bộ vòng lặp nên không để lại quỹ đạo;
# learn_eval_every=1 trong hai config bật phần đó. Chờ launcher chính (24 baseline
# chạy lại sau khi sửa bug gradient) xong để không tranh GPU.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; DONE=$R/done; mkdir -p "$DONE"

while pgrep -f "run_retrain_and_rerun.sh" >/dev/null 2>&1; do
  echo "[*] $(date +%T) chờ launcher chính xong..."; sleep 300
done

for s in class domain; do
  slug="RetrainingPerEpoch__${s}"
  [ -f "$DONE/$slug" ] && { echo "[skip] Retraining/$s (đã xong)"; continue; }
  echo ""; echo "[*] $(date +%T) Retraining/$s — đánh giá theo từng epoch"
  $PYTHON retrain_baseline.py --config "config/main_table_pacs_seed42/retrain_pacs_${s}.yaml" \
    2>&1 | tee "$R/logs/Retraining__${s}__perepoch.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && touch "$DONE/$slug"
  echo "--- quỹ đạo FA ($s) ---"
  grep -ohE "\[eval @ epoch [0-9]+\].*" "$R/logs/Retraining__${s}__perepoch.log" | tail -5
done
echo "[*] $(date +%T) xong cả hai"
