#!/usr/bin/env bash
# Retraining/domain (gold standard): train từ đầu trên tập retain, không bao giờ thấy
# domain 3. Lần trước bị dừng ở epoch 19/100 nên checkpoint trên đĩa không dùng được.
# learn_eval_every=1 trong config -> có quỹ đạo FA/RA/TA/MIA theo từng epoch.
# Chờ SG-Unlearning và bảng quét RepSelect xong để không tranh GPU.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42

while pgrep -f "run_sg_matched.sh|sweep_repselect.sh|run_two_fixes.sh" >/dev/null 2>&1; do
  echo "[*] $(date +%T) chờ job trước xong..."; sleep 180
done
echo "[*] $(date +%T) bắt đầu Retraining/domain (100 epoch)"
$PY retrain_baseline.py --config config/main_table_pacs_seed42/retrain_pacs_domain.yaml \
  2>&1 | tee "$R/logs/Retraining__domain.log"
st=${PIPESTATUS[0]}
[ "$st" -eq 0 ] && touch "$R/done/Retraining__domain"
echo "[*] $(date +%T) kết thúc (exit $st)"
echo "=== chỉ số cuối ==="
grep -oE "\[Final Metrics\].*" "$R/logs/Retraining__domain.log" | tail -1
echo "=== quỹ đạo FA (10 epoch cuối) ==="
grep -oE "\[eval @ epoch [0-9]+\].*" "$R/logs/Retraining__domain.log" | tail -10
