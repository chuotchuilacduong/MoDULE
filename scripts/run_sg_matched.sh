#!/usr/bin/env bash
# SG-Unlearning class + domain với tập unseen KHỚP domain/lớp cho proxy.
# alpha trả về 0.1 (mặc định) vì tăng lên 1.0 đã chứng minh không đổi kết quả:
# FA 99.50 -> 99.59, Proxy M Loss vẫn đứng ở mức ngẫu nhiên.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs; DONE=$R/done
while pgrep -f "run_two_fixes.sh" >/dev/null 2>&1; do
  echo "[*] $(date +%T) chờ RepSelect xong..."; sleep 180
done
for s in class domain; do
  echo ""; echo "[*] $(date +%T) launching: SG-Unlearning/$s (matched unseen)"
  $PYTHON unlearn.py --config "config/main_table_pacs_seed42/sg_unlearning_pacs_${s}.yaml" \
    2>&1 | tee "$LOG/SG-Unlearning__${s}.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && touch "$DONE/SG-Unlearning__${s}"
  echo "--- Proxy M Loss đầu/cuối ---"
  grep -oE "Proxy M Loss: [0-9.]+" "$LOG/SG-Unlearning__${s}.log" | sed -n '1p;$p' | tr '\n' ' '; echo
done
echo "### XONG ###"
