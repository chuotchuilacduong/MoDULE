#!/usr/bin/env bash
# Pass 2: chạy lại những job chưa có marker trong results/.../done/ (job thất bại ở
# pass 1). Launcher chính đã có sẵn cơ chế bỏ qua theo marker nên chỉ cần gọi lại nó.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R=results/main_table_pacs_seed42
while pgrep -f "run_retrain_and_rerun.sh|rerun_retrain_perepoch.sh" >/dev/null 2>&1; do
  echo "[*] $(date +%T) chờ pass 1 + retrain per-epoch xong..."; sleep 300
done
echo "[*] $(date +%T) pass 2 — job còn thiếu: $(( 24 - $(ls $R/done 2>/dev/null | grep -vc Retraining) ))"
exec bash scripts/run_retrain_and_rerun.sh
