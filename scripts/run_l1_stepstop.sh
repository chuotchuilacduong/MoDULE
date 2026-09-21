#!/usr/bin/env bash
# l1-sparse bản cũ (ascent, expert-only, lr 1e-4) + early-stop theo bước (FA đo mỗi 5 bước, dừng khi <= 10%).
#   PY=<python có torch> nohup bash scripts/run_l1_stepstop.sh > l1_stepstop.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/l1_stepstop; mkdir -p "$LOG"
for cfg in config/l1_stepstop/*.yaml; do
  n=$(basename "$cfg" .yaml); [ -f "$LOG/$n.done" ] && { echo "[skip] $n"; continue; }
  ck=$("$PY" -c "import yaml;print(yaml.safe_load(open('$cfg'))['pretrained_model_path'])"); [ -f "$ck" ] || { echo "[skip] $n — thiếu $ck"; continue; }
  echo ""; echo "######## $(date '+%m-%d %T')  $n"
  "$PY" -m unlearn --config "$cfg" > "$LOG/$n.log" 2>&1 && touch "$LOG/$n.done" || echo "[FAIL]"
  grep -E "early stop|Metrics:" "$LOG/$n.log" | tail -2
done
echo "### $(date '+%m-%d %T') XONG ###"
