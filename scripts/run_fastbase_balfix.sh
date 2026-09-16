#!/usr/bin/env bash
# Base-learn 30 epoch với L_bal đã vá. Chờ RepSelect/class bằng marker trong log.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; L=$R/logs/fastbase_balfix.log
until grep -q "### XONG ###" "$R/repselect_k3a2.log" 2>/dev/null; do sleep 60; done
echo "[*] $(date '+%m-%d %T') base-learn 30 epoch, L_bal đã vá"
$PY learn.py --config config/learn/fastbase_balfix_M8_k2.yaml > "$L" 2>&1
echo "[exit $?]"
echo "=== quỹ đạo loss ==="
grep -oE "epoch \[[0-9]+/30\] \| total_loss: [0-9.]+ \(ce: [0-9.]+, sp: [0-9.]+, bal: [0-9.]+, div: [0-9.]+\)" "$L"
echo "=== domain-mass spread ==="
grep -oE "\[domain-mass @ epoch [0-9]+\].*" "$L"
echo "=== chỉ số cuối ==="
grep -oE "\[Final Metrics\].*" "$L" | tail -1
echo "### FASTBASE XONG ###"
