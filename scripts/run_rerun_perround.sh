#!/usr/bin/env bash
# Chạy lại run 5jzeg9uy với cùng cấu hình, nhưng log class_mass/domain_mass theo TỪNG ROUND
# dưới dạng scalar phẳng (mỗi cặp lớp×expert một khoá) để W&B vẽ đường thay vì bảng cột.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=results/main_table_pacs_seed42/logs/base_balfix_rerun_perround.log
echo "[*] $(date '+%m-%d %T') base-learn 100 epoch, log mass mỗi 2 epoch"
$PY learn.py --config config/learn/base_balfix_rerun_perround.yaml > "$L" 2>&1
echo "[exit $?]"
grep -oE "\[Final Metrics\].*" "$L" | tail -1
grep -oE "\[class-mass @ epoch [0-9]+\].*" "$L" | tail -1
echo "### RERUN XONG ###"
