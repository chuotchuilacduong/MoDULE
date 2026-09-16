#!/usr/bin/env bash
# Base-learn ĐẦY ĐỦ 100 epoch với L_bal đã vá. Không có gì khác chạy song song.
# Cùng công thức với base cũ aaedd1da4d (M=8, k=2, lr 5e-4, lambda 0.5/0.5/0.5)
# nên so sánh được trực tiếp: cũ cho collapse rơi ngẫu nhiên 0.099-1.000 và cover 1-2/4,
# bản 30 epoch sau khi vá cho collapse 0.167/0.151 và cover 4/4.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; L=$R/logs/base_balfix_100ep.log
echo "[*] $(date '+%m-%d %T') base-learn 100 epoch (L_bal đã vá)"
$PY learn.py --config config/learn/base_balfix_M8_k2_100ep.yaml > "$L" 2>&1
echo "[exit $?]"
grep -oE "\[Final Metrics\].*" "$L" | tail -1
grep -oE "\[domain-mass @ epoch [0-9]+\].*" "$L" | tail -3
echo "### BASE100 XONG ###"
