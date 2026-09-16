#!/usr/bin/env bash
# Table 16 (I.2.3) — chỉ thiếu ModULE w/o L_sp; ba dòng kia tái sử dụng checkpoint đã có.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; L=$R/logs
until grep -q "### DENSE_TOTAL XONG ###" "$R/dense_total_rerun.log" 2>/dev/null; do sleep 60; done
echo "######## $(date '+%m-%d %T')  base-learn s16_no_lsp (30 epoch)"
$PY learn.py --config config/learn/s16/s16_no_lsp.yaml > "$L/s16_no_lsp.log" 2>&1
echo "[exit $?]"; grep -oE "\[Final Metrics\].*" "$L/s16_no_lsp.log" | tail -1
echo "### S16-LEARN XONG ###"
