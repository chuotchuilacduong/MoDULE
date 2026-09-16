#!/usr/bin/env bash
# Chạy lại lượt unlearn s14_dense_total sau khi vá mlp_ratio trong unlearn.py.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-1000/-home-admin1-projects-MoDULE/ce607d58-cf99-4c5b-87d3-01167772b49e/scratchpad/unl
R=results/main_table_pacs_seed42
until grep -q "### S14-UNL XONG ###" "$R/s13_s14_unl.log" 2>/dev/null; do sleep 60; done
echo "######## $(date '+%m-%d %T')  chạy lại unlearn s14_dense_total"
$PY config/experiments/run_moe_pipeline.py --config "$S/unl_s14_dense_total.yaml" --stage unlearn \
   --checkpoint "runs/_base_models/s14_dense_total/checkpoints/s14_dense_total.pt" --force \
   > "$S/log_s14_dense_total.txt" 2>&1
echo "[exit $?]"
grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$S/log_s14_dense_total.txt" | tail -1
grep -oE "(Traceback|RuntimeError|size mismatch).*" "$S/log_s14_dense_total.txt" | head -2
echo "### DENSE_TOTAL XONG ###"
