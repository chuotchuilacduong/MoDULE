#!/usr/bin/env bash
# S12 / Table 15 — batch-level vs EMA vs no-balance. Ba run tuần tự, batch_size cố định 128.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; L=$R/logs
for est in minibatch ema none; do
  echo ""; echo "######## $(date '+%m-%d %T')  S12 / balance_estimator=$est"
  $PY learn.py --config "config/learn/s12/s12_${est}.yaml" > "$L/s12_${est}.log" 2>&1
  echo "[exit $?]"
  grep -oE "\[\*\] balance estimator: .*" "$L/s12_${est}.log" | head -1
  grep -oE "\[Final Metrics\].*" "$L/s12_${est}.log" | tail -1
  grep -oE "\[domain-mass @ epoch 30\].*" "$L/s12_${est}.log" | tail -1
done
echo "### S12 XONG ###"
