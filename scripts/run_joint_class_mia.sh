#!/usr/bin/env bash
# Joint (non-sequential) class unlearning on PACS for the membership-leakage figure:
# each request {0}, {0,1}, {0,1,2} starts independently from the ORIGINAL base checkpoint.
# The 1-class runs are reused from the sequential experiment (configs verified identical);
# this script only runs the 2- and 3-class requests. Retrain models already exist under
# runs/sequential/pacs_retrain_class/ (50 epochs, ImageNet init, forget set never trained on).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PY:-/home/admin1/miniconda3/envs/Messi/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/joint_class_mia/logs; mkdir -p "$LOG"
for cfg in module_joint_c2 module_joint_c3 seuf_joint_c2 seuf_joint_c3; do
  ck="runs/joint_class/${cfg}/checkpoints"
  algo=$([ "${cfg:0:6}" = "module" ] && echo module || echo seuf)
  if [ -f "$ck/unlearned_${algo}_${cfg}.pt" ] && grep -q "unlearning complete" "$LOG/${cfg}.log" 2>/dev/null; then
    echo "[*] $cfg already finished; skipping"; continue
  fi
  echo ""; echo "=========================================="; echo "[*] $(date +%T) $cfg"; echo "=========================================="
  $PYTHON unlearn.py --config "config/joint_class/${cfg}.yaml" > "$LOG/${cfg}.log" 2>&1
  st=$?; [ "$st" -ne 0 ] && echo "[!] $cfg FAILED (exit $st)"
  tail -3 "$LOG/${cfg}.log"
done
echo ""; echo "### JOINT UNLEARNING RUNS DONE ###"
