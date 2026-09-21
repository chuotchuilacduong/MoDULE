#!/usr/bin/env bash
# Sequential domain removal on PACS (domain-wise counterpart of CoUn's Fig. 6):
# 3 stages, one domain each (3 sketch -> 1 cartoon -> 0 art_painting).
# Runs ModULE, SalUn and Boundary Shrink, each chained over the stages in
# config/sequential/pacs_*_seq_domain.yaml, plus the per-stage Retrain reference
# (shared by the three algorithms, trained once).
#
# Order: the three unlearning chains first (results early), then the Retrain
# references, then a final pass that only recomputes summary.csv with the gaps.
# Every step is resumable: finished stages are skipped, so the script can be
# re-run after an interruption.
#   bash scripts/run_sequential_pacs_domain.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PY:-/home/duong/miniconda3/envs/module/bin/python3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SETTING=domain
BASE=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
[ -f "$BASE" ] || { echo "[!] thiếu base checkpoint: $BASE"; exit 1; }
BASE12=runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt
[ -f "$BASE12" ] || { echo "[!] thiếu base M12/k4 cho ModULE: $BASE12"; exit 1; }

ALGOS=(module salun boundary_shrink)

run() {   # $1=config $2...=driver flags
  local cfg="$1"; shift
  local name; name=$(basename "$cfg" .yaml)
  echo ""; echo "=========================================="
  echo "[*] $(date +%T) $name $*"; echo "=========================================="
  $PYTHON sequential_unlearn.py --config "$cfg" "$@"
  local st=$?
  [ "$st" -ne 0 ] && echo "[!] $name FAILED (exit $st)"
  return "$st"
}

echo ""; echo "### PHASE 1 ($SETTING) — unlearning chains ###"
for a in "${ALGOS[@]}"; do
  run "config/sequential/pacs_${a}_seq_${SETTING}.yaml" || true
done

echo ""; echo "### PHASE 2 ($SETTING) — per-stage Retrain reference (shared) ###"
run "config/sequential/pacs_module_seq_${SETTING}.yaml" --retrain-only || true

echo ""; echo "### PHASE 3 ($SETTING) — summary.csv with gaps ###"
for a in "${ALGOS[@]}"; do
  run "config/sequential/pacs_${a}_seq_${SETTING}.yaml" --retrain || true
done

echo ""; echo "### XONG ($SETTING) ###"
for a in "${ALGOS[@]}"; do
  f="runs/sequential/pacs_${a}_seq_${SETTING}/summary.csv"; [ -f "$f" ] && { echo "--- $f"; cat "$f"; }
done
