#!/usr/bin/env bash
# Resumes the k_u ablation at the DOMAIN half only.
#
# The class half (ku={2,4,6,8}) already completed against the epoch-55 base
# checkpoint; re-running scripts/run_abla_unlearn_class_domain_ku.sh would redo
# those four runs from scratch. This script runs only the 4 domain configs, then
# the shared collector. Log paths and the wandb-dir bookkeeping match the main
# launcher so the collector sees identical filenames for both halves.
#
# Domain configs were cut from 20 to 14 epochs: in the class runs every metric
# was flat after ~epoch 14 (FA moved <1pt, MIA <0.01) and the latest MIA
# crossing was epoch 13. Domain runs take 25 optimizer steps/epoch vs 11 (forget
# set 3169 vs 1381), so they unlearn faster per epoch and cross earlier still.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

DOMAIN_KUS=(2 4 6 8)

run_and_log() {
  local config="$1"
  local run_name="$2"
  local log_file="$LOG_DIR/${run_name}.log"
  echo ""
  echo "=========================================="
  echo "[*] $(date +%T) launching: $run_name"
  echo "\$ $PYTHON unlearn.py --config $config"
  local before after new_dir
  before=$(ls -d wandb/run-*/ 2>/dev/null | sort)
  "$PYTHON" unlearn.py --config "$config" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  after=$(ls -d wandb/run-*/ 2>/dev/null | sort)
  new_dir=$(comm -13 <(echo "$before") <(echo "$after") | head -1)
  if [ -n "$new_dir" ]; then
    echo "$new_dir" > "$LOG_DIR/${run_name}.wandb_dir"
  fi
  if [ "$status" -ne 0 ]; then
    echo "[!] $run_name FAILED (exit $status). See $log_file"
    return "$status"
  fi
  echo "[*] $(date +%T) finished: $run_name"
}

for ku in "${DOMAIN_KUS[@]}"; do
  name="abla_unlearn_domain_ku_${ku}_M12_k4_seed42"
  cfg="config/experiments/${name}.yaml"
  run_and_log "$cfg" "$name" || { echo "[!] $name failed. Fix the underlying issue and rerun this exact config."; exit 1; }
done

echo ""
echo "[*] All 4 domain runs completed. Collecting results..."
"$PYTHON" scripts/collect_abla_unlearn_class_domain_ku_results.py
