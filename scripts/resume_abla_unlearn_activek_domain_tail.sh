#!/usr/bin/env bash
# Resumes the active-k ablation at the two outstanding domain runs.
#
# scripts/run_abla_unlearn_activek_class_domain.sh was killed part-way through
# domain active-k=10 (after its epoch 1). Runs 1-10 (all 6 class + domain
# active-k 2/4/6/8) completed and their logs are intact; re-running the full
# launcher would redo them from scratch (~13 h wasted). This runs only the
# remaining two configs, then the shared collector.
#
# Configs are untouched -- same k_u=12, epochs=20, fa_threshold=-1.0.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

REMAINING=(
  "abla_unlearn_domain_activek_10_ku12_M12_learnk4_seed42"
  "abla_unlearn_domain_activek_12_dense_ku12_M12_learnk4_seed42"
)

run_and_log() {
  local run_name="$1"
  local config="config/experiments/${run_name}.yaml"
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
  [ -n "$new_dir" ] && echo "$new_dir" > "$LOG_DIR/${run_name}.wandb_dir"
  if [ "$status" -ne 0 ]; then
    echo "[!] $run_name FAILED (exit $status). See $log_file"
    return "$status"
  fi
  echo "[*] $(date +%T) finished: $run_name"
}

for name in "${REMAINING[@]}"; do
  run_and_log "$name" || { echo "[!] $name failed. Fix the underlying issue and rerun this exact config."; exit 1; }
done

echo ""
echo "[*] Remaining domain runs completed. Collecting all 12 results..."
"$PYTHON" scripts/collect_abla_unlearn_activek_results.py
