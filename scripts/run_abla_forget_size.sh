#!/usr/bin/env bash
# Forget-set-size ablation (PACS / MoDULE / M=12 / learn_k=4 / ku=4 / seed=42).
#
# All five runs share ONE base checkpoint; the model is never retrained.
# Forget sets come from the repo's own construction in unlearn.py: random_split on
# the seeded train split. random_split permutes sum(lengths) == len(train_subset),
# which is identical across runs, then takes a prefix -- so the forget sets are
# nested (1% subset of 5% subset of 10% subset of 20% subset of 40%). Verified empirically.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

# reused epoch-55 M=12/gate_k=4 checkpoint (TA 91.99% on the test split).
# trained with lambda 0.5/2.0/2.0 rather than the 0.1/0.25/0.1 in the learn configs;
# that only affects how it was produced, not this unlearning phase.
BASE_CKPT="runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt"
PCTS=(1 5 10 20 40)

echo "=========================================="
echo "[*] shared base checkpoint: $BASE_CKPT"
if [ -f "$BASE_CKPT" ]; then
  echo "    (exists)"
else
  echo "[!] base checkpoint missing. Run scripts/run_abla_num_experts.sh first"
  echo "    (its phase 1 trains M=12 before any other M precisely for this)."
  exit 1
fi
echo "=========================================="

run_and_log() {
  local config="$1" run_name="$2"
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

for p in "${PCTS[@]}"; do
  name="abla_unlearn_forget_size_${p}pct_M12_ku4_k4_seed42"
  run_and_log "config/experiments/${name}.yaml" "$name" \
    || { echo "[!] $name failed. Fix the underlying issue and rerun this exact config."; exit 1; }
done

echo ""
echo "[*] All forget-size runs completed. Collecting..."
"$PYTHON" scripts/collect_abla_forget_size.py
