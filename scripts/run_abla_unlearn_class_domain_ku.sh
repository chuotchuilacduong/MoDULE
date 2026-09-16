#!/usr/bin/env bash
# Launcher for the k_u ablation (class + domain unlearning), PACS / MoDULE / M=12 / k=4 / seed=42.
#
# Contains exactly 8 primary experiment runs (4 class ku={2,4,6,8}, 4 domain ku={2,4,6,8}),
# via the repo's existing `python unlearn.py --config <yaml>` CLI. Also trains the M=12,k=4
# base checkpoint once, since no such checkpoint existed in the repo before this run.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

BASE_CONFIG="config/experiments/pacs_module_base_M12_k4.yaml"
# epoch-55 pre-divergence checkpoint (TA 91.99%). the epoch-100 final checkpoint
# from the same run scored TA 80.68% after diverging at epoch ~57 -- see the
# grad_clip_norm/lr_schedule note in pacs_module_base_M12_k4.yaml.
BASE_CKPT="runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt"

CLASS_KUS=(2 4 6 8)
DOMAIN_KUS=(2 4 6 8)

echo "=========================================="
echo "[*] Base checkpoint (PACS, M=12, k=4, seed=42): $BASE_CKPT"
echo "[*] Forgotten class (class-unlearning runs): 0 (dog)"
echo "[*] Forgotten domain (domain-unlearning runs): 3 (sketch)"
echo "[*] Exact commands:"
echo "    $PYTHON learn.py --config $BASE_CONFIG   (only if base checkpoint is missing)"
for ku in "${CLASS_KUS[@]}"; do
  echo "    $PYTHON unlearn.py --config config/experiments/abla_unlearn_class_ku_${ku}_M12_k4_seed42.yaml"
done
for ku in "${DOMAIN_KUS[@]}"; do
  echo "    $PYTHON unlearn.py --config config/experiments/abla_unlearn_domain_ku_${ku}_M12_k4_seed42.yaml"
done
echo "[*] GPU assignment: single GPU on this host (CUDA_VISIBLE_DEVICES=0) -> runs execute sequentially, no parallelization possible."
echo "=========================================="

# This host's one GPU was already occupied (~15/16GB) by an unrelated, separately
# started job (retrain_baseline.py, class0/domain3 gold-retrain baselines) when this
# launcher was written. Wait for it to finish before starting our own training so we
# don't OOM either job.
while pgrep -f "retrain_baseline.py" > /dev/null; do
  echo "[*] $(date +%T) waiting for existing retrain_baseline.py job(s) to free the GPU..."
  sleep 60
done
echo "[*] $(date +%T) GPU is free of retrain_baseline.py jobs, proceeding."

run_and_log() {
  local config="$1"
  local run_name="$2"
  local log_file="$LOG_DIR/${run_name}.log"
  local script="$3"
  echo ""
  echo "=========================================="
  echo "[*] $(date +%T) launching: $run_name"
  echo "\$ $PYTHON $script --config $config"
  local before after new_dir
  before=$(ls -d wandb/run-*/ 2>/dev/null | sort)
  "$PYTHON" "$script" --config "$config" 2>&1 | tee "$log_file"
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

if [ -f "$BASE_CKPT" ]; then
  echo "[skip] base checkpoint already exists: $BASE_CKPT"
else
  run_and_log "$BASE_CONFIG" "base_pacs_M12_k4_seed42" "learn.py" || { echo "[!] base training failed."; exit 1; }
fi

if [ ! -f "$BASE_CKPT" ]; then
  echo "[!] base checkpoint still missing after training: $BASE_CKPT"
  exit 1
fi

for ku in "${CLASS_KUS[@]}"; do
  name="abla_unlearn_class_ku_${ku}_M12_k4_seed42"
  cfg="config/experiments/${name}.yaml"
  run_and_log "$cfg" "$name" "unlearn.py" || { echo "[!] $name failed. Fix the underlying issue and rerun this exact config."; exit 1; }
done

for ku in "${DOMAIN_KUS[@]}"; do
  name="abla_unlearn_domain_ku_${ku}_M12_k4_seed42"
  cfg="config/experiments/${name}.yaml"
  run_and_log "$cfg" "$name" "unlearn.py" || { echo "[!] $name failed. Fix the underlying issue and rerun this exact config."; exit 1; }
done

echo ""
echo "[*] All 8 runs completed. Collecting results..."
"$PYTHON" scripts/collect_abla_unlearn_class_domain_ku_results.py
