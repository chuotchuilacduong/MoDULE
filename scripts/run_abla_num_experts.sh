#!/usr/bin/env bash
# Number-of-experts ablation (PACS / MoDULE / learn_k=4 / ku=4 / seed=42).
#
#   phase 1: train any missing base checkpoint (100 epochs, no early stopping)
#   phase 2: one unlearning run per M (20 epochs, no stopping/selection)
#   phase 3: collect
#
# Unlearning scenario is the repo's canonical one for this ablation, taken from the
# learn block of config/experiments/pacs_module_num_experts_sweep.yaml:
#   unlearn_setting: random, forget_ratio: 0.1  (10% of the training split)
#
# M=12 is trained FIRST so the forget-size ablation, which shares that checkpoint,
# unblocks as early as possible.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

# M=8 and M=12 reuse pre-existing checkpoints (see PHASE 1 below), so only these
# three are trained.
TRAIN_ORDER=(4 16 24)
UNLEARN_ORDER=(4 8 12 16 24)

ckpt_for() { echo "runs/_base_models/abla_num_experts_M$1_k4_seed42/checkpoints/abla_learn_num_experts_M$1_k4_seed42.pt"; }

run_and_log() {
  local script="$1" config="$2" run_name="$3"
  local log_file="$LOG_DIR/${run_name}.log"
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
  [ -n "$new_dir" ] && echo "$new_dir" > "$LOG_DIR/${run_name}.wandb_dir"
  if [ "$status" -ne 0 ]; then
    echo "[!] $run_name FAILED (exit $status). See $log_file"
    return "$status"
  fi
  echo "[*] $(date +%T) finished: $run_name"
}

echo "=========================================="
echo "[*] PHASE 0 — unlearning for M values whose checkpoint already exists"
echo "    (front-loaded so rows land before the remaining training finishes)"
echo "=========================================="
for M in "${UNLEARN_ORDER[@]}"; do
  name="abla_unlearn_num_experts_M${M}_ku4_k4_seed42"
  cfg="config/experiments/${name}.yaml"
  ck=$(grep -m1 '^pretrained_model_path:' "$cfg" | awk '{print $2}')
  if [ -f "$LOG_DIR/${name}.log" ] && grep -q "Unlearning finished" "$LOG_DIR/${name}.log" 2>/dev/null; then
    echo "[skip] $name already completed"
  elif [ -f "$ck" ]; then
    run_and_log "unlearn.py" "$cfg" "$name" \
      || { echo "[!] $name failed. Fix the underlying issue and rerun this exact config."; exit 1; }
  else
    echo "[defer] M=$M -- checkpoint not trained yet: $ck"
  fi
done
echo ""
echo "[*] partial collection after PHASE 0:"
"$PYTHON" scripts/collect_abla_num_experts.py || true

echo "=========================================="
echo "[*] PHASE 1 — base checkpoints"
echo "=========================================="
echo "[reuse] M=8  -> runs/_base_models/3ae530097a/checkpoints/learn_best.pt"
echo "        NOTE: that checkpoint was trained with gate_k=2 (not 4) and"
echo "        lambda 0.5/2.0/2.0, so the M=8 row is not learn_k=4 comparable."
echo "[reuse] M=12 -> runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt"
echo "        gate_k=4 as specified; trained with lambda 0.5/2.0/2.0 (epoch-55 best)."
for M in "${TRAIN_ORDER[@]}"; do
  ck="$(ckpt_for "$M")"
  name="abla_learn_num_experts_M${M}_k4_seed42"
  if [ -f "$ck" ]; then
    echo "[skip] M=$M checkpoint already present: $ck"
  else
    echo "[train] M=$M -> $ck"
    run_and_log "learn.py" "config/experiments/${name}.yaml" "$name" \
      || { echo "[!] learning for M=$M failed."; exit 1; }
    [ -f "$ck" ] || { echo "[!] expected checkpoint missing after training: $ck"; exit 1; }
  fi
done

echo ""
echo "=========================================="
echo "[*] PHASE 2 — unlearning for the remaining M (ku=4, 20 epochs, no stopping)"
echo "=========================================="
for M in "${UNLEARN_ORDER[@]}"; do
  name="abla_unlearn_num_experts_M${M}_ku4_k4_seed42"
  if [ -f "$LOG_DIR/${name}.log" ] && grep -q "Unlearning finished" "$LOG_DIR/${name}.log" 2>/dev/null; then
    echo "[skip] $name already completed in PHASE 0"
    continue
  fi
  run_and_log "unlearn.py" "config/experiments/${name}.yaml" "$name" \
    || { echo "[!] $name failed. Fix the underlying issue and rerun this exact config."; exit 1; }
done

echo ""
echo "[*] All number-of-experts runs completed. Collecting..."
"$PYTHON" scripts/collect_abla_num_experts.py
