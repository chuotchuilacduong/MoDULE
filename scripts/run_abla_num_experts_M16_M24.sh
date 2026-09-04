#!/usr/bin/env bash
# Portable launcher for the remaining half of the number-of-experts ablation:
# M=16 and M=24, learning (100 epochs) + unlearning (20 epochs) + collection.
#
# Self-contained and resumable -- rerunning skips whatever already finished.
# Intended to be run on a different machine from the M=4/8/12 rows.
#
#   PACS must be at ./dataset/data_folder/pacs (not in the repo).
#
# Usage:
#   bash scripts/run_abla_num_experts_M16_M24.sh
#   PYTHON=/path/to/python bash scripts/run_abla_num_experts_M16_M24.sh
#   BIG_GPU=1 bash scripts/run_abla_num_experts_M16_M24.sh    # >16GB VRAM
#   GPU=1 bash scripts/run_abla_num_experts_M16_M24.sh        # pick CUDA device
#
# BIG_GPU=1 switches the learning runs from batch 32 x accum 4 to batch 128 x
# accum 1. Both are an effective batch of 128 and are mathematically identical
# (LayerNorm only, no BatchNorm; verified to 2e-07 relative gradient error), but
# the un-accumulated form is faster where the activations fit. The 32x4 default
# exists because at M>=16 the full batch pins a 16GB card at 98% memory and
# throughput collapses (M=16: 2400s/epoch vs 281s/epoch accumulated).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

MS=(16 24)

learn_cfg()   { echo "config/experiments/abla_learn_num_experts_M$1_k4_seed42.yaml"; }
unlearn_cfg() { echo "config/experiments/abla_unlearn_num_experts_M$1_ku4_k4_seed42.yaml"; }
learn_name()   { echo "abla_learn_num_experts_M$1_k4_seed42"; }
unlearn_name() { echo "abla_unlearn_num_experts_M$1_ku4_k4_seed42"; }
ckpt_for()    { echo "runs/_base_models/abla_num_experts_M$1_k4_seed42/checkpoints/abla_learn_num_experts_M$1_k4_seed42.pt"; }

echo "=========================================="
echo "[*] number-of-experts ablation -- M=16 and M=24"
echo "[*] python : $PYTHON"
echo "[*] gpu    : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[*] dataset: ./dataset/data_folder/pacs"
[ -d "dataset/data_folder/pacs" ] || { echo "[!] PACS not found at dataset/data_folder/pacs"; exit 1; }
"$PYTHON" -c "import torch;print('[*] torch  :',torch.__version__,'| cuda',torch.cuda.is_available())" || exit 1

if [ "${BIG_GPU:-0}" = "1" ]; then
  echo "[*] BIG_GPU=1 -> learning at batch 128, grad_accum_steps 1 (same effective batch)"
  for M in "${MS[@]}"; do
    sed -i 's/^batch_size: 32$/batch_size: 128/; s/^grad_accum_steps: 4$/grad_accum_steps: 1/' "$(learn_cfg "$M")"
  done
fi
for M in "${MS[@]}"; do
  printf "[*] M=%-3s learn: " "$M"
  grep -hE "^num_experts:|^gate_k:|^epochs:|^batch_size:|^grad_accum_steps:" "$(learn_cfg "$M")" | tr '\n' ' '; echo
done
echo "=========================================="

run_and_log() {
  local script="$1" config="$2" run_name="$3"
  local log_file="$LOG_DIR/${run_name}.log"
  echo ""
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

for M in "${MS[@]}"; do
  ck="$(ckpt_for "$M")"
  if [ -f "$ck" ]; then
    echo "[skip] M=$M base checkpoint present: $ck"
  else
    run_and_log "learn.py" "$(learn_cfg "$M")" "$(learn_name "$M")" \
      || { echo "[!] learning for M=$M failed."; exit 1; }
    [ -f "$ck" ] || { echo "[!] expected checkpoint missing after training: $ck"; exit 1; }
  fi

  un="$(unlearn_name "$M")"
  if [ -f "$LOG_DIR/${un}.log" ] && grep -q "Unlearning finished" "$LOG_DIR/${un}.log" 2>/dev/null; then
    echo "[skip] $un already completed"
  else
    run_and_log "unlearn.py" "$(unlearn_cfg "$M")" "$un" \
      || { echo "[!] $un failed."; exit 1; }
  fi
done

echo ""
echo "[*] M=16 and M=24 done. Collecting..."
"$PYTHON" scripts/collect_abla_num_experts.py

cat <<'NOTE'

------------------------------------------------------------------
The M=4/8/12 rows were produced on another machine, so this table
will mark them MISSING here. To merge, copy back to the M=4/8/12 host:

    results/logs/abla_learn_num_experts_M16_k4_seed42.log
    results/logs/abla_learn_num_experts_M24_k4_seed42.log
    results/logs/abla_unlearn_num_experts_M16_ku4_k4_seed42.log
    results/logs/abla_unlearn_num_experts_M24_ku4_k4_seed42.log
    results/logs/abla_unlearn_num_experts_M{16,24}_*.wandb_dir
    runs/_base_models/abla_num_experts_M{16,24}_k4_seed42/checkpoints/*.pt

then rerun: python scripts/collect_abla_num_experts.py
------------------------------------------------------------------
NOTE
