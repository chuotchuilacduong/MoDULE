#!/usr/bin/env bash
# Active-routing-k ablation (class + domain unlearning), PACS / MoDULE / M=12 / learn_k=4 / ku=12 / seed=42.
#
# Exactly 12 primary runs: 6 class (unlearn_k = 2,4,6,8,10,12) + 6 domain (same),
# via the repo's `python unlearn.py --config <yaml>` CLI.
#
# unlearn_k is the number of experts combined per input during the UNLEARNING
# forward pass. It is distinct from:
#   M       = 12  total experts
#   learn_k = 4   routing top-k the base checkpoint was trained with (gate_k)
#   ku      = 12  experts eligible to update during unlearning
# Before this ablation, unlearn-time active-k was hard-wired to len(allowed_experts)
# == k_u (architecture/module.py:68), so it could not be varied independently. The
# `unlearn_active_k` config field now drives it, defaulting to the old behaviour
# when unset.
#
# No stopping logic: fa_threshold is -1.0 in every config, so FA can never reach it
# and all 20 epochs always run. Reported result is the epoch-20 result.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

BASE_CKPT="runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt"
ACTIVE_KS=(2 4 6 8 10 12)

cfg_name() {  # $1=scenario $2=k
  if [ "$2" = "12" ]; then echo "abla_unlearn_$1_activek_12_dense_ku12_M12_learnk4_seed42"
  else echo "abla_unlearn_$1_activek_$2_ku12_M12_learnk4_seed42"; fi
}

echo "=========================================="
echo "[*] PRE-LAUNCH VERIFICATION"
echo "=========================================="
echo " 1. base checkpoint      : $BASE_CKPT"
[ -f "$BASE_CKPT" ] && echo "                          (exists)" || { echo "[!] MISSING base checkpoint"; exit 1; }
echo " 2. forgotten class      : 0 (dog)          [class runs]"
echo " 3. forgotten domain     : 3 (sketch)       [domain runs]"
echo " 4. M (num_experts)      : $(grep -m1 '^num_experts:' config/experiments/$(cfg_name class 2).yaml | awk '{print $2}')"
echo " 5. learn_k (gate_k)     : $(grep -m1 '^gate_k:' config/experiments/$(cfg_name class 2).yaml | awk '{print $2}')"
echo " 6. ku (k_u)             : $(grep -m1 '^k_u:' config/experiments/$(cfg_name class 2).yaml | awk '{print $2}')"
echo " 7. unlearn_k values     : ${ACTIVE_KS[*]}"
echo " 8. seed                 : $(grep -m1 '^seed:' config/experiments/$(cfg_name class 2).yaml | awk '{print $2}')"
echo " 9. epochs               : $(grep -m1 '^epochs:' config/experiments/$(cfg_name class 2).yaml | awk '{print $2}')"
echo "10. stopping mechanism   : fa_threshold=$(grep -m1 '^fa_threshold:' config/experiments/$(cfg_name class 2).yaml | awk '{print $2}') -> unreachable, no early stop, no best-epoch selection"
echo "11. LR                   : $(grep -m1 '^lr:' config/experiments/$(cfg_name class 2).yaml | awk '{print $2}') (fixed, no sweep)"
echo "12. the exact 12 commands:"
for scen in class domain; do
  for k in "${ACTIVE_KS[@]}"; do
    echo "    $PYTHON unlearn.py --config config/experiments/$(cfg_name $scen $k).yaml"
  done
done
echo "13. GPU assignment       : single GPU on this host (CUDA_VISIBLE_DEVICES=0) -> sequential; no parallelization possible."
echo "=========================================="

# per-config sanity check: every run must differ ONLY in unlearn_k / scenario.
for scen in class domain; do
  for k in "${ACTIVE_KS[@]}"; do
    f="config/experiments/$(cfg_name $scen $k).yaml"
    got_k=$(grep -m1 '^unlearn_active_k:' "$f" | awk '{print $2}')
    got_ku=$(grep -m1 '^k_u:' "$f" | awk '{print $2}')
    got_ep=$(grep -m1 '^epochs:' "$f" | awk '{print $2}')
    got_fa=$(grep -m1 '^fa_threshold:' "$f" | awk '{print $2}')
    if [ "$got_k" != "$k" ] || [ "$got_ku" != "12" ] || [ "$got_ep" != "20" ] || [ "$got_fa" != "-1.0" ]; then
      echo "[!] config mismatch in $f (unlearn_active_k=$got_k k_u=$got_ku epochs=$got_ep fa_threshold=$got_fa)"; exit 1
    fi
  done
done
echo "[*] all 12 configs verified: k_u=12, epochs=20, fa_threshold=-1.0, unlearn_active_k as labelled."
echo ""

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

for scen in class domain; do
  for k in "${ACTIVE_KS[@]}"; do
    name="$(cfg_name $scen $k)"
    run_and_log "config/experiments/${name}.yaml" "$name" \
      || { echo "[!] $name failed. Fix the underlying issue and rerun this exact config."; exit 1; }
  done
done

echo ""
echo "[*] All 12 runs completed. Collecting results..."
"$PYTHON" scripts/collect_abla_unlearn_activek_results.py
