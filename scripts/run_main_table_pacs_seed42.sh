#!/usr/bin/env bash
# Main Table 1 — PACS only, seed 42, class + domain.
# Mọi baseline dùng chung base checkpoint learn_best.pt (checkpoint train-loss tốt nhất)
# thay vì learn.pt: config learn của pipeline không có grad clipping/LR schedule, và đúng
# công thức đó đã làm base M=12/k=4 diverge ở epoch ~57 (CE 0.056 -> 1.81, TA 91.99 -> 80.68).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/main_table_pacs_seed42/logs; mkdir -p "$LOG"
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt

while pgrep -f "learn.py --config .*aaedd1da4d" >/dev/null 2>&1; do
  echo "[*] $(date +%T) chờ base model train xong..."; sleep 180
done
[ -f "$BEST" ] || { echo "[!] thiếu base checkpoint: $BEST"; exit 1; }
echo "[*] $(date +%T) base sẵn sàng: $BEST"

validate_and_run() {   # $1=entry $2=config $3=disp $4...=extra args
  local entry="$1" cfg="$2" disp="$3"; shift 3
  local name scen lr
  name=$($PYTHON -c "import yaml;d=yaml.safe_load(open('$cfg'));u=d.get('unlearn',d);print(u['study_name'])")
  scen=$($PYTHON -c "import yaml;d=yaml.safe_load(open('$cfg'));u=d.get('unlearn',d);print(u.get('scenario') or u.get('unlearn_setting'))")
  lr=$($PYTHON   -c "import yaml;d=yaml.safe_load(open('$cfg'));u=d.get('unlearn',d);print(u['lr'])")
  echo ""; echo "=========================================="
  echo "Baseline: $disp"; echo "Dataset: PACS"; echo "Scenario: $scen"
  echo "Config: $cfg"; echo "W&B run name: $name"; echo "Seed: 42"
  echo "LR: $lr"; echo "Checkpoint: $BEST"; echo "=========================================="
  for token in "${disp// /}" PACS "$scen" "$(basename "$cfg" .yaml)"; do
    case "$name" in *"$token"*) ;; *) echo "[!] ABORT: '$token' thiếu trong run name '$name'"; return 1;; esac
  done
  $PYTHON "$entry" --config "$cfg" "$@" 2>&1 | tee "$LOG/${disp// /}__${scen}.log"
  local st=${PIPESTATUS[0]}
  [ "$st" -ne 0 ] && { echo "[!] $disp/$scen FAILED (exit $st)"; return "$st"; }
  echo "[*] $(date +%T) finished: $name"
}

echo "### PHASE 1 — pipeline baselines (--stage unlearn, dùng learn_best.pt) ###"
for s in class domain; do
  for b in grip seuf repselect salun; do
    case $b in grip) D=GRIP;; seuf) D=SEUF;; repselect) D=RepSelect;; salun) D=SalUn;; esac
    validate_and_run config/experiments/run_moe_pipeline.py "config/experiments/pacs_${b}_${s}.yaml" "$D" \
      --stage unlearn --checkpoint "$BEST" || true
  done
done

echo ""; echo "### PHASE 2 — baseline còn lại (unlearn.py) ###"
for s in class domain; do
  for stem in ft ga random_label scrub sg_unlearning boundary_expanding boundary_shrink l1_sparse module; do
    cfg="config/main_table_pacs_seed42/${stem}_pacs_${s}.yaml"
    D=$($PYTHON -c "import yaml;print(yaml.safe_load(open('$cfg'))['baseline_name'])")
    validate_and_run unlearn.py "$cfg" "$D" || true
  done
done
echo ""; echo "### XONG PHASE 1+2 ###"
