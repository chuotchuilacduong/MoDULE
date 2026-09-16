#!/usr/bin/env bash
# Chạy lại Phase 1 (GRIP/SEUF/RepSelect/SalUn x class,domain) sau khi sửa lỗi
# dataset: "PACS" -> "pacs" (unlearn.py so sánh phân biệt hoa thường).
# Chờ launcher chính (Phase 2) và SPM xong để không tranh GPU.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/main_table_pacs_seed42/logs; mkdir -p "$LOG"
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt

while pgrep -f "run_main_table_pacs_seed42.sh|run_main_table_spm.sh" >/dev/null 2>&1; do
  echo "[*] $(date +%T) chờ Phase 2 / SPM xong..."; sleep 300
done
echo "[*] $(date +%T) bắt đầu chạy lại Phase 1"

for s in class domain; do
  for b in grip seuf repselect salun; do
    case $b in grip) D=GRIP;; seuf) D=SEUF;; repselect) D=RepSelect;; salun) D=SalUn;; esac
    cfg="config/experiments/pacs_${b}_${s}.yaml"
    name=$($PYTHON -c "import yaml;print(yaml.safe_load(open('$cfg'))['unlearn']['study_name'])")
    echo ""; echo "=========================================="
    echo "Baseline: $D"; echo "Dataset: PACS"; echo "Scenario: $s"
    echo "Config: $cfg"; echo "W&B run name: $name"; echo "Seed: 42"
    echo "LR: $($PYTHON -c "import yaml;print(yaml.safe_load(open('$cfg'))['unlearn']['lr'])")"
    echo "Checkpoint: $BEST"; echo "=========================================="
    $PYTHON config/experiments/run_moe_pipeline.py --config "$cfg" \
      --stage unlearn --checkpoint "$BEST" --force 2>&1 | tee "$LOG/${D}__${s}.log"
    st=${PIPESTATUS[0]}
    [ "$st" -ne 0 ] && echo "[!] $D/$s FAILED (exit $st)" || echo "[*] $(date +%T) finished: $name"
  done
done
echo "### PHASE 1 (rerun) XONG ###"
