#!/usr/bin/env bash
# Chạy lại SG-Unlearning/domain (alpha 0.1->1.0) và RepSelect class+domain (bật LoRA adversary).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
run() {
  local entry="$1" cfg="$2" disp="$3" slug="$4"; shift 4
  echo ""; echo "[*] $(date +%T) launching: $disp"
  $PYTHON "$entry" --config "$cfg" "$@" 2>&1 | tee "$LOG/${slug}.log"
  [ ${PIPESTATUS[0]} -ne 0 ] && { echo "[!] $disp FAILED"; return 0; }
  touch "$DONE/$slug"; echo "[*] $(date +%T) finished: $disp"
}
run unlearn.py "config/main_table_pacs_seed42/sg_unlearning_pacs_domain.yaml" \
    "SG-Unlearning/domain" "SG-Unlearning__domain"
for s in class domain; do
  run config/experiments/run_moe_pipeline.py "config/experiments/pacs_repselect_${s}.yaml" \
      "RepSelect/$s" "RepSelect__${s}" --stage unlearn --checkpoint "$BEST" --force
done
echo "### XONG ###"
