#!/usr/bin/env bash
# 3 baseline: SG-Unlearning, SEUF, RepSelect (class + domain).
#   SG-Unlearning: batch 64 (lần trước OOM ở 128).
#   SEUF: seuf_include_head=true -- không đụng head thì không hạ được logit lớp cần quên.
#   RepSelect: tự mở khoá tham số nó nhắm tới thay vì bỏ qua tham số bị đóng băng,
#              nếu không nó chỉ còn 4 nhóm MoE ở block 10-11 sau residual, đo được
#              là đổi 20.4% trọng số mà FA không nhúc nhích.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt

run() {
  local entry="$1" cfg="$2" disp="$3" slug="$4"; shift 4
  [ -f "$DONE/$slug" ] && { echo "[skip] $disp"; return 0; }
  echo ""; echo "=========================================="
  echo "[*] $(date +%T) launching: $disp"
  echo "=========================================="
  $PYTHON "$entry" --config "$cfg" "$@" 2>&1 | tee "$LOG/${slug}.log"
  local st=${PIPESTATUS[0]}
  [ "$st" -ne 0 ] && { echo "[!] $disp FAILED (exit $st)"; return 0; }
  touch "$DONE/$slug"; echo "[*] $(date +%T) finished: $disp"
}

for s in class domain; do
  run unlearn.py "config/main_table_pacs_seed42/sg_unlearning_pacs_${s}.yaml" \
      "SG-Unlearning/$s" "SG-Unlearning__${s}"
  for b in seuf repselect; do
    case $b in seuf) D=SEUF;; repselect) D=RepSelect;; esac
    run config/experiments/run_moe_pipeline.py "config/experiments/pacs_${b}_${s}.yaml" \
        "$D/$s" "${D}__${s}" --stage unlearn --checkpoint "$BEST" --force
  done
done
echo ""; echo "### XONG ###"
