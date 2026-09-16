#!/usr/bin/env bash
# Retraining (gold standard) + chạy lại các baseline sau khi sửa bug phạm vi gradient
# (unlearn.py trước đây gọi _set_grad_mode("unlearning") cho MỌI thuật toán dùng
# ModuleArchitecture, đóng băng backbone + classifier head + router → chỉ 32.4%
# tham số trainable; các baseline "full-model" không hề full-model).
#
# CÓ RESUME: mỗi job xong ghi một marker vào results/.../done/. Chạy lại script sẽ
# bỏ qua job đã có marker — cần thiết vì tiến trình nền trong phiên này đã bị kill
# lặng lẽ vài lần, lần gần nhất mất 8 tiếng ở Retraining/class epoch 88/100.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42
LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt

run() {  # $1=entry $2=config $3=tên $4=slug $5...=extra
  local entry="$1" cfg="$2" disp="$3" slug="$4"; shift 4
  if [ -f "$DONE/$slug" ]; then echo "[skip] $disp (đã xong)"; return 0; fi
  echo ""; echo "=========================================="
  echo "[*] $(date +%T) launching: $disp   ($cfg)"
  echo "=========================================="
  $PYTHON "$entry" --config "$cfg" "$@" 2>&1 | tee "$LOG/${slug}.log"
  local st=${PIPESTATUS[0]}
  if [ "$st" -ne 0 ]; then echo "[!] $disp FAILED (exit $st)"; return 0; fi
  touch "$DONE/$slug"; echo "[*] $(date +%T) finished: $disp"
}

echo "### GIAI ĐOẠN 1 — Retraining ###"
# class đã cứu được từ checkpoint _best.pt (epoch ~88) và đã đánh giá xong
touch "$DONE/Retraining__class" 2>/dev/null
run retrain_baseline.py "config/main_table_pacs_seed42/retrain_pacs_domain.yaml" \
    "Retraining/domain" "Retraining__domain"

echo ""; echo "### GIAI ĐOẠN 2 — chạy lại baseline sau khi sửa bug ###"
for s in class domain; do
  for stem in ft ga random_label scrub sg_unlearning boundary_expanding boundary_shrink l1_sparse; do
    cfg="config/main_table_pacs_seed42/${stem}_pacs_${s}.yaml"
    D=$($PYTHON -c "import yaml;print(yaml.safe_load(open('$cfg'))['baseline_name'])")
    run unlearn.py "$cfg" "$D/$s" "${D// /}__${s}"
  done
  for b in grip seuf repselect salun; do
    case $b in grip) D=GRIP;; seuf) D=SEUF;; repselect) D=RepSelect;; salun) D=SalUn;; esac
    run config/experiments/run_moe_pipeline.py "config/experiments/pacs_${b}_${s}.yaml" \
        "$D/$s" "${D}__${s}" --stage unlearn --checkpoint "$BEST" --force
  done
done
echo ""; echo "### XONG ($(ls "$DONE" | wc -l) job có marker) ###"
