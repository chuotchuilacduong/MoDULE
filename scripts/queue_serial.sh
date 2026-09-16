#!/usr/bin/env bash
# Hàng đợi tuần tự trong MỘT tiến trình. Không dùng pgrep để chờ nhau nữa:
# cmdline của shell bọc (Claude Code bash tool) chứa nguyên văn nội dung script,
# nên `pgrep -f "<tên script>.sh"` khớp vào chính các shell bọc đó và không bao giờ
# rỗng -> deadlock vĩnh viễn, GPU đứng không 6.5h.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs; DONE=$R/done
mkdir -p "$LOG" "$DONE"
echo $$ > "$R/queue_serial.pid"
step() { echo ""; echo "######## $(date '+%m-%d %T')  $*"; }

# ---- 1-2. Retraining (yêu cầu hiện hành của user, ưu tiên cao nhất) ----
for scen in domain class; do
  step "Retraining/$scen  (100 epoch)"
  $PY retrain_baseline.py --config "config/main_table_pacs_seed42/retrain_pacs_$scen.yaml" \
      > "$LOG/Retraining__$scen.log" 2>&1
  st=$?; echo "[exit $st]"
  [ $st -eq 0 ] && touch "$DONE/Retraining__$scen"
  grep -oE "\[Final Metrics\].*" "$LOG/Retraining__$scen.log" | tail -1
done

# ---- 3. chấm checkpoint final-epoch (+ _best để đối chiếu) ----
step "eval checkpoint retrain"
for scen in domain class; do
  d=runs/main_table_pacs_seed42/retrain_pacs_$scen/retrain_pacs_$scen
  cfg=config/main_table_pacs_seed42/retrain_pacs_$scen.yaml
  if [ -f "$d/retrain_pacs_$scen.pt" ]; then
    echo "--- Retraining/$scen final-epoch"
    $PY scripts/eval_original.py --config "$cfg" --checkpoint "$d/retrain_pacs_$scen.pt" \
        --baseline Retraining --no-wandb
    echo "--- Retraining/$scen best-train-loss"
    $PY scripts/eval_original.py --config "$cfg" --checkpoint "$d/retrain_pacs_${scen}_best.pt" \
        --baseline Retraining_bestloss --no-wandb
  else
    echo "!! thiếu $d/retrain_pacs_$scen.pt"
  fi
done

# ---- 4. SG-Unlearning với unseen pool khớp domain/lớp ----
for scen in class domain; do
  step "SG-Unlearning/$scen (matched unseen)"
  $PY unlearn.py --config "config/main_table_pacs_seed42/sg_unlearning_pacs_$scen.yaml" \
      > "$LOG/SG-Unlearning__$scen.log" 2>&1
  st=$?; echo "[exit $st]"
  [ $st -eq 0 ] && touch "$DONE/SG-Unlearning__$scen"
  echo -n "Proxy M Loss đầu/cuối: "
  grep -oE "Proxy M Loss: [0-9.]+" "$LOG/SG-Unlearning__$scen.log" | sed -n '1p;$p' | tr '\n' ' '; echo
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/SG-Unlearning__$scen.log" | tail -1
done

# ---- 5. bảng quét RepSelect ----
step "sweep RepSelect (k_pcs x alpha)"
bash scripts/sweep_repselect_nowait.sh

step "### TOÀN BỘ HÀNG ĐỢI XONG ###"
