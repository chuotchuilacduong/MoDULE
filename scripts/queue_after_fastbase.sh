#!/usr/bin/env bash
# Sau fastbase: (1) quét RepSelect DOMAIN để kéo FA từ 66.55 về sát gold standard
# 49.07, (2) hai lượt retrain chẩn đoán lớp 4/5. Một tiến trình, tuần tự.
# epochs=1 vì delta tính một lần, trạng thái cuối bằng đúng 20 epoch (đã kiểm chứng:
# epoch1 alpha=0.1 -> đổi 9.1993e-04 = (0.1/2.0)x1.840e-02).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; LOG=$R/logs
S=/tmp/claude-1000/-home-admin1-projects-MoDULE/ce607d58-cf99-4c5b-87d3-01167772b49e/scratchpad/sweepdom
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
OUT=$S/results.tsv; printf "k_pcs\talpha\tFA\tRA\tTA\tMIA\n" > "$OUT"

until grep -q "### FASTBASE XONG ###" "$R/fastbase_balfix.log" 2>/dev/null; do sleep 60; done

echo ""; echo "######## $(date '+%m-%d %T')  QUÉT RepSelect / DOMAIN"
echo "  mốc: Retraining/domain FA=49.07 RA=99.77 TA=90.61 | hiện có k=3,a=2.0 -> FA=66.55"
for k in 1 2 3 4; do for a in 2.0 3.0 4.0; do
  cfg=$S/rsd_k${k}_a${a}.yaml
  $PY - "$cfg" "$k" "$a" <<'PYEOF'
import sys, yaml
dst,k,a = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
d = yaml.safe_load(open("config/experiments/pacs_repselect_domain.yaml"))
u = d["unlearn"]; u["k_pcs"]=k; u["rep_select_lr"]=a; u["epochs"]=1
u["use_lora_adversary"]=False; u["wandb_mode"]="disabled"
u["study_name"]=f"sweepdom_repselect_k{k}_a{a}"
d["name"]=f"sweepdom_rs_k{k}_a{a}"
yaml.safe_dump(d, open(dst,"w"), sort_keys=False, allow_unicode=True)
PYEOF
  echo "[*] $(date +%T) k_pcs=$k alpha=$a"
  $PY config/experiments/run_moe_pipeline.py --config "$cfg" --stage unlearn \
      --checkpoint "$BEST" --force > "$S/log_k${k}_a${a}.txt" 2>&1
  m=$(grep -oE "\[RepSelect\] epoch \[1/1\].*" "$S/log_k${k}_a${a}.txt" | tail -1)
  ra=$(echo "$m"|grep -oE "ra: [0-9.]+"|cut -d' ' -f2); fa=$(echo "$m"|grep -oE "fa: [0-9.]+"|cut -d' ' -f2)
  ta=$(echo "$m"|grep -oE "ta: [0-9.]+"|cut -d' ' -f2); mi=$(echo "$m"|grep -oE "mia: [0-9.]+"|cut -d' ' -f2)
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$k" "$a" "${fa:-FAIL}" "${ra:-}" "${ta:-}" "${mi:-}" >> "$OUT"
  echo "    -> FA=${fa:-FAIL} RA=${ra:-} TA=${ta:-} MIA=${mi:-}"
done; done
echo ""; echo "=== BẢNG QUÉT DOMAIN ==="; column -t "$OUT"

for c in 4 5; do
  echo ""; echo "######## $(date '+%m-%d %T')  Retrain chẩn đoán bỏ lớp $c (15 epoch)"
  $PY retrain_baseline.py --config "config/main_table_pacs_seed42/retrain_pacs_class$c.yaml" \
      > "$LOG/DIAG_Retraining__class$c.log" 2>&1
  echo "[exit $?]"
  grep -oE "eval @ epoch [0-9]+\] ra: [0-9.]+% \| fa: [0-9.]+% \| ta: [0-9.]+%" "$LOG/DIAG_Retraining__class$c.log"
done
echo ""; echo "######## $(date '+%m-%d %T')  ### TẤT CẢ XONG ###"
