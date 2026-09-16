#!/usr/bin/env bash
# Quét alpha (selection_weighting) x lambda (dampening_constant) cho SSD/class.
# alpha=10 mặc định chọn được gần như KHÔNG tham số nào -> kết quả trùng khít Original.
# alpha thấp => chọn nhiều tham số hơn; lambda thấp => tham số đã chọn bị co mạnh hơn.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42
S=/tmp/claude-1000/-home-admin1-projects-MoDULE/ce607d58-cf99-4c5b-87d3-01167772b49e/scratchpad/sweepssd
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
OUT=$S/results.tsv; printf "alpha\tlambda\tFA\tRA\tTA\tMIA\n" > "$OUT"
until grep -q "### SPM+SSD XONG ###" "$R/spm_ssd.log" 2>/dev/null; do sleep 30; done
echo "  mốc: Original FA=99.42 RA=99.44 TA=92.62 | Retraining FA=0.00 RA=99.36 TA=95.52"
for al in 0.05 0.1 0.5 1.0 2.0 5.0; do for lm in 0.1 1.0; do
  cfg=$S/ssd_a${al}_l${lm}.yaml
  $PY - "$cfg" "$al" "$lm" <<'PYEOF'
import sys, yaml
dst, al, lm = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
d = yaml.safe_load(open("config/experiments/pacs_ssd_class.yaml"))
u = d["unlearn"]; u["selection_weighting"]=al; u["dampening_constant"]=lm
u["study_name"]=f"sweepssd_a{al}_l{lm}"; u["wandb_mode"]="disabled"
d["name"]=f"sweepssd_a{al}_l{lm}"
yaml.safe_dump(d, open(dst,"w"), sort_keys=False, allow_unicode=True)
PYEOF
  echo "[*] $(date +%T) alpha=$al lambda=$lm"
  $PY config/experiments/run_moe_pipeline.py --config "$cfg" --stage unlearn \
      --checkpoint "$BEST" --force > "$S/log_a${al}_l${lm}.txt" 2>&1
  m=$(grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$S/log_a${al}_l${lm}.txt" | tail -1)
  ra=$(echo "$m"|grep -oE "RA: [0-9.]+"|cut -d' ' -f2); fa=$(echo "$m"|grep -oE "FA: [0-9.]+"|cut -d' ' -f2)
  ta=$(echo "$m"|grep -oE "TA: [0-9.]+"|cut -d' ' -f2); mi=$(echo "$m"|grep -oE "MIA: [0-9.]+"|cut -d' ' -f2)
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$al" "$lm" "${fa:-FAIL}" "${ra:-}" "${ta:-}" "${mi:-}" >> "$OUT"
  echo "    -> FA=${fa:-FAIL} RA=${ra:-} TA=${ta:-} MIA=${mi:-}"
done; done
echo ""; echo "=== BẢNG QUÉT SSD ==="; column -t "$OUT"
echo "### SWEEPSSD XONG ###"
