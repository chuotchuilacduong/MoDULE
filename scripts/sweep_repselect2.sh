#!/usr/bin/env bash
# Quét tinh giữa k_pcs=1 (FA 18.97, sập RA/TA) và k_pcs=8 (FA 86.39, RA/TA còn nguyên),
# alpha>=1.0. Mục tiêu: FA ~80 mà giữ RA/TA. Delta tính một lần trước vòng epoch nên
# epochs=1 cho trạng thái cuối giống hệt 20 epoch.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-1000/-home-admin1-projects-MoDULE/ce607d58-cf99-4c5b-87d3-01167772b49e/scratchpad/sweep2
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
OUT=$S/results.tsv; printf "k_pcs\talpha\tFA\tRA\tTA\tMIA\n" > "$OUT"
for k in 2 3 4 5 6 8; do for a in 1.0 1.5 2.0; do
  cfg=$S/rs_k${k}_a${a}.yaml
  $PY - "$cfg" "$k" "$a" <<'PYEOF'
import sys, yaml
dst,k,a = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
d = yaml.safe_load(open("config/experiments/pacs_repselect_class.yaml"))
u = d["unlearn"]; u["k_pcs"]=k; u["rep_select_lr"]=a; u["epochs"]=1
u["use_lora_adversary"]=False
u["study_name"]=f"sweep2_repselect_k{k}_a{a}"
d["name"]=f"sweep2_rs_k{k}_a{a}"
yaml.safe_dump(d, open(dst,"w"), sort_keys=False)
PYEOF
  echo "[*] $(date +%T) k_pcs=$k alpha=$a"
  $PY config/experiments/run_moe_pipeline.py --config "$cfg" --stage unlearn \
      --checkpoint "$BEST" --force > "$S/log_k${k}_a${a}.txt" 2>&1
  # log in ra dạng "[RepSelect] epoch [1/1] | alpha: .. | ra: ..% | fa: ..% | ta: ..% | mia: .."
  m=$(grep -oE "\[RepSelect\] epoch \[1/1\].*" "$S/log_k${k}_a${a}.txt" | tail -1)
  ra=$(echo "$m"|grep -oE "ra: [0-9.]+"|cut -d' ' -f2); fa=$(echo "$m"|grep -oE "fa: [0-9.]+"|cut -d' ' -f2)
  ta=$(echo "$m"|grep -oE "ta: [0-9.]+"|cut -d' ' -f2); mi=$(echo "$m"|grep -oE "mia: [0-9.]+"|cut -d' ' -f2)
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$k" "$a" "${fa:-FAIL}" "${ra:-}" "${ta:-}" "${mi:-}" >> "$OUT"
  echo "    -> FA=${fa:-FAIL} RA=${ra:-} TA=${ta:-}"
done; done
echo ""; echo "=== BẢNG QUÉT 2 ==="; column -t "$OUT"
