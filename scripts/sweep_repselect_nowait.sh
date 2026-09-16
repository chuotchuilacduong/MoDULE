#!/usr/bin/env bash
# Quét k_pcs x rep_select_lr cho RepSelect (class). Delta được tính MỘT lần trước vòng
# lặp epoch, epoch chỉ ramp alpha tới rep_select_lr -- nên trạng thái cuối sau 1 epoch
# bằng đúng sau 20 epoch. Dùng epochs=1 để quét nhanh, rồi chạy lại đủ 20 epoch ở
# cấu hình được chọn.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-1000/-home-admin1-projects-MoDULE/ce607d58-cf99-4c5b-87d3-01167772b49e/scratchpad/sweep
BEST=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
OUT=$S/results.tsv; echo -e "k_pcs\talpha\tFA\tRA\tTA\tMIA" > "$OUT"


for k in 1 8 32 128 512; do
  for a in 0.1 1.0; do
    cfg=$S/rs_k${k}_a${a}.yaml
    $PY - "$cfg" "$k" "$a" <<'PYEOF'
import sys, yaml
dst,k,a = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
d = yaml.safe_load(open("config/experiments/pacs_repselect_class.yaml"))
u = d["unlearn"]; u["k_pcs"]=k; u["rep_select_lr"]=a; u["epochs"]=1
u["use_lora_adversary"]=False          # tách biến: chỉ quét k_pcs va alpha
u["study_name"]=f"sweep_repselect_k{k}_a{a}"
d["name"]=f"sweep_rs_k{k}_a{a}"
yaml.safe_dump(d, open(dst,"w"), sort_keys=False)
PYEOF
    echo "[*] $(date +%T) k_pcs=$k alpha=$a"
    $PY config/experiments/run_moe_pipeline.py --config "$cfg" --stage unlearn \
        --checkpoint "$BEST" --force > "$S/log_k${k}_a${a}.txt" 2>&1
    m=$(grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$S/log_k${k}_a${a}.txt" | tail -1)
    ra=$(echo "$m"|grep -oE "RA: [0-9.]+"|cut -d' ' -f2); fa=$(echo "$m"|grep -oE "FA: [0-9.]+"|cut -d' ' -f2)
    ta=$(echo "$m"|grep -oE "TA: [0-9.]+"|cut -d' ' -f2); mi=$(echo "$m"|grep -oE "MIA: [0-9.]+"|cut -d' ' -f2)
    echo -e "${k}\t${a}\t${fa:-FAIL}\t${ra:-}\t${ta:-}\t${mi:-}" >> "$OUT"
    echo "    -> FA=${fa:-FAIL} RA=${ra:-} TA=${ta:-}"
  done
done
echo ""; echo "=== BẢNG QUÉT ==="; column -t "$OUT"
