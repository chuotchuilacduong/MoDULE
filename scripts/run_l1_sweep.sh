#!/usr/bin/env bash
# l1-sparse (Jia et al.: retain-FT + gamma*||W||_1 giảm dần) — sweep lr x gamma theo lưới của Fan et al. (SalUn),
# PACS class (dog), base aaedd1da4d (Table 1). Dừng sớm khi FA <= 10% như mọi baseline Table 1.
#   PY=<python có torch> nohup bash scripts/run_l1_sweep.sh > l1_sweep.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
[ -f "$BASE" ] || "$PY" scripts/fetch_wandb_ckpt.py --run g0inran5 --file checkpoints/learn_best.pt --out "$BASE" || exit 1
[ -d dataset/data_folder/pacs ] || "$PY" -m dataset.downloader.pacs || exit 1
LOG=results/l1_sweep; mkdir -p "$LOG"
for cfg in config/l1_sweep/pacs_class_*.yaml; do
  n=$(basename "$cfg" .yaml); [ -f "$LOG/$n.done" ] && { echo "[skip] $n"; continue; }
  echo ""; echo "######## $(date '+%m-%d %T')  $n"
  "$PY" -m unlearn --config "$cfg" > "$LOG/$n.log" 2>&1 && touch "$LOG/$n.done" || echo "[FAIL]"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/$n.log" | tail -1
done
echo "### $(date '+%m-%d %T') L1 SWEEP XONG ###"
"$PY" scripts/collect_l1_sweep.py
