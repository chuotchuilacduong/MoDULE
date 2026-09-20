#!/usr/bin/env bash
# GA + l1-sparse(ascent) expert-only ở lr nhỏ cho mọi setting ngoài PACS class (đã có run 07/09).
#   PY=<python có torch> nohup bash scripts/run_ga_l1_gentle.sh > gentle.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/ga_l1_gentle; mkdir -p "$LOG"
PB=runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
[ -f "$PB" ] || "$PY" scripts/fetch_wandb_ckpt.py --run g0inran5 --file checkpoints/learn_best.pt --out "$PB"
for cfg in config/ga_l1_gentle/*.yaml; do
  n=$(basename "$cfg" .yaml); [ -f "$LOG/$n.done" ] && { echo "[skip] $n"; continue; }
  ck=$("$PY" -c "import yaml;print(yaml.safe_load(open('$cfg'))['pretrained_model_path'])"); [ -f "$ck" ] || { echo "[skip] $n — thiếu $ck"; continue; }
  echo ""; echo "######## $(date '+%m-%d %T')  $n"
  "$PY" -m unlearn --config "$cfg" > "$LOG/$n.log" 2>&1 && touch "$LOG/$n.done" || echo "[FAIL]"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/$n.log" | tail -1
done
echo "### $(date '+%m-%d %T') XONG ###"
