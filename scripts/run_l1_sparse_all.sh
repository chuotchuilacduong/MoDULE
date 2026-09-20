#!/usr/bin/env bash
# Chạy lại l1-sparse (bản đúng: retain-FT + l1 giảm dần) cho mọi dataset/scenario đã có base trên máy này.
#   PY=<python có torch> nohup bash scripts/run_l1_sparse_all.sh > l1_all.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/l1_sparse_v2; mkdir -p "$LOG"
for cfg in config/main_table_officehome_seed42/l1_sparse_officehome_class.yaml \
           config/main_table_officehome_seed42/l1_sparse_officehome_domain.yaml \
           config/main_table_cifar100_seed42/l1_sparse_cifar100_class.yaml \
           config/main_table_tiny_imagenet_seed42/l1_sparse_tiny_imagenet_class.yaml \
           config/main_table_pacs_seed42/l1_sparse_pacs_class.yaml \
           config/main_table_pacs_seed42/l1_sparse_pacs_domain.yaml; do
  ck=$("$PY" -c "import yaml;print(yaml.safe_load(open('$cfg'))['pretrained_model_path'])")
  [ -f "$ck" ] || { echo "[skip] $cfg — thiếu base $ck"; continue; }
  n=$(basename "$cfg" .yaml); echo ""; echo "######## $(date '+%m-%d %T')  $n"
  "$PY" -m unlearn --config "$cfg" > "$LOG/$n.log" 2>&1; echo "[exit $?]"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/$n.log" | tail -1
done
echo "### $(date '+%m-%d %T') L1-SPARSE XONG ###"
