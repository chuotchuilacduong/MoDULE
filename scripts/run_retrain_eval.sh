#!/usr/bin/env bash
# Sau khi hai lượt retrain xong: chấm checkpoint FINAL-EPOCH (đúng quy ước main table
# "không chọn best epoch"), và chấm thêm _best.pt để đối chiếu.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0

while pgrep -f "run_retrain_domain.sh|run_retrain_class.sh" >/dev/null 2>&1; do sleep 120; done

for scen in domain class; do
  d=runs/main_table_pacs_seed42/retrain_pacs_$scen/retrain_pacs_$scen
  cfg=config/main_table_pacs_seed42/retrain_pacs_$scen.yaml
  fin=$d/retrain_pacs_$scen.pt
  if [ -f "$fin" ]; then
    echo "=== Retraining/$scen — final epoch ==="
    $PY scripts/eval_original.py --config "$cfg" --checkpoint "$fin" --baseline Retraining --no-wandb
    echo "=== Retraining/$scen — best train-loss (đối chiếu) ==="
    $PY scripts/eval_original.py --config "$cfg" --checkpoint "$d/retrain_pacs_${scen}_best.pt" \
        --baseline Retraining_bestloss --no-wandb
  else
    echo "!! thiếu $fin — lượt $scen chưa chạy hết 100 epoch"
  fi
done
echo "=== results/main_table_pacs_seed42/original_results.csv ==="
cat results/main_table_pacs_seed42/original_results.csv
