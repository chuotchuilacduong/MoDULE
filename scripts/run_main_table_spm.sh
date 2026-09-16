#!/usr/bin/env bash
# SPM cho main table — kiến trúc riêng (spm_resnet18), checkpoint riêng.
# Giữ nguyên hyperparameter gốc: learn 30 epoch lr 1e-3; unlearn 1 epoch lr 1e-3, fa_threshold 0.8.
# Chờ launcher chính xong để không tranh GPU.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/main_table_pacs_seed42/logs; mkdir -p "$LOG"
SPM_CKPT=checkpoint/config/learn/pacs_spm/pacs_spm.pt

while pgrep -f "run_main_table_pacs_seed42.sh" >/dev/null 2>&1; do
  echo "[*] $(date +%T) chờ launcher chính xong..."; sleep 300
done
echo "[*] $(date +%T) GPU rảnh, bắt đầu SPM"

if [ -f "$SPM_CKPT" ]; then echo "[skip] SPM base đã có: $SPM_CKPT"
else
  echo "[*] $(date +%T) train SPM base (spm_resnet18, 30 epoch)"
  $PYTHON learn.py --config config/learn/pacs_spm.yaml 2>&1 | tee "$LOG/SPM__learn.log"
  [ -f "$SPM_CKPT" ] || { echo "[!] thiếu checkpoint sau train: $SPM_CKPT"; exit 1; }
fi

for s in class domain; do
  cfg="config/experiments/pacs_spm_unlearn_${s}.yaml"
  name=$($PYTHON -c "import yaml;print(yaml.safe_load(open('$cfg'))['study_name'])")
  echo ""; echo "=========================================="
  echo "Baseline: SPM"; echo "Dataset: PACS"; echo "Scenario: $s"
  echo "Config: $cfg"; echo "W&B run name: $name"; echo "Seed: 42"
  echo "LR: $($PYTHON -c "import yaml;print(yaml.safe_load(open('$cfg'))['lr'])")"
  echo "Checkpoint: $SPM_CKPT"; echo "=========================================="
  $PYTHON unlearn.py --config "$cfg" 2>&1 | tee "$LOG/SPM__${s}.log"
  echo "[*] $(date +%T) finished: $name"
done
echo "### SPM XONG ###"
