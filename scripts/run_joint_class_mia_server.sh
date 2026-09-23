#!/usr/bin/env bash
# End-to-end JOINT class-unlearning membership-leakage experiment on PACS, for the server.
# Self-contained and resumable: every step is skipped if its artefact already exists.
#
#   cd ~/Github_repos/MoDULE && git pull
#   nohup bash scripts/run_joint_class_mia_server.sh > joint_mia.nohup.log 2>&1 &
#   echo $! > joint_mia.pid ; tail -f joint_mia.nohup.log
#
# Needs only the M=12/k=4 base checkpoint (the one the sequential domain runs used):
#   runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt
# Everything else (6 unlearning runs, 3 Retrain references, evaluation, figure) is produced here.
#
# Steps:
#   1. ModULE and SEUF unlearn {0}, {0,1}, {0,1,2} -- each JOINT: started independently from the
#      base checkpoint, k_u fixed at 4, 20 epochs, no FA early stopping.
#   2. Three Retrain references, from ImageNet init on the corresponding retain set (50 epochs).
#   3. ModULE expert-selection record + the controlled MIA evaluation and figure.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# absolute interpreter: the server's PATH does not always have the env active under nohup
PYTHON=${PY:-/home/duong/miniconda3/envs/module/bin/python3}
[ -x "$PYTHON" ] || PYTHON=$(command -v python3)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/joint_class_mia/logs; mkdir -p "$LOG"
BASE=runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt

echo "[*] python: $PYTHON"; $PYTHON -c "import torch;print('[*] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
[ -f "$BASE" ] || { echo "[!] MISSING base checkpoint: $BASE -- stop (do not retrain the base here)"; exit 1; }
[ -d dataset/data_folder/pacs ] || { echo "[!] MISSING PACS data: dataset/data_folder/pacs"; exit 1; }

echo ""; echo "### STEP 0/3 — pull checkpoints already trained elsewhere (W&B artifact) ###"
# the three Retrain references and the two 1-class unlearned models were trained on the
# other machine; downloading them skips ~3 h of GPU time here. Existing files are kept.
if [ "${SKIP_WANDB_PULL:-0}" != "1" ]; then
  $PYTHON scripts/wandb_ckpt_transfer.py pull 2>&1 | tee "$LOG/wandb_pull.log" || \
    echo "[!] W&B pull failed -- the steps below will train whatever is still missing"
fi

echo ""; echo "### STEP 1/3 — joint unlearning runs (ModULE, SEUF x 1/2/3 classes) ###"
for cfg in module_joint_c1 module_joint_c2 module_joint_c3 seuf_joint_c1 seuf_joint_c2 seuf_joint_c3; do
  algo=${cfg%%_*}
  ck="runs/joint_class/${cfg}/checkpoints/unlearned_${algo}_${cfg}.pt"
  if [ -f "$ck" ]; then
    # either finished here, or downloaded from the W&B artifact in step 0
    echo "[skip] $cfg: checkpoint present ($ck)"; continue
  fi
  echo ""; echo "[*] $(date +%F_%T) $cfg"
  $PYTHON unlearn.py --config "config/joint_class/${cfg}.yaml" > "$LOG/${cfg}.log" 2>&1
  st=$?
  if [ "$st" -ne 0 ]; then echo "[!] $cfg FAILED (exit $st); last lines:"; tail -5 "$LOG/${cfg}.log"; else
    grep "Metrics:" "$LOG/${cfg}.log" | tail -1; fi
done

echo ""; echo "### STEP 2/3 — Retrain references for {0}, {0,1}, {0,1,2} (50 epochs each) ###"
missing=0
for k in 0 0-1 0-1-2; do [ -f "runs/sequential/pacs_retrain_class/retrain_class_${k}/retrain_class_${k}.pt" ] || missing=1; done
if [ "$missing" -eq 0 ]; then
  echo "[skip] all three Retrain references already exist"
else
  $PYTHON sequential_unlearn.py --config config/joint_class/retrain_driver_c123.yaml --retrain-only \
    > "$LOG/retrain_c123.log" 2>&1
  st=$?; [ "$st" -ne 0 ] && { echo "[!] retrain FAILED (exit $st); last lines:"; tail -5 "$LOG/retrain_c123.log"; }
  grep -h "Final Metrics" "$LOG/retrain_c123.log" | tail -3
fi

echo ""; echo "### STEP 3/3 — expert-selection record + controlled MIA evaluation + figure ###"
$PYTHON scripts/joint_class_experts.py 2>&1 | tee "$LOG/expert_selection.log"
$PYTHON scripts/joint_class_mia.py 2>&1 | tee "$LOG/joint_class_mia_eval.log"

echo ""; echo "### DONE — artefacts ###"
ls -la results/joint_class_mia/ 2>/dev/null
echo ""; echo "--- results/joint_class_mia/joint_mia_metrics.csv ---"
cat results/joint_class_mia/joint_mia_metrics.csv 2>/dev/null
