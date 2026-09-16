#!/usr/bin/env bash
# Main table cho dataset KHÔNG có domain (chỉ class unlearning): cifar100 | tiny_imagenet.
#   cifar100      : quên 10/100 lớp (c10, 10 % dữ liệu)           ~2.8 h base + ~3 h unlearn + ~2.8 h retrain (50 epoch)
#   tiny_imagenet : quên 20/200 lớp (c20, 10 %)                   ~2x cifar100 (110k ảnh)
# Cấu trúc y hệt run_full_officehome.sh (marker skip, base không train lại nếu đã có).
#   PY=<python có torch> nohup bash scripts/run_full_dataset.sh cifar100 > cifar100.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS=${1:?dùng: run_full_dataset.sh cifar100|tiny_imagenet}
case $DS in
  cifar100)      TAG=c10; DATA=dataset/data_folder/cifar100/train;                       DL=dataset.downloader.cifar100;;
  tiny_imagenet) TAG=c20; DATA=dataset/data_folder/tiny_imagenet/tiny-imagenet-200/train; DL=dataset.downloader.tiny_imagenet;;
  *) echo "dataset không hỗ trợ: $DS"; exit 1;;
esac
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FORCE=0; [ "${2:-}" = "--force" ] && FORCE=1
CFG=config/main_table_${DS}_seed42
R=results/main_table_${DS}_seed42_${TAG}; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
BASE_DIR=runs/_base_models/${DS}_M8_k2_seed42
BEST=$BASE_DIR/checkpoints/learn_best.pt
SPM_CKPT=checkpoint/$CFG/${DS}_spm/${DS}_spm.pt

step() {
  local name="$1"; shift
  if [ $FORCE -eq 0 ] && [ -f "$DONE/$name" ]; then echo "[skip] $name"; return 0; fi
  echo ""; echo "######## $(date '+%m-%d %T')  BẮT ĐẦU $name"; echo "$ $*"
  "$@" > "$LOG/$name.log" 2>&1
  local st=$?
  if [ $st -eq 0 ]; then touch "$DONE/$name"; echo "[ok]   $(date '+%m-%d %T')  $name"
  else echo "[FAIL] $(date '+%m-%d %T')  $name (exit $st)"; grep -E "Traceback|Error" "$LOG/$name.log" | tail -3; fi
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/$name.log" | tail -1
  grep -oE "\[Final Metrics\].*" "$LOG/$name.log" | tail -1
  return $st
}

# 0. dữ liệu
if [ -d "$DATA" ] && [ "$(ls "$DATA" | wc -l)" -ge 100 ]; then echo "[skip] $DS đã có ($(ls "$DATA" | wc -l) lớp)"
else step 00_download_$DS "$PY" -m $DL || exit 1; fi

# 1. base ModULE (không phụ thuộc forget target -> skip nếu đã có)
mkdir -p "$BASE_DIR/checkpoints"
sed "s#^output_dir:.*#output_dir: $BASE_DIR/checkpoints#" "$CFG/base_${DS}_M8_k2.yaml" > "$BASE_DIR/learn.yaml"
if [ -f "$BEST" ]; then echo "[skip] base ModULE đã có: $BEST"; else
  step 01_base_module_M8_k2 "$PY" -m learn --config "$BASE_DIR/learn.yaml"; fi
[ -f "$BEST" ] || { echo "[!] thiếu $BEST — dừng"; exit 1; }

# 2. base SPM
if [ -f "$SPM_CKPT" ]; then echo "[skip] base SPM đã có"; else
  step 02_base_spm_resnet18 "$PY" -m learn --config "$CFG/${DS}_spm.yaml"; fi

# 3. pipeline baselines
for b in grip seuf repselect salun ssd; do
  step "03_${b}__class" "$PY" -m config.experiments.run_moe_pipeline \
    --config "$CFG/${DS}_${b}_class.yaml" --stage unlearn --checkpoint "$BEST"
done

# 4. unlearn.py baselines + SPM + Original
for stem in ft ga random_label scrub sg_unlearning boundary_expanding boundary_shrink l1_sparse module; do
  step "04_${stem}__class" "$PY" -m unlearn --config "$CFG/${stem}_${DS}_class.yaml"
done
step "04_spm__class" "$PY" -m unlearn --config "$CFG/${DS}_spm_unlearn_class.yaml"
step "04_original__class" "$PY" scripts/eval_original.py --config "$CFG/ft_${DS}_class.yaml"

# 5. retraining
step "05_retrain__class" "$PY" -m retrain_baseline --config "$CFG/retrain_${DS}_class.yaml"

echo ""; echo "### $(date '+%m-%d %T') $DS XONG ###"
for f in "$LOG"/0[345]_*.log; do
  printf "%-40s " "$(basename "$f" .log)"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+|\[Final Metrics\].*" "$f" | tail -1 || echo "(không có)"
done
