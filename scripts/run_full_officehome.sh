#!/usr/bin/env bash
# Main Table — OfficeHome, seed 42, class (forget class 0 = Alarm_Clock) + domain (forget domain 3 = Real World).
# Chạy tuần tự trong MỘT tiến trình (không pgrep -f để chờ script khác — đã gây deadlock trên PACS):
#   0. tải OfficeHome từ HF (flwrlabs/office-home) nếu chưa có
#   1. base ModULE M=8/k=2 (learn.py)         -> runs/_base_models/officehome_M8_k2_seed42/checkpoints/learn_best.pt
#   2. base SPM ResNet18 (learn.py)           -> checkpoint/config/main_table_officehome_seed42/officehome_spm/officehome_spm.pt
#   3. pipeline baselines: GRIP SEUF RepSelect SalUn SSD   (run_moe_pipeline.py --stage unlearn)
#   4. unlearn.py baselines: FT GA RandomLabel SCRUB SG BoundaryExp BoundaryShrink L1-sparse ModULE SPM
#   5. Retraining class + domain (retrain_baseline.py, 100 epoch)
# Mỗi bước có marker trong $DONE — chạy lại script sẽ bỏ qua bước đã xong. Dùng --force để chạy lại tất cả.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}                       # trên server: PY=/path/to/env/bin/python bash scripts/run_full_officehome.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1

CFG=config/main_table_officehome_seed42
DATA=dataset/data_folder/officehome
R=results/main_table_officehome_seed42; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
BASE_DIR=runs/_base_models/officehome_M8_k2_seed42
BEST=$BASE_DIR/checkpoints/learn_best.pt
SPM_CKPT=checkpoint/$CFG/officehome_spm/officehome_spm.pt

step() {   # $1=tên bước  $2...=lệnh. Log vào $LOG/<tên>.log, marker $DONE/<tên>
  local name="$1"; shift
  if [ $FORCE -eq 0 ] && [ -f "$DONE/$name" ]; then echo "[skip] $name (đã xong)"; return 0; fi
  echo ""; echo "######## $(date '+%m-%d %T')  BẮT ĐẦU $name"; echo "$ $*"
  "$@" > "$LOG/$name.log" 2>&1
  local st=$?
  if [ $st -eq 0 ]; then touch "$DONE/$name"; echo "[ok]   $(date '+%m-%d %T')  $name"
  else echo "[FAIL] $(date '+%m-%d %T')  $name (exit $st)"; grep -E "Traceback|Error" "$LOG/$name.log" | tail -3; fi
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/$name.log" | tail -1
  grep -oE "\[Final Metrics\].*" "$LOG/$name.log" | tail -1
  return $st
}

# ---------- 0. dữ liệu ----------
if [ -d "$DATA" ] && [ "$(find "$DATA" -name '*.jpg' | wc -l)" -ge 15000 ]; then
  echo "[skip] OfficeHome đã có: $(find "$DATA" -name '*.jpg' | wc -l) ảnh"
else
  rm -rf "$DATA"
  step 00_download_officehome "$PY" dataset/downloader/officehome.py || exit 1
  echo "[*] domains: $(ls "$DATA" | tr '\n' ' ')  | classes: $(ls "$DATA/$(ls "$DATA" | head -1)" | wc -l)"
fi

# ---------- 1. base ModULE ----------
# learn.py đặt tên checkpoint theo tên yaml, nên copy config thành learn.yaml để ra learn.pt / learn_best.pt
# (đúng bố cục mà run_moe_pipeline.py --stage unlearn mong đợi: learn.yaml nằm hai cấp trên .pt).
mkdir -p "$BASE_DIR/checkpoints"
sed "s#^output_dir:.*#output_dir: $BASE_DIR/checkpoints#" "$CFG/base_officehome_M8_k2.yaml" > "$BASE_DIR/learn.yaml"
step 01_base_module_M8_k2 "$PY" learn.py --config "$BASE_DIR/learn.yaml"
[ -f "$BEST" ] || { echo "[!] thiếu $BEST — dừng"; exit 1; }

# ---------- 2. base SPM ----------
step 02_base_spm_resnet18 "$PY" learn.py --config "$CFG/officehome_spm.yaml"
[ -f "$SPM_CKPT" ] || echo "[!] thiếu $SPM_CKPT — SPM unlearn sẽ fail, các baseline khác vẫn chạy"

# ---------- 3. pipeline baselines ----------
for s in class domain; do
  for b in grip seuf repselect salun ssd; do
    step "03_${b}__${s}" "$PY" config/experiments/run_moe_pipeline.py \
      --config "$CFG/officehome_${b}_${s}.yaml" --stage unlearn --checkpoint "$BEST"
  done
done

# ---------- 4. unlearn.py baselines ----------
for s in class domain; do
  for stem in ft ga random_label scrub sg_unlearning boundary_expanding boundary_shrink l1_sparse module; do
    step "04_${stem}__${s}" "$PY" unlearn.py --config "$CFG/${stem}_officehome_${s}.yaml"
  done
  step "04_spm__${s}" "$PY" unlearn.py --config "$CFG/officehome_spm_unlearn_${s}.yaml"
done

# ---------- 5. retraining ----------
for s in class domain; do
  step "05_retrain__${s}" "$PY" retrain_baseline.py --config "$CFG/retrain_officehome_${s}.yaml"
done

echo ""; echo "### $(date '+%m-%d %T') OFFICEHOME XONG ###"
echo "Tổng kết (dòng metric cuối mỗi log):"
for f in "$LOG"/0[345]_*.log; do
  printf "%-40s " "$(basename "$f" .log)"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+|\[Final Metrics\].*" "$f" | tail -1 || echo "(không có)"
done
