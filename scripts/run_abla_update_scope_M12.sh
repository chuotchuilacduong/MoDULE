#!/usr/bin/env bash
# Ablation update scope (Table 7/8) chạy lại trên cùng base với Table 9: M=12/k=4 best.pt, k_u=4, đủ 20 epoch.
# 12 run tuần tự, KHÔNG dùng sweep. Chạy từ gốc repo: PY=python nohup bash scripts/run_abla_update_scope_M12.sh > abla_scope.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=config/abla_update_scope_M12
R=results/abla_update_scope_M12; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
BASE=runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt
BASE_CFG=config/experiments/pacs_module_base_M12_k4.yaml

[ -d dataset/data_folder/pacs ] || { echo "[*] tải PACS"; "$PY" -m dataset.downloader.pacs || exit 1; }

if [ ! -f "$BASE" ]; then
  echo "[!] thiếu $BASE — train base M12/k4 (100 epoch, ~4h). Nếu đã có checkpoint ở máy khác thì scp sang rồi chạy lại."
  "$PY" -m learn --config "$BASE_CFG" > "$LOG/00_base_M12_k4.log" 2>&1 || { echo "[FAIL] base"; exit 1; }
fi
[ -f "$BASE" ] || { echo "[!] vẫn thiếu $BASE"; exit 1; }

for s in class domain; do
  for sc in selected_experts_only selected_experts_and_head selected_experts_and_router \
            selected_experts_and_last_block all_experts full_model; do
    name=abla_scope_${s}_${sc}_M12_k4_ku4_seed42
    if [ -f "$DONE/$name" ]; then echo "[skip] $name"; continue; fi
    echo ""; echo "######## $(date '+%m-%d %T')  $name"
    "$PY" -m unlearn --config "$CFG/$name.yaml" > "$LOG/$name.log" 2>&1
    st=$?; [ $st -eq 0 ] && touch "$DONE/$name" || { echo "[FAIL exit $st]"; grep -E "Traceback|Error" "$LOG/$name.log" | tail -2; }
    grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/$name.log" | tail -1
  done
done
echo ""; echo "### $(date '+%m-%d %T') UPDATE-SCOPE XONG ###"
for f in "$LOG"/abla_scope_*.log; do
  printf "%-62s " "$(basename "$f" .log)"; grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$f" | tail -1 || echo "(không có)"
done
