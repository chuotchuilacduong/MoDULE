#!/usr/bin/env bash
# Dòng "MoE" của Table 1: base-learn Standard MoE (M=12, k=4, chỉ có L_bal) rồi unlearn
# class + domain với đúng ngân sách k_u=4 như các dòng MoE khác trong bảng.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/main_table_pacs_seed42; L=$R/logs; DONE=$R/done
mkdir -p "$L" "$DONE" runs/moe_baseline
[ -d dataset/data_folder/pacs ] || "$PY" -m dataset.downloader.pacs || exit 1

echo "######## $(date '+%m-%d %T')  base-learn MoE M=12 k=4 (100 epoch)"
$PY -m learn --config config/learn/moe_baseline/moe_baseline_M12_k4.yaml > "$L/moe_baseline_learn.log" 2>&1
st=$?; echo "[exit $st]"
grep -oE "\[Final Metrics\].*" "$L/moe_baseline_learn.log" | tail -1
[ $st -ne 0 ] && { echo "!! base-learn lỗi, dừng"; exit 1; }

for scen in class domain; do
  echo ""; echo "######## $(date '+%m-%d %T')  unlearn MoE/$scen"
  $PY -m unlearn --config "config/learn/moe_baseline/unl_moe_${scen}.yaml" > "$L/MoE__${scen}.log" 2>&1
  s2=$?; echo "[exit $s2]"; [ $s2 -eq 0 ] && touch "$DONE/MoE__${scen}"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$L/MoE__${scen}.log" | tail -1
  grep -oE "(Traceback|Error|error:).*" "$L/MoE__${scen}.log" | head -2
done
echo "### MOE-BASELINE XONG ###"
