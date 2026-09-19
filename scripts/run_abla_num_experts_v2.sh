#!/usr/bin/env bash
# Table "Effect of number of experts M" — chạy lại NHẤT QUÁN: mọi M cùng recipe base
# (k=4, λ=0.5/2.0/2.0, cosine+clip, 100 epoch -- đúng recipe pacs_M12_k4_seed42 của Table 7-10) + ModULE unlearn DOMAIN (sketch) k_u=4, 20 epoch,
# không early-stop, kèm Eq.7 (RFO = Routing Overlap). SCEN=class để chạy thêm class (dog).
#   PY=<python có torch> nohup bash scripts/run_abla_num_experts_v2.sh > nexp.nohup.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}; SCEN=${SCEN:-domain}; MS=${MS:-"4 8 12 16 24"}
# KU=prop -> dùng config unlprop_* (k_u tỉ lệ M/3: 2/3/4/6/8) thay vì k_u=4 cố định
PFX=unl; [ "${KU:-}" = "prop" ] && PFX=unlprop; [ "${KU:-}" = "half" ] && PFX=unlhalf   # half: k_u = M/2 (2/4/6/8/12)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/nexp_v2; LOG=$R/logs; DONE=$R/done; mkdir -p "$LOG" "$DONE"
[ -d dataset/data_folder/pacs ] || "$PY" -m dataset.downloader.pacs || exit 1
for M in $MS; do
  ck=runs/_base_models/nexp_M${M}_k4_seed42/checkpoints/base_M${M}_k4_best.pt
  if [ -f "$ck" ] && [ -f "$DONE/base_M$M" ]; then echo "[skip] base_M$M"; else
    echo ""; echo "######## $(date '+%m-%d %T')  base_M$M"
    "$PY" -m learn --config "config/abla_num_experts_v2/base_M${M}_k4.yaml" > "$LOG/base_M$M.log" 2>&1
    st=$?; [ $st -eq 0 ] && touch "$DONE/base_M$M" || { echo "[FAIL exit $st]"; grep -E "Traceback|Error" "$LOG/base_M$M.log" | tail -2; continue; }
    grep -oE "\[Final Metrics\].*" "$LOG/base_M$M.log" | tail -1
  fi
  if [ -f "$DONE/${PFX}_${SCEN}_M$M" ]; then echo "[skip] unl_${SCEN}_M$M"; continue; fi
  echo "######## $(date '+%m-%d %T')  ${PFX}_${SCEN}_M$M"
  "$PY" -m unlearn --config "config/abla_num_experts_v2/${PFX}_${SCEN}_M${M}.yaml" > "$LOG/${PFX}_${SCEN}_M$M.log" 2>&1
  st=$?; [ $st -eq 0 ] && touch "$DONE/${PFX}_${SCEN}_M$M" || { echo "[FAIL exit $st]"; grep -E "Traceback|Error" "$LOG/${PFX}_${SCEN}_M$M.log" | tail -2; }
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$LOG/${PFX}_${SCEN}_M$M.log" | tail -1
  grep -oE "Retain-forget routing overlap \(RFO\): [0-9.]+" "$LOG/${PFX}_${SCEN}_M$M.log" | tail -1
done
echo ""; echo "### $(date '+%m-%d %T') NUM-EXPERTS ($SCEN) XONG ###"
echo "M | TA | FA | RA | MIA | RFO(overlap)"
for M in $MS; do
  L="$LOG/${PFX}_${SCEN}_M$M.log"; [ -f "$L" ] || continue
  m=$(grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$L" | tail -1)
  rfo=$(grep -oE "Retain-forget routing overlap \(RFO\): [0-9.]+" "$L" | tail -1 | grep -oE "[0-9.]+$")
  echo "$M | $m | RFO=$rfo"
done
