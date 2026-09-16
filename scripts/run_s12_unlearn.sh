#!/usr/bin/env bash
# Bước unlearn cho S12: lấy RA/FA của Table 12. Dùng cấu hình MoDULE class chuẩn
# (k_u=2, selection=diff, update_scope=selected_experts_and_head, 20 epoch, lr 1e-4)
# áp lên từng base checkpoint của ba chế độ ước lượng cân bằng.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-1000/-home-admin1-projects-MoDULE/ce607d58-cf99-4c5b-87d3-01167772b49e/scratchpad/unl
for est in minibatch ema none; do
  cfg=$S/unl_s12_${est}.yaml
  $PY - "$cfg" "$est" <<'PYEOF'
import sys, yaml
dst, est = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open("config/experiments/pacs_module_class_unlearn.yaml"))
d.pop("sweep", None)
u = d["unlearn"]; u["wandb_mode"]="disabled"; u["study_name"]=f"s12_unl_{est}"
d["name"]=f"s12_unl_{est}"
# base learn block phải khớp kiến trúc của checkpoint
lb = yaml.safe_load(open(f"config/learn/s12/s12_{est}.yaml"))
for k in ("num_experts","gate_k","expert_depth","expert_hidden_ratio","moe_layers","model_name","mlp_ratio"):
    if k in lb: d["learn"][k]=lb[k]
yaml.safe_dump(d, open(dst,"w"), sort_keys=False, allow_unicode=True)
PYEOF
  echo ""; echo "######## $(date '+%m-%d %T')  unlearn S12/$est"
  $PY config/experiments/run_moe_pipeline.py --config "$cfg" --stage unlearn \
     --checkpoint "runs/_base_models/s12_${est}/checkpoints/s12_${est}.pt" --force \
     > "$S/log_s12_${est}.txt" 2>&1
  echo "[exit $?]"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$S/log_s12_${est}.txt" | tail -1
  grep -oE "(Traceback|Error|error:).*" "$S/log_s12_${est}.txt" | head -2
done
echo "### S12-UNL XONG ###"
