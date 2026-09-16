#!/usr/bin/env bash
# Bước unlearn cho S13 (5 mục tiêu diversity) và S14 (5 kiến trúc).
# Hai dòng dense không có expert để chọn -> update_scope: full_model (theo chỉ đạo).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/admin1/miniconda3/envs/Messi/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/tmp/claude-1000/-home-admin1-projects-MoDULE/ce607d58-cf99-4c5b-87d3-01167772b49e/scratchpad/unl
R=results/main_table_pacs_seed42
until grep -q "### S12-UNL XONG ###" "$R/s12_unl.log" 2>/dev/null; do sleep 30; done

run_one () {   # $1 = tên config learn (vd s13_cka), $2 = thư mục config
  local nm="$1" grp="$2"
  local cfg=$S/unl_${nm}.yaml
  $PY - "$cfg" "$nm" "$grp" <<'PYEOF'
import sys, yaml
dst, nm, grp = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open("config/experiments/pacs_module_class_unlearn.yaml"))
d.pop("sweep", None)
u = d["unlearn"]; u["wandb_mode"]="disabled"; u["study_name"]=f"unl_{nm}"
d["name"]=f"unl_{nm}"
lb = yaml.safe_load(open(f"config/learn/{grp}/{nm}.yaml"))
for k in ("num_experts","gate_k","expert_depth","expert_hidden_ratio","moe_layers","model_name","mlp_ratio"):
    if k in lb: d["learn"][k]=lb[k]
# dense: khong co expert de chon -> cap nhat toan mo hinh
if str(lb.get("moe_layers","")).count("S") == 0:
    u["update_scope"] = "full_model"
    u["_note_scope"] = "dense control: khong co expert, dung full_model"
yaml.safe_dump(d, open(dst,"w"), sort_keys=False, allow_unicode=True)
print(f"    update_scope={u['update_scope']}  moe_layers={lb.get('moe_layers')}")
PYEOF
  echo "######## $(date '+%m-%d %T')  unlearn $nm"
  $PY config/experiments/run_moe_pipeline.py --config "$cfg" --stage unlearn \
     --checkpoint "runs/_base_models/${nm}/checkpoints/${nm}.pt" --force \
     > "$S/log_${nm}.txt" 2>&1
  echo "[exit $?]"
  grep -oE "RA: [0-9.]+% \| FA: [0-9.]+% \| TA: [0-9.]+% \| MIA: [0-9.]+" "$S/log_${nm}.txt" | tail -1
  grep -oE "(Traceback|Error|error:).*" "$S/log_${nm}.txt" | head -2
}
for n in none cosine orthogonality cka output_decorrelation; do run_one "s13_$n" s13; done
echo "### S13-UNL XONG ###"
for n in dense_active dense_total standard_moe moe_no_modular module; do run_one "s14_$n" s14; done
echo "### S14-UNL XONG ###"
