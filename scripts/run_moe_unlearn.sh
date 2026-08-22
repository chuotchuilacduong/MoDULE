#!/usr/bin/env bash
set -euo pipefail

# Unlearning-phase ablations using one manually selected learned checkpoint.
# Usage:
#   BASE_CHECKPOINT=runs/_base_models/<id>/checkpoints/learn.pt \
#     bash scripts/run_moe_unlearning_ablations.sh
#   DRY_RUN=1 BASE_CHECKPOINT=/path/to/learn.pt \
#     bash scripts/run_moe_unlearning_ablations.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the selected learning checkpoint}"
PIPELINE="scripts/run_moe_pipeline.py"
CONFIG="config/experiments/moe_pipeline.yaml"
EXTRA_ARGS=()
[[ "${DRY_RUN:-0}" == "1" ]] && EXTRA_ARGS+=(--dry-run)
[[ -n "${MAX_TRIALS:-}" ]] && EXTRA_ARGS+=(--max-trials "${MAX_TRIALS}")

COMMON=(--stage unlearn --config "${CONFIG}" --checkpoint "${BASE_CHECKPOINT}")

# U1. Number of selected experts
python "${PIPELINE}" "${COMMON[@]}" --name unlearn_k_u \
  --sweep 'unlearn.k_u=[1,2,4,8]' "${EXTRA_ARGS[@]}"

# U2. Forget-expert selection strategy
python "${PIPELINE}" "${COMMON[@]}" --name unlearn_selection \
  --sweep 'unlearn.selection_option=[diff,ratio,gradient,random]' "${EXTRA_ARGS[@]}"

# U3. Parameter update scope
python "${PIPELINE}" "${COMMON[@]}" --name unlearn_update_scope \
  --sweep 'unlearn.update_scope=[selected_experts_only,selected_experts_and_head,selected_experts_and_router,all_experts,full_model]' \
  "${EXTRA_ARGS[@]}"

# U4. Retain-mass penalty in the diff responsibility score
python "${PIPELINE}" "${COMMON[@]}" --name unlearn_alpha \
  --sweep 'unlearn.alpha=[0.0,0.5,1.0,2.0]' "${EXTRA_ARGS[@]}"

# U5. Retain classification loss
python "${PIPELINE}" "${COMMON[@]}" --name unlearn_beta \
  --sweep 'unlearn.beta=[0.0,0.5,1.0,2.0]' "${EXTRA_ARGS[@]}"

# U6. Retain distillation loss
python "${PIPELINE}" "${COMMON[@]}" --name unlearn_gamma \
  --sweep 'unlearn.gamma=[0.0,0.5,1.0,2.0]' "${EXTRA_ARGS[@]}"

# U7. Expert-separation loss
python "${PIPELINE}" "${COMMON[@]}" --name unlearn_eta \
  --sweep 'unlearn.eta=[0.0,0.1,0.5,1.0]' "${EXTRA_ARGS[@]}"

# U8. Unlearning learning rate
python "${PIPELINE}" "${COMMON[@]}" --name unlearn_lr \
  --sweep 'unlearn.lr=[0.00001,0.00005,0.0001,0.0005]' "${EXTRA_ARGS[@]}"
