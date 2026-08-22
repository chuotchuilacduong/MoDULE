#!/usr/bin/env bash
set -euo pipefail

# Learning-phase MoE ablations. These commands train base models only.
# Usage:
#   bash scripts/run_moe_learning_ablations.sh
#   DRY_RUN=1 bash scripts/run_moe_learning_ablations.sh
#   MAX_TRIALS=2 bash scripts/run_moe_learning_ablations.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PIPELINE="scripts/run_moe_pipeline.py"
CONFIG="config/experiments/moe_pipeline.yaml"
EXTRA_ARGS=()
[[ "${DRY_RUN:-0}" == "1" ]] && EXTRA_ARGS+=(--dry-run)
[[ -n "${MAX_TRIALS:-}" ]] && EXTRA_ARGS+=(--max-trials "${MAX_TRIALS}")

# L1. Expert capacity
python "${PIPELINE}" --stage learn --config "${CONFIG}" \
  --name learn_num_experts \
  --sweep 'learn.num_experts=[8,12]' \
  "${EXTRA_ARGS[@]}"

# L2. Active experts per token
python "${PIPELINE}" --stage learn --config "${CONFIG}" \
  --name learn_gate_k \
  --sweep 'learn.gate_k=[2,3,4]' \
  "${EXTRA_ARGS[@]}"

# L3. Expert depth
python "${PIPELINE}" --stage learn --config "${CONFIG}" \
  --name learn_expert_depth \
  --sweep 'learn.expert_depth=[1,2,3]' \
  "${EXTRA_ARGS[@]}"

# L4. Expert hidden-width ratio
python "${PIPELINE}" --stage learn --config "${CONFIG}" \
  --name learn_expert_hidden_ratio \
  --sweep 'learn.expert_hidden_ratio=[1,2,4]' \
  "${EXTRA_ARGS[@]}"

# L5. Sparse-routing loss
python "${PIPELINE}" --stage learn --config "${CONFIG}" \
  --name learn_lambda_sparse \
  --sweep 'learn.lambda_sparse=[0.0,0.1,0.5,1.0]' \
  "${EXTRA_ARGS[@]}"

# L6. Load-balancing loss
python "${PIPELINE}" --stage learn --config "${CONFIG}" \
  --name learn_lambda_balance \
  --sweep 'learn.lambda_balance=[0.0,0.1,0.5,1.0]' \
  "${EXTRA_ARGS[@]}"

# L7. Expert-diversity loss
python "${PIPELINE}" --stage learn --config "${CONFIG}" \
  --name learn_lambda_div \
  --sweep 'learn.lambda_div=[0.0,0.1,0.5,1.0]' \
  "${EXTRA_ARGS[@]}"

# L8. Independent training seeds for the final selected configuration
python "${PIPELINE}" --stage learn --config "${CONFIG}" \
  --name learn_seeds \
  --sweep 'learn.seed=[1,2,3,4,5]' \
  "${EXTRA_ARGS[@]}"
