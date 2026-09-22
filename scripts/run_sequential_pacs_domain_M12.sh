#!/usr/bin/env bash
# Sequential domain removal on PACS with EVERY method on the same Original model
# (base pacs_M12_k4_seed42, M=12/k=4) and a matching M=12/k=4 Retrain reference.
#   phase 1: the 3 per-stage Retrain references (shared)      -- 3 x 50 epochs
#   phase 2: SalUn, Boundary Shrink, SCRUB, GRIP chains        -- 3 stages each
#   phase 3: summary.csv (+ gaps) for the 4 chains and for the ModULE v2 chain,
#            which is already trained (config/sequential/pacs_module_seq_domain_v2.yaml).
# Resumable: finished stages / retrains are skipped.
#   bash scripts/run_sequential_pacs_domain_M12.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PY:-/home/duong/miniconda3/envs/module/bin/python3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt
[ -f "$BASE" ] || { echo "[!] thiếu base: $BASE"; exit 1; }

ALGOS=(salun boundary_shrink scrub grip)
run() { local cfg="$1"; shift; echo ""; echo "=== $(date +%T) $(basename "$cfg" .yaml) $* ==="
        $PYTHON sequential_unlearn.py --config "$cfg" "$@"; local st=$?
        [ "$st" -ne 0 ] && echo "[!] $(basename "$cfg") FAILED (exit $st)"; return "$st"; }

echo "### PHASE 1 — Retrain M12/k4 per stage ###"
run config/sequential/pacs_salun_seq_domain_M12.yaml --retrain-only || true

echo "### PHASE 2 — baseline chains on the M12/k4 base ###"
for a in "${ALGOS[@]}"; do run "config/sequential/pacs_${a}_seq_domain_M12.yaml" || true; done

echo "### PHASE 3 — summaries with gaps ###"
for a in "${ALGOS[@]}"; do run "config/sequential/pacs_${a}_seq_domain_M12.yaml" --retrain || true; done
run config/sequential/pacs_module_seq_domain_v2.yaml --retrain || true

echo "### XONG ###"
for f in runs/sequential/pacs_*_seq_domain_M12/summary.csv runs/sequential/pacs_module_seq_domain_v2/summary.csv; do
  [ -f "$f" ] && { echo "--- $f"; cat "$f"; }
done
