#!/usr/bin/env bash
# Explanation-quality runs for the supervised configurations.
#   HAM_GPU=0 nohup bash scripts/run_expl.sh > logs/expl.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-2}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/sup logs/faithful logs/batch_status logs/batch_summary
ST="logs/batch_status/O.txt"; T0=$(date +%s); DONE=0; TOTAL=15
status () { { echo "batch=O (interpretability ) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo "results: logs/batch_summary/loc_table.txt , faith_table.txt"; } > "$ST"; }

echo "===== 1) mask supervised + $(date '+%H:%M') ====="
for K in 0 1 2 3 4; do
  if [ -f "outputs/localize/loc_supg0.2_global_f${K}.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "1) fold $K"
  python -m experiments.localize_sup2 --stage train --fold $K --lam 0.2 > logs/sup/supg_f$K.log 2>&1 || continue
  python -m experiments.localize_sup2 --stage eval  --fold $K --lam 0.2 > logs/sup/supg_eval_f$K.log 2>&1 || continue
 DONE=$((DONE+1)); echo "--- [ f$K] $(grep -o 'readout acc=[0-9.]*' logs/sup/supg_eval_f$K.log)"
  python -m experiments.loc_table > logs/batch_summary/loc_table.txt 2>/dev/null
done

echo "===== 2) supervised attention / $(date '+%H:%M') ====="
for K in 0 1 2 3 4; do
 status "2) / fold $K"
  python -m experiments.faithful --fold $K --names sup0.2_bias_local supg0.2_global \
 > logs/faithful/sup_f$K.log 2>&1 || { echo "--- [2) f$K] failed"; continue; }
  DONE=$((DONE+1)); grep -E "^  (sup|gc4|center|random)" logs/faithful/sup_f$K.log | sed 's/^/    /'
  python -m experiments.faith_table > logs/batch_summary/faith_table.txt 2>/dev/null
done

echo "===== 3) seed 43 () $(date '+%H:%M') ====="
for K in 0 1 2 3 4; do
  if [ -f "outputs/localize/seed43/loc_supg0.2_global_f${K}.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "3) seed 43 fold $K"
  python -m experiments.localize_sup2 --stage train --fold $K --seed 43 --lam 0.2 > logs/sup/supg_s43_f$K.log 2>&1 || continue
  python -m experiments.localize_sup2 --stage eval  --fold $K --seed 43 --lam 0.2 > logs/sup/supg_s43_eval_f$K.log 2>&1 || continue
  DONE=$((DONE+1)); python -m experiments.loc_table > logs/batch_summary/loc_table.txt 2>/dev/null
done
status "done"; echo " end $(date '+%m-%d %H:%M')"
