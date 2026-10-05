#!/usr/bin/env bash
# Attention supervision with a fraction of the masks.
#   FRACS="0.1 0.25 0.5" HAM_GPU=0 nohup bash scripts/run_frac.sh > logs/frac.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-1}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FRACS="${FRACS:-0.1 0.25 0.5}"; FOLDS="${FOLDS:-0 1 2 3 4}"; LAM="${LAM:-0.2}"
mkdir -p logs/frac logs/batch_status logs/batch_summary
ST="logs/batch_status/frac.txt"; T0=$(date +%s); DONE=0
TOTAL=$(( $(echo $FRACS | wc -w) * $(echo $FOLDS | wc -w) ))
status () { { echo "batch=frac (mask supervised) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL fold elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo "results: python -m experiments.loc_table | grep frac"; } > "$ST"; }
echo "===== mask GPU=$HAM_GPU fracs=$FRACS start $(date '+%m-%d %H:%M') ====="
for F in $FRACS; do
  for K in $FOLDS; do
    if [ -f "outputs/localize/loc_frac${F}_bias_local_f${K}.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "frac=$F fold $K"
    python -m experiments.sup_frac --stage train --fold $K --frac $F --lam $LAM \
 > logs/frac/f${F}_k${K}.log 2>&1 || { echo "--- [frac $F f$K training] failed"; tail -6 logs/frac/f${F}_k${K}.log; continue; }
    python -m experiments.sup_frac --stage eval --fold $K --frac $F \
 > logs/frac/f${F}_k${K}_eval.log 2>&1 || { echo "--- [frac $F f$K measurement] failed"; tail -6 logs/frac/f${F}_k${K}_eval.log; continue; }
    DONE=$((DONE+1))
    echo "--- [frac $F f$K] $(date '+%H:%M')  $(grep -E '^  ltv ' logs/frac/f${F}_k${K}_eval.log | tail -1)"
    python -m experiments.loc_table > logs/batch_summary/loc_table.txt 2>/dev/null
  done
done
status "done"; echo " end $(date '+%m-%d %H:%M')"
