#!/usr/bin/env bash
# Injection of the refined features back into the mid-level backbone representation.
#   MODES="A B" HAM_GPU=0 nohup bash scripts/run_mid.sh > logs/mid.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-1}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODES="${MODES:-A B}"; FOLDS="${FOLDS:-0 1 2 3 4}"; SEED="${SEED:-42}"
mkdir -p logs/mid logs/batch_status logs/batch_summary
ST="logs/batch_status/P.txt"; T0=$(date +%s); DONE=0
TOTAL=$(( $(echo $MODES | wc -w) * $(echo $FOLDS | wc -w) ))
status () { { echo "batch=P ( table ) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL fold elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo "results: logs/batch_summary/mid_table.txt"; } > "$ST"; }
echo "===== table GPU=$HAM_GPU modes=$MODES start $(date '+%m-%d %H:%M') ====="
for M in $MODES; do
  for K in $FOLDS; do
    if [ -f "outputs/residual/mid${M}_f${K}_s${SEED}_val_predictions.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "mode=$M fold $K"
    python -m experiments.midres --mode $M --fold $K --seed $SEED > "logs/mid/mid${M}_f$K.log" 2>&1 \
 || { echo "--- [mid$M f$K] failed"; tail -6 "logs/mid/mid${M}_f$K.log"; continue; }
    DONE=$((DONE+1))
 echo "--- [mid$M f$K] $(date '+%H:%M') $(grep -o 'val acc=[0-9.]*' logs/mid/mid${M}_f$K.log) $(grep -o ' [-0-9.]*' logs/mid/mid${M}_f$K.log | tail -1)"
    python -m experiments.residual_cv --models flat midA midB > logs/batch_summary/mid_table.txt 2>/dev/null
  done
done
status "done"; echo " end $(date '+%m-%d %H:%M')"
