#!/usr/bin/env bash
# Final configurations evaluated on the held-out test split.
#   HAM_GPU=0 nohup bash scripts/run_final.sh > logs/final.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-1}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODELS="${MODELS:-d3lr d3}"; SEEDS="${SEEDS:-42 43 44 45 46}"
mkdir -p logs/final logs/batch_status logs/batch_summary
ST="logs/batch_status/N.txt"; T0=$(date +%s); DONE=0
TOTAL=$(( $(echo $MODELS | wc -w) * $(echo $SEEDS | wc -w) ))
status () { { echo "batch=N (correction evaluation 5seed) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo "results: logs/batch_summary/final_table.txt"; } > "$ST"; }
echo "===== correction evaluation GPU=$HAM_GPU models=$MODELS start $(date '+%m-%d %H:%M') ====="
for M in $MODELS; do
  for S in $SEEDS; do
    if [ -f "outputs/residual/${M}_full_s${S}_test_predictions.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "$M seed $S"
    if [ "$M" = "d3lr" ] || [ "$M" = "d1lr" ]; then
      python -m experiments.residual_lr --name $M --fold -1 --seed $S > logs/final/${M}_s$S.log 2>&1 \
 || { echo "--- [$M s$S] failed"; tail -6 logs/final/${M}_s$S.log; continue; }
    else
      python -m experiments.residual --model $M --fold -1 --seed $S > logs/final/${M}_s$S.log 2>&1 \
 || { echo "--- [$M s$S] failed"; tail -6 logs/final/${M}_s$S.log; continue; }
    fi
    DONE=$((DONE+1))
    echo "--- [$M s$S] $(date '+%H:%M')  $(grep -o 'test acc=[0-9.]*' logs/final/${M}_s$S.log)"
    python -m experiments.final_table > logs/batch_summary/final_table.txt 2>/dev/null
  done
done
status "done"; echo " end $(date '+%m-%d %H:%M')"
