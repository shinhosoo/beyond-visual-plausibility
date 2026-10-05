#!/usr/bin/env bash
# LTV as the classification head, with the masking order corrected.
#   HAM_GPU=0 nohup bash scripts/run_explore_fix.sh > logs/explore_fix.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-2}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAMES="${NAMES:-ltv}"; FOLDS="${FOLDS:-0 1 2}"; SEED="${SEED:-42}"
mkdir -p logs/explore outputs/explore
echo "===== LTV modified GPU=$HAM_GPU names=$NAMES folds=$FOLDS start $(date '+%H:%M:%S') ====="
for N in $NAMES; do
  for K in $FOLDS; do
    TAG="${N}_f${K}_s${SEED}"; LOG="logs/explore/${TAG}.log"
 if [ -f "outputs/explore/${TAG}_val_predictions.npz" ]; then echo "--- [$TAG] done, "; continue; fi
    echo "--- [$TAG]  $(date '+%H:%M:%S')"
    python -m experiments.explore_fix --name "$N" --fold "$K" --seed "$SEED" > "$LOG" 2>&1 \
 || { echo " failed:"; tail -20 "$LOG" | sed 's/^/ /'; exit 1; }
    grep -o "\[explore.*val acc=[0-9.]*" "$LOG" | sed 's/^/    /'
  done
done
echo " end $(date '+%H:%M:%S')"
