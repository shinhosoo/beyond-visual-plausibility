#!/usr/bin/env bash
# Correction configurations with a larger head learning rate.
#   HAM_GPU=0 nohup bash scripts/run_residual_lr.sh > logs/residual_lr.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-2}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAMES="${NAMES:-d1lr}"; FOLDS="${FOLDS:-0 1 2 3 4}"; SEED="${SEED:-42}"
mkdir -p logs/residual outputs/residual
echo "===== training 10 GPU=$HAM_GPU names=$NAMES start $(date '+%H:%M:%S') ====="
for N in $NAMES; do
  for K in $FOLDS; do
    TAG="${N}_f${K}_s${SEED}"; LOG="logs/residual/${TAG}.log"
 if [ -f "outputs/residual/${TAG}_val_predictions.npz" ]; then echo "--- [$TAG] done, "; continue; fi
    echo "--- [$TAG]  $(date '+%H:%M:%S')"
    python -m experiments.residual_lr --name "$N" --fold "$K" --seed "$SEED" > "$LOG" 2>&1 \
 || { echo " failed:"; tail -20 "$LOG" | sed 's/^/ /'; exit 1; }
    grep -o "\[residual.*val acc=[0-9.]*" "$LOG" | sed 's/^/    /'
  done
done
echo " end $(date '+%H:%M:%S')"
