#!/usr/bin/env bash
# Class-aware augmentation (cross-validated).
#   HAM_GPU=0 nohup bash scripts/run_aug.sh > logs/aug.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-2}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${MODEL:-nonmel}"; AUGS="${AUGS:-none S M B}"; FOLDS="${FOLDS:-0 1 2 3 4}"; SEED="${SEED:-42}"
case "$MODEL" in flat) PRE=flat;; mel) PRE=mel;; nonmel) PRE=nm;; esac
mkdir -p logs/aug logs/batch_status logs/batch_summary
ST="logs/batch_status/aug_${MODEL}.txt"; T0=$(date +%s); DONE=0
TOTAL=$(( $(echo $AUGS | wc -w) * $(echo $FOLDS | wc -w) ))
status () { { echo "batch=aug_$MODEL GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL fold elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo "results: logs/batch_summary/aug_table.txt"; } > "$ST"; }
echo "===== augmentation model=$MODEL GPU=$HAM_GPU augs=$AUGS start $(date '+%m-%d %H:%M') ====="
for A in $AUGS; do
  for K in $FOLDS; do
    TAG="${PRE}${A}"
    if [ -f "outputs/residual/${TAG}_f${K}_s${SEED}_val_predictions.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "$MODEL aug=$A fold $K"
    python -m experiments.aug_cv --model $MODEL --aug $A --fold $K --seed $SEED > "logs/aug/${TAG}_f$K.log" 2>&1 \
 || { echo "--- [$TAG f$K] failed"; tail -6 "logs/aug/${TAG}_f$K.log"; continue; }
    DONE=$((DONE+1))
    echo "--- [$TAG f$K] $(date '+%H:%M')  $(grep -o 'val acc=[0-9.]*' logs/aug/${TAG}_f$K.log)"
    python -m experiments.aug_table > logs/batch_summary/aug_table.txt 2>/dev/null
  done
done
status "done"
echo " end $(date '+%m-%d %H:%M')"
