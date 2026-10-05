#!/usr/bin/env bash
# Lesion-constrained training: lesion-centred cropping and background attenuation.
#   STEPS="lcrop bgdim" HAM_GPU=0 nohup bash scripts/run_bg.sh > logs/bg.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-0}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FOLDS="${FOLDS:-0 1 2 3 4}"; SEED="${SEED:-42}"; STEPS="${STEPS:-zoom2 lcrop bgdim}"
mkdir -p logs/zoom logs/aug2 logs/batch_status logs/batch_summary outputs/zoom
ST="logs/batch_status/R.txt"; T0=$(date +%s); DONE=0
TOTAL=$(( $(echo $STEPS | wc -w) * $(echo $FOLDS | wc -w) ))
status () { { echo "batch=R (  augmentation) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL fold elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo "results: logs/batch_summary/zoom2_table.txt , bg_table.txt"; } > "$ST"; }
echo "===== GPU=$HAM_GPU steps=$STEPS start $(date '+%m-%d %H:%M') ====="
for S in $STEPS; do
  for K in $FOLDS; do
    if [ "$S" = "zoom2" ]; then
      [ -f "outputs/zoom/zoom2_f${K}_s${SEED}.npz" ] && { DONE=$((DONE+1)); continue; }
 status "1) fold $K"
      python -m experiments.zoom2 --fold $K --seed $SEED > logs/zoom/z2_f$K.log 2>&1 \
 || { echo "--- [zoom2 f$K] failed"; tail -6 logs/zoom/z2_f$K.log; continue; }
      python -m experiments.zoom2_table > logs/batch_summary/zoom2_table.txt 2>/dev/null
    else
      [ -f "outputs/residual/${S}_f${K}_s${SEED}_val_predictions.npz" ] && { DONE=$((DONE+1)); continue; }
 status "${S} fold $K"
      python -m experiments.aug2 --aug $S --fold $K --seed $SEED > logs/aug2/${S}_f$K.log 2>&1 \
 || { echo "--- [$S f$K] failed"; tail -6 logs/aug2/${S}_f$K.log; continue; }
      echo "--- [$S f$K] $(date '+%H:%M')  $(grep -o 'val acc=[0-9.]*' logs/aug2/${S}_f$K.log)"
      python -m experiments.residual_cv --models flat lcrop bgdim > logs/batch_summary/bg_table.txt 2>/dev/null
    fi
    DONE=$((DONE+1))
  done
done
status "done"; echo " end $(date '+%m-%d %H:%M')"
