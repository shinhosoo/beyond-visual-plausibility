#!/usr/bin/env bash
# Re-input of the magnified lesion region (inference only).
#   HAM_GPU=0 nohup bash scripts/run_zoom.sh > logs/zoom.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-1}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FOLDS="${FOLDS:-0 1 2 3 4}"; READOUT="${READOUT:-sup0.2_bias_local}"
mkdir -p logs/zoom logs/batch_status logs/batch_summary outputs/zoom
ST="logs/batch_status/Q.txt"; T0=$(date +%s); DONE=0; TOTAL=$(echo $FOLDS | wc -w)
status () { { echo "batch=Q ( , ) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL fold elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo "results: logs/batch_summary/zoom_table.txt"; } > "$ST"; }
echo "===== GPU=$HAM_GPU readout=$READOUT start $(date '+%m-%d %H:%M') ====="
for K in $FOLDS; do
  if [ -f "outputs/zoom/zoom_f${K}_s42.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "fold $K"
  python -m experiments.zoom --fold $K --readout $READOUT > logs/zoom/f$K.log 2>&1 \
 || { echo "--- [f$K] failed"; tail -6 logs/zoom/f$K.log; continue; }
  DONE=$((DONE+1)); grep -E "^  (base|rand|center|maskbox|ltvbox|ltvonly)" logs/zoom/f$K.log | sed 's/^/    /'
  python -m experiments.zoom_table > logs/batch_summary/zoom_table.txt 2>/dev/null
done
status "done"; echo " end $(date '+%m-%d %H:%M')"
