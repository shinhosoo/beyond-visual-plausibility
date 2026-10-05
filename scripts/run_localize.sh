#!/usr/bin/env bash
# Explanation-quality measurement for a single configuration.
#   HAM_GPU=0 bash scripts/run_localize.sh
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-0}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FOLDS="${FOLDS:-0 1 2 3 4}"; MODE="${MODE:-orig}"
mkdir -p logs/localize outputs/localize
echo "===== measurement GPU=$HAM_GPU mode=$MODE folds=$FOLDS start $(date '+%H:%M:%S') ====="
for K in $FOLDS; do
 [ -f "outputs/residual/flat_f${K}_s42.pth" ] || { echo " flat_f${K} checkpoint missing"; exit 1; }
  if [ ! -f "outputs/localize/readout_${MODE}_f${K}.pth" ]; then
    echo "--- [readout ${MODE} f$K]  $(date '+%H:%M:%S')"
    python -m experiments.localize --stage train --fold $K --ltv-mode $MODE > "logs/localize/train_${MODE}_f$K.log" 2>&1 \
 || { echo " failed:"; tail -20 "logs/localize/train_${MODE}_f$K.log"; exit 1; }
    grep -o "best readout.*" "logs/localize/train_${MODE}_f$K.log" | sed 's/^/    /'
  fi
 echo "--- [measurement ${MODE} f$K] $(date '+%H:%M:%S')"
  python -m experiments.localize --stage eval --fold $K --ltv-mode $MODE > "logs/localize/eval_${MODE}_f$K.log" 2>&1 \
 || { echo " failed:"; tail -20 "logs/localize/eval_${MODE}_f$K.log"; exit 1; }
  grep -E "^\[localize|^  (ltv|gc4|gc2|center) " "logs/localize/eval_${MODE}_f$K.log" | sed 's/^/    /'
done
echo " end $(date '+%H:%M:%S')"
