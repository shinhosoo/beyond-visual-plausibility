#!/usr/bin/env bash
# Mask supervision of the LTV attention at several loss weights.
#   LAMS="0.2 1.0 5.0" HAM_GPU=0 nohup bash scripts/run_next.sh > logs/sup.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-2}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAMES="${NAMES:-orig bias_local k2_bias_local}"
LAMS="${LAMS:-1.0 5.0}"
FOLDS="${FOLDS:-0 1 2 3 4}"
mkdir -p logs/faithful logs/sup logs/batch_status logs/batch_summary outputs/faithful
ST="logs/batch_status/E.txt"
T0=$(date +%s); DONE=0
TOTAL=$(( $(echo $FOLDS | wc -w) * (1 + $(echo $LAMS | wc -w)) ))
status () { { echo "batch=E (1) + 2)masksupervised) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"; } > "$ST"; }

echo "===== 1) / $(date '+%m-%d %H:%M') ====="
for K in $FOLDS; do
  if [ -f "outputs/faithful/faith_f${K}_s42.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "1) fold $K"
  python -m experiments.faithful --fold $K --names $NAMES > logs/faithful/f$K.log 2>&1 \
 || { echo " 1) fold $K failed"; tail -5 logs/faithful/f$K.log; continue; }
  grep -E "^  (orig|bias|k2|gc4|gc2|center|random)" logs/faithful/f$K.log | sed 's/^/    /'
  DONE=$((DONE+1)); python -m experiments.faith_table > logs/batch_summary/faith_table.txt 2>/dev/null
done

echo "===== 2) mask supervised training $(date '+%m-%d %H:%M') ====="
for LAM in $LAMS; do
  TAG="sup${LAM%.0}_bias_local"; [ "$LAM" = "1.0" ] && TAG="sup1_bias_local"; [ "$LAM" = "5.0" ] && TAG="sup5_bias_local"
  for K in $FOLDS; do
    if [ -f "outputs/localize/loc_${TAG}_f${K}.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "2) lam=$LAM fold $K"
    if [ ! -f "outputs/localize/readout_${TAG}_f${K}.pth" ]; then
      python -m experiments.localize_sup --stage train --fold $K --lam $LAM > logs/sup/${TAG}_f$K.log 2>&1 \
 || { echo " 2) $TAG f$K training failed"; tail -5 logs/sup/${TAG}_f$K.log; continue; }
    fi
    python -m experiments.localize_sup --stage eval --fold $K --lam $LAM > logs/sup/${TAG}_eval_f$K.log 2>&1 \
 || { echo " 2) $TAG f$K measurement failed"; tail -5 logs/sup/${TAG}_eval_f$K.log; continue; }
    grep -E "^\[localize|^  ltv " logs/sup/${TAG}_eval_f$K.log | sed 's/^/    /'
    DONE=$((DONE+1)); python -m experiments.loc_table > logs/batch_summary/loc_table.txt 2>/dev/null
  done
done
status "done"
echo " end $(date '+%m-%d %H:%M')"
