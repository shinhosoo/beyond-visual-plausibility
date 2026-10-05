#!/usr/bin/env bash
# Repeat the pipeline with a different backbone; the original sources are not modified.
#   BB=convnext_tiny STEP=flat HAM_GPU=0 nohup bash scripts/run_backbone.sh > logs/bb.log 2>&1 &
#   STEP: flat (single-stage) | unsup (unsupervised LTV) | sup (mask-supervised LTV)
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-0}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BB="${BB:-convnext_tiny}"; OUT="$PWD/outputs_${BB}"
STEP="${STEP:-flat}"; FOLDS="${FOLDS:-0 1 2 3 4}"; SEED="${SEED:-42}"
mkdir -p "$OUT/residual" "$OUT/localize" logs/bb logs/batch_status logs/batch_summary
cp -n "$PWD/outputs/residual/cv_folds.csv" "$OUT/residual/cv_folds.csv" 2>/dev/null
R() { python -m experiments.bb_run --backbone "$BB" --out "$OUT" -- "$@"; }
ST="logs/batch_status/bb_${STEP}.txt"; T0=$(date +%s); DONE=0; TOTAL=$(echo $FOLDS | wc -w)
status () { { echo "batch=bb_$STEP ($BB) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL fold elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo ": $OUT ( outputs )"; } > "$ST"; }
echo "===== backbone [$BB] step=$STEP GPU=$HAM_GPU start $(date '+%m-%d %H:%M') ====="
for K in $FOLDS; do
  case "$STEP" in
    flat)
      [ -f "$OUT/residual/flat_f${K}_s${SEED}_val_predictions.npz" ] && { DONE=$((DONE+1)); continue; }
 status "1-stage fold $K"
      R experiments.residual --model flat --fold $K --seed $SEED > logs/bb/flat_f$K.log 2>&1 \
 || { echo "--- [flat f$K] failed"; tail -8 logs/bb/flat_f$K.log; continue; }
      echo "--- [flat f$K] $(grep -o 'val acc=[0-9.]*' logs/bb/flat_f$K.log)" ;;
    unsup)
      [ -f "$OUT/localize/loc_orig_f${K}.npz" ] && { DONE=$((DONE+1)); continue; }
 status "unsupervised LTV fold $K"
      R experiments.localize --stage train --fold $K --ltv-mode orig > logs/bb/unsup_tr_f$K.log 2>&1 \
 || { echo "--- [unsup f$K training] failed"; tail -8 logs/bb/unsup_tr_f$K.log; continue; }
      R experiments.localize --stage eval --fold $K --ltv-mode orig > logs/bb/unsup_ev_f$K.log 2>&1 \
 || { echo "--- [unsup f$K measurement] failed"; tail -8 logs/bb/unsup_ev_f$K.log; continue; }
      grep -E "^  (ltv|gc4|gc2|center) " logs/bb/unsup_ev_f$K.log | sed 's/^/    /' ;;
    sup)
      [ -f "$OUT/localize/loc_sup0.2_bias_local_f${K}.npz" ] && { DONE=$((DONE+1)); continue; }
 status "mask supervised fold $K"
      R experiments.localize_sup --stage train --fold $K --lam 0.2 > logs/bb/sup_tr_f$K.log 2>&1 \
 || { echo "--- [sup f$K training] failed"; tail -8 logs/bb/sup_tr_f$K.log; continue; }
      R experiments.localize_sup --stage eval --fold $K --lam 0.2 > logs/bb/sup_ev_f$K.log 2>&1 \
 || { echo "--- [sup f$K measurement] failed"; tail -8 logs/bb/sup_ev_f$K.log; continue; }
      grep -E "^  (ltv|gc4|gc2|center) " logs/bb/sup_ev_f$K.log | sed 's/^/    /' ;;
  esac
 DONE=$((DONE+1)); status " fold "
  HAM_OUTPUT_ROOT="$OUT" python -m experiments.loc_table > logs/batch_summary/bb_${BB}_loc.txt 2>/dev/null
done
status "done"; echo " end $(date '+%m-%d %H:%M')"
