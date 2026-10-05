#!/usr/bin/env bash
# Hierarchical correction with mask-supervised attention.
#   HAM_GPU=0 nohup bash scripts/run_hier_sup.sh > logs/hier_sup.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-2}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEEDS="${SEEDS:-42 43}"; FOLDS="${FOLDS:-0 1 2 3 4}"; LAM="${LAM:-1.0}"
mkdir -p logs/hier_sup logs/batch_status logs/batch_summary outputs/residual
ST="logs/batch_status/G.txt"; T0=$(date +%s); DONE=0
TOTAL=$(( $(echo $SEEDS | wc -w) * $(echo $FOLDS | wc -w) ))
status () { { echo "batch=G (masksupervised + correction) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL fold elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo ": logs/batch_summary/hier_sup.txt (fold 4/5 + p<0.05 + seed 43 )"; } > "$ST"; }

echo "===== mask supervised + correction GPU=$HAM_GPU lam=$LAM start $(date '+%m-%d %H:%M') ====="
for S in $SEEDS; do
  for K in $FOLDS; do
    if [ -f "outputs/residual/h1sup_f${K}_s${S}_val_predictions.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "seed $S fold $K"
    python -m experiments.hier_sup --fold $K --seed $S --lam $LAM > logs/hier_sup/f${K}_s${S}.log 2>&1 \
 || { echo "--- [f$K s$S] failed"; tail -8 logs/hier_sup/f${K}_s${S}.log; continue; }
    DONE=$((DONE+1))
    echo "--- [f$K s$S] $(date '+%H:%M')  $(grep -o 'val acc=.*' logs/hier_sup/f${K}_s${S}.log)"
 { echo "== $(date '+%m-%d %H:%M') (seed 42)"; python -m experiments.residual_cv --models flat h1sup
      if ls outputs/residual/h1sup_f*_s43_val_predictions.npz > /dev/null 2>&1; then
 echo; echo "== seed 43"; python -m experiments.residual_cv --models flat h1sup --seed 43; fi
    } > logs/batch_summary/hier_sup.txt 2>/dev/null
  done
done
status "done"
echo " end $(date '+%m-%d %H:%M')"
