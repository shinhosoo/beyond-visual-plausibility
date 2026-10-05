#!/usr/bin/env bash
# Per-seed retraining: Stage 1 from scratch, then the Stage 2 branches of four settings.
# Results go to outputs_seed<SEED>/; the existing outputs/ directory is left untouched.
#   SEEDS="43 44" HAM_GPU=0 nohup bash scripts/run_seed.sh > logs/seed.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-0}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEEDS="${SEEDS:-43 44 45 46}"
TAGS="${TAGS:-baseline teme ltv full}"
mkdir -p logs/seedrun logs/batch_status
ST="logs/batch_status/seedrun.txt"; T0=$(date +%s); DONE=0
TOTAL=$(( $(echo $SEEDS | wc -w) * ($(echo $TAGS | wc -w) * 3 + 1) ))
status () { { echo "batch=seedrun (Stage 1 retraining) GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL elapsed $(( ($(date +%s)-T0)/60 ))"; echo "current: $1"
 echo ": outputs_seed<seed>/ ( outputs/ )"; } > "$ST"; }
R() { OUT="$1"; shift; python -m experiments.bb_run --backbone mambaout_tiny.in1k --out "$OUT" -- "$@"; }

echo "===== seed retraining seeds=$SEEDS GPU=$HAM_GPU start $(date '+%m-%d %H:%M') ====="
for S in $SEEDS; do
  OUT="$PWD/outputs_seed$S"
  mkdir -p "$OUT/experiments"
  if [ ! -f "$OUT/stage1_mela_ltv_softrouting_multiscale_7class.pth" ]; then
 status "seed $S: Stage 1 training"
    python -m experiments.seed_run --seed $S --out "$OUT" -- SkinCancer.main.run_stage1 \
      > logs/seedrun/s1_s$S.log 2>&1 \
 || { echo "--- [s$S Stage1] failed"; tail -8 logs/seedrun/s1_s$S.log; continue; }
 echo "--- [s$S Stage1] done $(date '+%H:%M')"
  fi
  DONE=$((DONE+1))
  for T in $TAGS; do
    case "$T" in
      baseline) OPT="--teme 0 --ltv 0" ;;
      teme)     OPT="--teme 1 --ltv 0" ;;
      ltv)      OPT="--teme 0 --ltv 1" ;;
      full)     OPT="--teme 1 --ltv 1" ;;
    esac
    if [ -f "$OUT/experiments/tbl_${T}_s${S}_test_predictions.npz" ]; then DONE=$((DONE+3)); continue; fi
    for STG in mel nonmel; do
 status "seed $S: $T / $STG training"
      R "$OUT" experiments.run --stage $STG --tag "tbl_${T}_s${S}" $OPT --seed $S \
        > logs/seedrun/${T}_${STG}_s$S.log 2>&1 \
 || { echo "--- [s$S $T $STG] failed"; tail -8 logs/seedrun/${T}_${STG}_s$S.log; }
      DONE=$((DONE+1))
    done
 status "seed $S: $T prediction"
    R "$OUT" experiments.run --stage predict --tag "tbl_${T}_s${S}" $OPT --seed $S --split test \
      > logs/seedrun/${T}_pred_s$S.log 2>&1 \
 || { echo "--- [s$S $T predict] failed"; tail -8 logs/seedrun/${T}_pred_s$S.log; }
    DONE=$((DONE+1))
 echo "--- [s$S $T] $(date '+%H:%M') done"
  done
done
status "done"; echo " end $(date '+%m-%d %H:%M')"
