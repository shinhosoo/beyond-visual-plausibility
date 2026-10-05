#!/usr/bin/env bash
# Multi-seed runs of the ablation settings on the standard split.
#   HAM_GPU=0 nohup bash scripts/run_tbl_batch.sh > logs/tbl.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-1}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BATCH="${BATCH:-C}"
DEF="flat:45 flat:46"
for S in 43 44 45 46; do for N in baseline ltv teme simple; do DEF="$DEF run:$N:$S"; done; done
DEF="$DEF run:full:45 run:full:46 cv:flat:43 cv:flat:44 cv:d3lr:43"
JOBS="${JOBS:-$DEF}"
mkdir -p logs/tbl logs/batch_status logs/batch_summary
ST="logs/batch_status/${BATCH}.txt"
TOTAL=$(echo $JOBS | wc -w); DONE=0; RUN=0; T0=$(date +%s); LAST="-"

status () {
  local now=$(date +%s); local el=$(( (now-T0)/60 )); local eta="-"
  [ $RUN -gt 0 ] && eta=$(awk "BEGIN{printf \"%.1f\", ($TOTAL-$DONE)*($now-$T0)/60/$RUN/60}")
 { echo "batch=$BATCH GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL done ( run elapsed ${el}, ${eta})"
 echo "current: $1"; echo " done: $LAST"; } > "$ST"
}
tv () {
  case "$1" in baseline) echo "0 0 0";; ltv) echo "0 1 0";; teme) echo "1 0 0";; full) echo "1 1 0";; simple) echo "0 0 1";; esac
}

echo "===== table seed batch $BATCH GPU=$HAM_GPU start $(date '+%m-%d %H:%M') $TOTAL ====="
for J in $JOBS; do
  IFS=: read -r TYPE A B <<< "$J"
  case "$TYPE" in
    flat)
      if [ -f "outputs/hier/flat_s${A}_test_predictions.npz" ]; then DONE=$((DONE+1)); continue; fi
 status "1-stage seed $A"
 python -m experiments.flat_seed --seed $A > logs/tbl/flat_s$A.log 2>&1 || { echo "--- [$J] failed"; continue; } ;;
    run)
      TAG="tbl_${A}_s${B}"
      if [ -f "outputs/experiments/${TAG}_test_predictions.npz" ]; then DONE=$((DONE+1)); continue; fi
      read -r TE LT SI <<< "$(tv $A)"
 status "$A seed $B"
      ok=1
      for BR in mel nonmel; do
        python -m experiments.run --stage $BR --tag $TAG --teme $TE --ltv $LT --simple-head $SI --seed $B \
          > logs/tbl/${TAG}_$BR.log 2>&1 || { ok=0; break; }
      done
      for SP in val test; do
        [ $ok = 1 ] && { python -m experiments.run --stage predict --tag $TAG --teme $TE --ltv $LT --simple-head $SI \
          --split $SP > logs/tbl/${TAG}_pred_$SP.log 2>&1 || ok=0; }
      done
 [ $ok = 1 ] || { echo "--- [$J] failed"; continue; } ;;
    cv)
      MOD="residual"; ARG="--model $A"; [ "$A" = "d3lr" ] && { MOD="residual_lr"; ARG="--name $A"; }
      ok=1
      for K in 0 1 2 3 4; do
        [ -f "outputs/residual/${A}_f${K}_s${B}_val_predictions.npz" ] && continue
 status "CV $A seed $B fold $K"
        python -m experiments.$MOD $ARG --fold $K --seed $B > logs/tbl/cv_${A}_s${B}_f$K.log 2>&1 || { ok=0; break; }
      done
 [ $ok = 1 ] || { echo "--- [$J] failed"; continue; } ;;
  esac
  DONE=$((DONE+1)); RUN=$((RUN+1)); LAST="$J $(date '+%H:%M')"
 echo "--- [$J] done $(date '+%H:%M')"
  python -m experiments.acc_table > logs/batch_summary/acc_table.txt 2>/dev/null
 status " "
done
python -m experiments.acc_table > logs/batch_summary/acc_table.txt 2>/dev/null
status "done"
echo " end $(date '+%m-%d %H:%M')"
