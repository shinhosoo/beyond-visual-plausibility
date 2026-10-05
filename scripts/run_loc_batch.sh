#!/usr/bin/env bash
# Explanation-quality measurement for the LTV design variants.
#   BATCH=A bash scripts/run_loc_batch.sh    original module and masking variants
#   BATCH=B bash scripts/run_loc_batch.sh    slot count, deeper features, no TEME, no global path
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-2}}"
export HAM_MASK_DIR="${HAM_MASK_DIR:-$PWD/masks_ham10000}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BATCH="${BATCH:-A}"
case "$BATCH" in
  A) DEF="bias:42 noteme:42 norm:42 orig:43 orig:44 bias_local:43 bias_local:44 bias:43 bias:44 noteme:43 noteme:44" ;;
  B) DEF="s3:42 s3_bias_local:42 noteme_bias_local:42 k1_bias_local:42 k2_bias_local:42 k8_bias_local:42 e20_bias_local:42 s3_bias_local:43 noteme_bias_local:43 k2_bias_local:43" ;;
  D) DEF="s3:43 s3:44 s3_bias_local:44 noteme_bias_local:44 k1_bias_local:43 k1_bias_local:44 k2_bias_local:44 k8_bias_local:43 k8_bias_local:44 e20_bias_local:43 e20_bias_local:44" ;;
  *) DEF="" ;;
esac
JOBS="${JOBS:-$DEF}"
mkdir -p logs/localize logs/batch_status logs/batch_summary outputs/localize
ST="logs/batch_status/${BATCH}.txt"
TOTAL=0; for J in $JOBS; do TOTAL=$((TOTAL+5)); done
DONE=0; RUN=0; T0=$(date +%s); LAST="-"

status () {
  local now=$(date +%s); local el=$(( (now-T0)/60 ))
  local per="-"; local eta="-"
  if [ $RUN -gt 0 ]; then per=$(( (now-T0)/60/RUN )); eta=$(awk "BEGIN{printf \"%.1f\", ($TOTAL-$DONE)*($now-$T0)/60/$RUN/60}"); fi
  {
 echo "batch=$BATCH GPU=$HAM_GPU $(date '+%m-%d %H:%M')"
 echo "progress: $DONE / $TOTAL fold done ( run elapsed ${el}, fold ${per}, ${eta})"
 echo "current: $1"
 echo " results: $LAST"
 echo " jobs: $JOBS"
  } > "$ST"
}

echo "===== measurement batch $BATCH GPU=$HAM_GPU start $(date '+%m-%d %H:%M') $(echo $JOBS | wc -w) ($TOTAL fold) ====="
for J in $JOBS; do
  N="${J%%:*}"; S="${J##*:}"
  D="outputs/localize"; [ "$S" != "42" ] && D="outputs/localize/seed$S"
  for K in 0 1 2 3 4; do
    if [ -f "$D/loc_${N}_f${K}.npz" ]; then DONE=$((DONE+1)); continue; fi
    if pgrep -f -- "--fold $K --seed $S --name $N\$" > /dev/null || \
       { [ "$S" = "42" ] && pgrep -f -- "--fold $K --ltv-mode $N\$" > /dev/null; }; then
 echo "--- [$N s$S f$K] run , "; continue
    fi
 status "$N (seed $S) fold $K"
    LG="logs/localize/x_${N}_s${S}_f${K}"
    if [ ! -f "$D/readout_${N}_f${K}.pth" ]; then
      python -m experiments.localize_x --stage train --fold $K --seed $S --name $N > "${LG}_train.log" 2>&1 \
 || { echo "--- [$N s$S f$K] training failed"; tail -5 "${LG}_train.log"; continue; }
    fi
    python -m experiments.localize_x --stage eval --fold $K --seed $S --name $N > "${LG}_eval.log" 2>&1 \
 || { echo "--- [$N s$S f$K] measurement failed"; tail -5 "${LG}_eval.log"; continue; }
    DONE=$((DONE+1)); RUN=$((RUN+1))
    LAST="$N s$S f$K: $(grep -o '^  ltv .*' "${LG}_eval.log" | head -1 | tr -s ' ')"
    echo "--- [$N s$S f$K] $(date '+%H:%M')  $LAST"
    python -m experiments.loc_table > logs/batch_summary/loc_table.txt 2>/dev/null
 status " fold "
  done
done
python -m experiments.loc_table > logs/batch_summary/loc_table.txt 2>/dev/null
status "done"
echo " end $(date '+%m-%d %H:%M')"
