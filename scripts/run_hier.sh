#!/usr/bin/env bash
# Hierarchical design variants (cross-validated).
#   HAM_GPU=0 nohup bash scripts/run_hier.sh > logs/hier.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
_REQ_GPU="${HAM_GPU:-}"
. ./env.sh > /dev/null
export HAM_GPU="${_REQ_GPU:-${HAM_GPU:-3}}"
mkdir -p logs/hier outputs/hier

S1="outputs/stage1_mela_ltv_softrouting_multiscale_7class.pth"
FLAT="outputs/onestage_7class.pth"
[ -f "$S1" ] || { echo "Stage 1 checkpoint missing: $S1"; exit 1; }
[ -f "$FLAT" ] || { echo "1-stage checkpoint missing: $FLAT (python -m ablation.onestage )"; exit 1; }

echo "======================================================================"
echo " GPU=$HAM_GPU start $(date '+%Y-%m-%d %H:%M:%S')"
echo "   Stage 1 checkpoint: $(stat -c '%y' "$S1" | cut -d. -f1)"
echo "   single-stage checkpoint: $(stat -c '%y' "$FLAT" | cut -d. -f1)"
echo "======================================================================"

echo; echo "--- 1-stage val/test prediction (val comparison) ---"
for SP in val test; do
    python -m experiments.hier --stage flatpred --split $SP > "logs/hier/flat_${SP}.log" 2>&1 \
 || { echo " failed:"; tail -20 "logs/hier/flat_${SP}.log"; exit 1; }
 grep "" "logs/hier/flat_${SP}.log" | sed 's/.*\[hier/ [hier/'
done

run_variant () {
    local TAG="$1" MODE="$2" INIT="$3"
    echo; echo "--- [$TAG] mode=$MODE init=$INIT  $(date '+%H:%M:%S') ---"
    for BR in mel nonmel; do
        local LOG="logs/hier/${TAG}_${BR}.log"
        python -m experiments.hier --stage $BR --tag "$TAG" --mode $MODE --init $INIT \
 > "$LOG" 2>&1 || { echo " [$BR] failed:"; tail -25 "$LOG" | sed 's/^/ /'; return 1; }
 grep -E "|best val AUC" "$LOG" | sed 's/^/ /'
    done
    for SP in val test; do
        python -m experiments.hier --stage predict --tag "$TAG" --mode $MODE --split $SP \
 > "logs/hier/${TAG}_pred_${SP}.log" 2>&1 || { echo " [predict $SP] failed"; return 1; }
    done
 echo " done $(date '+%H:%M:%S')"
}

run_variant V0 subset   imagenet
run_variant V1 subset   stage1
run_variant V2 allother imagenet
run_variant V3 allother stage1

echo
python -m experiments.hier_eval --tags V0 V1 V2 V3 2>&1 | tee logs/hier/summary.log
echo
echo " end $(date '+%Y-%m-%d %H:%M:%S')"
