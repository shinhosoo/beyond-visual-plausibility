#!/usr/bin/env bash
# Single-stage classifier on the standard split, seeds 42-46.
#   HAM_GPU=0 nohup bash scripts/run_seeds.sh > logs/seeds.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
. ./env.sh > /dev/null
export HAM_GPU="${HAM_GPU:-3}"
mkdir -p logs/experiments outputs/experiments

echo "start: $(date '+%H:%M:%S')"
for SD in 43 44; do
  for NAME in full focal_a2; do
    if [ "$NAME" = "focal_a2" ]; then
      EXTRA="--mel-loss focal --focal-alpha 0.2"
    else
      EXTRA=""
    fi
    TAG="seed${SD}_${NAME}"
    echo
    echo "--- [$TAG] $(date '+%H:%M:%S') ---"
    for BR in mel nonmel; do
      python -m experiments.run --stage $BR --tag "$TAG" --teme 1 --ltv 1 \
        --seed $SD $EXTRA > "logs/experiments/${TAG}_${BR}.log" 2>&1 \
 || { echo " [$BR] failed:"; tail -20 "logs/experiments/${TAG}_${BR}.log" | sed 's/^/ /'; exit 1; }
      grep "best val AUC" "logs/experiments/${TAG}_${BR}.log" | tail -1 | sed 's/^/    /'
    done
    for SP in val test; do
      python -m experiments.run --stage predict --tag "$TAG" --teme 1 --ltv 1 \
        --split $SP > "logs/experiments/${TAG}_pred_${SP}.log" 2>&1 \
 || { echo " [predict $SP] failed"; exit 1; }
    done
  done
done

echo
echo "===== seed comparison ====="
python -m experiments.tune \
  --tags abl_full seed43_full seed44_full s3_focal_a2 seed43_focal_a2 seed44_focal_a2 \
  --baseline abl_baseline 2>&1 | tail -25
echo "end: $(date '+%H:%M:%S')"
