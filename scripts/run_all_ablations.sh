#!/usr/bin/env bash
# Seed-42 runs of the ablation settings (single-stage, hard and soft routing, TEME, LTV,
# both modules, simple head). These produce the seed-42 column of the multi-seed tables.
#   bash scripts/run_all_ablations.sh
set -u
cd "$(dirname "$0")/.."

WHICH="${*:-t9 t3 t10}"
GPU="${HAM_GPU:-3}"
WD="${HAM_WEIGHT_DECAY:-0.05}"
AUG="${HAM_AUGMENT:-code}"
SLOTS="${HAM_N_SLOTS:-4}"
mkdir -p logs outputs runs

. ./env.sh > /dev/null
export HAM_GPU="$GPU" HAM_WEIGHT_DECAY="$WD" HAM_AUGMENT="$AUG" HAM_N_SLOTS="$SLOTS"

S1="outputs/stage1_mela_ltv_softrouting_multiscale_7class.pth"
[ -f "$S1" ] || { echo "Stage 1 checkpoint missing: $S1"; exit 1; }

echo "======================================================================"
echo " Ablation full run"
echo " configurations: $WHICH"
echo " settings: wd=$WD aug=$AUG n_slots=$SLOTS GPU=$GPU"
echo " Stage 1 checkpoint: $(stat -c '%y' "$S1" | cut -d. -f1)"
echo "======================================================================"

variant () {
    local TAG="$1" TEME="$2" LTV="$3" SIMPLE="${4:-0}"
    echo
    echo "--- [$TAG]  TEME=$TEME LTV=$LTV simple=$SIMPLE -------------------"
    for BR in mel nonmel; do
        echo "    stage2 $BR ..."
        python -m ablation.run --stage "$BR" --tag "$TAG" \
            --teme "$TEME" --ltv "$LTV" --simple-head "$SIMPLE" \
            > "logs/abl_${TAG}_${BR}.log" 2>&1 \
 || { echo " failed:"; tail -30 "logs/abl_${TAG}_${BR}.log" | sed 's/^/ /'; exit 1; }
        grep -E "best val AUC" "logs/abl_${TAG}_${BR}.log" | tail -1 | sed 's/^/      /'
    done
    echo "    routing (soft) ..."
    python -m ablation.run --stage routing --tag "$TAG" \
        --teme "$TEME" --ltv "$LTV" --simple-head "$SIMPLE" --routing soft \
        > "logs/abl_${TAG}_soft.log" 2>&1 \
 || { echo " failed:"; tail -30 "logs/abl_${TAG}_soft.log" | sed 's/^/ /'; exit 1; }
    grep -E "^   (Accuracy|Sensitivity|Specificity|AUC|F1|PPV|NPV)" "logs/abl_${TAG}_soft.log" \
        | head -7 | sed 's/^/      /'
    mkdir -p "runs/abl_${TAG}"
    cp outputs/abl_${TAG}_*.npz outputs/abl_${TAG}_*.csv "runs/abl_${TAG}"/ 2>/dev/null
    cp outputs/abl_${TAG}_stage2_*.pth "runs/abl_${TAG}"/ 2>/dev/null
}

if echo "$WHICH" | grep -qw t9; then
    echo; echo "##################### Table 9  TEME x LTV #####################"
    variant baseline 0 0
    variant ltv      0 1
    variant teme     1 0
    variant full     1 1
fi

if echo "$WHICH" | grep -qw t3; then
    echo; echo "##################### Table 3  routing #####################"
 echo; echo "--- [1-stage] 7-class -----------------------------"
    python -m ablation.onestage > logs/abl_onestage.log 2>&1 \
 || { echo " failed:"; tail -30 logs/abl_onestage.log | sed 's/^/ /'; exit 1; }
    grep -E "^   (Accuracy|Sensitivity|Specificity|AUC|F1|PPV|NPV)" logs/abl_onestage.log \
        | head -7 | sed 's/^/      /'
    mkdir -p runs/abl_onestage && cp outputs/onestage_7class_* runs/abl_onestage/ 2>/dev/null

    if [ -f outputs/abl_baseline_stage2_mel.pth ]; then
        echo; echo "--- [2-stage hard routing] ---------------------------------"
        python -m ablation.run --stage routing --tag baseline --teme 0 --ltv 0 --routing hard \
            > logs/abl_baseline_hard.log 2>&1 \
 || { echo " failed:"; tail -30 logs/abl_baseline_hard.log | sed 's/^/ /'; exit 1; }
        grep -E "^   (Accuracy|Sensitivity|Specificity|AUC|F1|PPV|NPV)" logs/abl_baseline_hard.log \
            | head -7 | sed 's/^/      /'
        cp outputs/abl_baseline_hard_* runs/abl_baseline/ 2>/dev/null
    else
 echo " [] hard routing t9 baseline results ."
    fi
 echo " [] 2-stage soft routing = Table 9 baseline "
fi

if echo "$WHICH" | grep -qw t10; then
    echo; echo "##################### Table 10  simple head #####################"
    variant simplehead 0 0 1
fi

echo
echo "======================================================================"
echo " Summary"
echo "======================================================================"
printf "%-16s%9s%9s%9s%9s%9s%9s%9s\n" "settings" "Acc" "Sens" "Spec" "AUC" "F1" "PPV" "NPV"
row () {
    local NAME="$1" LOG="$2"
    [ -f "$LOG" ] || return
    printf "%-16s" "$NAME"
    for M in Accuracy Sensitivity Specificity AUC F1 PPV NPV; do
        printf "%9s" "$(grep -E "^   $M " "$LOG" | head -1 | awk -F'|' '{gsub(/ /,"",$2); print $2}')"
    done
    echo
}
row 1-stage       logs/abl_onestage.log
row hard-routing  logs/abl_baseline_hard.log
row baseline      logs/abl_baseline_soft.log
row +LTV          logs/abl_ltv_soft.log
row +TEME         logs/abl_teme_soft.log
row +TEME+LTV     logs/abl_full_soft.log
row simple-head   logs/abl_simplehead_soft.log
