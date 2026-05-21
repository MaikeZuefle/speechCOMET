#!/bin/bash
# Evaluate QE baselines on ContraProST pairwise accuracy.

CONTRAPROST_DIR="data/contraProST"
SPEECHQE_MODEL="h-j-han/SpeechQE-TowerInstruct-7B-en2de"

for METHOD in asr_comet asr_comet_partial blaser speechqe; do
    echo "=== $METHOD ==="
    SPEECHQE_ARGS=""
    if [ "$METHOD" = "speechqe" ]; then
        SPEECHQE_ARGS="--speechqe-model-de $SPEECHQE_MODEL --speechqe-model-zh $SPEECHQE_MODEL"
    fi
    python baselines-QE/baseline_eval.py --method "$METHOD" --task contraprost \
        --contraprost-dir "$CONTRAPROST_DIR" $SPEECHQE_ARGS
done
