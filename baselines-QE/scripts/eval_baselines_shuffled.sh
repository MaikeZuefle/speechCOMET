#!/bin/bash
# Evaluate QE baselines with shuffled source inputs.

SPEECHQE_MODEL="h-j-han/SpeechQE-TowerInstruct-7B-en2de"

for METHOD in asr_comet asr_comet_partial blaser speechqe; do
    echo ""
    echo "=== $METHOD (shuffled src) ==="

    SPEECHQE_ARGS=""
    if [ "$METHOD" = "speechqe" ]; then
        SPEECHQE_ARGS="--speechqe-model-de $SPEECHQE_MODEL"
    fi

    python QE-baselines/baseline_eval_shuffled.py \
        --method "$METHOD" \
        --split dev \
        $SPEECHQE_ARGS
done
