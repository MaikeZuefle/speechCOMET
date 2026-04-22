#!/bin/bash
# Evaluate QE baselines with shuffled source inputs.
# Run from repo root: bash QE-baselines/scripts/run_qe_shuffled_src.sh

SPLIT=dev_asr

SPEECHQE_MODEL_DE="h-j-han/SpeechQE-TowerInstruct-7B-en2de"

for METHOD in   speechqe blaser; do
    echo ""
    echo "======================================================"
    echo "QE baseline (shuffled src): $METHOD"
    echo "======================================================"

    if [ "$METHOD" = "speechqe" ]; then
        SPEECHQE_ARGS="--speechqe-model-de $SPEECHQE_MODEL_DE"
    else
        SPEECHQE_ARGS=""
    fi

    python QE-baselines/run_eval_shuffled_src.py \
        --method "$METHOD" \
        --split "$SPLIT" \
        $SPEECHQE_ARGS
done

echo ""
echo "All QE shuffled-source evaluations done."
