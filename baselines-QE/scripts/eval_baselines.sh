#!/bin/bash
# Evaluate all QE baselines on IWSLT dev, MuST-SHE, and ContraProST.

MUSTSHE_DIR="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"
CONTRAPROST_DIR="data/contraProST"
SPEECHQE_MODEL="h-j-han/SpeechQE-TowerInstruct-7B-en2de"

for METHOD in asr_comet asr_comet_partial blaser speechqe; do
    echo ""
    echo "=== $METHOD ==="

    SPEECHQE_ARGS=""
    if [ "$METHOD" = "speechqe" ]; then
        SPEECHQE_ARGS="--speechqe-model-de $SPEECHQE_MODEL --speechqe-model-zh $SPEECHQE_MODEL"
    fi

    python baselines-QE/baseline_eval.py --method "$METHOD" --task dev $SPEECHQE_ARGS

    python baselines-QE/baseline_eval.py --method "$METHOD" --task mustshe \
        --mustshe-dir "$MUSTSHE_DIR" $SPEECHQE_ARGS

    python baselines-QE/baseline_eval.py --method "$METHOD" --task contraprost \
        --contraprost-dir "$CONTRAPROST_DIR" $SPEECHQE_ARGS
done
