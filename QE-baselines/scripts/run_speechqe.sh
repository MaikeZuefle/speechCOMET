#!/bin/bash
# Run all evaluations for the SpeechQE baseline.
# Run from repo root: bash QE-baselines/scripts/run_speechqe.sh

METHOD=speechqe
MUSTSHE_DIR="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"
CONTRAPROST_DIR="data/contraProST"

MODEL_DE="h-j-han/SpeechQE-TowerInstruct-7B-en2de"
# en-zh is zero-shot: same model, no zh-specific training
MODEL_ZH="h-j-han/SpeechQE-TowerInstruct-7B-en2de"

# echo "=== $METHOD: dev ==="
# python QE-baselines/run_eval.py --method $METHOD --task dev_asr \
#     --speechqe-model-de "$MODEL_DE" \
#     --speechqe-model-zh "$MODEL_ZH"

echo "=== $METHOD: MuST-SHE ==="
python QE-baselines/run_eval.py --method $METHOD --task mustshe \
    --mustshe-dir "$MUSTSHE_DIR" \
    --speechqe-model-de "$MODEL_DE" \
    --speechqe-model-zh "$MODEL_ZH"

echo "=== $METHOD: ContraProST ==="
python QE-baselines/run_eval.py --method $METHOD --task contraprost \
    --contraprost-dir "$CONTRAPROST_DIR" \
    --speechqe-model-de "$MODEL_DE" \
    --speechqe-model-zh "$MODEL_ZH"
