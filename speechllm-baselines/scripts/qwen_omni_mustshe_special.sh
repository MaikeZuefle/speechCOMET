#!/bin/bash
cd "$(dirname "$0")/../src" || exit 1

MODEL_NAME="Qwen/Qwen2.5-Omni-7B"
MUSTSHE_DIR="../../data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"
OUTPUT_NAME="speechllm-baselines/results/Qwen_Qwen2.5-Omni-7B"

python generate_qwen_omni.py \
    --model-name "$MODEL_NAME" \
    --mustshe-dir "$MUSTSHE_DIR" \
    --output-name "$OUTPUT_NAME" \
    --prompt mustshe_gender
