#!/bin/bash
cd "$(dirname "$0")/../src" || exit 1

MODEL_NAME="Qwen/Qwen2.5-Omni-7B"
CONTRAPROST_DIR="../../data/contraProST"
OUTPUT_NAME="speechllm-baselines/results/Qwen_Qwen2.5-Omni-7B"

python generate_qwen_omni.py \
    --model-name "$MODEL_NAME" \
    --contraprost-dir "$CONTRAPROST_DIR" \
    --output-name "$OUTPUT_NAME" \
    --prompt contraprost_prosody
