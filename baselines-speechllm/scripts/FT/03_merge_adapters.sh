#!/bin/bash
# Run from baselines-speechllm/:  bash scripts/FT/03_merge_adapters.sh

cd "$(dirname "$0")/../.."  # run from baselines-speechllm/

BASE_MODEL="Qwen/Qwen2.5-Omni-7B"

MERGES=(
    "saves/qwen2.5-omni-7b/lora/text:saves/qwen2.5-omni-7b/merged/iwslt26_text"
    "saves/qwen2.5-omni-7b/lora/audio:saves/qwen2.5-omni-7b/merged/iwslt26_audio"
    "saves/qwen2.5-omni-7b/lora/textaudio:saves/qwen2.5-omni-7b/merged/iwslt26_textaudio"
)

for ENTRY in "${MERGES[@]}"; do
    LORA_PATH="${ENTRY%%:*}"
    SAVE_PATH="${ENTRY##*:}"
    echo "Merging LoRA: $LORA_PATH -> $SAVE_PATH"
    python LlamaFactory/scripts/qwen_omni_merge.py merge_lora \
        --model_path="$BASE_MODEL" \
        --lora_path="$LORA_PATH" \
        --save_path="$SAVE_PATH"
    echo "Done: $SAVE_PATH"
done