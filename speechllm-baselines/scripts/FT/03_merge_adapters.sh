#!/bin/bash
# Run from speechllm-baselines/:  bash scripts/FT/03_merge_adapters.sh

cd "$(dirname "$0")/../.."  # run from speechllm-baselines/

BASE_MODEL="Qwen/Qwen2.5-Omni-7B"

MERGES=(
    # already merged
    # "saves/qwen2.5-omni-7b/lora/context_word:saves/qwen2.5-omni-7b/merged/context_word"
    # "saves/qwen2.5-omni-7b/lora/target_word:saves/qwen2.5-omni-7b/merged/target_word"
    # "saves/qwen2.5-omni-7b/lora/both:saves/qwen2.5-omni-7b/merged/both"
    # "saves/qwen2.5-omni-7b/lora/fleurs_context_1:saves/qwen2.5-omni-7b/merged/fleurs_context_1"
    # "saves/qwen2.5-omni-7b/lora/fleurs_context_5:saves/qwen2.5-omni-7b/merged/fleurs_context_5"
    # "saves/qwen2.5-omni-7b/lora/fleurs_context_10:saves/qwen2.5-omni-7b/merged/fleurs_context_10"
    # "saves/qwen2.5-omni-7b/lora/fleurs_context_mixed:saves/qwen2.5-omni-7b/merged/fleurs_context_mixed"
    # combined
    "saves/qwen2.5-omni-7b/lora/context_word_fleurs_mixed:saves/qwen2.5-omni-7b/merged/context_word_fleurs_mixed"
    "saves/qwen2.5-omni-7b/lora/target_word_fleurs_mixed:saves/qwen2.5-omni-7b/merged/target_word_fleurs_mixed"
    "saves/qwen2.5-omni-7b/lora/both_fleurs_mixed:saves/qwen2.5-omni-7b/merged/both_fleurs_mixed"
    # iwslt26 metrics fine-tuning
    # "saves/qwen2.5-omni-7b/lora/text:saves/qwen2.5-omni-7b/merged/iwslt26_text"
    # "saves/qwen2.5-omni-7b/lora/audio:saves/qwen2.5-omni-7b/merged/iwslt26_audio"
    # "saves/qwen2.5-omni-7b/lora/textaudio:saves/qwen2.5-omni-7b/merged/iwslt26_textaudio"
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