#!/bin/bash
# Sequential, resumable run of every model configuration that has been probed
# against MuST-SHE gender labels. Each block is skipped automatically if its
# output file already exists, so re-running this script only executes
# new/missing configs. Runs prepare step first if JSONL splits don't exist.
#
# Usage:
#   bash probe_mustshe_all.sh [speech_model]
#
# speech_model: HF repo suffix under maikezu/ (default: harris)
# Results written to results/probing_mustshe_*.txt

set -euo pipefail

MODEL=${1:-harris}
TRAIN=mustshe_train.jsonl
DEV=mustshe_dev.jsonl
EMBED_CACHE=embeddings_cache
SYS_PROMPT="You are an evaluator. Given the source and/or audio and a translation, respond with only a single float score between 0 and 1 indicating translation quality. Output nothing else."
mkdir -p results "$EMBED_CACHE"

# Prepare data splits if not already present
if [[ ! -f "$TRAIN" || ! -f "$DEV" ]]; then
    echo "=== Preparing MuST-SHE probing splits ==="
    python prepare_mustshe_probe.py
fi

# --- SONAR audio probe (speechCOMET) ---
for seed in 0 42 420; do
OUT="results/probing_mustshe_${MODEL}_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SONAR audio encoder: maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes gender \
    --model "maikezu/$MODEL" \
    --model-type speechcomet \
    --src-audio-field src_audio \
    --embeddings-dir "$EMBED_CACHE" \
    --epochs 30 \
    --seed $seed > "$OUT"
echo "Written: $OUT"
done

# --- Text encoder control (cometkiwi-da on English SRC text) ---
for seed in 0 42 420; do
OUT="results/probing_mustshe_text_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing text encoder: Unbabel/wmt22-cometkiwi-da (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes gender \
    --model Unbabel/wmt22-cometkiwi-da \
    --model-type comet \
    --embeddings-dir "$EMBED_CACHE" \
    --epochs 30 --seed $seed > "$OUT"
echo "Written: $OUT"
done

# --- SONAR audio probe, repeat (equivalent args to the block above: default
#     --src-audio-field is already 'src_audio'). Kept for parity with history;
#     shares the same output file so it is a no-op once the block above ran. ---
for seed in 0 42 420; do
OUT="results/probing_mustshe_${MODEL}_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SONAR audio encoder: maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes gender \
    --model "maikezu/$MODEL" \
    --model-type speechcomet \
    --embeddings-dir "$EMBED_CACHE" \
    --epochs 30 \
    --seed $seed > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM audio-only probe (evaluator system prompt, last user token) ---
for seed in 0 42 420; do
OUT="results/probing_mustshe_qwen_audio_sys_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (audio, sys-eval): frozen Qwen2.5-Omni-7B (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes gender \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-modality audio \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir "$EMBED_CACHE" \
    --epochs 40 \
    --lr 3e-4 \
    --seed $seed > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM text-only probe (evaluator system prompt, last user token) ---
for seed in 0 42 420; do
OUT="results/probing_mustshe_qwen_text_sys_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (text, sys-eval): frozen Qwen2.5-Omni-7B (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes gender \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-modality text \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir "$EMBED_CACHE" \
    --epochs 40 \
    --lr 3e-4 \
    --seed $seed > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM audio+text probe (evaluator system prompt, last user token) ---
for seed in 0 42 420; do
OUT="results/probing_mustshe_qwen_audiotext_sys_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (audiotext, sys-eval): frozen Qwen2.5-Omni-7B (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes gender \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-modality audiotext \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir "$EMBED_CACHE" \
    --epochs 40 \
    --lr 3e-4 \
    --seed $seed > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM audio-only probe, fine-tuned adapter (evaluator system prompt) ---
for seed in 0 42 420; do
OUT="results/probing_mustshe_qwen_ft_${MODEL}_audio_sys_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (audio, sys-eval): maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes gender \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-adapter "maikezu/$MODEL" \
    --qwen-modality audio \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir "$EMBED_CACHE" \
    --epochs 50 \
    --lr 3e-4 \
    --seed $seed > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM text-only probe, fine-tuned adapter (evaluator system prompt) ---
for seed in 0 42 420; do
OUT="results/probing_mustshe_qwen_ft_${MODEL}_text_sys_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (text, sys-eval): maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes gender \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-adapter "maikezu/$MODEL" \
    --qwen-modality text \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir "$EMBED_CACHE" \
    --epochs 50 \
    --lr 3e-4 \
    --seed $seed > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM audio+text probe, fine-tuned adapter (evaluator system prompt) ---
for seed in 0 42 420; do
OUT="results/probing_mustshe_qwen_ft_${MODEL}_audiotext_sys_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (audiotext, sys-eval): maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes gender \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-adapter "maikezu/$MODEL" \
    --qwen-modality audiotext \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir "$EMBED_CACHE" \
    --epochs 50 \
    --lr 3e-4 \
    --seed $seed > "$OUT"
echo "Written: $OUT"
done

echo "=== Done ==="
