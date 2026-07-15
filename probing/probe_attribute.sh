#!/bin/bash
# Probing sweep for a single categorical attribute
# across COMETKiwi-DA, COMETKiwi-RoBERTa variants, SpeechCOMET variants, and
# Qwen2.5-Omni (frozen and fine-tuned). Each run is skipped automatically if
# its output file already exists.
#
# Replaces the near-duplicate probe_emotion.sh / probe_intonation.sh, which
# differed only in attribute name and data file prefix.
#
# Usage:
#   bash probe_attribute.sh <attribute> <data_prefix>
#
#   attribute:   JSONL column name to probe (e.g. emotion, intonation)
#   data_prefix: prefix of <prefix>_train.jsonl / <prefix>_dev.jsonl
#
# Examples:
#   bash probe_attribute.sh emotion contraprost

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash probe_attribute.sh <attribute> <data_prefix>" >&2
    exit 1
fi

ATTRIBUTE=$1
DATA_PREFIX=$2
TRAIN="${DATA_PREFIX}_train.jsonl"
DEV="${DATA_PREFIX}_dev.jsonl"
SYS_PROMPT="You are an evaluator. Given the source and/or audio and a translation, respond with only a single float score between 0 and 1 indicating translation quality. Output nothing else."

mkdir -p results embeddings_cache

# --- COMETKiwi-DA (text, default model) ---
for SEED in 0 42 420; do
OUT="results/probing_${ATTRIBUTE}_cometkiwi-da_seed$SEED.txt"
[ -f "$OUT" ] && continue
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRIBUTE \
    --embeddings-dir embeddings_cache \
    --seed $SEED > "$OUT"
done

# --- COMETKiwi RoBERTa variants (text) ---
for MODEL in "COMETKiwi-RoBERTa-IWSLT" "COMETKiwi-RoBERTa-WMT"; do
for SEED in 0 42 420; do
OUT="results/probing_${ATTRIBUTE}_${MODEL}_seed$SEED.txt"
[ -f "$OUT" ] && continue
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRIBUTE \
    --model maikezu/$MODEL \
    --model-type speechcomet \
    --embeddings-dir embeddings_cache \
    --seed $SEED > "$OUT"
done
done

# --- SpeechCOMET variants ---
for MODEL in "SpeechCOMET-SONAR" "SpeechCOMET-Whisper" "SpeechCOMET-textaudio" \
             "frozen-ablation-SpeechCOMET-SONAR" "frozen-ablation-SpeechCOMET-Whisper" \
             "frozen-ablation-SpeechCOMET-textaudio"; do
for SEED in 0 42 420; do
OUT="results/probing_${ATTRIBUTE}_${MODEL}_seed$SEED.txt"
[ -f "$OUT" ] && continue
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRIBUTE \
    --model maikezu/$MODEL \
    --model-type speechcomet \
    --src-audio-field src_audio \
    --embeddings-dir embeddings_cache \
    --seed $SEED > "$OUT"
done
done

# --- SpeechLLM frozen (no adapter) — audio, text, audiotext ---
for MODALITY in "audio" "text" "audiotext"; do
for SEED in 0 42 420; do
OUT="results/probing_${ATTRIBUTE}_qwen_${MODALITY}_seed$SEED.txt"
[ -f "$OUT" ] && continue
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRIBUTE \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-modality $MODALITY \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir embeddings_cache \
    --epochs 50 \
    --lr 3e-4 \
    --seed $SEED > "$OUT"
done
done

# --- SpeechLLM fine-tuned — audio (Speech-FT), text (Text-FT), audiotext (SpTxt-FT) ---
for SEED in 0 42 420; do
OUT="results/probing_${ATTRIBUTE}_qwen_ft_audio_seed$SEED.txt"
[ -f "$OUT" ] && continue
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRIBUTE \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-adapter maikezu/main-SpeechLLM-Speech-FT \
    --qwen-modality audio \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir embeddings_cache \
    --epochs 50 \
    --lr 3e-4 \
    --seed $SEED > "$OUT"
done

for SEED in 0 42 420; do
OUT="results/probing_${ATTRIBUTE}_qwen_ft_text_seed$SEED.txt"
[ -f "$OUT" ] && continue
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRIBUTE \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-adapter maikezu/main-SpeechLLM-Text-FT \
    --qwen-modality text \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir embeddings_cache \
    --epochs 50 \
    --lr 3e-4 \
    --seed $SEED > "$OUT"
done

for SEED in 0 42 420; do
OUT="results/probing_${ATTRIBUTE}_qwen_ft_audiotext_seed$SEED.txt"
[ -f "$OUT" ] && continue
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRIBUTE \
    --model Qwen/Qwen2.5-Omni-7B \
    --model-type qwen \
    --src-audio-field src_audio \
    --qwen-adapter maikezu/main-SpeechLLM-SpTxt-FT \
    --qwen-modality audiotext \
    --system-prompt "$SYS_PROMPT" \
    --embeddings-dir embeddings_cache \
    --epochs 50 \
    --lr 3e-4 \
    --seed $SEED > "$OUT"
done
