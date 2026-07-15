#!/bin/bash
# Sequential, resumable run of every model configuration that has been probed
# for morphological features using IWSLT 2026 probing data (speechCOMET SONAR
# encoder and Qwen2.5-Omni assistant-token hidden state). Each block is
# skipped automatically if its output file already exists, so re-running this
# script only executes new/missing configs.
#
# NOTE: unlike the historical probe_speech.sh, every fine-tuned-adapter output
# filename here includes ${MODEL}. In probe_speech.sh the actual `>` redirect
# for those blocks omitted ${MODEL} (only the "Written:" echo included it), so
# running the script for a second model would have silently overwritten the
# first model's results file. Since resume-skip treats "file exists" as "this
# config is done", that same omission would make it silently *skip* the
# second model's run instead of overwriting it. Fixed here so multi-model
# sweeps behave correctly.
#
# Usage:
#   bash probe_speech_all.sh [speech_model]
#
# speech_model: HF repo suffix under maikezu/ (default: harris)
# Results written to results/probing_*.txt

set -euo pipefail

MODEL=${1:-harris}
TRAIN=train_probe_morph_wordtype.jsonl
DEV=dev_probe_morph_wordtype.jsonl
ATTRS="src_VerbForm src_Tense src_Mood src_ObjNumber src_SubjNumber src_HasEntity"
SYS_PROMPT="You are an evaluator. Given the source and/or audio and a translation, respond with only a single float score between 0 and 1 indicating translation quality. Output nothing else."

mkdir -p results embeddings_cache

# --- speechCOMET (SONAR) probe ---
for seed in 0 42 420; do
OUT="results/probing_${MODEL}_seed${seed}.txt"
[ -f "$OUT" ] && continue
echo "=== Probing speechCOMET SONAR encoder: maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRS \
    --train-frac 0.1 \
    --seed $seed \
    --src-lang en \
    --model-type speechcomet \
    --model "maikezu/$MODEL" \
    --src-audio-field src_audio \
    > "$OUT"
echo "Written: $OUT"
done

# --- Qwen2.5-Omni (assistant-token, probe-prompt) probe ---
for seed in 42 420; do
OUT="results/probing_qwen_prompt_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing Qwen2.5-Omni: Qwen/Qwen2.5-Omni-7B (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRS \
    --train-frac 0.1 \
    --seed $seed \
    --src-lang en \
    --model-type qwen \
    --model Qwen/Qwen2.5-Omni-7B \
    --lr 3e-4 \
    --src-audio-field src_audio \
    --probe-prompt "$SYS_PROMPT" \
    > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM audio-only probe, fine-tuned adapter (probe-prompt) ---
for seed in 0 42 420; do
OUT="results/probing_qwen_ft_prompt_${MODEL}_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (audio, probe-prompt): maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRS \
    --train-frac 0.1 \
    --seed $seed \
    --src-lang en \
    --model-type qwen \
    --model Qwen/Qwen2.5-Omni-7B \
    --src-audio-field src_audio \
    --qwen-adapter "maikezu/$MODEL" \
    --qwen-modality audio \
    --lr 3e-4 \
    --embeddings-dir embeddings_cache \
    --probe-prompt "$SYS_PROMPT" \
    > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM audio-only probe, frozen (evaluator system prompt, last user token) ---
for seed in 0 42 420; do
OUT="results/probing_qwen_audio_sys_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (audio, sys-eval): frozen Qwen2.5-Omni-7B (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRS \
    --train-frac 0.1 \
    --seed $seed \
    --src-lang en \
    --model-type qwen \
    --model Qwen/Qwen2.5-Omni-7B \
    --src-audio-field src_audio \
    --qwen-modality audio \
    --system-prompt "$SYS_PROMPT" \
    --lr 3e-4 \
    --embeddings-dir embeddings_cache \
    > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM text-only probe, frozen (evaluator system prompt, last user token) ---
for seed in 0 42 420; do
OUT="results/probing_qwen_text_sys_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (text, sys-eval): frozen Qwen2.5-Omni-7B (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRS \
    --train-frac 0.1 \
    --seed $seed \
    --src-lang en \
    --model-type qwen \
    --model Qwen/Qwen2.5-Omni-7B \
    --src-audio-field src_audio \
    --qwen-modality text \
    --system-prompt "$SYS_PROMPT" \
    --lr 3e-4 \
    --embeddings-dir embeddings_cache \
    > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM audio+text probe, frozen (evaluator system prompt, last user token) ---
for seed in 0 42 420; do
OUT="results/probing_qwen_audiotext_sys_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (audiotext, sys-eval): frozen Qwen2.5-Omni-7B (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRS \
    --train-frac 0.1 \
    --seed $seed \
    --src-lang en \
    --model-type qwen \
    --model Qwen/Qwen2.5-Omni-7B \
    --src-audio-field src_audio \
    --qwen-modality audiotext \
    --system-prompt "$SYS_PROMPT" \
    --lr 3e-4 \
    --embeddings-dir embeddings_cache \
    > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM audio-only probe, fine-tuned adapter (evaluator system prompt) ---
for seed in 0 42 420; do
OUT="results/probing_qwen_ft_audio_sys_${MODEL}_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (audio, sys-eval): maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRS \
    --train-frac 0.1 \
    --seed $seed \
    --src-lang en \
    --model-type qwen \
    --model Qwen/Qwen2.5-Omni-7B \
    --src-audio-field src_audio \
    --qwen-adapter "maikezu/$MODEL" \
    --qwen-modality audio \
    --system-prompt "$SYS_PROMPT" \
    --lr 3e-4 \
    --embeddings-dir embeddings_cache \
    > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM text-only probe, fine-tuned adapter (evaluator system prompt) ---
for seed in 0 42 420; do
OUT="results/probing_qwen_ft_text_sys_${MODEL}_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (text, sys-eval): maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRS \
    --train-frac 0.1 \
    --seed $seed \
    --src-lang en \
    --model-type qwen \
    --model Qwen/Qwen2.5-Omni-7B \
    --src-audio-field src_audio \
    --qwen-adapter "maikezu/$MODEL" \
    --qwen-modality text \
    --system-prompt "$SYS_PROMPT" \
    --lr 3e-4 \
    --embeddings-dir embeddings_cache \
    > "$OUT"
echo "Written: $OUT"
done

# --- SpeechLLM audio+text probe, fine-tuned adapter (evaluator system prompt) ---
for seed in 0 42 420; do
OUT="results/probing_qwen_ft_audiotext_sys_${MODEL}_seed$seed.txt"
[ -f "$OUT" ] && continue
echo "=== Probing SpeechLLM (audiotext, sys-eval): maikezu/$MODEL (seed $seed) ==="
python probe_comet.py \
    --train-data "$TRAIN" \
    --dev-data "$DEV" \
    --attributes $ATTRS \
    --train-frac 0.1 \
    --seed $seed \
    --src-lang en \
    --model-type qwen \
    --model Qwen/Qwen2.5-Omni-7B \
    --src-audio-field src_audio \
    --qwen-adapter "maikezu/$MODEL" \
    --qwen-modality audiotext \
    --system-prompt "$SYS_PROMPT" \
    --lr 3e-4 \
    --embeddings-dir embeddings_cache \
    > "$OUT"
echo "Written: $OUT"
done

echo "=== Done ==="
