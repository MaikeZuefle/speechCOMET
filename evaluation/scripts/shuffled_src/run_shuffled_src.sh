#!/bin/bash
# Evaluate SpeechCOMET models with shuffled source inputs.
# Run from repo root: bash evaluation/scripts/shuffled_src/run_shuffled_src.sh

CHECKPOINT_FOLDER=trained_models

# ── Standard shuffled-src (shuffle everything) ────────────────────────────────
# format: model_name:modality:split
MODELS=(
    # harris-FT-sonar:audio:dev_asr
    # harris-20ep:audio:dev_asr
    # lewis-10ep:text:dev_asr
    # skye-20ep:text:dev_asr
    # mull-attn-10ep:audio:dev_asr
    # mull-attn-from-ckpt:audio:dev_asr
    # orkney-sum-20ep:audiotext:dev
    # orkney-sum-from-text-ckpt-20ep:audiotext:dev
    # orkney-sum-from-text-ckpt-BIG:audiotext:dev
    # orkney-sum-from-text-ckpt-FT-sonar:audiotext:dev
)

run_shuffled() {
    local model_name=$1 modality=$2 split=$3 shuffle_mod=${4:-both}
    local output_dir="$CHECKPOINT_FOLDER/$model_name"
    local subdir
    case "$shuffle_mod" in
        both)  subdir="shuffled_src"   ;;
        audio) subdir="shuffled_audio" ;;
        text)  subdir="shuffled_text"  ;;
    esac

    echo ""
    echo "=== $model_name ($modality, shuffle=$shuffle_mod) ==="

    if ls "$output_dir/$subdir/output_scores_${split}_"*.jsonl &>/dev/null; then
        echo "  Scores already exist in $subdir/, skipping inference."
    else
        python evaluation/run_inf_shuffled_src.py \
            --model-folder "$output_dir" \
            --modality "$modality" \
            --split "$split" \
            --shuffle-modality "$shuffle_mod"
    fi

    cd evaluation/iwslt26-metrics/
    for scores_file in "../../$output_dir/$subdir/output_scores_${split}_"*.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_${split}_//")
        input_file="../../$output_dir/$subdir/input_data_${split}_${lang_pair}.jsonl"
        corr_file="../../$output_dir/$subdir/correlation_${split}_${lang_pair}.txt"
        echo "Evaluating $lang_pair ($shuffle_mod shuffled) ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file" | tee "$corr_file"
    done
    cd ../..
}

for ENTRY in "${MODELS[@]}"; do
    IFS=: read -r model modality split <<< "$ENTRY"
    run_shuffled "$model" "$modality" "$split" both
done

# ── Audio-only / text-only ablation (audiotext models) ────────────────────────
# Shuffle one modality while keeping the other real, to isolate which
# source modality the model conditions on.
AUDIOTEXT_MODELS=(
    orkney-sum-from-text-ckpt-20ep:dev
    orkney-sum-from-text-ckpt-BIG:dev
    orkney-sum-from-text-ckpt-FT-sonar:dev
)

for ENTRY in "${AUDIOTEXT_MODELS[@]}"; do
    IFS=: read -r model split <<< "$ENTRY"
    run_shuffled "$model" audiotext "$split" audio
    run_shuffled "$model" audiotext "$split" text
done
