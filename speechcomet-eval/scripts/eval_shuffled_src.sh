#!/bin/bash
# Evaluate SpeechCOMET models with shuffled source inputs.
# For audiotext models, also runs audio-only and text-only shuffle ablations automatically.

CHECKPOINT_FOLDER=trained_models

# format: model_name:modality:split
MODELS=(
    # audio-sonar:audio:dev
    # audio-whisper:audio:dev
    # text:text:dev
    # audiotext-sonar:audiotext:dev
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
        python evaluation/iwslt_eval_shuffled.py \
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
    if [ "$modality" = "audiotext" ]; then
        run_shuffled "$model" "$modality" "$split" audio
        run_shuffled "$model" "$modality" "$split" text
    fi
done
