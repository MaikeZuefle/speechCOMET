#!/bin/bash

MODEL_NAMES=(
    # mull-attn-10ep
    # mull-attn-lora-10ep
    # mull-avg-20ep
    # mull-avg-lora-10ep
    # shetland
    # skye
    # shetland-20ep
    harris-20ep-continue
)
MODALITY=audio
SPLIT=dev_asr  # dev or dev_asr
HF=false  # true if all models are from HF


# define paths
CHECKPOINT_FOLDER=trained_models # ignore if HF models
HF_USER=maikezu # ignore if local models


# --------------
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    if [ "$HF" = false ]; then
        OUTPUT_DIR=$CHECKPOINT_FOLDER/$MODEL_NAME
        MODEL_ARG="--model-folder $OUTPUT_DIR"
    else
        OUTPUT_DIR=${HF_USER}_${MODEL_NAME}
        MODEL_ARG="--hf-model $HF_USER/$MODEL_NAME"
    fi

    echo "=== Running inference for $MODEL_NAME ==="

    # generation
    python evaluation/run_inf.py \
      $MODEL_ARG \
      --dataset maikezu/scottish-metrics \
      --modality $MODALITY \
      --split $SPLIT

    # evaluation — run once per lang pair
    cd evaluation/iwslt26-metrics/
    for scores_file in ../../$OUTPUT_DIR/output_scores_${SPLIT}_*.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_${SPLIT}_//")
        input_file="../../$OUTPUT_DIR/input_data_${SPLIT}_${lang_pair}.jsonl"
        echo "Evaluating $lang_pair ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file"
    done
    cd ../..

    # WER correlation analysis (only meaningful for dev_asr)
    if [ "$SPLIT" = "dev_asr" ]; then
        python evaluation/04-wer_correlation_analysis.py \
            --model-dir $OUTPUT_DIR \
            --split $SPLIT
    fi

    # MuST-SHE pairwise accuracy
    python evaluation/mustshe_eval.py \
        --mustshe-dir data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv \
        --modality $MODALITY \
        --batch-size 32 \
        $MODEL_ARG
done