#!/bin/bash

MODEL_NAMES=(
    mull-attn-10ep
    mull-attn-lora-10ep
    mull-avg-10ep
    mull-avg-lora-10ep
    # shetland
    # skye
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
    python eval_scripts/run_inf.py \
      $MODEL_ARG \
      --dataset maikezu/scottish-metrics \
      --modality $MODALITY \
      --split $SPLIT

    # evaluation — run once per lang pair
    cd eval_scripts/iwslt26-metrics/
    for scores_file in ../../$OUTPUT_DIR/output_scores_${SPLIT}_*.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_${SPLIT}_//")
        input_file="../../$OUTPUT_DIR/input_data_${SPLIT}_${lang_pair}.jsonl"
        echo "Evaluating $lang_pair ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file"
    done
    cd ../..
done