#!/bin/bash
HF=false  # true if it's a model from HF
MODEL_NAME=speech_audiotext_sum # or shetland, skye ...
MODALITY=audiotext



# define paths
CHECKPOINT_FOLDER=default # ignore if it's HF model
HF_USER=maikezu # ignore if it's local model
# --------------
if [ "$HF" = false ]; then
    OUTPUT_DIR=$CHECKPOINT_FOLDER/$MODEL_NAME
    MODEL_ARG="--model-folder $OUTPUT_DIR"
else
    OUTPUT_DIR=${HF_USER}_${MODEL_NAME}
    MODEL_ARG="--hf-model $HF_USER/$MODEL_NAME"
fi

# generation
python eval_scripts/run_inf.py \
  $MODEL_ARG \
  --dataset maikezu/iwslt2026-metrics-shared-train-dev \
  --modality $MODALITY

# evaluation — run once per lang pair
cd eval_scripts/iwslt26-metrics/
for scores_file in ../../$OUTPUT_DIR/output_scores_*.jsonl; do
    # extract lang pair from filename, e.g. output_scores_en-de.jsonl -> en-de
    lang_pair=$(basename "$scores_file" .jsonl | sed 's/output_scores_//')
    input_file="../../$OUTPUT_DIR/input_data_${lang_pair}.jsonl"

    echo "Evaluating $lang_pair ..."
    python evaluation/__main__.py -i "$input_file" -m "$scores_file"
done