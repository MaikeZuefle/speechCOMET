#!/bin/bash

HF=false  # true if it's a model from HF
MODEL_NAME=speech_audio_from_text_checkpoint # or shetland, skye ...

#######################################
MODALITY=audio
CHECKPOINT_FOLDER=/net/tscratch/people/plgzuefle/iwslt2026/speechCOMET/default # ignore if it's HF model
HF_USER=maikezu # ignore if it's local model
# --------------

if [ "$HF" = false ]; then
    OUTPUT_DIR=$CHECKPOINT_FOLDER/$MODEL_NAME
    MODEL_ARG="--model-folder $OUTPUT_DIR"
else
    OUTPUT_DIR=${HF_USER}_${MODEL_NAME}
    MODEL_ARG="--hf-model $HF_USER/$MODEL_NAME"
fi

python eval_scripts/run_inf.py \
  $MODEL_ARG \
  --dataset maikezu/iwslt2026-metrics-shared-train-dev \
  --modality $MODALITY

cd eval_scripts/iwslt26-metrics
python evaluation.py -i ../../$OUTPUT_DIR/input_data.jsonl -m ../../$OUTPUT_DIR/output_scores.jsonl