MODEL_NAME=speech_audiotext_concat
MODALITY=audiotext

CHECKPOINT_FOLDER=/net/tscratch/people/plgzuefle/iwslt2026/speechCOMET/default
MODEL_DIR=$CHECKPOINT_FOLDER/$MODEL_NAME

python eval_scripts/run_inf.py \
  --model-folder $MODEL_DIR \
  --dataset maikezu/iwslt2026-metrics-shared-train-dev \
  --modality $MODALITY

cd eval_scripts/iwslt26-metrics

python evaluation -i $MODEL_DIR/input_data.jsonl -m $MODEL_DIR/output_scores.jsonl