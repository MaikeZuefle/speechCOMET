#!/bin/bash
# Run from repo root: bash evaluation/scripts/run_inf_textaudio.sh

MODEL_NAMES=(
    # orkney-avg-20ep
    # orkney-sum-20ep
    # orkney-concat-20ep
    # orkney-sum-from-text-ckpt-20ep

    # orkney-sum-from-text-ckpt
    # orkney-sum-from-text-ckpt-FT-sonar
    orkney-sum-from-text-ckpt-BIG
)
MODALITY=audiotext
SPLIT=dev_asr
WER_CSV="data/wer_analysis/wer_dev_asr.csv"
CHECKPOINT_FOLDER=trained_models
MUSTSHE_DIR="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"
CONTRAPROST_DIR="data/contraProST"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    OUTPUT_DIR="$CHECKPOINT_FOLDER/$MODEL_NAME"
    MODEL_ARG="--model-folder $OUTPUT_DIR"

    echo ""
    echo "======================================================"
    echo "=== $MODEL_NAME ==="
    echo "======================================================"

    # inference
    python evaluation/run_inf.py \
        $MODEL_ARG \
        --dataset maikezu/scottish-metrics \
        --modality $MODALITY \
        --split $SPLIT

    # correlation evaluation
    cd evaluation/iwslt26-metrics/
    for scores_file in ../../$OUTPUT_DIR/output_scores_${SPLIT}_*.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_${SPLIT}_//")
        input_file="../../$OUTPUT_DIR/input_data_${SPLIT}_${lang_pair}.jsonl"
        corr_file="../../$OUTPUT_DIR/correlation_${SPLIT}_${lang_pair}.txt"
        echo "Evaluating $lang_pair ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file" | tee "$corr_file"
    done
    cd ../..

    # WER correlation analysis
    python evaluation/wer_correlation_analysis.py \
        --wer-csv "$WER_CSV" \
        --model-dir "$OUTPUT_DIR" \
        --split "$SPLIT"

    # shuffled source analysis
    python evaluation/run_inf_shuffled_src.py \
        $MODEL_ARG \
        --modality $MODALITY \
        --split $SPLIT

    cd evaluation/iwslt26-metrics/
    for scores_file in ../../$OUTPUT_DIR/shuffled_src/output_scores_${SPLIT}_*.jsonl; do
        lang_pair=$(basename "$scores_file" .jsonl | sed "s/output_scores_${SPLIT}_//")
        input_file="../../$OUTPUT_DIR/shuffled_src/input_data_${SPLIT}_${lang_pair}.jsonl"
        echo "Evaluating $lang_pair (shuffled src) ..."
        python evaluation/__main__.py -i "$input_file" -m "$scores_file"
    done
    cd ../..

    # MuST-SHE pairwise accuracy
    python evaluation/mustshe_eval.py \
        --mustshe-dir "$MUSTSHE_DIR" \
        --modality $MODALITY \
        --batch-size 8 \
        $MODEL_ARG

    # ContraProST pairwise accuracy
    python evaluation/contraprost_eval.py \
        --data-dir "$CONTRAPROST_DIR" \
        --modality $MODALITY \
        --batch-size 8 \
        $MODEL_ARG
done
