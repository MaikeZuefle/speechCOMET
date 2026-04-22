#!/bin/bash
# Evaluate QE baselines (COMET, COMET-partial, BLASER) on all tasks.
# Run from repo root: bash QE-baselines/scripts/run_qe_baselines.sh

MUSTSHE_DIR="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"
CONTRAPROST_DIR="data/contraProST"
WER_CSV="data/wer_analysis/wer_dev_asr.csv"

for METHOD in asr_comet asr_comet_partial; do
    echo ""
    echo "======================================================"
    echo "QE baseline: $METHOD"
    echo "======================================================"

    echo "--- dev ---"
    python QE-baselines/run_eval.py --method "$METHOD" --task dev

    echo "--- dev_asr ---"
    python QE-baselines/run_eval.py --method "$METHOD" --task dev_asr --wer-csv "$WER_CSV"

    echo "--- MuST-SHE ---"
    python QE-baselines/run_eval.py --method "$METHOD" --task mustshe \
        --mustshe-dir "$MUSTSHE_DIR"

    echo "--- ContraProST ---"
    python QE-baselines/run_eval.py --method "$METHOD" --task contraprost \
        --contraprost-dir "$CONTRAPROST_DIR"
done

echo ""
echo "======================================================"
echo "QE baseline: blaser"
echo "======================================================"

echo "--- dev ---"
python QE-baselines/run_eval.py --method blaser --task dev

# dev_asr omitted: BLASER uses audio input, scores are identical to dev

echo "--- MuST-SHE ---"
python QE-baselines/run_eval.py --method blaser --task mustshe \
    --mustshe-dir "$MUSTSHE_DIR"

echo "--- ContraProST ---"
python QE-baselines/run_eval.py --method blaser --task contraprost \
    --contraprost-dir "$CONTRAPROST_DIR"

echo ""
echo "All QE baseline evaluations done."
