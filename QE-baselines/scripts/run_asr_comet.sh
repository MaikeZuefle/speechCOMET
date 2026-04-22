#!/bin/bash
# Run all evaluations for the ASR-COMET baseline.
# Run from repo root: bash QE-baselines/scripts/run_asr_comet.sh

METHOD=asr_comet
MUSTSHE_DIR="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv"
CONTRAPROST_DIR="data/contraProST"

echo "=== $METHOD: dev ==="
python QE-baselines/run_eval.py --method $METHOD --task dev

echo "=== $METHOD: dev_asr ==="
python QE-baselines/run_eval.py --method $METHOD --task dev_asr

echo "=== $METHOD: MuST-SHE ==="
python QE-baselines/run_eval.py --method $METHOD --task mustshe --mustshe-dir "$MUSTSHE_DIR"

echo "=== $METHOD: ContraProST ==="
python QE-baselines/run_eval.py --method $METHOD --task contraprost --contraprost-dir "$CONTRAPROST_DIR"
