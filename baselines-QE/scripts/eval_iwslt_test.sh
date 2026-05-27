#!/bin/bash
# Evaluate text-based QE baselines on IWSLT test set.

DATASET=maikezu/iwslt2026-metrics-shared-test

for METHOD in asr_comet asr_comet_partial; do
    echo "=== $METHOD ==="
    python baselines-QE/baseline_eval.py --method "$METHOD" --task dev \
        --split test --dataset "$DATASET" --no-correlation
done
