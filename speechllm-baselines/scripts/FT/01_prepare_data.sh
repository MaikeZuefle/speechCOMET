#!/bin/bash
# Prepare LlamaFactory training data from maikezu/iwslt2026-metrics-shared-train-dev.
# Run from speechllm-baselines/:  bash scripts/FT/01_prepare_data.sh
# For a quick test run:           bash scripts/FT/01_prepare_data.sh --limit 50

cd "$(dirname "$0")/../.."  # run from speechllm-baselines/

python src/prepare_llama_factory_data.py "$@"
