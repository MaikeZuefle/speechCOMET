#!/bin/bash
# Fine-tune Qwen2.5-Omni-7B (text+audio modality) with LlamaFactory.
# Run from baselines-speechllm/:  bash scripts/FT/02_ft_textaudio.sh

cd "$(dirname "$0")/../.."  # run from baselines-speechllm/

export TORCHAUDIO_USE_BACKEND_DISPATCHER=0

llamafactory-cli train configs/qwen_omni_lora_textaudio.yaml
