"""
Compare original vs. shuffled-source scores for all models.

Prints a table showing mean score and the drop when the source is shuffled.
A large drop means the model actually attends to the source content.

Usage (run from repo root):
    python evaluation/compare_shuffled_src.py --split dev_asr
"""
import argparse
import json
import os
import statistics

import numpy as np


LANG_PAIRS = ["en-de", "en-zh"]

MODEL_DIRS = {
    # SpeechCOMET
    "harris-FT-sonar":                      "trained_models/harris-FT-sonar",
    "harris-20ep":                           "trained_models/harris-20ep",
    "lewis-10ep":                            "trained_models/lewis-10ep",
    "skye-20ep":                             "trained_models/skye-20ep",
    "shetland-20ep":                         "trained_models/shetland-20ep",
    "mull-avg-20ep":                         "trained_models/mull-avg-20ep",
    "mull-attn-10ep":                        "trained_models/mull-attn-10ep",
    "orkney-avg-20ep":                       "trained_models/orkney-avg-20ep",
    "orkney-sum-20ep":                       "trained_models/orkney-sum-20ep",
    "orkney-concat-20ep":                    "trained_models/orkney-concat-20ep",
    # QE baselines
    "qe-comet":                              "QE-baselines/results/qe-comet",
    "qe-comet-partial":                      "QE-baselines/results/qe-comet-partial",
    "qe-blaser":                             "QE-baselines/results/qe-blaser",
    "qe-speechqe":                           "QE-baselines/results/qe-speechqe",
    # SpeechLLM
    "Qwen_Qwen2.5-Omni-7B_text":            "speechllm-baselines/results/Qwen_Qwen2.5-Omni-7B",
    "Qwen_Qwen2.5-Omni-7B_audio":           "speechllm-baselines/results/Qwen_Qwen2.5-Omni-7B",
    "Qwen_Qwen2.5-Omni-7B_audiotext":       "speechllm-baselines/results/Qwen_Qwen2.5-Omni-7B",
    "Qwen_Omni-7B-iwslt26-text":            "speechllm-baselines/results/Qwen_Qwen2.5-Omni-7B-iwslt26-text",
    "Qwen_Omni-7B-iwslt26-audio":           "speechllm-baselines/results/Qwen_Qwen2.5-Omni-7B-iwslt26-audio",
    "Qwen_Omni-7B-iwslt26-audiotext":       "speechllm-baselines/results/Qwen_Qwen2.5-Omni-7B-iwslt26-textaudio",
}

# SpeechLLM scores have a modality suffix on the filename; map model key → suffix
SCORE_SUFFIX = {
    "Qwen_Qwen2.5-Omni-7B_text":       "text",
    "Qwen_Qwen2.5-Omni-7B_audio":      "audio",
    "Qwen_Qwen2.5-Omni-7B_audiotext":  "audiotext",
    "Qwen_Omni-7B-iwslt26-text":       "text",
    "Qwen_Omni-7B-iwslt26-audio":      "audio",
    "Qwen_Omni-7B-iwslt26-audiotext":  "audiotext",
}


def load_scores(directory, split, lang_pair, suffix=""):
    fname = f"output_scores_{split}_{lang_pair}"
    if suffix:
        fname += f"_{suffix}"
    fname += ".jsonl"
    path = os.path.join(directory, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev_asr")
    args = parser.parse_args()

    header = f"{'Model':<40} {'lang':>5} {'orig':>7} {'shuf':>7} {'drop':>7}"
    print(header)
    print("-" * len(header))

    for model_key, model_dir in MODEL_DIRS.items():
        if not os.path.isdir(model_dir):
            continue
        suffix = SCORE_SUFFIX.get(model_key, "")
        shuffled_dir = os.path.join(model_dir, "shuffled_src")
        if not os.path.isdir(shuffled_dir):
            continue

        for lp in LANG_PAIRS:
            orig = load_scores(model_dir, args.split, lp, suffix)
            shuf = load_scores(shuffled_dir, args.split, lp, suffix)
            if orig is None or shuf is None:
                continue
            orig_mean = statistics.mean(orig)
            shuf_mean = statistics.mean(shuf)
            drop = orig_mean - shuf_mean
            print(f"{model_key:<40} {lp:>5} {orig_mean:>7.3f} {shuf_mean:>7.3f} {drop:>+7.3f}")


if __name__ == "__main__":
    main()
