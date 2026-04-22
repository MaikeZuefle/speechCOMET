"""
Collect contraprost_results.csv from all model folders into a single wide CSV
ready to copy-paste into a spreadsheet.

Finds all trained_models/*/contraprost_results.csv and
speechllm-baselines/*/contraprost/contraprost_results_*.csv

Usage:
    python evaluation/collect_contraprost_results.py
    python evaluation/collect_contraprost_results.py --output results/contraprost_combined.csv
"""
import argparse
import glob
import os

import pandas as pd

LANG_RENAME = {"de": "en-German", "es": "en-Spanish", "ja": "en-Japanese"}
CAT_RENAME = {
    "Sentence Stress":  "Sentence Stress",
    "Prosodic Breaks":  "Prosodic Breaks",
    "Intonation":       "Intonation",
    "Emotional Prosody": "Emotion Prosody",
    "Pragmatic Prosody": "Politeness",
}
CAT_ORDER = ["Sentence Stress", "Prosodic Breaks", "Intonation", "Emotion Prosody", "Politeness", "Overall"]
LANG_ORDER = ["en-German", "en-Spanish", "en-Japanese"]
MODEL_ORDER = ["lewis", "skye", "harris", "shetland", "orkney", "mull"]


def load_results(path, model_name):
    df = pd.read_csv(path)
    df = df[df["lang"] != "ALL"]          # keep per-lang rows only
    df["lang"] = df["lang"].map(LANG_RENAME).fillna(df["lang"])
    df["category"] = df["category"].map(CAT_RENAME).fillna(df["category"])
    # rename ALL category to Overall
    df["category"] = df["category"].replace("ALL", "Overall")
    df["pairwise_acc"] = pd.to_numeric(df["pairwise_acc"], errors="coerce")
    df["model"] = model_name
    return df[["model", "lang", "category", "pairwise_acc"]]


def collect(search_dirs):
    frames = []
    for pattern in search_dirs:
        for path in sorted(glob.glob(pattern)):
            # Derive model name from path
            parts = path.split(os.sep)
            if "contraprost_results_" in os.path.basename(path):
                # speechllm-baselines: …/ModelName/contraprost/contraprost_results_audio.csv
                modality = os.path.basename(path).replace("contraprost_results_", "").replace(".csv", "")
                model_name = parts[-3] + f" ({modality})"
            else:
                # trained_models: …/ModelName/contraprost_results.csv
                model_name = parts[-2]
            try:
                df = load_results(path, model_name)
                frames.append(df)
                print(f"  Loaded {model_name}")
            except Exception as e:
                print(f"  WARNING: skipping {path}: {e}")

    if not frames:
        raise FileNotFoundError("No contraprost_results.csv files found.")
    return pd.concat(frames, ignore_index=True)


def pivot_wide(df):
    pivot = df.pivot_table(
        index="model", columns=["lang", "category"],
        values="pairwise_acc", aggfunc="first"
    )
    # Order columns
    langs = [l for l in LANG_ORDER if l in pivot.columns.get_level_values(0)]
    cats  = [c for c in CAT_ORDER if c in pivot.columns.get_level_values(1)]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([langs, cats]))

    # Sort rows by MODEL_ORDER prefix (lewis < skye < harris < ...)
    def model_sort_key(name):
        name_lower = name.lower()
        for i, prefix in enumerate(MODEL_ORDER):
            if name_lower.startswith(prefix):
                return (i, name_lower)
        return (len(MODEL_ORDER), name_lower)

    pivot = pivot.iloc[sorted(range(len(pivot)), key=lambda i: model_sort_key(pivot.index[i]))]
    return pivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/contraprost_analysis/contraprost_combined.csv")
    args = parser.parse_args()

    search_dirs = [
        "trained_models/*/contraprost_results.csv",
        "speechllm-baselines/*/contraprost/contraprost_results_*.csv",
        "QE-baselines/results/*/contraprost_results.csv",
    ]

    print("Collecting ContraProST results...")
    df = collect(search_dirs)
    pivot = pivot_wide(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pivot.to_csv(args.output)
    print(f"\nSaved to {args.output}")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
