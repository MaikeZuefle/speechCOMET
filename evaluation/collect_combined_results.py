"""
Collect MuST-SHE and ContraProST results into one combined wide table.

Columns (left to right):
  MuST-SHE  — en-es / en-fr / en-it overall, then 1F/1M breakdown per lang
  ContraProST — en-de / en-es / en-ja overall, then prosody category breakdown per lang

Usage:
    python evaluation/collect_combined_results.py
    python evaluation/collect_combined_results.py --output results/combined.csv
"""
import argparse
import glob
import os
import re

import pandas as pd

# ── MuST-SHE settings ─────────────────────────────────────────────────────────
MUSTSHE_LANGS   = ["es", "fr", "it"]
MUSTSHE_CATS    = ["1F", "1M"]          # category values in mustshe_results.csv
MUSTSHE_OVERALL = "ALL"

# ── ContraProST settings ──────────────────────────────────────────────────────
CONTRAPROST_LANG_RENAME = {
    "de": "en-de", "es": "en-es", "ja": "en-ja",
}
CONTRAPROST_CAT_RENAME = {
    "Sentence Stress":   "Stress",
    "Prosodic Breaks":   "Breaks",
    "Intonation":        "Intonation",
    "Emotional Prosody": "Emotion",
    "Pragmatic Prosody": "Politeness",
}
CONTRAPROST_LANGS = ["de", "es", "ja"]
CONTRAPROST_CATS  = list(CONTRAPROST_CAT_RENAME.values())

# ── Model ordering ────────────────────────────────────────────────────────────
# Map subdir prompt suffixes to a canonical row suffix so mustshe and
# contraprost special-prompt runs land in the same row.
PROMPT_SUFFIX_MAP = {
    "mustshe_gender":      "special_prompt",
    "contraprost_prosody": "special_prompt",
}

MODEL_ORDER = [
    # QE baselines
    "qe-comet-partial",
    "qe-comet",
    "qe-speechqe",
    "qe-blaser",
    # SpeechCOMET — text
    "lewis",
    # SpeechCOMET — audio
    "skye",
    "harris",
    "shetland",
    "bute",
    # SpeechCOMET — audiotext
    "orkney",
    "mull",
    # SpeechLLM
    "Qwen",
]

def model_sort_key(name):
    name_lower = name.lower()
    for i, prefix in enumerate(MODEL_ORDER):
        if name_lower.startswith(prefix.lower()):
            return (i, name_lower)
    return (len(MODEL_ORDER), name_lower)

DEFAULT_OUTPUT = "data/contraprost_analysis/combined_results.csv"


# ── MuST-SHE loader ───────────────────────────────────────────────────────────

def load_mustshe(path, model_name):
    df = pd.read_csv(path)
    df["pairwise_acc"] = pd.to_numeric(df["pairwise_acc"], errors="coerce")
    row = {"model": model_name}
    for lang in MUSTSHE_LANGS:
        # overall
        overall = df[(df["lang"] == lang) & (df["category"] == MUSTSHE_OVERALL)]
        row[f"mustshe_{lang}"] = overall["pairwise_acc"].iloc[0] if len(overall) else float("nan")
        # breakdown
        for cat in MUSTSHE_CATS:
            sub = df[(df["lang"] == lang) & (df["category"] == cat)]
            row[f"mustshe_{lang}_{cat}"] = sub["pairwise_acc"].iloc[0] if len(sub) else float("nan")
    return row


def collect_mustshe(search_dirs):
    rows = {}
    for pattern in search_dirs:
        for path in sorted(glob.glob(pattern)):
            parts = path.split(os.sep)
            if "mustshe_results_" in os.path.basename(path):
                modality = os.path.basename(path).replace("mustshe_results_", "").replace(".csv", "")
                subdir = parts[-2]  # e.g. "mustshe" or "mustshe_mustshe_gender"
                raw_suffix = subdir[len("mustshe"):].lstrip("_")  # "" or "mustshe_gender"
                canonical = PROMPT_SUFFIX_MAP.get(raw_suffix, raw_suffix)
                suffix = f"_{canonical}" if canonical else ""
                model_name = parts[-3] + suffix + f" ({modality})"
            else:
                model_name = parts[-2]
            try:
                rows[model_name] = load_mustshe(path, model_name)
            except Exception as e:
                print(f"  WARNING: skipping MuST-SHE {path}: {e}")
    return rows


# ── ContraProST loader ────────────────────────────────────────────────────────

def load_contraprost(path, model_name):
    df = pd.read_csv(path)
    df = df[df["lang"] != "ALL"]
    df["lang"] = df["lang"].map(CONTRAPROST_LANG_RENAME).fillna(df["lang"])
    df["category"] = df["category"].map(CONTRAPROST_CAT_RENAME).fillna(df["category"])
    df["pairwise_acc"] = pd.to_numeric(df["pairwise_acc"], errors="coerce")
    row = {"model": model_name}
    for lang_key in CONTRAPROST_LANGS:
        lang_col = CONTRAPROST_LANG_RENAME[lang_key]
        # overall (category == ALL in original → not in df after filter, so recompute from per-cat mean)
        sub = df[df["lang"] == lang_col]
        row[f"contraprost_{lang_key}"] = sub["pairwise_acc"].mean() if len(sub) else float("nan")
        # breakdown
        for cat in CONTRAPROST_CATS:
            entry = sub[sub["category"] == cat]
            row[f"contraprost_{lang_key}_{cat}"] = entry["pairwise_acc"].iloc[0] if len(entry) else float("nan")
    return row


def collect_contraprost(search_dirs):
    rows = {}
    for pattern in search_dirs:
        for path in sorted(glob.glob(pattern)):
            parts = path.split(os.sep)
            if "contraprost_results_" in os.path.basename(path):
                modality = os.path.basename(path).replace("contraprost_results_", "").replace(".csv", "")
                subdir = parts[-2]
                raw_suffix = subdir[len("contraprost"):].lstrip("_")
                canonical = PROMPT_SUFFIX_MAP.get(raw_suffix, raw_suffix)
                suffix = f"_{canonical}" if canonical else ""
                model_name = parts[-3] + suffix + f" ({modality})"
            else:
                model_name = parts[-2]
            try:
                rows[model_name] = load_contraprost(path, model_name)
            except Exception as e:
                print(f"  WARNING: skipping ContraProST {path}: {e}")
    return rows


# ── Build combined table ──────────────────────────────────────────────────────

def build_table(mustshe_rows, contraprost_rows):
    all_models = sorted(
        set(mustshe_rows.keys()) | set(contraprost_rows.keys()),
        key=model_sort_key,
    )
    records = []
    for model in all_models:
        row = {"model": model}
        if model in mustshe_rows:
            row.update({k: v for k, v in mustshe_rows[model].items() if k != "model"})
        if model in contraprost_rows:
            row.update({k: v for k, v in contraprost_rows[model].items() if k != "model"})
        records.append(row)

    df = pd.DataFrame(records)

    # define column order: overall first, then detailed
    mustshe_overall    = [f"mustshe_{l}"       for l in MUSTSHE_LANGS]
    contraprost_overall= [f"contraprost_{l}"   for l in CONTRAPROST_LANGS]
    mustshe_detail     = [f"mustshe_{l}_{c}"   for l in MUSTSHE_LANGS     for c in MUSTSHE_CATS]
    contraprost_detail = [f"contraprost_{l}_{c}" for l in CONTRAPROST_LANGS for c in CONTRAPROST_CATS]

    ordered_cols = ["model"] + [
        c for c in mustshe_overall + contraprost_overall + mustshe_detail + contraprost_detail
        if c in df.columns
    ]
    df = df[ordered_cols]

    numeric_cols = [c for c in ordered_cols if c != "model"]
    df[numeric_cols] = df[numeric_cols].round(3)
    return df


def refresh_combined_table(output_path=DEFAULT_OUTPUT):
    """Regenerate the combined MuST-SHE + ContraProST table. Called after each eval run."""
    mustshe_search = [
        "trained_models/*/mustshe_results.csv",
        "speechllm-baselines/results/*/mustshe/mustshe_results_*.csv",
        "speechllm-baselines/results/*/mustshe_*/mustshe_results_*.csv",
        "QE-baselines/results/*/mustshe_results.csv",
    ]
    contraprost_search = [
        "trained_models/*/contraprost_results.csv",
        "speechllm-baselines/results/*/contraprost*/contraprost_results_*.csv",
        "QE-baselines/results/*/contraprost_results.csv",
    ]
    mustshe_rows = collect_mustshe(mustshe_search)
    contraprost_rows = collect_contraprost(contraprost_search)
    df = build_table(mustshe_rows, contraprost_rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Updated combined table → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mustshe_search = [
        "trained_models/*/mustshe_results.csv",
        "speechllm-baselines/results/*/mustshe/mustshe_results_*.csv",
        "speechllm-baselines/results/*/mustshe_*/mustshe_results_*.csv",
        "QE-baselines/results/*/mustshe_results.csv",
    ]
    contraprost_search = [
        "trained_models/*/contraprost_results.csv",
        "speechllm-baselines/results/*/contraprost*/contraprost_results_*.csv",
        "QE-baselines/results/*/contraprost_results.csv",
    ]

    print("Collecting MuST-SHE results...")
    mustshe_rows = collect_mustshe(mustshe_search)
    print(f"  Found {len(mustshe_rows)} models")

    print("Collecting ContraProST results...")
    contraprost_rows = collect_contraprost(contraprost_search)
    print(f"  Found {len(contraprost_rows)} models")

    df = build_table(mustshe_rows, contraprost_rows)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
