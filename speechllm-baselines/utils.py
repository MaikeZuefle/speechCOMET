"""Shared data-loading utilities for speechllm baselines."""
import glob
import os
import re

import pandas as pd


WAV_DIR_RELATIVE = "../wav"


def load_mustshe_csv_files(tsv_dir: str) -> pd.DataFrame:
    """
    Load all MONOLINGUAL *.[12][FM].csv files from tsv_dir.
    Returns a DataFrame with columns: src, mt, score, src_audio,
    lang, category, audio_path (absolute).
    Raises FileNotFoundError if no files are found.
    Files without a gender-category suffix are skipped (they are duplicates of 1F).
    """
    pattern = os.path.join(tsv_dir, "MONOLINGUAL.*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {pattern}")

    wav_dir = os.path.abspath(os.path.join(tsv_dir, WAV_DIR_RELATIVE))
    frames = []
    for path in csv_files:
        basename = os.path.basename(path)
        m = re.match(r"MONOLINGUAL\.([a-z]+)_v[\d.]+\.tsv\.([12][FM])\.csv", basename)
        if not m:
            print(f"  Skipping {basename} (no category suffix — likely a duplicate)")
            continue
        lang, category = m.group(1), m.group(2)
        df = pd.read_csv(path)
        df["lang"] = lang
        df["category"] = category
        df["audio_path"] = df["src_audio"].apply(
            lambda p: os.path.join(wav_dir, os.path.basename(p))
        )
        frames.append(df)
        print(f"  Loaded {len(df):4d} rows  {basename}  (lang={lang}, category={category})")

    return pd.concat(frames, ignore_index=True)


def check_missing_audio(df: pd.DataFrame, missing_txt: str | None = None) -> pd.DataFrame:
    """
    Abort with a clear error if any audio files are missing.
    Prints a per-language breakdown and optionally refers to a missing.txt.
    Returns the original df unchanged (callers decide what to do).
    """
    missing_mask = ~df["audio_path"].apply(os.path.exists)
    if not missing_mask.any():
        return df

    missing_files = df.loc[missing_mask, "audio_path"].apply(os.path.basename).unique()
    print(f"\n*** WARNING: {missing_mask.sum()} rows ({len(missing_files)} wav files) MISSING ***")
    for lang, grp in df[missing_mask].groupby("lang"):
        n_pairs = grp["audio_path"].nunique()
        print(f"  {lang}: {n_pairs} pairs missing")
    if missing_txt:
        print(f"  Full list: {missing_txt}")
    print()
    return df


def pairwise_accuracy(df: pd.DataFrame, score_col: str = "model_score"):
    """
    For each unique audio_path, compare score_col of the correct row (score=100)
    vs the wrong row (score=0). Returns (accuracy, mean_gap, n_pairs).
    """
    correct = df[df["score"] == 100].set_index("audio_path")[score_col]
    wrong   = df[df["score"] == 0  ].set_index("audio_path")[score_col]
    shared  = correct.index.intersection(wrong.index)
    if len(shared) == 0:
        return float("nan"), float("nan"), 0
    wins = (correct.loc[shared].values > wrong.loc[shared].values).sum()
    gap  = (correct.loc[shared].values - wrong.loc[shared].values).mean()
    return wins / len(shared), gap, len(shared)


def compute_mustshe_results(df: pd.DataFrame, score_col: str = "model_score") -> pd.DataFrame:
    """Compute pairwise accuracy per lang × category, per lang, and overall."""
    rows = []
    for (lang, cat), grp in df.groupby(["lang", "category"]):
        acc, gap, n = pairwise_accuracy(grp, score_col)
        rows.append({"lang": lang, "category": cat, "n_pairs": n,
                     "pairwise_acc": acc, "mean_score_gap": gap})
    for lang, grp in df.groupby("lang"):
        acc, gap, n = pairwise_accuracy(grp, score_col)
        rows.append({"lang": lang, "category": "ALL", "n_pairs": n,
                     "pairwise_acc": acc, "mean_score_gap": gap})
    acc, gap, n = pairwise_accuracy(df, score_col)
    rows.append({"lang": "ALL", "category": "ALL", "n_pairs": n,
                 "pairwise_acc": acc, "mean_score_gap": gap})
    return pd.DataFrame(rows).sort_values(["lang", "category"])


def print_mustshe_pivot(results: pd.DataFrame, modality: str = ""):
    """Print a lang × category pivot of pairwise_acc."""
    lang_results = results[results["lang"] != "ALL"]
    pivot = lang_results.pivot(index="lang", columns="category", values="pairwise_acc")
    if "ALL" in pivot.columns:
        cols = [c for c in pivot.columns if c != "ALL"] + ["ALL"]
        pivot = pivot[cols]
    pivot.columns.name = None
    pivot.index.name = "lang"
    header = f"=== MuST-SHE Pairwise Accuracy ({modality}) ===" if modality else "=== MuST-SHE Pairwise Accuracy ==="
    print(f"\n{header}")
    print(pivot.map(lambda x: f"{x:.3f}" if x == x else "—").to_string())
    overall = results[(results["lang"] == "ALL") & (results["category"] == "ALL")].iloc[0]
    print(f"Overall: {overall['pairwise_acc']:.3f}  (n={int(overall['n_pairs'])}, mean gap={overall['mean_score_gap']:.4f})\n")
