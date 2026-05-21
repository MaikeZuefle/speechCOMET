"""
Evaluate a SpeechCOMET model on MuST-SHE pairwise accuracy.

For each (audio, correct_translation, wrong_translation) triple, we check
whether the model scores the correct translation higher than the wrong one.
Reports pairwise accuracy and mean score gap per language and gender category.
"""
import argparse
import glob
import os
import re

import pandas as pd

from eval_utils import load_model, run_inference, pairwise_accuracy


WAV_DIR_RELATIVE = "../wav"


def load_csv_files(tsv_dir):
    """Load all MONOLINGUAL *.csv files, filter known-missing audio, and tag each row."""
    pattern = os.path.join(tsv_dir, "MONOLINGUAL.*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {tsv_dir}")

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

    df = pd.concat(frames, ignore_index=True)

    # Filter out known-missing audio files (documented in missing.txt)
    missing_txt = os.path.join(os.path.dirname(tsv_dir), "missing.txt")
    missing_mask = ~df["audio_path"].apply(os.path.exists)
    if missing_mask.any():
        missing_files = df.loc[missing_mask, "audio_path"].apply(os.path.basename).unique()
        print(f"\n*** WARNING: {missing_mask.sum()} rows ({len(missing_files)} wav files) MISSING ***")
        for lang, grp in df[missing_mask].groupby("lang"):
            print(f"  {lang}: {grp['audio_path'].nunique()} pairs missing")
        print(f"  Full list: {missing_txt}\n")
        df = df[~missing_mask].reset_index(drop=True)

    return df


def compute_results(df, score_col="model_score"):
    rows = []

    for (lang, cat), group in df.groupby(["lang", "category"]):
        acc, gap, n = pairwise_accuracy(group, score_col=score_col)
        rows.append({"lang": lang, "category": cat, "n_pairs": n,
                     "pairwise_acc": acc, "mean_score_gap": gap})

    for lang, group in df.groupby("lang"):
        acc, gap, n = pairwise_accuracy(group, score_col=score_col)
        rows.append({"lang": lang, "category": "ALL", "n_pairs": n,
                     "pairwise_acc": acc, "mean_score_gap": gap})

    acc, gap, n = pairwise_accuracy(df, score_col=score_col)
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


def main():
    parser = argparse.ArgumentParser(description="MuST-SHE pairwise accuracy evaluation")
    parser.add_argument("--mustshe-dir", required=True,
                        help="Path to MuST-SHE-v1.2-data/tsv/ directory")
    parser.add_argument("--model-folder", default=None,
                        help="Path to Lightning log directory (uses best epoch checkpoint)")
    parser.add_argument("--hf-model", default=None,
                        help="HuggingFace model repo id (e.g. maikezu/shetland)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save results CSV (defaults to model folder or hf model name)")
    parser.add_argument("--modality", type=str, default="audio",
                        choices=["audio", "text", "textaudio", "audiotext"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.hf_model and not args.model_folder:
        parser.error("Either --hf-model or --model-folder must be provided")

    model, output_dir = load_model(model_folder=args.model_folder, hf_model=args.hf_model)
    if args.output_dir:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading MuST-SHE CSV files from {args.mustshe_dir}")
    df = load_csv_files(args.mustshe_dir)
    df = run_inference(df, model, args.batch_size, args.modality)

    scores_path = os.path.join(output_dir, "mustshe_scores.csv")
    df.to_csv(scores_path, index=False)
    print(f"\nSaved raw scores to {scores_path}")

    results = compute_results(df)
    print_mustshe_pivot(results)

    results["pairwise_acc"] = results["pairwise_acc"].map(lambda x: f"{x:.3f}" if x == x else "nan")
    results["mean_score_gap"] = results["mean_score_gap"].map(lambda x: f"{x:.4f}" if x == x else "nan")

    results_path = os.path.join(output_dir, "mustshe_results.csv")
    results.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
