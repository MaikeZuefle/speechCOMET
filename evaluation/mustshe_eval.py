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
from collections import defaultdict

import pandas as pd
import speechcomet
from speechcomet import download_model
from tqdm import tqdm


WAV_DIR_RELATIVE = "../wav"  # relative to the tsv/ directory


def load_csv_files(tsv_dir):
    """Load all MONOLINGUAL *.csv files and tag each row with lang and category."""
    pattern = os.path.join(tsv_dir, "MONOLINGUAL.*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {tsv_dir}")

    frames = []
    for path in csv_files:
        basename = os.path.basename(path)
        # Extract language and category from filename, e.g. MONOLINGUAL.es_v1.2.tsv.1F.csv
        # Skip files without a category suffix — they are duplicates of the 1F file
        m = re.match(r"MONOLINGUAL\.([a-z]+)_v[\d.]+\.tsv\.([12][FM])\.csv", basename)
        if not m:
            print(f"Skipping {basename} (no category suffix)")
            continue
        lang = m.group(1)
        category = m.group(2)
        df = pd.read_csv(path)
        df["lang"] = lang
        df["category"] = category
        # Resolve audio paths relative to tsv_dir
        wav_dir = os.path.abspath(os.path.join(tsv_dir, WAV_DIR_RELATIVE))
        df["audio_path"] = df["src_audio"].apply(
            lambda p: os.path.join(wav_dir, os.path.basename(p))
        )
        frames.append(df)
        print(f"  Loaded {len(df):4d} rows from {basename}  (lang={lang}, category={category})")

    return pd.concat(frames, ignore_index=True)


def build_sample(row, modality):
    if modality == "audio":
        return {"src_audio": row["audio_path"], "mt": row["mt"]}
    elif modality == "text":
        return {"src": row["src"], "mt": row["mt"]}
    elif modality in ("textaudio", "audiotext"):
        return {"src_audio": row["audio_path"], "src": row["src"], "mt": row["mt"]}
    else:
        raise ValueError(f"Unknown modality: {modality}")


def run_inference(df, model, batch_size, modality):
    """Run model inference on all rows and add a 'model_score' column."""
    if modality != "text":
        missing_mask = ~df["audio_path"].apply(os.path.exists)
        if missing_mask.any():
            missing_files = df.loc[missing_mask, "audio_path"].apply(os.path.basename).unique()
            print(f"\n*** WARNING: {missing_mask.sum()} rows ({len(missing_files)} wav files) missing — SKIPPED ***")
            for lang, grp in df[missing_mask].groupby("lang"):
                n_pairs = grp["audio_path"].nunique()
                print(f"  {lang}: {n_pairs} pairs skipped")
            print(f"  See data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/missing.txt for full list\n")
            df = df[~missing_mask].reset_index(drop=True)
    samples = [build_sample(row, modality) for _, row in df.iterrows()]
    print(f"\nRunning inference on {len(samples)} samples...")
    result = model.predict(samples=samples, gpus=1, num_workers=0, batch_size=batch_size)
    df = df.copy()
    df["model_score"] = result.scores
    return df


def pairwise_accuracy(df):
    """
    For each audio file compute whether model_score(score=100) > model_score(score=0).
    Returns pairwise accuracy (0–1) and mean score gap.
    """
    correct_rows = df[df["score"] == 100].set_index("audio_path")
    wrong_rows   = df[df["score"] == 0  ].set_index("audio_path")
    shared = correct_rows.index.intersection(wrong_rows.index)

    if len(shared) == 0:
        return float("nan"), float("nan"), 0

    correct_scores = correct_rows.loc[shared, "model_score"]
    wrong_scores   = wrong_rows.loc[shared, "model_score"]
    wins = (correct_scores.values > wrong_scores.values).sum()
    gap  = (correct_scores.values - wrong_scores.values).mean()
    return wins / len(shared), gap, len(shared)


def compute_results(df):
    rows = []

    # Per language × category
    for (lang, cat), group in df.groupby(["lang", "category"]):
        acc, gap, n = pairwise_accuracy(group)
        rows.append({"lang": lang, "category": cat, "n_pairs": n,
                     "pairwise_acc": acc, "mean_score_gap": gap})

    # Per language (across categories)
    for lang, group in df.groupby("lang"):
        acc, gap, n = pairwise_accuracy(group)
        rows.append({"lang": lang, "category": "ALL", "n_pairs": n,
                     "pairwise_acc": acc, "mean_score_gap": gap})

    # Overall
    acc, gap, n = pairwise_accuracy(df)
    rows.append({"lang": "ALL", "category": "ALL", "n_pairs": n,
                 "pairwise_acc": acc, "mean_score_gap": gap})

    return pd.DataFrame(rows).sort_values(["lang", "category"])


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
                        choices=["audio", "text", "textaudio", "audiotext"],
                        help="Input modality (default: audio)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.hf_model and not args.model_folder:
        parser.error("Either --hf-model or --model-folder must be provided")

    # Load model
    if args.hf_model:
        model = speechcomet.load_from_checkpoint(download_model(args.hf_model))
        output_dir = args.output_dir or args.hf_model.replace("/", "_")
    else:
        ckpt_dir = os.path.join(args.model_folder, "checkpoints")
        matches = glob.glob(os.path.join(ckpt_dir, "epoch=*-*.ckpt"))
        checkpoint = max(
            matches,
            key=lambda p: int(os.path.basename(p).split("epoch=")[1].split("-")[0])
        )
        print(f"Loading checkpoint: {checkpoint}")
        model = speechcomet.load_from_checkpoint(checkpoint)
        output_dir = args.output_dir or args.model_folder

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading MuST-SHE CSV files from {args.mustshe_dir}")
    df = load_csv_files(args.mustshe_dir)

    df = run_inference(df, model, args.batch_size, args.modality)

    # Save raw scores
    scores_path = os.path.join(output_dir, "mustshe_scores.csv")
    df.to_csv(scores_path, index=False)
    print(f"\nSaved raw scores to {scores_path}")

    # Compute and print results
    results = compute_results(df)

    # Pivot to lang × category for easy reading
    pivot = results[results["lang"] != "ALL"].pivot(
        index="lang", columns="category", values="pairwise_acc"
    ).rename(columns={"1F": "1F (female)", "1M": "1M (male)", "ALL": "overall"})
    pivot.index.name = "lang"

    print("\n=== MuST-SHE Pairwise Accuracy ===")
    print(pivot.map(lambda x: f"{x:.3f}").to_string())
    overall_acc, overall_gap, overall_n = pairwise_accuracy(df)
    print(f"\nOverall: {overall_acc:.3f}  (n={overall_n}, mean gap={overall_gap:.4f})")

    results["pairwise_acc"] = results["pairwise_acc"].map(lambda x: f"{x:.3f}" if x == x else "nan")
    results["mean_score_gap"] = results["mean_score_gap"].map(lambda x: f"{x:.4f}" if x == x else "nan")

    results_path = os.path.join(output_dir, "mustshe_results.csv")
    results.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
