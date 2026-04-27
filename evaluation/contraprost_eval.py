"""
Evaluate a SpeechCOMET model on ContraProST pairwise accuracy.

For each (audio, correct_translation, wrong_translation) pair, we check
whether the model scores the correct translation higher than the wrong one.
Reports pairwise accuracy and mean score gap per language pair, prosodic
category, and overall.
"""
import argparse
import os

import pandas as pd

from eval_utils import load_model, run_inference, pairwise_accuracy, load_contraprost_csv_files
from collect_contraprost_results import refresh_combined_csv


def compute_results(df, score_col="model_score"):
    # Same audio files appear across languages — use (lang, src_audio) as composite key
    # for any cross-language groupings to avoid duplicate key collisions.
    df = df.copy()
    df["_key"] = df["lang"] + "||" + df["src_audio"]

    rows = []

    for (lang, cat), group in df.groupby(["lang", "category"]):
        acc, gap, n = pairwise_accuracy(group, key_col="src_audio", score_col=score_col)
        rows.append({"lang": lang, "category": cat, "n_pairs": n,
                     "pairwise_acc": acc, "mean_score_gap": gap})

    for lang, group in df.groupby("lang"):
        acc, gap, n = pairwise_accuracy(group, key_col="src_audio", score_col=score_col)
        rows.append({"lang": lang, "category": "ALL", "n_pairs": n,
                     "pairwise_acc": acc, "mean_score_gap": gap})

    for cat, group in df.groupby("category"):
        acc, gap, n = pairwise_accuracy(group, key_col="_key", score_col=score_col)
        rows.append({"lang": "ALL", "category": cat, "n_pairs": n,
                     "pairwise_acc": acc, "mean_score_gap": gap})

    acc, gap, n = pairwise_accuracy(df, key_col="_key", score_col=score_col)
    rows.append({"lang": "ALL", "category": "ALL", "n_pairs": n,
                 "pairwise_acc": acc, "mean_score_gap": gap})

    return pd.DataFrame(rows).sort_values(["lang", "category"])


def print_contraprost_results(results: pd.DataFrame, modality: str = ""):
    """Print pairwise accuracy per language and per prosodic category."""
    header = f"=== ContraProST Pairwise Accuracy ({modality}) ===" if modality else "=== ContraProST Pairwise Accuracy ==="
    print(f"\n{header}")
    lang_rows = results[(results["lang"] != "ALL") & (results["category"] == "ALL")]
    for _, row in lang_rows.iterrows():
        print(f"  {row['lang']:6s}  acc={row['pairwise_acc']:.3f}  gap={row['mean_score_gap']:.4f}  (n={row['n_pairs']})")
    overall = results[(results["lang"] == "ALL") & (results["category"] == "ALL")].iloc[0]
    print(f"  {'ALL':6s}  acc={overall['pairwise_acc']:.3f}  gap={overall['mean_score_gap']:.4f}  (n={overall['n_pairs']})")
    print(f"\n  {'Category':<30s}  acc    gap")
    cat_rows = results[(results["lang"] == "ALL") & (results["category"] != "ALL")]
    for _, row in cat_rows.iterrows():
        print(f"  {row['category']:<30s}  {row['pairwise_acc']:.3f}  {row['mean_score_gap']:.4f}  (n={row['n_pairs']})")
    print()


def main():
    parser = argparse.ArgumentParser(description="ContraProST pairwise accuracy evaluation")
    parser.add_argument("--data-dir", default="data/contraProST",
                        help="Directory containing en_*_expanded.csv files")
    parser.add_argument("--model-folder", default=None,
                        help="Path to Lightning log directory (uses best epoch checkpoint)")
    parser.add_argument("--hf-model", default=None,
                        help="HuggingFace model repo id (e.g. maikezu/shetland)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save results CSV (defaults to model folder)")
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

    print(f"\nLoading ContraProST CSV files from {args.data_dir}")
    df = load_contraprost_csv_files(args.data_dir)

    df = run_inference(df, model, args.batch_size, args.modality, audio_col="src_audio")

    scores_path = os.path.join(output_dir, "contraprost_scores.csv")
    df.to_csv(scores_path, index=False)
    print(f"\nSaved raw scores to {scores_path}")

    results = compute_results(df)
    print_contraprost_results(results)

    results["pairwise_acc"]   = results["pairwise_acc"].map(lambda x: f"{x:.3f}" if x == x else "nan")
    results["mean_score_gap"] = results["mean_score_gap"].map(lambda x: f"{x:.4f}" if x == x else "nan")

    results_path = os.path.join(output_dir, "contraprost_results.csv")
    results.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    refresh_combined_csv()


if __name__ == "__main__":
    main()
