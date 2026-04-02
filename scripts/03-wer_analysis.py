"""
WER analysis for maikezu/scottish-metrics.
- dev split: human reference transcriptions (src_text)
- dev_asr split: Whisper ASR transcriptions (src_text)
Joins on (audio_path, doc_id), computes WER per example, saves CSV and plots distribution.
"""
import argparse
import os
import pandas as pd
from datasets import load_dataset
import jiwer
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ExpandCommonEnglishContractions
import numpy as np
import matplotlib.pyplot as plt


normalizer = Compose([
    ToLowerCase(),
    ExpandCommonEnglishContractions(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


def compute_wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return float("nan")
    return jiwer.wer(normalizer(reference), normalizer(hypothesis))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="maikezu/scottish-metrics")
    parser.add_argument("--output_plot", default="data/wer_distribution.png")
    parser.add_argument("--output_csv", default="data/wer_dev_asr.csv")
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    dev     = load_dataset(args.dataset, split="dev")
    dev_asr = load_dataset(args.dataset, split="dev_asr")
    print(f"  dev:     {len(dev)} examples, columns: {dev.column_names}")
    print(f"  dev_asr: {len(dev_asr)} examples, columns: {dev_asr.column_names}")

    # Build lookup: (audio_path, doc_id) -> human src_text
    # dev has one row per (audio_path, doc_id, tgt_system) but src_text is same for all
    ref_lookup = {(ex["audio_path"], ex["doc_id"]): ex["src_text"] for ex in dev}

    print("Computing WER...")
    rows, missing = [], 0
    for ex in dev_asr:
        key = (ex["audio_path"], ex["doc_id"])
        if key not in ref_lookup:
            missing += 1
            continue
        ref = ref_lookup[key]          # human transcription from dev
        hyp = ex["src_text"]           # whisper transcription (varies per tgt_system)
        rows.append({
            "audio_path": ex["audio_path"],
            "doc_id":     ex["doc_id"],
            "tgt_system": ex["tgt_system"],
            "tgt_lang":   ex["tgt_lang"],
            "wer": compute_wer(ref, hyp),
        })

    if missing:
        print(f"  Warning: {missing} dev_asr examples had no matching dev entry")

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"WER CSV saved to {args.output_csv} ({len(df)} rows)")

    valid = df["wer"].dropna().values

    print(f"\nWER statistics ({len(valid)} examples):")
    print(f"  Mean:    {valid.mean():.3f}")
    print(f"  Median:  {np.median(valid):.3f}")
    print(f"  Std:     {valid.std():.3f}")
    print(f"  Min:     {valid.min():.3f}")
    print(f"  Max:     {valid.max():.3f}")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  p{p:02d}:     {np.percentile(valid, p):.3f}")

    print("\nExamples above WER thresholds:")
    for t in [0.3, 0.5, 0.7, 1.0]:
        n = (valid > t).sum()
        print(f"  WER > {t:.1f}: {n} ({100*n/len(valid):.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(valid, bins=50, edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("WER")
    axes[0].set_ylabel("Count")
    axes[0].set_title("WER Distribution (dev_asr vs dev)")
    axes[0].axvline(np.median(valid), color="red", linestyle="--", label=f"Median={np.median(valid):.2f}")
    axes[0].axvline(valid.mean(), color="orange", linestyle="--", label=f"Mean={valid.mean():.2f}")
    axes[0].legend()

    axes[1].hist(valid, bins=50, edgecolor="white", linewidth=0.3, cumulative=True, density=True)
    axes[1].set_xlabel("WER")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("Cumulative WER Distribution")
    for t in [0.3, 0.5, 0.7]:
        axes[1].axvline(t, color="gray", linestyle=":", alpha=0.7, label=f"WER={t}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    print(f"Plot saved to {args.output_plot}")


if __name__ == "__main__":
    main()
