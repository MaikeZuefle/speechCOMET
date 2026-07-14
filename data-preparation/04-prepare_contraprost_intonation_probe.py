#!/usr/bin/env python3
"""Prepare ContraProST intonation data for probing.

Reads en_de.csv, en_es.csv, en_ja.csv from the ml-speech-is-more-than-words
dataset, extracts the Intonation subset, and writes stratified 80/20 train/dev
JSONL splits keyed on the English source audio.

Each CSV row has the same sentence recorded twice: prosody_1 is a statement
(trailing '.') and prosody_2 is a question (trailing '?'). The label is derived
from the trailing punctuation. Audio paths are identical across the three
language files, so the union collapses to 173 sentences × 2 = 346 unique
audio files with a perfectly balanced Statement/Question split.

Split strategy: sentences are split 80/20 at the sentence level so both
recordings of the same sentence always land in the same split. Since every
sentence contributes exactly one Statement and one Question audio, any
sentence-level split is inherently class-balanced.

Output fields per line:
  src        - English source sentence (without trailing punctuation)
  mt         - "" (dummy; required by probe_comet.py but unused for src_emb)
  src_audio  - absolute path to 16 kHz wav file
  intonation - Statement | Question
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


DATASET_ROOT = Path.home() / "datasets/ml-speech-is-more-than-words"
CSV_FILES = ["en_de.csv", "en_es.csv", "en_ja.csv"]

DEFAULT_TRAIN = Path("intonation_train.jsonl")
DEFAULT_DEV   = Path("intonation_dev.jsonl")


def load_unique_audio_rows(dataset_root: Path) -> list[dict]:
    """Read all three CSVs, derive Statement/Question labels, deduplicate by audio path.

    Returns a list of dicts with keys: sentence, src_audio, intonation.
    Deduplication keeps the first occurrence (de > es > ja).
    """
    seen: dict[str, dict] = {}  # audio_path -> row

    for csv_name in CSV_FILES:
        csv_path = dataset_root / "data" / csv_name
        df = pd.read_csv(csv_path)
        inton = df[df["category"] == "Intonation"]

        for _, row in inton.iterrows():
            for audio_col, prosody_col, trans_col in [
                ("audio_1", "prosody_1", "translation_1"),
                ("audio_2", "prosody_2", "translation_2"),
            ]:
                rel_path = row[audio_col]
                if rel_path in seen:
                    continue
                prosody_text = str(row[prosody_col]).strip()
                label = "Question" if prosody_text.endswith("?") else "Statement"
                seen[rel_path] = {
                    "sentence":   row["sentence"].strip(),
                    "src":        prosody_text,
                    "mt":         str(row[trans_col]).strip(),
                    "src_audio":  str((dataset_root / rel_path).resolve()),
                    "intonation": label,
                }

    return list(seen.values())


def sentence_split(rows: list[dict], train_frac: float, seed: int):
    """Split 80/20 at the sentence level.

    Both recordings of the same sentence land in the same split. Since every
    sentence contributes exactly one Statement and one Question, the split is
    class-balanced by construction.
    """
    rng = random.Random(seed)

    # Group audio rows by sentence
    from collections import defaultdict
    sent_to_rows: dict[str, list] = defaultdict(list)
    for row in rows:
        sent_to_rows[row["sentence"]].append(row)

    sentences = list(sent_to_rows.keys())
    rng.shuffle(sentences)
    n_train = max(1, round(len(sentences) * train_frac))

    train_rows = [r for s in sentences[:n_train] for r in sent_to_rows[s]]
    dev_rows   = [r for s in sentences[n_train:] for r in sent_to_rows[s]]

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)

    return train_rows, dev_rows


def to_jsonl_row(row: dict) -> dict:
    return {
        "src":        row["src"],
        "mt":         row["mt"],
        "src_audio":  row["src_audio"],
        "intonation": row["intonation"],
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonl_row(row), ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} lines → {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT),
                        help="Path to ml-speech-is-more-than-words root")
    parser.add_argument("--train-out", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev-out",   default=str(DEFAULT_DEV))
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Fraction of sentences assigned to train (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset_root)

    print("Loading CSVs and deriving intonation labels ...")
    all_rows = load_unique_audio_rows(root)
    print(f"  {len(all_rows)} unique audio files")

    from collections import Counter
    counts = Counter(r["intonation"] for r in all_rows)
    for label in ("Statement", "Question"):
        print(f"    {label:10s}: {counts[label]}")

    missing = [r for r in all_rows if not Path(r["src_audio"]).exists()]
    if missing:
        print(f"  WARNING: {len(missing)} audio files not found on disk")
        for r in missing[:5]:
            print(f"    {r['src_audio']}")

    print(f"\nSplitting (train_frac={args.train_frac}, seed={args.seed}) ...")
    train_rows, dev_rows = sentence_split(all_rows, args.train_frac, args.seed)

    train_counts = Counter(r["intonation"] for r in train_rows)
    dev_counts   = Counter(r["intonation"] for r in dev_rows)
    print(f"  Train: {len(train_rows)} examples  "
          f"(Statement={train_counts['Statement']}, Question={train_counts['Question']})")
    print(f"  Dev:   {len(dev_rows)} examples  "
          f"(Statement={dev_counts['Statement']}, Question={dev_counts['Question']})")

    write_jsonl(train_rows, Path(args.train_out))
    write_jsonl(dev_rows,   Path(args.dev_out))


if __name__ == "__main__":
    main()
