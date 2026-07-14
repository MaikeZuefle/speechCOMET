#!/usr/bin/env python3
"""Prepare ContraProST emotional-prosody data for probing.

Reads en_de.csv, en_es.csv, en_ja.csv from the ml-speech-is-more-than-words
dataset, extracts the Emotional Prosody subset, and writes stratified 80/20
train/dev JSONL splits keyed on the English source audio.

Each CSV row has two audio files with different emotional prosodies (audio_1/
audio_2), encoded as subcategory "EmotionA-EmotionB".  This script unpacks
each row into two (audio, emotion) examples, deduplicates by audio path across
the three language files, filters to the four most common emotion classes
(Neutral, Angry, Happy, Surprised), and splits at the sentence level so both
audio recordings of the same sentence always land in the same split.

Split stratification: sentences are grouped by their pair subcategory
(e.g. "Neutral-Angry") and the 80/20 split is applied within each group.

Output fields per line:
  src        - English source sentence
  mt         - "" (dummy; required by probe_comet.py but unused for src_emb)
  src_audio  - absolute path to 16 kHz wav file
  emotion    - Neutral | Angry | Happy | Surprised
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


DATASET_ROOT = Path.home() / "datasets/ml-speech-is-more-than-words"
CSV_FILES = ["en_de.csv", "en_es.csv", "en_ja.csv"]
KEEP_EMOTIONS = {"Neutral", "Angry", "Happy", "Surprised"}

DEFAULT_TRAIN = Path("contraprost_train.jsonl")
DEFAULT_DEV   = Path("contraprost_dev.jsonl")


def load_unique_audio_rows(dataset_root: Path) -> list[dict]:
    """Read all three CSVs, unpack emotion pairs, deduplicate by audio path.

    Returns a list of dicts with keys: sentence, subcat, src_audio, emotion.
    Deduplication keeps the first occurrence (de > es > ja).
    """
    seen: dict[str, dict] = {}  # audio_path -> row

    for csv_name in CSV_FILES:
        csv_path = dataset_root / "data" / csv_name
        df = pd.read_csv(csv_path)
        ep = df[df["category"] == "Emotional Prosody"].copy()

        emo1 = ep["subcategory"].str.split("-").str[0]
        emo2 = ep["subcategory"].str.split("-").str[1]

        for _, row in ep.iterrows():
            for audio_col, prosody_col, trans_col, emo in [
                ("audio_1", "prosody_1", "translation_1", emo1[row.name]),
                ("audio_2", "prosody_2", "translation_2", emo2[row.name]),
            ]:
                rel_path = row[audio_col]
                if rel_path in seen:
                    continue
                seen[rel_path] = {
                    "sentence": row["sentence"],
                    "subcat":   row["subcategory"],
                    "src":      str(row[prosody_col]).strip(),
                    "mt":       str(row[trans_col]).strip(),
                    "src_audio": str((dataset_root / rel_path).resolve()),
                    "emotion":   emo,
                }

    return list(seen.values())


def stratified_sentence_split(rows: list[dict], train_frac: float, seed: int):
    """Split at sentence level, stratified by subcategory pair.

    Both audio recordings of the same sentence land in the same split.
    Within each subcategory group, shuffles sentences then takes train_frac
    for train and the remainder for dev.
    """
    rng = random.Random(seed)

    # Group audio rows by sentence
    sent_to_rows: dict[str, list] = defaultdict(list)
    for row in rows:
        sent_to_rows[row["sentence"]].append(row)

    # Group sentences by their pair subcategory
    subcat_to_sents: dict[str, list] = defaultdict(list)
    for sent, sent_rows in sent_to_rows.items():
        subcat_to_sents[sent_rows[0]["subcat"]].append(sent)

    train_sents: set[str] = set()
    dev_sents:   set[str] = set()

    for subcat, sents in subcat_to_sents.items():
        rng.shuffle(sents)
        n_train = max(1, round(len(sents) * train_frac))
        train_sents.update(sents[:n_train])
        dev_sents.update(sents[n_train:])

    train_rows = [r for s in train_sents for r in sent_to_rows[s]]
    dev_rows   = [r for s in dev_sents   for r in sent_to_rows[s]]

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)

    return train_rows, dev_rows


def to_jsonl_row(row: dict) -> dict:
    return {
        "src":       row["src"],
        "mt":        row["mt"],
        "src_audio": row["src_audio"],
        "emotion":   row["emotion"],
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

    print("Loading CSVs and unpacking emotion pairs ...")
    all_rows = load_unique_audio_rows(root)
    print(f"  {len(all_rows)} unique audio files before emotion filter")

    filtered = [r for r in all_rows if r["emotion"] in KEEP_EMOTIONS]
    print(f"  {len(filtered)} after filtering to {sorted(KEEP_EMOTIONS)}")

    # Verify audio files exist and report any missing
    missing = [r for r in filtered if not Path(r["src_audio"]).exists()]
    if missing:
        print(f"  WARNING: {len(missing)} audio files not found on disk:")
        for r in missing[:5]:
            print(f"    {r['src_audio']}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    empty = [r for r in filtered if Path(r["src_audio"]).exists() and Path(r["src_audio"]).stat().st_size == 0]
    if empty:
        print(f"  WARNING: {len(empty)} empty (0-byte) audio files removed:")
        for r in empty:
            print(f"    {r['src_audio']}")
    filtered = [r for r in filtered if Path(r["src_audio"]).stat().st_size > 0]

    # Per-emotion counts
    from collections import Counter
    counts = Counter(r["emotion"] for r in filtered)
    print("  Emotion distribution:")
    for emo in sorted(KEEP_EMOTIONS):
        print(f"    {emo:12s}: {counts[emo]}")

    print(f"\nSplitting (train_frac={args.train_frac}, seed={args.seed}) ...")
    train_rows, dev_rows = stratified_sentence_split(filtered, args.train_frac, args.seed)

    train_counts = Counter(r["emotion"] for r in train_rows)
    dev_counts   = Counter(r["emotion"] for r in dev_rows)
    print(f"  Train: {len(train_rows)} examples")
    for emo in sorted(KEEP_EMOTIONS):
        print(f"    {emo:12s}: {train_counts[emo]}")
    print(f"  Dev:   {len(dev_rows)} examples")
    for emo in sorted(KEEP_EMOTIONS):
        print(f"    {emo:12s}: {dev_counts[emo]}")

    write_jsonl(train_rows, Path(args.train_out))
    write_jsonl(dev_rows,   Path(args.dev_out))


if __name__ == "__main__":
    main()
