#!/usr/bin/env python
"""Prepare MuST-SHE data for speaker-gender probing.

Reads MONOLINGUAL.{es,fr,it}_v1.2.tsv, filters to category-1 He/She speakers,
deduplicates across language files (same underlying English audio segment can appear
in all three), and writes stratified 50/50 train/dev JSONL splits.

Deduplication key: (TALK, SRC, SPEAKER) — same talk + same English source text +
same speaker means the same audio recording. When duplicates exist, ES is preferred
over FR over IT so the chosen wav is deterministic.

Output fields per line:
  src        - English source text (SRC column)
  mt         - reference translation (from whichever language file was kept)
  src_audio  - absolute path to 16 kHz wav file
  gender     - speaker gender label: "He" or "She"
"""

import argparse
import csv
import json
import random
from pathlib import Path


DATASET_ROOT = Path.home() / "datasets/MuST-SHE_speechCOMET/MuST-SHE_v1.2/MuST-SHE-v1.2-data"
DEFAULT_WAV_DIR = DATASET_ROOT / "wav"
DEFAULT_TRAIN = Path("mustshe_train.jsonl")
DEFAULT_DEV = Path("mustshe_dev.jsonl")

# Preference order for deduplication: first match wins
TSV_FILES = [
    DATASET_ROOT / "tsv/MONOLINGUAL.es_v1.2.tsv",
    DATASET_ROOT / "tsv/MONOLINGUAL.fr_v1.2.tsv",
    DATASET_ROOT / "tsv/MONOLINGUAL.it_v1.2.tsv",
]


def load_all_tsv(tsv_paths: list, wav_dir: Path):
    seen = {}  # (TALK, SRC, SPEAKER) -> row dict, first seen wins
    skipped_wav = 0

    for tsv_path in tsv_paths:
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                gender = row["GENDER"].strip()
                category = row["CATEGORY"].strip()
                # Category 1 only: gender is recoverable from the utterance itself.
                # Keep only congruent cases: He+1M and She+1F.
                if gender == "He" and category != "1M":
                    continue
                if gender == "She" and category != "1F":
                    continue
                if gender not in ("He", "She"):
                    continue

                wav_path = wav_dir / f"{row['ID'].strip()}.wav"
                if not wav_path.exists():
                    skipped_wav += 1
                    continue

                key = (row["TALK"].strip(), row["SRC"].strip(), row["SPEAKER"].strip())
                if key in seen:
                    continue  # already have a wav for this segment; keep first (ES-preferred)

                seen[key] = {
                    "src": row["SRC"].strip(),
                    "mt": row["REF"].strip(),
                    "src_audio": str(wav_path.resolve()),
                    "gender": gender,
                }

    if skipped_wav:
        print(f"  Skipped {skipped_wav} rows with missing wav files.")
    return list(seen.values())


def stratified_split(rows, seed):
    rng = random.Random(seed)
    he = [r for r in rows if r["gender"] == "He"]
    she = [r for r in rows if r["gender"] == "She"]
    rng.shuffle(he)
    rng.shuffle(she)
    mid_he = len(he) // 2
    mid_she = len(she) // 2
    train = he[:mid_he] + she[:mid_she]
    dev = he[mid_he:] + she[mid_she:]
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def write_jsonl(rows, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} lines → {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-dir", default=str(DEFAULT_WAV_DIR), help="Directory containing wav files")
    parser.add_argument("--train-out", default=str(DEFAULT_TRAIN), help="Output path for train JSONL")
    parser.add_argument("--dev-out", default=str(DEFAULT_DEV), help="Output path for dev JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle and split")
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)

    print(f"Loading ES, FR, IT TSV files ...")
    rows = load_all_tsv(TSV_FILES, wav_dir)

    he_count = sum(1 for r in rows if r["gender"] == "He")
    she_count = sum(1 for r in rows if r["gender"] == "She")
    print(f"  {len(rows)} unique segments after deduplication (He={he_count}, She={she_count})")

    train, dev = stratified_split(rows, args.seed)
    train_he = sum(1 for r in train if r["gender"] == "He")
    train_she = sum(1 for r in train if r["gender"] == "She")
    dev_he = sum(1 for r in dev if r["gender"] == "He")
    dev_she = sum(1 for r in dev if r["gender"] == "She")
    print(f"  Train: {len(train)} (He={train_he}, She={train_she})")
    print(f"  Dev:   {len(dev)} (He={dev_he}, She={dev_she})")

    write_jsonl(train, Path(args.train_out))
    write_jsonl(dev, Path(args.dev_out))


if __name__ == "__main__":
    main()
