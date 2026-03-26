"""
Generate TTS audio for text_train.csv and text_dev.csv using Kokoro,
then push to HuggingFace as maikezu/ben_nevis_tts.

Columns match maikezu/iwslt2026-metrics-shared-train-dev:
  src_text, tgt_text, audio, score

train.csv → "train" split, dev.csv → "validation" split.
Within each split, TTS is deduplicated (same src → same audio).

Storage: parquet shards instead of individual WAVs to stay within
cluster file limits.
"""

import hashlib
import os

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, load_dataset
from kokoro import KPipeline
from tqdm import tqdm

# ── config ─────────────────────────────────────────────────────────────────────
TRAIN_CSV    = "data/train/text_train.csv"
DEV_CSV      = "data/dev/text_dev.csv"
HF_REPO      = "maikezu/ben_nevis_tts"
SHARD_DIR    = "data/tts_shards"
SAMPLE_RATE  = 24000
BATCH_SIZE   = 1000  # unique sources per shard

VOICES = {
    'a': [  # American English
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck",
    ],
    'b': [  # British English
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
}
ALL_VOICES = [(lang, v) for lang, vs in VOICES.items() for v in vs]


def pick_voice(src_text: str):
    h = int(hashlib.md5(src_text.encode()).hexdigest(), 16)
    return ALL_VOICES[h % len(ALL_VOICES)]


def load_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["src", "mt"])
    df = df[(df["src"].astype(str).str.strip() != "") & (df["mt"].astype(str).str.strip() != "")]
    df["src"] = df["src"].astype(str)
    df["mt"]  = df["mt"].astype(str)
    return df


def process_and_push(df, split_name, pipelines):
    shard_dir = os.path.join(SHARD_DIR, split_name)
    os.makedirs(shard_dir, exist_ok=True)

    # deduplicate TTS within this split
    src_to_rows = {}
    for _, row in df.iterrows():
        src_to_rows.setdefault(row["src"], []).append(row)

    unique_srcs = list(src_to_rows.keys())
    batches = [unique_srcs[i:i + BATCH_SIZE] for i in range(0, len(unique_srcs), BATCH_SIZE)]

    for shard_idx, batch_srcs in enumerate(tqdm(batches, desc=f"Shards ({split_name})")):
        path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.parquet")
        if os.path.exists(path):
            continue

        records = []
        for src_text in tqdm(batch_srcs, desc=f"  TTS", leave=False):
            lang, voice = pick_voice(src_text)
            chunks = [audio for _, _, audio in pipelines[lang](src_text, voice=voice)]
            if not chunks:
                print(f"  Warning: no audio for: {src_text[:80]!r}")
                continue
            audio_array = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

            for row in src_to_rows[src_text]:
                records.append({
                    "src_text": row["src"],
                    "tgt_text": row["mt"],
                    "score":    float(row["score"]),
                    "audio":    {"array": audio_array, "sampling_rate": SAMPLE_RATE},
                })

        shard = Dataset.from_list(records).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        shard.to_parquet(path)

    shard_files = sorted(
        os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".parquet")
    )
    dataset = load_dataset("parquet", data_files=shard_files)["train"]
    print(f"Pushing {len(dataset)} rows as split='{split_name}'...")
    dataset.push_to_hub(HF_REPO, split=split_name)


# ── main ───────────────────────────────────────────────────────────────────────
pipelines = {lang: KPipeline(lang_code=lang) for lang in VOICES}

train_df = load_csv(TRAIN_CSV)
dev_df   = load_csv(DEV_CSV)
print(f"Loaded {len(train_df)} train + {len(dev_df)} dev rows")

process_and_push(train_df, "train",      pipelines)
process_and_push(dev_df,   "validation", pipelines)

print(f"Done! https://huggingface.co/datasets/{HF_REPO}")
