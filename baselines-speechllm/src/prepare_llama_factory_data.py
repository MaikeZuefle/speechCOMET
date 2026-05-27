"""
Convert maikezu/iwslt2026-metrics-shared-train-dev to LlamaFactory sharegpt format.

Creates JSON files in data/llama_factory/:
  iwslt26_{modality}.json       -- train (train + train_synthetic)
  iwslt26_{modality}_dev.json   -- validation (dev split)

Modalities: text, audio, textaudio

Audio is decoded from HF bytes via soundfile (no torchcodec) and saved to
  data/llama_factory/audio/{split}_{idx}.wav

Run from baselines-speechllm/:
  python src/prepare_llama_factory_data.py
  python src/prepare_llama_factory_data.py --limit 100  # quick test
"""
import argparse
import io
import json
import os

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are an evaluator. Given the source and/or audio and a translation, "
    "respond with only a single float score between 0 and 1 indicating translation quality. "
    "Output nothing else."
)
OUT_DIR = "data/llama_factory"
AUDIO_DIR = os.path.join(OUT_DIR, "audio")

TRAIN_SPLITS = ["train"]
DEV_SPLIT = "dev"


def decode_audio_bytes(audio_bytes: bytes):
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    return data, sr


def get_audio_bytes(audio_obj):
    if isinstance(audio_obj, dict):
        return audio_obj.get("bytes")
    if hasattr(audio_obj, "_hf_encoded"):
        return getattr(audio_obj, "_hf_encoded", {}).get("bytes")
    return None


def build_entry_text(src_text, mt_text, score_norm):
    return {
        "conversations": [
            {"from": "human", "value": f"Source: {src_text}\nTranslation: {mt_text}"},
            {"from": "gpt",   "value": f"{score_norm:.4f}"},
        ]
    }


def build_entry_audio(audio_path, mt_text, score_norm):
    return {
        "conversations": [
            {"from": "human", "value": f"<audio>Translation: {mt_text}"},
            {"from": "gpt",   "value": f"{score_norm:.4f}"},
        ],
        "audios": [audio_path],
    }


def build_entry_textaudio(audio_path, src_text, mt_text, score_norm):
    return {
        "conversations": [
            {"from": "human", "value": f"<audio>Source: {src_text}\nTranslation: {mt_text}"},
            {"from": "gpt",    "value": f"{score_norm:.4f}"},
        ],
        "audios": [audio_path],
    }


def process_split(split_name: str, limit: int | None = None):
    print(f"\n  Loading {split_name}...")
    ds = load_dataset("maikezu/iwslt2026-metrics-shared-train-dev", split=split_name)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  {len(ds)} examples")

    text_entries, audio_entries, textaudio_entries = [], [], []
    skipped_audio = 0

    for idx, ex in enumerate(tqdm(ds, desc=f"  {split_name}", leave=False)):
        src_text  = ex["src_text"]
        mt_text   = ex["tgt_text"]
        score     = ex["score"] / 100.0

        text_entries.append(build_entry_text(src_text, mt_text, score))

        audio_bytes = get_audio_bytes(ex["audio"])
        if audio_bytes:
            try:
                wav_name = f"{split_name}_{idx}.wav"
                wav_path = os.path.join(AUDIO_DIR, wav_name)
                if not os.path.exists(wav_path):
                    data, sr = decode_audio_bytes(audio_bytes)
                    sf.write(wav_path, data, sr)
                rel_path = f"data/llama_factory/audio/{wav_name}"
                audio_entries.append(build_entry_audio(rel_path, mt_text, score))
                textaudio_entries.append(build_entry_textaudio(rel_path, src_text, mt_text, score))
            except Exception as e:
                print(f"\n  WARNING: audio error at {split_name}_{idx}: {e}")
                skipped_audio += 1
        else:
            skipped_audio += 1

    if skipped_audio:
        print(f"  *** {skipped_audio}/{len(ds)} examples missing audio — excluded from audio/textaudio sets ***")

    return text_entries, audio_entries, textaudio_entries


def write_json(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(entries):6d} examples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per split (for quick testing)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # --- training splits ---
    print("=== Training data ===")
    train_text, train_audio, train_textaudio = [], [], []
    for split in TRAIN_SPLITS:
        t, a, ta = process_split(split, args.limit)
        train_text.extend(t)
        train_audio.extend(a)
        train_textaudio.extend(ta)

    print("\nWriting training JSONs...")
    write_json(train_text,      os.path.join(OUT_DIR, "iwslt26_text.json"))
    write_json(train_audio,     os.path.join(OUT_DIR, "iwslt26_audio.json"))
    write_json(train_textaudio, os.path.join(OUT_DIR, "iwslt26_textaudio.json"))

    # --- validation split ---
    print("\n=== Validation data ===")
    dev_text, dev_audio, dev_textaudio = process_split(DEV_SPLIT, args.limit)

    print("\nWriting validation JSONs...")
    write_json(dev_text,      os.path.join(OUT_DIR, "iwslt26_text_dev.json"))
    write_json(dev_audio,     os.path.join(OUT_DIR, "iwslt26_audio_dev.json"))
    write_json(dev_textaudio, os.path.join(OUT_DIR, "iwslt26_textaudio_dev.json"))

    print("\nDone.")


if __name__ == "__main__":
    main()
